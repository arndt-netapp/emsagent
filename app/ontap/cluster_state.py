"""Live cluster-state reads for the Stage 2 agent.

Where `client.py` pulls the EMS event *history*, this pulls what the cluster
looks like *right now* — the thing that turns "these events fired" into "these
events fired and the aggregate is at 94%".

Two rules hold throughout, both inherited from tools.py's clamping discipline,
and any new area added here must obey both:

1. **Every function returns an already-summarized structure, never a raw REST
   payload.** A cluster with 600 disks would otherwise put 600 JSON objects
   into the model's context, at full price, riding along for every remaining
   turn — the exact hole MAX_TOOL_RESULT_EVENTS was added to close. Counts and
   outliers are what a diagnosis needs; the full inventory is not. Note this is
   about what is *returned*, not what is *fetched*: the reads below walk every
   page so their counts are true, then summarize what they found.
2. **Nothing here raises on an unreachable or unconfigured cluster.** These
   calls happen mid-investigation; a dead cluster must degrade the evidence
   available to the agent, not kill a run the user has already paid for. All
   failures come back as {"available": False, "reason": ...} for the model to
   read and work around.

Field access is defensive throughout (nested .get chains, no KeyError paths):
these responses vary across ONTAP versions, and a missing nested key must
degrade one number rather than raise inside an investigation the user has
already paid for.

**Live performance telemetry is deliberately out of scope.** There is no
latency, IOPS or throughput number anywhere in this module, and the prompts are
written not to promise one. Answering "is this actually slow?" needs historical
trending over a time-series backend, which is a different system rather than a
bigger version of this one — NetApp Harvest and its MCP server
(https://netapp.github.io/harvest/nightly/mcp/overview/) already do it against
Prometheus/VictoriaMetrics. A `performance_issue` candidate can be
characterized from the EMS events, but not confirmed against current load.

**Deliberately unimplemented, and the obvious next areas** if someone extends
this PoC — each absent by scope decision, not oversight:
  - environmental sensors (`/cluster/sensors`), which reports `threshold_state`
    against warning/critical thresholds; the closest thing ONTAP offers to a
    purpose-built corroborator for `predictive_failure` on shelves/PSUs/thermals.
  - SnapMirror (`/snapmirror/relationships`: `healthy`, `lag_time`,
    `unhealthy_reason`), for the protection events EMS emits constantly.
  - network ports (`/network/ethernet/ports`), for link-flap events.
Adding one is a new `area` value in tools.py's `get_cluster_state` — cheap,
because tool *schemas* ride in every turn's request but enum values barely do.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

from app.config import settings
from app.ontap.client import OntapClientError, _new_session, validate_host

logger = logging.getLogger(__name__)

# Hard ceiling on how many individual records any summary will name
# explicitly, however many the cluster reports.
MAX_LISTED_ITEMS = 15

# Records requested per page. ONTAP treats max_records as a per-response
# ceiling, not a total, so this sets how many round trips a walk makes rather
# than how much it returns — same meaning as client.py's `page_size`.
PAGE_SIZE = 500

# Disk states that mean "this disk is a problem", per the ONTAP disk state
# enum. Everything else (present, spare, zeroing, ...) is normal operation.
UNHEALTHY_DISK_STATES = {"broken", "maintenance", "pending", "reconstructing", "removed", "unfail"}


def credentials_configured() -> bool:
    """Whether cluster credentials are available at all.

    Note this says nothing about WHICH cluster — the host is never
    configuration. It comes from the analysis run's scope, i.e. the cluster the
    events under investigation were actually fetched from."""
    return bool(settings.ontap_user and settings.ontap_password)


def _unavailable(reason: str) -> Dict[str, Any]:
    return {"available": False, "reason": reason}


def _with_incomplete(result: Dict[str, Any], incomplete: Optional[str]) -> Dict[str, Any]:
    """Attach a partial-walk note, if there was one.

    A partial result is still `available: True` — the agent got real evidence,
    just not all of it. The key is present only when it applies, so a complete
    read stays clean."""
    if incomplete:
        result["incomplete_reason"] = incomplete
    return result


def _authenticated_session() -> requests.Session:
    """One session for a whole walk, using the single shared credential pair
    (see settings.ontap_user).

    Built once per walk rather than once per request: a multi-page walk that
    rebuilt this each time would discard keep-alive and re-handshake TLS on
    every page, and `_new_session` mounts the Retry policy that exists
    precisely to survive a long walk (see client.py:_retry_policy)."""
    if not settings.ontap_user or not settings.ontap_password:
        raise OntapClientError("ONTAP_USER / ONTAP_PASSWORD are not configured")
    return _new_session(settings.ontap_user, settings.ontap_password, settings.ontap_verify_tls)


def _get_url(
    session: requests.Session, url: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """One authenticated GET. Raises OntapClientError; callers convert that
    into an `available: False` result or a partial one."""
    try:
        response = session.get(url, params=params, timeout=settings.ontap_timeout_seconds)
    except requests.RequestException as exc:
        raise OntapClientError(f"GET {url} failed: {exc}")
    if response.status_code != 200:
        raise OntapClientError(f"GET {url} failed: {response.status_code} {response.text[:200]}")
    try:
        return response.json()
    except ValueError as exc:
        raise OntapClientError(f"GET {url} returned non-JSON: {exc}")


def _get_all(
    host: str, path: str, params: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Walk every page of a collection endpoint, following `_links.next`.

    Returns (records, incomplete_reason). `incomplete_reason` is None on a
    complete walk and a human-readable string when the walk stopped early —
    the caller reports it so the model reads the counts as a floor rather than
    a total.

    Three properties, each load-bearing:

    - **A failure part-way keeps what already arrived.** Losing 6,000 fetched
      volumes because page 13 of 20 timed out would make a big cluster strictly
      worse than a small one, on exactly the clusters that need paging at all.
      This mirrors routes_clusters, which keeps the events a died-part-way
      fetch had already collected.
    - **A failure on the FIRST page is still an error**, and raises. Nothing
      arrived, so there is no partial answer to report — same rule as an event
      fetch that got nothing, and it keeps a dead or unauthenticated cluster
      reported as `available: False` rather than as an empty cluster.
    - **The walk cannot spin.** client.py's fetch can follow `next` blindly
      because `count` bounds it; this walk is deliberately unbounded, so a
      cluster that returns a repeating pagination link would loop forever
      inside a tool call, with the Stage 2 cost cap powerless to stop it (it
      bounds turns, not wall-clock). Visited URLs are tracked and a repeat
      ends the walk."""
    session = _authenticated_session()
    payload = _get_url(session, f"https://{host}{path}", params)
    records: List[Dict[str, Any]] = list(payload.get("records") or [])

    visited = set()
    while True:
        next_href = ((payload.get("_links") or {}).get("next") or {}).get("href")
        if not next_href:
            return records, None
        next_url = urljoin(f"https://{host}/", next_href.lstrip("/"))
        if next_url in visited:
            logger.warning("pagination link repeated at %s; ending walk", next_url)
            return records, (
                f"stopped after {len(records)} records: the cluster returned a repeating "
                "pagination link, so this is a partial list"
            )
        visited.add(next_url)
        try:
            payload = _get_url(session, next_url)
        except OntapClientError as exc:
            logger.warning("paginated walk of %s stopped early: %s", path, exc)
            return records, (
                f"stopped after {len(records)} records: {exc}. Treat the counts below as a "
                "lower bound, not a total"
            )
        records.extend(payload.get("records") or [])


def _require_host(host: Optional[str]) -> str:
    """The host is always supplied by the caller, from the run's cluster scope.

    There is deliberately no fallback to a configured default: a default would
    be used for every investigation regardless of which cluster the evidence
    came from, so with two clusters registered an investigation of cluster B
    would silently report cluster A's state as corroboration."""
    if not host:
        raise OntapClientError(
            "no cluster host for this investigation — its events came from an uploaded log "
            "file rather than a cluster fetch, so there is no live cluster to query"
        )
    # The host reaches here from data (a run's scope_cluster, ultimately the
    # upload form's free-text cluster field), and this module attaches the
    # shared ONTAP credentials to whatever it is handed. Validate before any
    # request is built, not after.
    return validate_host(host)


def _pct(used: Optional[int], total: Optional[int]) -> Optional[float]:
    # `used is None`, not `not used`: zero bytes used is a real, knowable 0.0%,
    # and reporting the emptiest aggregate in the cluster as "usage unknown"
    # (which then sorts as -1 in both callers) is a worse answer than the
    # truth. A zero or missing `total` genuinely has no percentage.
    if used is None or not total:
        return None
    return round(100.0 * used / total, 1)


def get_aggregate_capacity(host: Optional[str], node: Optional[str] = None) -> Dict[str, Any]:
    """Per-aggregate space usage, sorted fullest first. The single most useful
    corroborating signal for a volume/space EMS event."""
    try:
        host = _require_host(host)
        params: Dict[str, Any] = {
            "fields": "name,node.name,state,space.block_storage",
            "max_records": PAGE_SIZE,
        }
        if node:
            params["node.name"] = node
        records, incomplete = _get_all(host, "/api/storage/aggregates", params)
    except OntapClientError as exc:
        return _unavailable(str(exc))

    aggregates = []
    for record in records:
        block = (record.get("space") or {}).get("block_storage") or {}
        size, used = block.get("size"), block.get("used")
        aggregates.append(
            {
                "name": record.get("name"),
                "node": (record.get("node") or {}).get("name"),
                "state": record.get("state"),
                "size_bytes": size,
                "used_bytes": used,
                "used_pct": _pct(used, size),
            }
        )
    aggregates.sort(key=lambda a: a["used_pct"] if a["used_pct"] is not None else -1, reverse=True)
    return _with_incomplete(
        {
            "available": True,
            "cluster": host,
            "aggregate_count": len(aggregates),
            "listed": min(len(aggregates), MAX_LISTED_ITEMS),
            "aggregates": aggregates[:MAX_LISTED_ITEMS],
        },
        incomplete,
    )


def get_disk_health(host: Optional[str], node: Optional[str] = None) -> Dict[str, Any]:
    """Disk inventory reduced to a state histogram plus the individual disks in
    a bad state. A healthy 600-disk cluster comes back as one small histogram
    and an empty problem list."""
    try:
        host = _require_host(host)
        params: Dict[str, Any] = {
            "fields": "name,state,node.name,container_type,model,bay",
            "max_records": PAGE_SIZE,
        }
        if node:
            params["node.name"] = node
        records, incomplete = _get_all(host, "/api/storage/disks", params)
    except OntapClientError as exc:
        return _unavailable(str(exc))

    by_state: Dict[str, int] = {}
    unhealthy: List[Dict[str, Any]] = []
    for record in records:
        state = record.get("state") or "unknown"
        by_state[state] = by_state.get(state, 0) + 1
        if state in UNHEALTHY_DISK_STATES:
            unhealthy.append(
                {
                    "name": record.get("name"),
                    "state": state,
                    "node": (record.get("node") or {}).get("name"),
                    "model": record.get("model"),
                    "bay": record.get("bay"),
                    "container_type": record.get("container_type"),
                }
            )
    return _with_incomplete(
        {
            "available": True,
            "cluster": host,
            "disk_count": len(records),
            "by_state": by_state,
            # unhealthy_count is the true number of bad disks; unhealthy_disks
            # is the trimmed list. Reporting both is what stops the model
            # reading a 15-item list as "there are 15".
            "unhealthy_count": len(unhealthy),
            "listed": min(len(unhealthy), MAX_LISTED_ITEMS),
            "unhealthy_disks": unhealthy[:MAX_LISTED_ITEMS],
        },
        incomplete,
    )


def get_node_ha_status(host: Optional[str]) -> Dict[str, Any]:
    """Per-node HA state: whether takeover is currently possible, and why not
    if it isn't. Directly corroborates or refutes the failover-risk candidates
    Stage 1 is most likely to raise."""
    try:
        host = _require_host(host)
        records, incomplete = _get_all(
            host,
            "/api/cluster/nodes",
            {"fields": "name,state,uptime,model,ha", "max_records": PAGE_SIZE},
        )
    except OntapClientError as exc:
        return _unavailable(str(exc))

    nodes = []
    for record in records:
        ha = record.get("ha") or {}
        partners = [p.get("name") for p in (ha.get("partners") or []) if p.get("name")]
        nodes.append(
            {
                "name": record.get("name"),
                "state": record.get("state"),
                "uptime_seconds": record.get("uptime"),
                "model": record.get("model"),
                "ha_enabled": ha.get("enabled"),
                "takeover_possible": (ha.get("takeover") or {}).get("state"),
                "giveback_state": (ha.get("giveback") or {}).get("state"),
                "partners": partners,
            }
        )
    # Every node is listed, not MAX_LISTED_ITEMS of them: a cluster's node
    # count is bounded by ONTAP's own maximum (24), so this cannot become an
    # inventory dump — and trimming it would hide the takeover state of nodes
    # 16+ in exactly the area where "which node cannot fail over" is the whole
    # question.
    return _with_incomplete(
        {"available": True, "cluster": host, "node_count": len(nodes), "nodes": nodes},
        incomplete,
    )


def get_volume_state(
    host: Optional[str], name: Optional[str] = None, unhealthy_only: bool = True
) -> Dict[str, Any]:
    """Volume states and space usage. Defaults to unhealthy_only because a
    cluster can have thousands of volumes and an investigation almost always
    cares about the offline/restricted ones, not the inventory.

    Every volume is fetched — the filtering happens here, not at the REST
    layer. This used to stop at max_records=500 and then report
    `volume_count: 500`, which on a 3,000-volume cluster was simply a false
    statement: the histogram was wrong and an offline volume among the
    unfetched 2,500 was invisible, with nothing telling the model to doubt it.
    The walk costs round trips, not tokens — the result is summarized either
    way — so completeness here is nearly free."""
    try:
        host = _require_host(host)
        params: Dict[str, Any] = {
            "fields": "name,state,svm.name,space.size,space.used,aggregates",
            "max_records": PAGE_SIZE,
        }
        if name:
            params["name"] = f"*{name}*"
        records, incomplete = _get_all(host, "/api/storage/volumes", params)
    except OntapClientError as exc:
        return _unavailable(str(exc))

    by_state: Dict[str, int] = {}
    volumes = []
    for record in records:
        state = record.get("state") or "unknown"
        by_state[state] = by_state.get(state, 0) + 1
        space = record.get("space") or {}
        used_pct = _pct(space.get("used"), space.get("size"))
        # "Interesting" = not online, or nearly full. Everything else is noise
        # in a context window the cost cap is paying for.
        interesting = state != "online" or (used_pct is not None and used_pct >= 90)
        if interesting or not unhealthy_only:
            volumes.append(
                {
                    "name": record.get("name"),
                    "svm": (record.get("svm") or {}).get("name"),
                    "state": state,
                    "used_pct": used_pct,
                    "aggregates": [a.get("name") for a in (record.get("aggregates") or []) if a.get("name")],
                }
            )
    volumes.sort(key=lambda v: v["used_pct"] if v["used_pct"] is not None else -1, reverse=True)
    return _with_incomplete(
        {
            "available": True,
            "cluster": host,
            "volume_count": len(records),
            "by_state": by_state,
            "filtered_to": "non-online or >=90% full" if unhealthy_only else "all",
            # How many volumes actually matched the filter, alongside the
            # trimmed list — the shape get_disk_health already uses. Now that
            # every volume is fetched, the matching set is far likelier to
            # exceed MAX_LISTED_ITEMS than it was at 500, so a trim the model
            # cannot see would be the same lie in a new place.
            "matching_count": len(volumes),
            "listed": min(len(volumes), MAX_LISTED_ITEMS),
            "volumes": volumes[:MAX_LISTED_ITEMS],
        },
        incomplete,
    )
