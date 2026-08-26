import logging
import re
from typing import Any, Dict, Iterator, Optional
from urllib.parse import urljoin

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# A cluster value must be a bare hostname or IPv4 address: no scheme, no
# userinfo (`@`), no port (`:`), no path (`/`).
#
# This is a security boundary, not tidiness. Every "cluster" in this app is
# also the REST host it gets queried at — `f"https://{host}{path}"` in
# cluster_state, with the shared ONTAP_USER/ONTAP_PASSWORD attached and
# ontap_verify_tls defaulting to False. The value reaches that point from the
# free-text `cluster` field on the upload form, so without this an uploaded log
# file labelled `attacker.example.com` makes the Stage 2 agent post the one
# credential pair this app holds to a host of the uploader's choosing.
#
# IPv6 is not accepted. It never worked on this path anyway: routes_clusters
# rewrites `:` to `_` when naming the fetch output, so an IPv6 literal could
# not round-trip through cluster_from_filename either.
CLUSTER_HOST_PATTERN = r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,253}[A-Za-z0-9])?$"
_CLUSTER_HOST_RE = re.compile(CLUSTER_HOST_PATTERN)


class OntapClientError(Exception):
    pass


def validate_host(host: str) -> str:
    """Return `host` if it is a usable cluster host, else raise.

    Raises OntapClientError specifically so cluster_state's callers degrade
    rather than fail: they already turn that exception into
    `{"available": False, "reason": ...}`, so a bad host costs the agent one
    avenue of evidence instead of the investigation."""
    if not host or not _CLUSTER_HOST_RE.match(host):
        raise OntapClientError(
            f"{host!r} is not a valid cluster host — expected a hostname or IPv4 address "
            "with no scheme, port or path"
        )
    return host


def _retry_policy() -> Retry:
    """Retry transient network failures on the paginated GET walk.

    A large fetch is not one request, it is `count / page_size` sequential
    requests over one keep-alive connection — 10,000 events is dozens of them,
    taking minutes. Somewhere in a walk that long ONTAP will close the pooled
    connection (idle timeout, session recycling, a management-plane restart),
    and the next request goes out on a socket the server has already hung up:
    `RemoteDisconnected: Remote end closed connection without response`. A
    `requests.Session` retries nothing by default, so that killed the whole
    fetch — which is why 1,000 events worked and 10,000 did not.

    Retrying is safe here because every request in the walk is a GET: no state
    changes on the cluster, and the response either arrives whole or not at
    all. The backoff is exponential (0.5s, 1s, 2s, ...), which also covers a
    cluster briefly too busy to answer."""
    return Retry(
        total=5,
        connect=5,
        read=5,
        status=3,
        backoff_factor=0.5,
        # Retried automatically; anything else (401, 404, and the 400 the
        # order_by probe below relies on) is returned to the caller as-is.
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        # Hand back the final response instead of raising, so the status
        # checks below stay the single place a bad status is reported.
        raise_on_status=False,
    )


def _new_session(user: str, password: str, verify_tls: bool) -> requests.Session:
    session = requests.Session()
    session.auth = (user, password)
    session.verify = verify_tls
    session.headers.update({"Accept": "application/json"})
    session.mount("https://", HTTPAdapter(max_retries=_retry_policy()))
    if not verify_tls:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session


def _get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """GET with every network failure surfaced as OntapClientError.

    Callers (and the API routes above them) handle OntapClientError; a raw
    requests.ConnectionError escaping instead reached FastAPI as an unhandled
    exception, so a cluster that dropped the connection produced a 500 with a
    stack trace rather than a 502 saying which cluster failed and why."""
    try:
        return session.get(url, timeout=30, **kwargs)
    except requests.RequestException as exc:
        raise OntapClientError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc


def fetch_ems_events(
    host: str,
    user: str,
    password: str,
    count: int,
    severity: Optional[str] = None,
    log_message: Optional[str] = None,
    verify_tls: bool = False,
    # Every page is one round trip, so this sets how many requests a fetch
    # makes: 10,000 events was 100 requests at the old page_size of 100. ONTAP
    # treats max_records as a ceiling, not a promise — it returns fewer (with a
    # `next` link) when a page would exceed its own return_timeout — so asking
    # for more pages fewer times costs nothing and cannot fail on its own.
    page_size: int = 500,
) -> Iterator[Dict[str, Any]]:
    """Yield up to `count` EMS event records from GET /api/support/ems/events,
    newest first when the target ONTAP version supports `order_by`."""
    session = _new_session(user, password, verify_tls)
    base_url = f"https://{host}/api/support/ems/events"

    params: Dict[str, Any] = {"max_records": min(page_size, count)}
    if severity:
        params["message.severity"] = severity
    if log_message:
        params["log_message"] = log_message

    ordered_params = dict(params, order_by="time desc")
    response = _get(session, base_url, params=ordered_params)
    if response.status_code == 400:
        logger.warning(
            "cluster rejected order_by=time desc (status 400); falling back to default event order"
        )
        response = _get(session, base_url, params=params)
    if response.status_code != 200:
        raise OntapClientError(f"GET {base_url} failed: {response.status_code} {response.text}")

    yielded = 0
    while True:
        payload = response.json()
        for record in payload.get("records", []):
            if yielded >= count:
                return
            yield record
            yielded += 1
        if yielded >= count:
            return

        next_href = payload.get("_links", {}).get("next", {}).get("href")
        if not next_href:
            return
        next_url = urljoin(f"https://{host}/", next_href.lstrip("/"))
        response = _get(session, next_url)
        if response.status_code != 200:
            raise OntapClientError(f"GET {next_url} failed: {response.status_code} {response.text}")
