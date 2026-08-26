"""Cluster-state reads, against scripted responses.

Same caveat as test_ontap_client.py: these lock in how the code SUMMARIZES a
response — the clamping, the sorting, the degrade-not-raise paths — not that
the response shape itself is right. Only a real cluster proves that.
"""

import pytest
import requests

from app.config import settings
from app.ontap import cluster_state


HOST = "cluster.test"


@pytest.fixture(autouse=True)
def configured_credentials(monkeypatch):
    monkeypatch.setattr(settings, "ontap_user", "admin")
    monkeypatch.setattr(settings, "ontap_password", "secret")


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = "error body"

    def json(self):
        return self._payload


def _fake_get(payload, status_code=200, capture=None):
    def get(url, params=None, timeout=None):
        if capture is not None:
            capture["url"] = url
            capture["params"] = params
            capture["timeout"] = timeout
        return _FakeResponse(payload, status_code)

    return get


@pytest.fixture
def patch_session(monkeypatch):
    def apply(payload, status_code=200, capture=None):
        class _FakeSession:
            get = staticmethod(_fake_get(payload, status_code, capture))

        monkeypatch.setattr(cluster_state, "_new_session", lambda *a, **k: _FakeSession())

    return apply


def _page(records, next_href=None):
    """One page of a paginated collection response."""
    payload = {"records": records}
    if next_href:
        payload["_links"] = {"next": {"href": next_href}}
    return payload


@pytest.fixture
def patch_paged_session(monkeypatch):
    """Serve a sequence of pages, one per GET, and count sessions created.

    The plain `patch_session` fixture returns the same payload for every
    request, which can only ever exercise a single-page walk."""

    def apply(pages):
        state = {"calls": [], "sessions": 0}

        class _FakeSession:
            @staticmethod
            def get(url, params=None, timeout=None):
                state["calls"].append(url)
                page = pages[len(state["calls"]) - 1]
                if isinstance(page, Exception):
                    raise page
                return _FakeResponse(page)

        def new_session(*a, **k):
            state["sessions"] += 1
            return _FakeSession()

        monkeypatch.setattr(cluster_state, "_new_session", new_session)
        return state

    return apply


def test_credentials_configured_needs_both(monkeypatch):
    """Credentials are configuration; the HOST is not — it comes from the run's
    cluster scope, so there is nothing host-shaped to check here."""
    assert cluster_state.credentials_configured() is True
    monkeypatch.setattr(settings, "ontap_password", None)
    assert cluster_state.credentials_configured() is False


def test_aggregates_sorted_fullest_first(patch_session):
    patch_session(
        {
            "records": [
                {"name": "aggr1", "node": {"name": "n1"}, "state": "online",
                 "space": {"block_storage": {"size": 1000, "used": 200}}},
                {"name": "aggr2", "node": {"name": "n2"}, "state": "online",
                 "space": {"block_storage": {"size": 1000, "used": 940}}},
            ]
        }
    )
    result = cluster_state.get_aggregate_capacity(HOST)
    assert result["available"] is True
    assert [a["name"] for a in result["aggregates"]] == ["aggr2", "aggr1"]
    assert result["aggregates"][0]["used_pct"] == 94.0


def test_aggregates_tolerate_missing_space_fields(patch_session):
    """Field shapes are unconfirmed against a real cluster, so a missing
    nested key must not raise inside an investigation."""
    patch_session({"records": [{"name": "aggr1"}]})
    result = cluster_state.get_aggregate_capacity(HOST)
    assert result["aggregates"][0]["used_pct"] is None


def test_disk_health_summarizes_instead_of_dumping(patch_session):
    """A 600-disk cluster must come back as a histogram plus the bad disks —
    never 600 JSON objects into a context window the cost cap is paying for."""
    records = [{"name": f"d{i}", "state": "present", "node": {"name": "n1"}} for i in range(600)]
    records.append({"name": "bad1", "state": "broken", "node": {"name": "n1"}, "bay": 3})
    patch_session({"records": records})

    result = cluster_state.get_disk_health(HOST)
    assert result["disk_count"] == 601
    assert result["by_state"] == {"present": 600, "broken": 1}
    assert result["unhealthy_count"] == 1
    assert result["unhealthy_disks"][0]["name"] == "bad1"
    assert len(result["unhealthy_disks"]) <= cluster_state.MAX_LISTED_ITEMS


def test_disk_health_caps_listed_unhealthy_disks(patch_session):
    patch_session({"records": [{"name": f"d{i}", "state": "broken"} for i in range(50)]})
    result = cluster_state.get_disk_health(HOST)
    assert result["unhealthy_count"] == 50
    assert len(result["unhealthy_disks"]) == cluster_state.MAX_LISTED_ITEMS


def test_ha_status_extracts_takeover_state(patch_session):
    patch_session(
        {
            "records": [
                {
                    "name": "n1",
                    "state": "up",
                    "ha": {
                        "enabled": True,
                        "takeover": {"state": "not_possible"},
                        "giveback": {"state": "nothing_to_giveback"},
                        "partners": [{"name": "n2"}],
                    },
                }
            ]
        }
    )
    node = cluster_state.get_node_ha_status(HOST)["nodes"][0]
    assert node["takeover_possible"] == "not_possible"
    assert node["partners"] == ["n2"]


def test_volumes_filtered_to_interesting_by_default(patch_session):
    patch_session(
        {
            "records": [
                {"name": "healthy", "state": "online", "space": {"size": 100, "used": 10}},
                {"name": "full", "state": "online", "space": {"size": 100, "used": 95}},
                {"name": "offline", "state": "offline", "space": {"size": 100, "used": 1}},
            ]
        }
    )
    result = cluster_state.get_volume_state(HOST)
    assert result["volume_count"] == 3
    assert sorted(v["name"] for v in result["volumes"]) == ["full", "offline"]


def test_volumes_can_return_everything(patch_session):
    patch_session({"records": [{"name": "healthy", "state": "online", "space": {"size": 100, "used": 10}}]})
    result = cluster_state.get_volume_state(HOST, unhealthy_only=False)
    assert [v["name"] for v in result["volumes"]] == ["healthy"]


def test_unreachable_cluster_degrades_rather_than_raises(monkeypatch):
    """These calls happen mid-investigation. A dead cluster must cost the agent
    one avenue of evidence, not the whole run the user already paid for."""
    class _ExplodingSession:
        @staticmethod
        def get(url, params=None, timeout=None):
            raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(cluster_state, "_new_session", lambda *a, **k: _ExplodingSession())
    result = cluster_state.get_disk_health(HOST)
    assert result["available"] is False
    assert "no route to host" in result["reason"]


def test_http_error_degrades_rather_than_raises(patch_session):
    patch_session({}, status_code=401)
    result = cluster_state.get_node_ha_status(HOST)
    assert result["available"] is False
    assert "401" in result["reason"]


def test_missing_host_is_reported_not_raised():
    """A candidate whose events came from an uploaded log file has no cluster
    behind it. There is deliberately no configured default to fall back on: it
    would point every investigation at one cluster regardless of where its
    evidence came from."""
    result = cluster_state.get_aggregate_capacity(None)
    assert result["available"] is False
    assert "uploaded log file" in result["reason"]


@pytest.mark.parametrize(
    "host",
    [
        "https://attacker.example.com/x",  # a scheme and a path
        "admin@attacker.example.com",  # userinfo, so the real host is the tail
        "cluster.test:8443/api",
        "cluster.test/../../x",
        "-leading-dash.example.com",
    ],
)
def test_a_host_that_is_not_a_hostname_is_refused_before_any_request(host, monkeypatch):
    """The host arrives as DATA — a run's scope_cluster, ultimately the free
    text someone typed in the upload form's cluster box — and this module
    attaches the shared ONTAP credentials to whatever it is handed, with TLS
    verification off by default. So an arbitrary string here posts the one
    credential pair this app holds to a host of the uploader's choosing.

    It degrades rather than raising, like every other failure here: a bad host
    costs the agent one avenue of evidence, not the investigation."""

    def explode(*args, **kwargs):
        raise AssertionError("a session must not be built for an invalid host")

    monkeypatch.setattr(cluster_state, "_new_session", explode)

    result = cluster_state.get_aggregate_capacity(host)

    assert result["available"] is False
    assert "valid cluster host" in result["reason"]


def test_a_zero_percent_used_aggregate_is_reported_as_zero_not_unknown(patch_session):
    """`not used` treats 0 bytes as missing, so the emptiest aggregate in the
    cluster came back as "usage unknown" and then sorted as if it were -1%."""
    patch_session(
        _page([{"name": "aggr_empty", "space": {"block_storage": {"size": 1000, "used": 0}}}])
    )

    result = cluster_state.get_aggregate_capacity(HOST)

    assert result["aggregates"][0]["used_pct"] == 0.0


def test_missing_credentials_are_reported_not_raised(monkeypatch):
    monkeypatch.setattr(settings, "ontap_user", None)
    result = cluster_state.get_aggregate_capacity(HOST)
    assert result["available"] is False
    assert "ONTAP_USER" in result["reason"]


def test_timeout_is_applied(patch_session, monkeypatch):
    """A hung cluster would otherwise stall a background task holding a SQLite
    connection for as long as the OS lets it."""
    monkeypatch.setattr(settings, "ontap_timeout_seconds", 7)
    capture = {}
    patch_session({"records": []}, capture=capture)
    cluster_state.get_aggregate_capacity(HOST)
    assert capture["timeout"] == 7


def test_node_filter_is_passed_through(patch_session):
    capture = {}
    patch_session({"records": []}, capture=capture)
    cluster_state.get_disk_health(HOST, node="n1")
    assert capture["params"]["node.name"] == "n1"


def test_volume_walk_follows_every_page(patch_paged_session):
    """Stopping at one page and reporting its length as volume_count was a
    false statement, not a truncation: on a 3,000-volume cluster the histogram
    was wrong and an offline volume beyond the first page was invisible."""
    state = patch_paged_session(
        [
            _page(
                [{"name": "v1", "state": "online", "space": {"size": 100, "used": 10}}],
                next_href="/api/storage/volumes?start=2",
            ),
            _page(
                [{"name": "v2", "state": "online", "space": {"size": 100, "used": 20}}],
                next_href="/api/storage/volumes?start=3",
            ),
            _page([{"name": "v3", "state": "offline", "space": {"size": 100, "used": 30}}]),
        ]
    )

    result = cluster_state.get_volume_state(HOST)
    assert result["available"] is True
    assert result["volume_count"] == 3
    assert result["by_state"] == {"online": 2, "offline": 1}
    assert [v["name"] for v in result["volumes"]] == ["v3"]
    assert "incomplete_reason" not in result
    assert len(state["calls"]) == 3


def test_walk_uses_one_session_for_every_page(patch_paged_session):
    """A session per page would re-handshake TLS each time and throw away the
    Retry policy that exists to survive a long walk."""
    state = patch_paged_session(
        [
            _page([{"name": "d1", "state": "present"}], next_href="/api/storage/disks?start=2"),
            _page([{"name": "d2", "state": "present"}]),
        ]
    )
    cluster_state.get_disk_health(HOST)
    assert len(state["calls"]) == 2
    assert state["sessions"] == 1


def test_walk_keeps_what_arrived_when_a_later_page_fails(patch_paged_session):
    """Losing 6,000 fetched volumes because page 13 timed out would make a big
    cluster strictly worse than a small one. The partial result is still
    available:True — it is real evidence, just not all of it."""
    state = patch_paged_session(
        [
            _page(
                [{"name": "v1", "state": "offline", "space": {"size": 100, "used": 10}}],
                next_href="/api/storage/volumes?start=2",
            ),
            requests.ConnectionError("connection reset by peer"),
        ]
    )

    result = cluster_state.get_volume_state(HOST)
    assert result["available"] is True
    assert result["volume_count"] == 1
    assert [v["name"] for v in result["volumes"]] == ["v1"]
    assert "connection reset by peer" in result["incomplete_reason"]
    assert "lower bound" in result["incomplete_reason"]
    assert len(state["calls"]) == 2


def test_failure_on_the_first_page_is_still_unavailable(patch_paged_session):
    """Nothing arrived, so there is no partial answer to report — and an
    unreachable cluster must not read as an empty one."""
    patch_paged_session([requests.ConnectionError("no route to host")])
    result = cluster_state.get_volume_state(HOST)
    assert result["available"] is False
    assert "no route to host" in result["reason"]


def test_repeating_pagination_link_ends_the_walk(patch_paged_session):
    """The walk is deliberately unbounded, so a cluster that returns a
    self-referencing next link would spin forever inside a tool call — and the
    Stage 2 cost cap bounds turns, not wall-clock, so nothing would stop it."""
    state = patch_paged_session(
        [
            _page([{"name": "d1", "state": "broken"}], next_href="/api/storage/disks?start=2"),
            _page([{"name": "d2", "state": "broken"}], next_href="/api/storage/disks?start=2"),
        ]
    )

    result = cluster_state.get_disk_health(HOST)
    assert result["available"] is True
    assert result["disk_count"] == 2
    assert "repeating pagination link" in result["incomplete_reason"]
    assert len(state["calls"]) == 2


def test_volumes_report_how_many_matched_beyond_the_listed_ones(patch_session):
    """Now that every volume is fetched, the matching set is far likelier to
    exceed MAX_LISTED_ITEMS than it was at max_records=500 — a trim the model
    cannot see would be the same lie in a new place."""
    patch_session(
        {
            "records": [
                {"name": f"v{i}", "state": "offline", "space": {"size": 100, "used": i}}
                for i in range(40)
            ]
        }
    )
    result = cluster_state.get_volume_state(HOST)
    assert result["volume_count"] == 40
    assert result["matching_count"] == 40
    assert result["listed"] == cluster_state.MAX_LISTED_ITEMS
    assert len(result["volumes"]) == cluster_state.MAX_LISTED_ITEMS


def test_every_node_is_listed(patch_session):
    """Node count is bounded by ONTAP's own cluster maximum, so listing all of
    them cannot become an inventory dump — and trimming would hide the takeover
    state of nodes 16+ in the one area where that is the whole question."""
    patch_session(
        {
            "records": [
                {"name": f"n{i}", "state": "up", "ha": {"takeover": {"state": "not_possible"}}}
                for i in range(24)
            ]
        }
    )
    result = cluster_state.get_node_ha_status(HOST)
    assert result["node_count"] == 24
    assert len(result["nodes"]) == 24
