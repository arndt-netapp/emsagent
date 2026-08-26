import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.ontap.client import OntapClientError, fetch_ems_events

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "ems_events_response.json").read_text())


def _response(status_code: int, payload) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


def test_fetch_stops_within_first_page_without_following_next():
    with patch("requests.Session.get") as mock_get:
        mock_get.return_value = _response(200, FIXTURE["page_1"])

        records = list(
            fetch_ems_events(host="cluster.example.com", user="admin", password="pw", count=2)
        )

    assert len(records) == 2
    assert records[0]["message"]["name"] == "raid.autoPart.disabled"
    mock_get.assert_called_once()
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["order_by"] == "time desc"
    assert kwargs["params"]["max_records"] == 2


def test_fetch_follows_pagination_across_pages():
    with patch("requests.Session.get") as mock_get:
        mock_get.side_effect = [
            _response(200, FIXTURE["page_1"]),
            _response(200, FIXTURE["page_2"]),
        ]

        records = list(
            fetch_ems_events(host="cluster.example.com", user="admin", password="pw", count=3, page_size=2)
        )

    assert len(records) == 3
    assert [r["index"] for r in records] == [4602, 4603, 4604]
    assert mock_get.call_count == 2
    second_call_url = mock_get.call_args_list[1][0][0]
    assert "start.index=4603" in second_call_url


def test_fetch_falls_back_when_order_by_unsupported():
    with patch("requests.Session.get") as mock_get:
        mock_get.side_effect = [
            _response(400, {"error": {"message": "invalid query parameter order_by"}}),
            _response(200, FIXTURE["page_1"]),
        ]

        records = list(
            fetch_ems_events(host="cluster.example.com", user="admin", password="pw", count=2)
        )

    assert len(records) == 2
    first_params = mock_get.call_args_list[0][1]["params"]
    second_params = mock_get.call_args_list[1][1]["params"]
    assert "order_by" in first_params
    assert "order_by" not in second_params


def test_fetch_passes_auth_and_filters():
    with patch("requests.Session.get") as mock_get:
        mock_get.return_value = _response(200, FIXTURE["page_1"])

        list(
            fetch_ems_events(
                host="cluster.example.com",
                user="admin",
                password="secret",
                count=1,
                severity="alert",
                log_message="*disk*",
            )
        )

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["message.severity"] == "alert"
    assert kwargs["params"]["log_message"] == "*disk*"


def test_fetch_raises_on_persistent_error():
    with patch("requests.Session.get") as mock_get:
        mock_get.return_value = _response(500, {"error": {"message": "internal error"}})

        with pytest.raises(OntapClientError):
            list(fetch_ems_events(host="cluster.example.com", user="admin", password="pw", count=1))


def test_a_dropped_connection_mid_walk_is_reported_as_an_ontap_error():
    """A large fetch is dozens of sequential requests over one keep-alive
    connection, and ONTAP closing that connection part-way used to raise a raw
    requests.ConnectionError. Nothing in the app catches that, so it reached
    FastAPI as an unhandled exception: a 500 with a stack trace instead of a
    502 naming the cluster, and — because the route's cleanup only ran for
    OntapClientError — a half-written log file orphaned in the watch
    directory."""
    with patch("requests.Session.get") as mock_get:
        mock_get.side_effect = [
            _response(200, FIXTURE["page_1"]),
            requests.exceptions.ConnectionError(
                "('Connection aborted.', RemoteDisconnected('Remote end closed connection "
                "without response'))"
            ),
        ]

        events = []
        with pytest.raises(OntapClientError) as excinfo:
            for record in fetch_ems_events(
                host="cluster.example.com", user="admin", password="pw", count=99, page_size=2
            ):
                events.append(record)

    # The pages that did arrive were yielded before the failure, which is what
    # lets the caller keep them instead of losing the whole walk.
    assert len(events) == 2
    assert "ConnectionError" in str(excinfo.value)


def test_transient_connection_failures_are_retried_for_the_paginated_walk():
    """Pins the retry policy against the installed urllib3 rather than
    asserting on its fields: a RemoteDisconnected surfaces as a ProtocolError,
    and this asserts urllib3 actually classifies that as retryable for GET
    (which is why it is safe — every request in the walk is a read).

    Without a policy mounted, requests retries nothing at all, and one dropped
    connection anywhere in a 10,000-event fetch killed the whole thing."""
    from urllib3.exceptions import ProtocolError

    from app.ontap.client import _new_session, _retry_policy

    policy = _retry_policy()
    after = policy.increment(
        method="GET",
        url="/api/support/ems/events",
        error=ProtocolError("Connection aborted.", "RemoteDisconnected"),
    )
    assert after.total < policy.total  # it retried rather than re-raising

    # ...and the policy is actually mounted on the scheme the client uses.
    session = _new_session("admin", "pw", verify_tls=False)
    assert session.get_adapter("https://cluster.example.com").max_retries.total == policy.total


def test_a_write_is_never_retried_even_if_one_is_added_later():
    """Guard on the retry policy's blast radius: retrying is only safe because
    the fetch walk is all GETs. If a POST/PATCH is ever added to this client,
    the policy must not silently replay it."""
    from urllib3.exceptions import ProtocolError

    from app.ontap.client import _retry_policy

    with pytest.raises(Exception):
        _retry_policy().increment(
            method="POST", url="/api/anything", error=ProtocolError("Connection aborted.")
        )
