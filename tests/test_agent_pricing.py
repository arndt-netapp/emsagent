from app.agent import pricing
from app.agent.pricing import estimate_cost_usd


def test_rates_match_published_sonnet_5_pricing():
    """Pin the four constants to Anthropic's published Claude Sonnet 5 rates
    (platform.claude.com/docs/en/about-claude/pricing, verified 2026-08-18).

    This exists because the constants once carried $3/$15 — the rate that was
    scheduled to replace the $2/$10 "introductory" pricing before Anthropic
    cancelled that increase. Nothing in the app noticed: every cost in the UI
    was silently 1.5x too high, and STAGE2_COST_CAP_USD stopped
    investigations at two thirds of its nominal budget. A pinned test
    turns the next such drift into a failure instead of a silent overcharge."""
    assert pricing.SONNET_5_INPUT_PRICE_PER_MTOK == 2.00
    assert pricing.SONNET_5_OUTPUT_PRICE_PER_MTOK == 10.00
    assert pricing.SONNET_5_CACHE_WRITE_PRICE_PER_MTOK == 2.50
    assert pricing.SONNET_5_CACHE_READ_PRICE_PER_MTOK == 0.20


def test_cache_multipliers_track_the_base_input_rate():
    """The cache rates are defined by the API as multiples of base input
    (1.25x for a 5-minute write, 0.1x for a read), so they must move together
    with it — fixing the base rate alone and leaving these behind would
    reintroduce the same class of bug in the terms that dominate a
    cache-heavy request."""
    base = pricing.SONNET_5_INPUT_PRICE_PER_MTOK
    assert pricing.SONNET_5_CACHE_WRITE_PRICE_PER_MTOK == base * 1.25
    assert round(pricing.SONNET_5_CACHE_READ_PRICE_PER_MTOK, 6) == round(base * 0.1, 6)


def test_plain_input_and_output_only():
    # 1M input tokens @ $2, 1M output tokens @ $10
    assert estimate_cost_usd(1_000_000, 1_000_000) == 12.0


def test_cache_write_priced_higher_than_plain_input():
    # 1M cache-write tokens should cost more than 1M plain input tokens
    # (1.25x) — this is the exact gap that was previously dropped entirely.
    plain = estimate_cost_usd(1_000_000, 0)
    cache_write = estimate_cost_usd(0, 0, cache_creation_input_tokens=1_000_000)
    assert cache_write > plain
    assert cache_write == 2.50


def test_cache_read_priced_much_lower_than_plain_input():
    plain = estimate_cost_usd(1_000_000, 0)
    cache_read = estimate_cost_usd(0, 0, cache_read_input_tokens=1_000_000)
    assert cache_read < plain
    assert cache_read == 0.20


def test_all_four_components_sum():
    cost = estimate_cost_usd(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert cost == 2.00 + 10.00 + 2.50 + 0.20


def test_realistic_10k_event_stage1_scale_is_well_under_a_dollar():
    """Sanity check against the actual concern this was built for: a ~140k-token
    compact corpus (10k events at ~14 tokens/event) plus a modest output should
    land well under $1, not the naive-linear-extrapolation fear of $17+.

    Stage 1 sends the corpus as plain input — it no longer sets a cache
    breakpoint, because its re-run guard meant the 1.25x write never got the
    read that would have paid for it (see stage1.run_stage1)."""
    cost = estimate_cost_usd(input_tokens=140_000, output_tokens=8_000)
    assert 0.20 < cost < 1.00


def test_stage1_estimate_is_linear_in_event_count():
    """Stage 1 is one call over the whole corpus, so its price grows with event
    count and nothing bounds it — which is why the scope selector shows the
    estimate before the button is pressed rather than after the bill."""
    small = pricing.estimate_stage1_cost_usd(1_000)
    large = pricing.estimate_stage1_cost_usd(10_000)

    # 10x the events for ~10x the price, allowing for the fixed overhead and
    # output terms that don't scale.
    assert 9 < (large - pricing.estimate_stage1_cost_usd(0)) / (
        small - pricing.estimate_stage1_cost_usd(0)
    ) < 11


def test_stage1_estimate_lands_near_the_measured_10k_event_run():
    """Pinned to a real measurement, not a guess: 10,000 events against a real
    cluster cost ~$1.25 (2026-08-19). An estimate that doesn't recover that
    number is telling the user something false about a run they haven't
    authorized yet."""
    assert 1.00 < pricing.estimate_stage1_cost_usd(10_000) < 1.60


def test_stage1_estimate_routes_through_the_shared_cost_formula():
    """Not an independent dollars-per-event constant: a rate change has to land
    in one place, and Stage 1 sets no cache_control so there are no cache
    terms to account for."""
    expected = estimate_cost_usd(
        input_tokens=pricing.STAGE1_OVERHEAD_TOKENS + 5_000 * pricing.STAGE1_INPUT_TOKENS_PER_EVENT,
        output_tokens=pricing.STAGE1_TYPICAL_OUTPUT_TOKENS,
    )
    assert pricing.estimate_stage1_cost_usd(5_000) == expected
