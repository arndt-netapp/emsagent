# Claude Sonnet 5 list pricing, per million tokens (USD), verified against
# https://platform.claude.com/docs/en/about-claude/pricing on 2026-08-18.
# Used to render a cost estimate for analysis runs/investigations; not a
# billing figure.
#
# These are deliberately hardcoded to Sonnet 5 rather than keyed off
# settings.claude_model. If you point CLAUDE_MODEL at a different model,
# every cost shown in the UI — and the STAGE2_COST_CAP_USD ceiling below,
# which is enforced against these same numbers — will be wrong. Opus 5, for
# instance, is 2.5x these rates, so a $0.50 cap would really stop at $1.25.
#
# NOTE: $2/$10 launched as "introductory pricing through 2026-08-31" and an
# increase to $3/$15 was scheduled for 2026-09-01. Anthropic has since
# cancelled that increase and made $2/$10 the standard price. An earlier
# version of this file carried the $3/$15 figures, which overstated every
# reported cost by exactly 1.5x and made the Stage 2 cap fire at two thirds of
# its nominal value in real spend.
SONNET_5_INPUT_PRICE_PER_MTOK = 2.00
SONNET_5_OUTPUT_PRICE_PER_MTOK = 10.00
# Prompt-cache write/read prices (1.25x / 0.1x of base input) — these are
# billed as SEPARATE usage fields (cache_creation_input_tokens /
# cache_read_input_tokens), not folded into input_tokens by the raw
# Anthropic SDK, so they need their own multipliers or the dominant term of
# a cache-heavy request goes uncounted. Only the 5-minute TTL is used
# anywhere in this codebase; a 1-hour write would be 2x base input instead.
SONNET_5_CACHE_WRITE_PRICE_PER_MTOK = 2.50
SONNET_5_CACHE_READ_PRICE_PER_MTOK = 0.20

# Hard ceiling on a single Stage 2 candidate investigation, enforced in code
# (app/agent/graph.py's routing) after every model call.
STAGE2_COST_CAP_USD = 0.50

# --- Stage 1 forward estimate ------------------------------------------------
#
# Stage 1 is one call over the entire compact corpus, so its cost is linear in
# event count and the corpus is the whole bill. These constants exist so the
# scope selector can say what a run will cost BEFORE it is paid for — a full
# autosupport EMS log can be an order of magnitude larger than a cluster fetch,
# and "10,000 events" does not read as "$1.25" to anyone who hasn't measured it.
#
# The per-event figure is measured, not derived: ~55-60 tokens/event on real
# cluster data (2026-08-19), where events rarely repeat back-to-back and
# compaction produces roughly one row per event. On highly repetitive data it
# is far lower (~14), so this ESTIMATE IS AN UPPER-ISH BOUND THAT CAN BE WRONG
# BY 4x IN THE CHEAP DIRECTION. It is a warning, not a quote. To re-derive it
# from your own data, use analysis_runs.input_tokens / events_considered.
STAGE1_INPUT_TOKENS_PER_EVENT = 60
# System prompt + glossary + dismissals: bounded by event-name variety and
# feedback count rather than volume, so it's a constant at any interesting size.
STAGE1_OVERHEAD_TOKENS = 2000
# Output is a top-N judgment, not a per-event transform, so it doesn't grow
# with the corpus either. Hard-capped at STAGE1_MAX_TOKENS (16000); this is a
# typical figure, not that ceiling.
STAGE1_TYPICAL_OUTPUT_TOKENS = 4000


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    return round(
        input_tokens / 1_000_000 * SONNET_5_INPUT_PRICE_PER_MTOK
        + output_tokens / 1_000_000 * SONNET_5_OUTPUT_PRICE_PER_MTOK
        + cache_creation_input_tokens / 1_000_000 * SONNET_5_CACHE_WRITE_PRICE_PER_MTOK
        + cache_read_input_tokens / 1_000_000 * SONNET_5_CACHE_READ_PRICE_PER_MTOK,
        4,
    )


def estimate_stage1_cost_usd(event_count: int) -> float:
    """What a Stage 1 run over `event_count` events is likely to cost.

    Routed through estimate_cost_usd rather than a dollars-per-event constant
    so a price change lands in exactly one place. Stage 1 sets no
    cache_control (see stage1.py), so there are no cache terms here."""
    return estimate_cost_usd(
        input_tokens=STAGE1_OVERHEAD_TOKENS + event_count * STAGE1_INPUT_TOKENS_PER_EVENT,
        output_tokens=STAGE1_TYPICAL_OUTPUT_TOKENS,
    )
