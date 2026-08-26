from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: Optional[str] = None
    claude_model: str = "claude-sonnet-5"
    ems_watch_dir: Path = REPO_ROOT / "data" / "logs_incoming"
    database_path: Path = REPO_ROOT / "data" / "emsagent.db"
    max_agent_iterations: int = 15

    # Reasoning effort (output_config.effort) per stage: low | medium | high |
    # xhigh | max. Sonnet 5 thinks by default and thinking tokens bill at the
    # OUTPUT rate, so leaving both stages at the API default of "high" — as
    # this app did originally — pays top-of-range reasoning cost on work that
    # mostly doesn't need it.
    #
    # Stage 1 is one analytical pass over pre-deduplicated rows whose whole
    # value is spotting cross-node/cross-event correlation, so it keeps real
    # reasoning budget at "medium". Stage 2's turns are mostly mechanical
    # ("read this tool result, pick the next tool"), and lower effort also
    # means fewer, more consolidated tool calls — which buys more useful
    # turns under the same STAGE2_COST_CAP_USD. Raise either if you see
    # quality regress; they are per-stage precisely so they can be tuned
    # independently.
    stage1_effort: str = "medium"
    stage2_effort: str = "low"

    # One credential pair, used for every cluster the agent is asked about.
    # A PoC simplification, chosen deliberately: the alternative is a clusters
    # table with per-cluster secrets, which is real credential management this
    # project has no business implementing. Consequence: every cluster reachable
    # from here must accept the same login.
    #
    # Unlike the ingestion fetch form — where a human is present and the
    # password is used for that one request and discarded — Stage 2's cluster
    # tools run inside a background investigation with nobody to prompt, so the
    # credential has to be resolvable from config alone.
    # Note there is deliberately NO host setting. The host is not configuration:
    # every analysis run is scoped to one cluster, and that cluster value IS the
    # host it was fetched from (routes_clusters passes it straight to
    # fetch_ems_events). A configured default host would be read as the target
    # for EVERY investigation, so with two clusters registered, investigating a
    # candidate from cluster B would query cluster A and present its disk health
    # as evidence for B.
    ontap_user: Optional[str] = None
    ontap_password: Optional[str] = None
    ontap_verify_tls: bool = False
    # Kept short on purpose: these calls happen inside the investigate loop,
    # where a hung cluster would otherwise stall a background task holding a
    # SQLite connection.
    ontap_timeout_seconds: int = 15


settings = Settings()
settings.ems_watch_dir.mkdir(parents=True, exist_ok=True)
settings.database_path.parent.mkdir(parents=True, exist_ok=True)
