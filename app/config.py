"""Engine configuration. Everything that affects latency is an explicit toggle,
so each lever can be turned on and off in an A/B and measured in isolation."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---------- model provider ----------
    # foundry -> Azure AI Foundry (services.ai.azure.com), v1 API - the target stack
    # azure   -> classic Azure OpenAI (<resource>.openai.azure.com + api-version)
    # openai  -> OpenAI directly
    # mock    -> simulates TTFT and token rate; measures the harness's own overhead
    llm_provider: Literal["foundry", "azure", "openai", "mock"] = "foundry"

    openai_api_key: str | None = None

    # Foundry: accepts both the resource host and the project URL
    # (https://<resource>.services.ai.azure.com/api/projects/<project>) - the
    # /api/projects/... path belongs to the projects SDK and is dropped for
    # inference, which lives under /openai/v1/.
    azure_ai_endpoint: str | None = None
    azure_ai_api_key: str | None = None

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"

    # tiers: nano (locale/intent), mini (common questions), frontier (hard ones)
    nano_model: str = "gpt-5.6-terra"
    mini_model: str = "gpt-5.6-terra"
    frontier_model: str = "gpt-5.6-terra"
    embedding_model: str = "text-embedding-3-small"

    # Per-provider tier overrides. The tier vars above hold Azure deployment
    # names; these let an OpenAI model name sit alongside them, so LLM_PROVIDER
    # becomes the only line that changes when switching providers. Without them,
    # switching means rewriting three names and losing the other provider's set -
    # which makes a provider A/B a manual edit instead of a toggle.
    # Empty => fall back to the tier var above.
    openai_nano_model: str | None = None
    openai_mini_model: str | None = None
    openai_frontier_model: str | None = None
    openai_embedding_model: str | None = None

    max_output_tokens: int = 250  # reference: ~250 output tokens per turn
    temperature: float = 0.2

    # gpt-5.x accepts reasoning_effort; the valid value varies per model
    # (gpt-5.6-terra accepts only "none"; gpt-5.2 accepts "minimal").
    # On the hot path the rule is explicit: always non-reasoning.
    reasoning_effort: str | None = "none"

    # ---------- price per million tokens (cost per interaction) ----------
    # Price is NOT hardcoded here. It comes from data/model_prices.csv, produced
    # by `uv run python scripts/fetch_prices.py`, and is resolved PER MODEL - the
    # three tiers and the model inside Stagehand can have different prices, and a
    # single global in/out pair would bill all of them at the same rate.
    # See app/pricing.py.
    #
    # The fields below are OVERRIDES: empty => catalog. Filling both (in and out)
    # forces that price for every model - the enterprise-contract case, where
    # list price does not apply.
    price_in_per_mtok: float | None = None
    price_cached_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None
    price_currency: str = "USD"

    # CSV path (empty => data/model_prices.csv in the repo)
    price_catalog_path: str | None = None
    # Provider prefix preferred when matching a deployment name against the
    # catalog. None => "azure/" when the provider is foundry/azure, "" on
    # OpenAI direct. Pinned-region deployments use "azure/us/", which costs
    # ~10% more and has its own row in the CSV.
    price_catalog_prefix: str | None = None
    # deployment with a custom name -> catalog model:
    #   PRICE_MODEL_ALIASES="my-prod-deploy=gpt-5.6-terra,rag-nano=gpt-5.4-nano"
    # There is deliberately no fuzzy matching: a silently wrong price is worse
    # than a visible placeholder.
    price_model_aliases: str | None = None

    # mock: reference budget targets, to validate the instrumentation itself
    mock_ttft_ms: int = 1050
    mock_tokens_per_s: int = 170
    mock_output_tokens: int = 250

    # ---------- latency levers ----------
    detect_locale: bool = False  # when the channel already knows the locale, this is waste
    classify_intent: bool = True
    speculative_retrieval: bool = True  # retrieval in parallel with intent
    cache_l1_enabled: bool = True  # exact
    # The canonical tier ships ON. Finding 06's conclusion was not "no semantic
    # cache", it was "replace similarity with a canonical key" - shipping the
    # key off would be shipping the recommendation unadopted. It also costs no
    # embedding, so unlike L2 there is no latency argument for keeping it off.
    cache_canonical_enabled: bool = True  # (entity, attribute) - finding 06
    # L2 off by default: see scripts/calibrate_l2.py and the findings - there is
    # no threshold that separates a paraphrase from a near-match with the
    # opposite answer.
    cache_l2_enabled: bool = False  # semantic (costs 1 embedding - measured)
    cache_l2_threshold: float = 0.95
    cache_ttl_s: int = 3600
    cached_replay_tokens_per_s: int = 0  # 0 = deliver a cache hit at once (measures the real floor)

    # ---------- agent topology ----------
    # true  -> agent loop: the model picks the tools (>= 2 hops when it uses one)
    # false -> fixed pipeline: retrieval always, 1 hop
    # Comparing the two is the most useful measurement this engine makes.
    agentic: bool = True

    # ---------- tools ----------
    # kb_search       local ChromaDB + in-process ONNX embedding  ~20-40 ms
    # metric_lookup   in-memory dict                             ~0.01 ms
    # latency_budget  pure arithmetic                            ~0.01 ms
    # web_search      Browserbase Search - internet search        ~1000 ms
    # web_fetch       Browserbase Fetch - page as markdown        ~1000-3000 ms
    # simulate_tool is in the default list on purpose: it is the only slow tool
    # that needs no credential, so the hop and parallelism lessons are
    # reproducible on a fresh clone. It is a declared simulator and says so in
    # the trace (`simulated: true`).
    enabled_tools: str = (
        "kb_search,metric_lookup,latency_budget,simulate_tool,summarize,web_search,web_fetch"
    )

    # Dedicated flag for the only tool that opens a browser session.
    # Deliberately outside the default: 12 s per call, it creates a Browserbase
    # session and needs `uv sync --extra browse`. Enabling it here avoids editing
    # the list above; being in the list also enables it (both ways add up).
    enable_web_browse: bool = False
    max_tool_hops: int = 3  # ceiling on model round trips per turn

    # ---------- what the trace carries ----------
    # The raw exchange with the model - the messages sent on each hop, the text
    # or tool_calls that came back, and each tool's actual output. Without it
    # the trace answers "how long" and "how much" but not "what happened", and
    # the three questions you actually debug with are: did it call the tool, did
    # the tool return anything usable, and what did the model see when it wrote
    # the answer. On by default because this engine exists to be looked at; the
    # per-field cap keeps a 6,000 character page from becoming the trace.
    trace_payloads: bool = True
    trace_payload_chars: int = 4000

    chroma_collection: str = "kb_local"
    chroma_persist_dir: str | None = None  # empty => in-memory index, rebuilt at startup

    # browserbase is the default; duckduckgo is the keyless fallback (important
    # so the repo runs out of the box). If the chosen backend has no key the code
    # falls back to duckduckgo and the TRACE records which backend actually ran -
    # a number measured with one backend cannot be read as the other's.
    web_search_backend: Literal[
        "browserbase", "duckduckgo", "tavily", "brave", "serper"
    ] = "browserbase"
    web_search_timeout_s: float = 20.0

    browserbase_api_key: str | None = None
    browserbase_project_id: str | None = None  # not required by Search/Fetch
    # Fetch can return up to 5 MB of markdown. Without a cap, one agent turn
    # swallows the whole page as input tokens - it is a cost and latency lever,
    # so it stays explicit.
    web_fetch_max_chars: int = 6000
    web_fetch_format: Literal["markdown", "raw"] = "markdown"

    # Stagehand (web_browse): real browser with act/extract. An order of
    # magnitude slower than Search/Fetch - off by default.
    stagehand_model: str = "openai/gpt-4.1"
    stagehand_timeout_s: float = 90.0

    tavily_api_key: str | None = None
    brave_api_key: str | None = None
    serper_api_key: str | None = None

    # ---------- retrieval ----------
    # local  -> in-memory hybrid index over the local corpus (no cloud dependency)
    # stub   -> just sleeps retrieval_stub_ms (isolates the model's latency)
    # search -> real Azure AI Search
    retriever: Literal["local", "stub", "search"] = "local"
    retrieval_stub_ms: int = 150  # reference: ~150 ms for a remote index query
    retrieval_top_k: int = 3

    azure_search_endpoint: str | None = None
    azure_search_api_key: str | None = None
    azure_search_index: str = "kb-content"

    # ---------- external state ----------
    redis_url: str | None = None  # empty => in-process fallback (flagged in the trace)

    # ---------- harness ----------
    default_locale: str = "en-US"
    trace_buffer_size: int = 5000
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def enabled_tools_list(self) -> list[str]:
        tools = [t.strip() for t in self.enabled_tools.split(",") if t.strip()]
        if self.enable_web_browse and "web_browse" not in tools:
            tools.append("web_browse")
        return tools

    def tier_model(self, tier: str) -> str:
        """The model/deployment name for a tier, honouring a per-provider override.

        Everything that needs to name a model goes through here - the LLM client,
        the price resolution and /healthz - so a tier can never be billed under
        one name and called under another.
        """
        base = {
            "nano": self.nano_model,
            "mini": self.mini_model,
            "frontier": self.frontier_model,
        }[tier]
        if self.llm_provider == "openai":
            override = {
                "nano": self.openai_nano_model,
                "mini": self.openai_mini_model,
                "frontier": self.openai_frontier_model,
            }[tier]
            return override or base
        return base

    @property
    def embedding_model_effective(self) -> str:
        if self.llm_provider == "openai" and self.openai_embedding_model:
            return self.openai_embedding_model
        return self.embedding_model

    @property
    def is_mock(self) -> bool:
        return self.llm_provider == "mock"

    @property
    def foundry_base_url(self) -> str | None:
        """Normalise the Foundry endpoint to the v1 inference base.

        Accepts the three forms the Azure portal shows:
          https://<r>.services.ai.azure.com
          https://<r>.services.ai.azure.com/api/projects/<project>
          https://<r>.cognitiveservices.azure.com
        and always returns <host>/openai/v1/.
        """
        if not self.azure_ai_endpoint:
            return None
        raw = self.azure_ai_endpoint.strip().rstrip("/")
        for marker in ("/api/projects", "/openai"):
            if marker in raw:
                raw = raw.split(marker)[0]
        return f"{raw}/openai/v1/"


@lru_cache
def get_settings() -> Settings:
    return Settings()
