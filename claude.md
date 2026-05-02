# NEXUS — Claude Code Instructions

## IDENTITY
Principal-level AI/GenAI engineer. Think before coding. No eager generation.

## PROJECT
Autonomous Multi-Agent Competitive Intelligence System.
Query → Research → Hybrid RAG → Analysis → Fact-check → HITL gate → Structured report → Langfuse traces + RAGAS eval.

## STACK
- Orchestration: LangGraph (supervisor pattern)
- Vector DB: Qdrant (BM25 sparse + bge-m3 dense, hybrid search)
- Reranking: Cohere Rerank
- Observability: Langfuse
- Evaluation: RAGAS + DeepEval
- Structured output: instructor + Pydantic v2
- Guardrails: Guardrails AI
- LLM: Claude Sonnet (async Anthropic SDK)
- Backend: FastAPI + Celery + Redis
- Settings: pydantic-settings
- Logging: structlog (JSON)
- Tests: pytest + pytest-asyncio

## STRUCTURE
```
nexus/
├── api/main.py, routers/research.py, routers/reports.py, dependencies.py
├── agents/supervisor.py, research_agent.py, kb_builder_agent.py, analysis_agent.py, factcheck_agent.py
├── rag/chunker.py, embedder.py, retriever.py, hyde.py
├── schemas/state.py, report.py, config.py
├── evaluation/ragas_eval.py, golden_dataset.py
├── observability/langfuse_client.py
├── workers/tasks.py
├── tests/unit/, tests/integration/
├── docker-compose.yml, .env.example, pyproject.toml
```

## THREE-ENGINEER CHECK (mandatory before any architectural decision)
```
┌─ VERIFICATION ──────────────────────────────────────┐
│ Pragmatist  : simplest thing that works at 3am?    │
│ Reliability : failure modes + retries + logging?   │
│ AI Systems  : right LLM pattern? token efficient?  │
│ Decision    : what we're building and why          │
└─────────────────────────────────────────────────────┘
```

## CODING RULES
- Pydantic v2 + type hints on every function. No `Any` without comment.
- Every external call has try/except with structlog error + meaningful exception.
- No `print()`. Always `structlog`.
- All secrets via `Settings(BaseSettings)`. Zero hardcoded values.
- FastAPI endpoints and LLM calls are always `async def`.
- No `pass`, `# TODO`, `raise NotImplementedError`. If not built now, not in file.
- One test per non-trivial function in `tests/unit/` or `tests/integration/`.
- `docker compose up` always works after any infra change.
- `.env.example` always in sync with `Settings`.
- LangGraph native APIs only. No deprecated LangChain imports.

## SCOPE RULE
One component per session. State exact scope before touching files. Finish and test before moving on.

## SESSION END (required)
```
┌─ DONE ──────────────────────────────────────────────┐
│ Built    : [files changed]                         │
│ Decisions: [key choices]                           │
│ Test now : [exact command]                         │
│ Next     : [one next task]                         │
└─────────────────────────────────────────────────────┘
```

## PHASES
- [x] 1: FastAPI + NexusState + Docker + Research Agent
- [x] 2: RAG pipeline (hybrid search + rerank + HyDE)
- [x] 3: Agent orchestration + HITL + state persistence
- [x] 4: Structured output + Guardrails AI
- [x] 5: Langfuse + RAGAS evaluation
- [ ] 6: React UI + polish + README

**Current: Phase 5 COMPLETE — begin Phase 6**