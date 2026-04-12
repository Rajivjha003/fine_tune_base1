"""
Forecasting API routes with RAG-augmented context retrieval.

POST /api/forecast — demand forecasting (RAG-augmented)
POST /api/mio-plan — MIO plan generation (RAG-augmented)
POST /api/chat — agentic chat (RAG-augmented)
POST /api/rag/query — direct RAG query
POST /api/rag/index — trigger knowledge base rebuild
GET  /api/rag/stats — index statistics
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Request / Response Models ─────────────────────────────────────────────


class ForecastRequest(BaseModel):
    """Request body for demand forecasting."""

    sku_id: str = Field(..., description="SKU identifier from the catalog")
    horizon_days: int = Field(default=30, ge=1, le=365, description="Forecast horizon in days")
    context: str = Field(default="", description="Additional context (recent trends, promotions)")
    include_reasoning: bool = Field(default=True, description="Include reasoning in response")
    use_rag: bool = Field(default=True, description="Augment prompt with retrieved knowledge base context")


class MIOPlanRequest(BaseModel):
    """Request body for MIO plan generation."""

    sku_id: str = Field(..., description="SKU identifier")
    current_stock: float = Field(..., ge=0, description="Current on-hand inventory units")
    avg_monthly_sales: float | None = Field(default=None, description="Override average monthly sales")
    context: str = Field(default="", description="Additional context (season, promotions)")
    use_rag: bool = Field(default=True, description="Augment with MIO rules from knowledge base")


class ChatRequest(BaseModel):
    """Request body for agentic chat."""

    message: str = Field(..., min_length=1, description="User message")
    conversation_id: str | None = Field(default=None, description="ID for multi-turn conversations")
    include_sources: bool = Field(default=True, description="Include source references")
    use_rag: bool = Field(default=True, description="Augment with knowledge base context")


class RAGQueryRequest(BaseModel):
    """Request body for direct RAG queries."""

    query: str = Field(..., min_length=1, description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context documents")
    include_sources: bool = Field(default=True)


class ForecastResponse(BaseModel):
    sku_id: str
    horizon_days: int
    forecast: str
    reasoning: str = ""
    model_used: str = ""
    latency_ms: float = 0.0
    trace_id: str = ""
    sources_used: list[str] = Field(default_factory=list)


class MIOPlanResponse(BaseModel):
    sku_id: str
    plan: str
    reasoning: str = ""
    model_used: str = ""
    latency_ms: float = 0.0
    trace_id: str = ""
    sources_used: list[str] = Field(default_factory=list)


class ChatResponseModel(BaseModel):
    message: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    model_used: str = ""
    latency_ms: float = 0.0
    trace_id: str = ""
    conversation_id: str = ""


# ── Routes ────────────────────────────────────────────────────────────────


@router.post("/forecast", response_model=ForecastResponse)
async def create_forecast(request: Request, body: ForecastRequest) -> ForecastResponse:
    """
    Generate a demand forecast for a specific SKU.

    Pipeline:
    1. Retrieve SKU data + seasonal info from knowledge base (if use_rag=True)
    2. Build augmented prompt with retrieved context
    3. Query the inference gateway
    4. Return forecast with reasoning and sources
    """
    trace_id = getattr(request.state, "trace_id", "")

    try:
        if body.use_rag:
            result = await _rag_augmented_completion(
                query=_build_forecast_query(body),
                prompt_builder=lambda ctx: _build_forecast_prompt(body, context=ctx),
            )
        else:
            from inference.gateway import InferenceGateway

            gateway = InferenceGateway()
            result = await gateway.complete(prompt=_build_forecast_prompt(body))
            result["sources"] = []

    except Exception as e:
        logger.error("Forecast inference failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Inference failed: {e}")

    return ForecastResponse(
        sku_id=body.sku_id,
        horizon_days=body.horizon_days,
        forecast=result.get("text", ""),
        reasoning="" if not body.include_reasoning else result.get("text", ""),
        model_used=result.get("model", ""),
        latency_ms=result.get("latency_ms", 0.0),
        trace_id=trace_id,
        sources_used=[s.get("source_file", "") for s in result.get("sources", [])],
    )


@router.post("/mio-plan", response_model=MIOPlanResponse)
async def create_mio_plan(request: Request, body: MIOPlanRequest) -> MIOPlanResponse:
    """Generate an MIO-based inventory replenishment plan with RAG context."""
    trace_id = getattr(request.state, "trace_id", "")

    try:
        if body.use_rag:
            result = await _rag_augmented_completion(
                query=_build_mio_query(body),
                prompt_builder=lambda ctx: _build_mio_prompt(body, context=ctx),
            )
        else:
            from inference.gateway import InferenceGateway

            gateway = InferenceGateway()
            result = await gateway.complete(prompt=_build_mio_prompt(body))
            result["sources"] = []

    except Exception as e:
        logger.error("MIO plan inference failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Inference failed: {e}")

    return MIOPlanResponse(
        sku_id=body.sku_id,
        plan=result.get("text", ""),
        reasoning=result.get("text", ""),
        model_used=result.get("model", ""),
        latency_ms=result.get("latency_ms", 0.0),
        trace_id=trace_id,
        sources_used=[s.get("source_file", "") for s in result.get("sources", [])],
    )


@router.post("/chat", response_model=ChatResponseModel)
async def chat(request: Request, body: ChatRequest) -> ChatResponseModel:
    """Agentic chat endpoint with optional RAG augmentation."""
    trace_id = getattr(request.state, "trace_id", "")

    try:
        if body.use_rag:
            from rag.query_engine import RAGQueryEngine

            engine = RAGQueryEngine()
            rag_result = await engine.query(
                body.message,
                include_sources=body.include_sources,
            )

            return ChatResponseModel(
                message=rag_result.get("answer", ""),
                sources=rag_result.get("sources", []),
                model_used=rag_result.get("model_used", ""),
                latency_ms=rag_result.get("latency_ms", 0.0),
                trace_id=trace_id,
                conversation_id=body.conversation_id or trace_id,
            )
        else:
            from inference.gateway import InferenceGateway

            gateway = InferenceGateway()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are MerchFine, a retail demand forecasting and inventory "
                        "planning assistant. Provide accurate, data-driven analysis. "
                        "If you lack sufficient data, clearly state what's needed."
                    ),
                },
                {"role": "user", "content": body.message},
            ]
            result = await gateway.chat(messages=messages)

            msg = result.get("message", {})
            return ChatResponseModel(
                message=msg.get("content", "") if isinstance(msg, dict) else str(msg),
                model_used=result.get("model", ""),
                latency_ms=result.get("latency_ms", 0.0),
                trace_id=trace_id,
                conversation_id=body.conversation_id or trace_id,
            )

    except Exception as e:
        logger.error("Chat inference failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Inference failed: {e}")


# ── RAG-specific routes ──────────────────────────────────────────────────


@router.post("/rag/query")
async def rag_query(body: RAGQueryRequest) -> dict[str, Any]:
    """Direct RAG query against the knowledge base."""
    from rag.query_engine import RAGQueryEngine

    engine = RAGQueryEngine()
    try:
        result = await engine.query(
            body.query,
            top_k=body.top_k,
            include_sources=body.include_sources,
        )
        return result
    except Exception as e:
        logger.error("RAG query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")


@router.post("/rag/index")
async def rebuild_index(force: bool = False) -> dict[str, Any]:
    """Trigger a knowledge base index rebuild."""
    from rag.indexer import KnowledgeBaseIndexer

    indexer = KnowledgeBaseIndexer()
    try:
        result = indexer.build_index(force=force)
        return result
    except Exception as e:
        logger.error("Index rebuild failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {e}")


@router.get("/rag/stats")
async def rag_stats() -> dict[str, Any]:
    """Get knowledge base index statistics."""
    from rag.indexer import KnowledgeBaseIndexer

    indexer = KnowledgeBaseIndexer()
    return indexer.get_index_stats()


# ── Helper Functions ──────────────────────────────────────────────────────


async def _rag_augmented_completion(
    query: str,
    prompt_builder,
) -> dict[str, Any]:
    """
    Retrieve context via RAG, build augmented prompt, call inference.

    Returns dict with 'text', 'model', 'latency_ms', 'sources' keys.
    """
    import time

    from inference.gateway import InferenceGateway
    from rag.retriever import HybridRetriever

    start = time.time()
    retriever = HybridRetriever()

    # Retrieve context
    retrieved = await retriever.retrieve(query)

    # Format context
    context_str = retriever.format_context(retrieved, max_tokens=2048)

    # Build final prompt
    prompt = prompt_builder(context_str)

    # Get completion
    gateway = InferenceGateway()
    result = await gateway.complete(prompt=prompt)

    elapsed_ms = (time.time() - start) * 1000
    result["latency_ms"] = round(elapsed_ms, 1)
    result["sources"] = [
        {
            "source_file": r.get("source_file", "unknown"),
            "score": round(r.get("score", 0), 3),
        }
        for r in retrieved
    ]

    return result


def _build_forecast_query(body: ForecastRequest) -> str:
    """Build a retrieval query optimized for the knowledge base."""
    return f"SKU {body.sku_id} demand forecast sales seasonality {body.horizon_days} days"


def _build_mio_query(body: MIOPlanRequest) -> str:
    """Build a retrieval query for MIO planning."""
    return f"SKU {body.sku_id} MIO inventory reorder stock {body.current_stock} units"


def _build_forecast_prompt(body: ForecastRequest, context: str = "") -> str:
    """Build a structured forecast prompt with optional RAG context."""
    parts = [
        f"Generate a demand forecast for SKU '{body.sku_id}' over the next {body.horizon_days} days.",
    ]

    if context:
        parts.extend([
            "",
            "=== KNOWLEDGE BASE CONTEXT ===",
            context,
            "=== END CONTEXT ===",
            "",
        ])

    parts.extend([
        "Provide:",
        "1. Predicted daily or weekly demand quantities",
        "2. Confidence level (high/medium/low)",
        "3. Key factors influencing the forecast",
        "4. Risks and caveats",
    ])

    if body.context:
        parts.extend(["", f"Additional context: {body.context}"])

    return "\n".join(parts)


def _build_mio_prompt(body: MIOPlanRequest, context: str = "") -> str:
    """Build a structured MIO plan prompt with optional RAG context."""
    parts = [
        f"Generate an inventory replenishment plan for SKU '{body.sku_id}'.",
        "",
        f"Current stock on hand: {body.current_stock} units",
    ]
    if body.avg_monthly_sales is not None:
        parts.append(f"Average monthly sales: {body.avg_monthly_sales} units")

    if context:
        parts.extend([
            "",
            "=== MIO RULES & SKU DATA ===",
            context,
            "=== END CONTEXT ===",
            "",
        ])

    parts.extend([
        "",
        "Provide:",
        "1. Current MIO (Months of Inventory Outstanding)",
        "2. Recommended reorder quantity",
        "3. Reorder urgency (immediate/soon/planned)",
        "4. Reasoning based on MIO rules and seasonal factors",
    ])

    if body.context:
        parts.extend(["", f"Additional context: {body.context}"])

    return "\n".join(parts)
