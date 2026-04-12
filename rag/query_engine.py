"""
RAG-augmented query engine.

Ties together the indexer, retriever, and inference gateway into a
single interface that:
1. Retrieves relevant context from the knowledge base
2. Builds an augmented prompt with the retrieved context
3. Sends the prompt to the inference gateway
4. Returns the response with source attribution

Usage:
    from rag.query_engine import RAGQueryEngine
    engine = RAGQueryEngine()
    result = await engine.query("What is the MIO target for sneakers?")
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)

# Prompt template for RAG-augmented queries
RAG_PROMPT_TEMPLATE = """You are MerchFine, a retail demand forecasting and inventory planning AI assistant.

Answer the user's question using ONLY the context provided below. If the context does not contain enough information to answer the question fully, clearly state what information is missing.

=== CONTEXT ===
{context}
=== END CONTEXT ===

RULES:
1. Base your answer strictly on the provided context
2. Cite the source document when referencing specific data
3. If numbers are involved, include the exact values from the context
4. If the context is insufficient, say "I don't have enough information about..." and specify what's missing
5. Format your response clearly with bullet points or tables when appropriate

USER QUESTION: {query}

ANSWER:"""


class RAGQueryEngine:
    """
    End-to-end RAG query engine.

    Pipeline:
    1. Retrieve relevant context from knowledge base
    2. Build augmented prompt
    3. Send to inference gateway
    4. Return response with sources
    """

    def __init__(self):
        self.settings = get_settings()
        self._retriever = None
        self._gateway = None

    def _get_retriever(self):
        """Lazy-initialize the hybrid retriever."""
        if self._retriever is None:
            from rag.retriever import HybridRetriever

            self._retriever = HybridRetriever()
        return self._retriever

    def _get_gateway(self):
        """Lazy-initialize the inference gateway."""
        if self._gateway is None:
            from inference.gateway import InferenceGateway

            self._gateway = InferenceGateway()
        return self._gateway

    async def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        temperature: float | None = None,
        max_context_tokens: int = 2048,
        include_sources: bool = True,
    ) -> dict[str, Any]:
        """
        Execute a RAG-augmented query.

        Args:
            question: The user's natural language question.
            top_k: Number of context documents to retrieve.
            temperature: LLM temperature override.
            max_context_tokens: Max tokens for context window.
            include_sources: Whether to include source docs in response.

        Returns:
            Dict with keys: answer, sources, context_used, latency_ms
        """
        start = time.time()
        retriever = self._get_retriever()
        gateway = self._get_gateway()

        # Step 1: Retrieve context
        retrieved = await retriever.retrieve(question, top_k=top_k)

        # Step 2: Format context
        context_str = retriever.format_context(
            retrieved,
            max_tokens=max_context_tokens,
        )

        # Step 3: Build augmented prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context_str,
            query=question,
        )

        # Step 4: Send to inference gateway
        response = await gateway.complete(
            prompt=prompt,
            temperature=temperature,
        )

        # Step 5: Build result
        elapsed_ms = (time.time() - start) * 1000

        result: dict[str, Any] = {
            "answer": response.get("text", ""),
            "model_used": response.get("model", ""),
            "latency_ms": round(elapsed_ms, 1),
            "context_documents": len(retrieved),
        }

        if include_sources:
            result["sources"] = [
                {
                    "source_file": r.get("source_file", "unknown"),
                    "relevance_score": round(r.get("score", 0), 3),
                    "retrieval_type": r.get("retrieval_type", "unknown"),
                    "preview": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                }
                for r in retrieved
            ]

        result["usage"] = response.get("usage", {})

        logger.info(
            "RAG query completed in %.0fms (%d context docs, model=%s).",
            elapsed_ms,
            len(retrieved),
            result["model_used"],
        )

        return result

    async def query_with_chat_history(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        *,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        RAG query with conversation context for multi-turn scenarios.

        Rewrites the query using chat history before retrieval to
        handle co-references (e.g., "What about that product?" → resolves "that").
        """
        retriever = self._get_retriever()
        gateway = self._get_gateway()

        # If we have chat history, condense the question
        effective_query = question
        if chat_history and len(chat_history) > 0:
            effective_query = await self._condense_question(question, chat_history)

        # Standard RAG flow with the condensed question
        return await self.query(effective_query, top_k=top_k)

    async def _condense_question(
        self,
        question: str,
        chat_history: list[dict[str, str]],
    ) -> str:
        """
        Rewrite the user's question to be self-contained using chat history.

        Resolves pronouns and co-references so the retriever gets a
        clean, standalone query.
        """
        gateway = self._get_gateway()

        # Build condensation prompt
        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in chat_history[-4:]  # Last 4 turns max
        )

        condense_prompt = (
            "Given the following conversation history and a follow-up question, "
            "rewrite the follow-up question to be a standalone question that "
            "can be understood without the conversation history.\n\n"
            f"CONVERSATION HISTORY:\n{history_text}\n\n"
            f"FOLLOW-UP QUESTION: {question}\n\n"
            "STANDALONE QUESTION:"
        )

        response = await gateway.complete(
            prompt=condense_prompt,
            temperature=0.0,
            max_tokens=256,
        )

        condensed = response.get("text", question).strip()
        if condensed:
            logger.debug("Condensed query: '%s' -> '%s'", question, condensed)
            return condensed

        return question
