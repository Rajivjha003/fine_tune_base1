"""
Hybrid retriever combining dense (embedding) + sparse (BM25) search.

At query time:
1. Dense retrieval via ChromaDB vector similarity
2. Sparse retrieval via BM25 keyword matching
3. Score fusion using Reciprocal Rank Fusion (RRF)
4. Auto-merging: if enough leaf nodes from the same parent are retrieved,
   merge upward to return the full parent context
5. Optional cross-encoder reranking

All parameters sourced from config/rag.yaml.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import get_settings
from core.exceptions import RAGRetrievalError

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid dense + sparse retriever with auto-merge and optional reranking.

    Implements the full retrieval pipeline from config/rag.yaml:
    - Dense: HuggingFace embedding → ChromaDB cosine similarity
    - Sparse: BM25 over the raw text corpus
    - Fusion: Reciprocal Rank Fusion (alpha-weighted)
    - Auto-merge: Parent context recovery from leaf matches
    - Rerank: Cross-encoder reranking (optional, CPU-based)
    """

    def __init__(self):
        self.settings = get_settings()
        self._rag_config = self.settings.rag
        self._bm25_index = None
        self._corpus: list[dict[str, Any]] = []  # [{text, metadata}, ...]
        self._index = None

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant context for a query.

        Pipeline:
        1. Dense retrieval (embedding similarity)
        2. BM25 sparse retrieval (if enabled)
        3. Reciprocal Rank Fusion
        4. Auto-merge parent nodes
        5. Rerank (if enabled)
        6. Return final_top_k results

        Args:
            query: The user's question or search query.
            top_k: Override for final number of results.

        Returns:
            List of dicts with keys: text, score, metadata, source_file
        """
        final_k = top_k or self._rag_config.retrieval.final_top_k
        initial_k = self._rag_config.retrieval.top_k

        try:
            # Step 1: Dense retrieval
            dense_results = await self._dense_retrieve(query, top_k=initial_k)

            # Step 2: Sparse retrieval (BM25)
            sparse_results = []
            if self._rag_config.retrieval.enable_bm25:
                sparse_results = self._bm25_retrieve(query, top_k=initial_k)

            # Step 3: Fusion
            if sparse_results:
                alpha = self._rag_config.retrieval.hybrid_alpha
                fused = self._reciprocal_rank_fusion(
                    dense_results, sparse_results, alpha=alpha
                )
            else:
                fused = dense_results

            # Step 4: Auto-merge parent context
            merged = self._auto_merge(fused)

            # Step 5: Rerank
            if self._rag_config.retrieval.enable_reranker:
                merged = await self._rerank(query, merged)

            # Step 6: Trim to final_top_k
            results = merged[:final_k]

            logger.info(
                "Retrieved %d results for query (dense=%d, sparse=%d, fused=%d, merged=%d).",
                len(results),
                len(dense_results),
                len(sparse_results),
                len(fused),
                len(merged),
            )

            return results

        except Exception as e:
            logger.error("Retrieval failed: %s", e, exc_info=True)
            raise RAGRetrievalError(f"Retrieval failed: {e}") from e

    async def _dense_retrieve(self, query: str, top_k: int = 12) -> list[dict[str, Any]]:
        """Retrieve via embedding similarity from ChromaDB."""
        if self._index is None:
            from rag.indexer import KnowledgeBaseIndexer

            indexer = KnowledgeBaseIndexer()
            self._index = indexer.get_retrieval_index()

        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                "text": node.get_text(),
                "score": node.get_score() if hasattr(node, "get_score") else 0.0,
                "metadata": node.metadata if hasattr(node, "metadata") else {},
                "source_file": (
                    node.metadata.get("source_file", "unknown")
                    if hasattr(node, "metadata")
                    else "unknown"
                ),
                "node_id": node.node_id if hasattr(node, "node_id") else "",
                "retrieval_type": "dense",
            })

        return results

    def _bm25_retrieve(self, query: str, top_k: int = 12) -> list[dict[str, Any]]:
        """Retrieve via BM25 keyword matching."""
        if self._bm25_index is None:
            self._build_bm25_index()

        if self._bm25_index is None:
            return []

        from rank_bm25 import BM25Okapi

        query_tokens = query.lower().split()
        scores = self._bm25_index.get_scores(query_tokens)

        # Get top-k indices
        import numpy as np

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._corpus[idx]
                results.append({
                    "text": doc["text"],
                    "score": float(scores[idx]),
                    "metadata": doc.get("metadata", {}),
                    "source_file": doc.get("metadata", {}).get("source_file", "unknown"),
                    "node_id": f"bm25_{idx}",
                    "retrieval_type": "sparse",
                })

        return results

    def _build_bm25_index(self) -> None:
        """Build BM25 index from the knowledge base documents."""
        try:
            from rank_bm25 import BM25Okapi
            from pathlib import Path

            source_dir = Path(self._rag_config.knowledge_base.source_dir)
            if not source_dir.exists():
                logger.warning("Knowledge base source dir not found: %s", source_dir)
                return

            # Read all documents
            extensions = set(self._rag_config.knowledge_base.supported_extensions)
            self._corpus = []

            for ext in extensions:
                for f in source_dir.rglob(f"*{ext}"):
                    try:
                        content = f.read_text(encoding="utf-8")
                        # Split into paragraphs for finer-grained BM25
                        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                        for para in paragraphs:
                            self._corpus.append({
                                "text": para,
                                "metadata": {
                                    "source_file": f.name,
                                    "source_path": str(f.resolve()),
                                },
                            })
                    except Exception as e:
                        logger.warning("BM25: Failed to read %s: %s", f, e)

            if not self._corpus:
                logger.warning("BM25: No documents found.")
                return

            # Tokenize corpus
            tokenized_corpus = [doc["text"].lower().split() for doc in self._corpus]
            self._bm25_index = BM25Okapi(tokenized_corpus)

            logger.info("BM25 index built: %d paragraphs from knowledge base.", len(self._corpus))

        except ImportError:
            logger.warning("rank-bm25 not installed — BM25 retrieval disabled.")

    @staticmethod
    def _reciprocal_rank_fusion(
        dense: list[dict[str, Any]],
        sparse: list[dict[str, Any]],
        *,
        alpha: float = 0.5,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion.

        RRF score = alpha * 1/(k+rank_dense) + (1-alpha) * 1/(k+rank_sparse)

        Args:
            dense: Dense retrieval results.
            sparse: Sparse retrieval results.
            alpha: Weight for dense results (0=sparse only, 1=dense only).
            k: RRF constant (default 60).

        Returns:
            Fused results sorted by combined score.
        """
        # Build lookup by text content (for deduplication)
        combined: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(dense):
            text = doc["text"]
            rrf_score = alpha * (1.0 / (k + rank + 1))
            if text in combined:
                combined[text]["rrf_score"] += rrf_score
            else:
                combined[text] = {**doc, "rrf_score": rrf_score}

        for rank, doc in enumerate(sparse):
            text = doc["text"]
            rrf_score = (1 - alpha) * (1.0 / (k + rank + 1))
            if text in combined:
                combined[text]["rrf_score"] += rrf_score
            else:
                combined[text] = {**doc, "rrf_score": rrf_score}

        # Sort by fused score
        fused = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Copy rrf_score to score
        for doc in fused:
            doc["score"] = doc.pop("rrf_score")
            doc["retrieval_type"] = "hybrid"

        return fused

    def _auto_merge(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Auto-merge leaf nodes into parent context when threshold is met.

        If >= auto_merge_threshold fraction of a parent's children are
        retrieved, replace the children with the full parent text.
        """
        threshold = self._rag_config.retrieval.auto_merge_threshold

        # Group results by parent node
        parent_groups: dict[str, list[dict[str, Any]]] = {}
        standalone: list[dict[str, Any]] = []

        for result in results:
            parent_id = result.get("metadata", {}).get("parent_id")
            if parent_id:
                if parent_id not in parent_groups:
                    parent_groups[parent_id] = []
                parent_groups[parent_id].append(result)
            else:
                standalone.append(result)

        # Check merge conditions
        merged: list[dict[str, Any]] = list(standalone)

        for parent_id, children in parent_groups.items():
            total_children = children[0].get("metadata", {}).get("sibling_count", len(children))
            ratio = len(children) / max(total_children, 1)

            if ratio >= threshold:
                # Merge: combine all children text into parent
                parent_text = "\n\n".join(c["text"] for c in children)
                avg_score = sum(c.get("score", 0) for c in children) / len(children)
                merged.append({
                    "text": parent_text,
                    "score": avg_score,
                    "metadata": {
                        **children[0].get("metadata", {}),
                        "auto_merged": True,
                        "merged_children": len(children),
                    },
                    "source_file": children[0].get("source_file", "unknown"),
                    "retrieval_type": "auto_merged",
                })
                logger.debug(
                    "Auto-merged %d/%d children for parent %s",
                    len(children),
                    total_children,
                    parent_id,
                )
            else:
                # Keep individual children
                merged.extend(children)

        # Re-sort by score
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged

    async def _rerank(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Rerank results using a cross-encoder model.

        CPU-based, adds ~200ms latency but improves precision.
        """
        try:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(
                self._rag_config.reranker.model_name,
                device=self._rag_config.reranker.device,
            )

            # Prepare pairs
            pairs = [(query, r["text"]) for r in results]

            # Score in batches
            scores = reranker.predict(
                pairs,
                batch_size=self._rag_config.reranker.batch_size,
            )

            # Update scores
            for result, score in zip(results, scores):
                result["original_score"] = result.get("score", 0)
                result["score"] = float(score)
                result["reranked"] = True

            # Sort by reranked score
            results.sort(key=lambda x: x["score"], reverse=True)

            logger.info("Reranked %d results with cross-encoder.", len(results))
            return results

        except ImportError:
            logger.warning("sentence-transformers not installed — reranking skipped.")
            return results
        except Exception as e:
            logger.error("Reranking failed: %s", e)
            return results

    def format_context(
        self,
        results: list[dict[str, Any]],
        *,
        max_tokens: int = 2048,
        separator: str = "\n---\n",
    ) -> str:
        """
        Format retrieved results into a context string for the LLM prompt.

        Includes source attribution and respects token budget.
        """
        parts = []
        estimated_tokens = 0

        for i, r in enumerate(results, 1):
            source = r.get("source_file", "unknown")
            score = r.get("score", 0)
            text = r["text"]

            # Rough token estimate (4 chars per token)
            text_tokens = len(text) // 4
            if estimated_tokens + text_tokens > max_tokens:
                # Truncate this passage to fit
                remaining = max_tokens - estimated_tokens
                text = text[: remaining * 4]
                parts.append(f"[Source: {source} | Relevance: {score:.2f}]\n{text}...")
                break

            parts.append(f"[Source: {source} | Relevance: {score:.2f}]\n{text}")
            estimated_tokens += text_tokens

        return separator.join(parts)
