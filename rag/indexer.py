"""
Hierarchical document indexer using LlamaIndex.

Implements a 3-level tree-structured index:
  Parent (2048 tokens) → Child (512 tokens) → Leaf (128 tokens)

At query time, leaf-level matches propagate upward — if enough siblings
from the same parent chunk are retrieved, the parent is auto-merged to
preserve full context. This dramatically improves answer quality for
multi-paragraph domain docs like MIO rules and seasonal calendars.

Usage:
    from rag.indexer import KnowledgeBaseIndexer
    indexer = KnowledgeBaseIndexer()
    indexer.build_index()          # Full rebuild from source_dir
    indexer.incremental_update()   # Detect changes and update
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.events import event_bus
from core.exceptions import RAGIndexError

logger = logging.getLogger(__name__)

# Event name for index rebuild
RAG_INDEX_REBUILT = "rag.index.rebuilt"


class KnowledgeBaseIndexer:
    """
    Builds and maintains a hierarchical vector index over the knowledge base.

    Pipeline:
    1. Scan source directory for supported files
    2. Compute content hashes (detect changes since last build)
    3. Parse documents into LlamaIndex Document objects
    4. Split into 3-level hierarchy (parent → child → leaf)
    5. Embed leaf nodes via configured embedding model
    6. Store in ChromaDB with hierarchical metadata
    """

    def __init__(self):
        self.settings = get_settings()
        self._rag_config = self.settings.rag
        self._source_dir = Path(self._rag_config.knowledge_base.source_dir)
        self._persist_dir = Path(self._rag_config.vector_store.persist_dir)
        self._hash_file = self._persist_dir / ".content_hashes.json"
        self._embed_model = None  # Lazy init

    def build_index(self, *, force: bool = False) -> dict[str, Any]:
        """
        Full index build from scratch.

        Args:
            force: Rebuild even if no changes detected.

        Returns:
            Summary dict with document counts, timing, etc.
        """
        start = time.time()
        logger.info("Building knowledge base index from %s...", self._source_dir)

        # Ensure directories exist
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # Scan source files
        source_files = self._scan_sources()
        if not source_files:
            raise RAGIndexError(f"No supported files found in {self._source_dir}")

        # Compute hashes
        current_hashes = self._compute_hashes(source_files)
        stored_hashes = self._load_hashes()

        if not force and current_hashes == stored_hashes:
            logger.info("No changes detected. Skipping rebuild. Use force=True to override.")
            return {"status": "no_changes", "files": len(source_files)}

        # Parse documents
        documents = self._parse_documents(source_files)
        logger.info("Parsed %d documents from %d files.", len(documents), len(source_files))

        # Create hierarchical nodes
        nodes = self._create_hierarchical_nodes(documents)
        logger.info(
            "Created %d hierarchical nodes (parent=%d, child=%d, leaf=%d).",
            nodes["total"],
            nodes["parent_count"],
            nodes["child_count"],
            nodes["leaf_count"],
        )

        # Build vector index
        index = self._build_vector_index(nodes["all_nodes"])

        # Save hashes
        self._save_hashes(current_hashes)

        elapsed = time.time() - start
        summary = {
            "status": "rebuilt",
            "files": len(source_files),
            "documents": len(documents),
            "total_nodes": nodes["total"],
            "parent_nodes": nodes["parent_count"],
            "child_nodes": nodes["child_count"],
            "leaf_nodes": nodes["leaf_count"],
            "elapsed_seconds": round(elapsed, 1),
        }

        logger.info("Index built in %.1fs. Summary: %s", elapsed, summary)

        # Emit event
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        loop.run_until_complete(
            event_bus.emit(
                RAG_INDEX_REBUILT,
                data=summary,
                source="indexer",
            )
        )

        return summary

    def incremental_update(self) -> dict[str, Any]:
        """
        Detect changed files and update only those.

        Compares content hashes from last build to current state.
        Only re-indexes files that have changed or been added.
        """
        source_files = self._scan_sources()
        current_hashes = self._compute_hashes(source_files)
        stored_hashes = self._load_hashes()

        changed = []
        new = []
        deleted = []

        for path, h in current_hashes.items():
            if path not in stored_hashes:
                new.append(path)
            elif stored_hashes[path] != h:
                changed.append(path)

        for path in stored_hashes:
            if path not in current_hashes:
                deleted.append(path)

        if not changed and not new and not deleted:
            logger.info("No changes detected.")
            return {"changed": 0, "new": 0, "deleted": 0}

        logger.info(
            "Incremental update: %d changed, %d new, %d deleted.",
            len(changed),
            len(new),
            len(deleted),
        )

        # For changed/new files, re-index them
        files_to_index = [Path(p) for p in changed + new]
        if files_to_index:
            documents = self._parse_documents(files_to_index)
            nodes = self._create_hierarchical_nodes(documents)
            self._add_to_index(nodes["all_nodes"])

        # For deleted files, remove from index
        if deleted:
            self._remove_from_index(deleted)

        # Update hashes
        self._save_hashes(current_hashes)

        return {
            "changed": len(changed),
            "new": len(new),
            "deleted": len(deleted),
        }

    def get_index_stats(self) -> dict[str, Any]:
        """Return statistics about the current index."""
        try:
            chroma = self._get_chroma_collection()
            count = chroma.count()
            return {
                "collection": self._rag_config.vector_store.collection_name,
                "document_count": count,
                "persist_dir": str(self._persist_dir),
                "embedding_model": self._rag_config.embedding.model_name,
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Private Methods ──────────────────────────────────────────────────

    def _scan_sources(self) -> list[Path]:
        """Scan the source directory for supported files."""
        if not self._source_dir.exists():
            logger.warning("Source directory does not exist: %s", self._source_dir)
            return []

        extensions = set(self._rag_config.knowledge_base.supported_extensions)
        files = []
        for ext in extensions:
            files.extend(self._source_dir.rglob(f"*{ext}"))

        files.sort(key=lambda p: p.name)
        return files

    def _compute_hashes(self, files: list[Path]) -> dict[str, str]:
        """Compute SHA256 hashes for each file."""
        hashes = {}
        for f in files:
            sha = hashlib.sha256(f.read_bytes()).hexdigest()
            hashes[str(f.resolve())] = sha
        return hashes

    def _load_hashes(self) -> dict[str, str]:
        """Load previously stored content hashes."""
        if self._hash_file.exists():
            return json.loads(self._hash_file.read_text(encoding="utf-8"))
        return {}

    def _save_hashes(self, hashes: dict[str, str]) -> None:
        """Save content hashes to disk."""
        self._hash_file.write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    def _parse_documents(self, files: list[Path]) -> list:
        """Parse files into LlamaIndex Document objects."""
        from llama_index.core import Document

        documents = []
        for f in files:
            try:
                content = f.read_text(encoding="utf-8")
                doc = Document(
                    text=content,
                    metadata={
                        "source_file": f.name,
                        "source_path": str(f.resolve()),
                        "file_type": f.suffix,
                        "content_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
                    },
                )
                documents.append(doc)
                logger.debug("Parsed: %s (%d chars)", f.name, len(content))
            except Exception as e:
                logger.error("Failed to parse %s: %s", f, e)

        return documents

    def _create_hierarchical_nodes(self, documents: list) -> dict[str, Any]:
        """
        Create a 3-level node hierarchy from documents.

        Level 1 (Parent): 2048 tokens — full context passages
        Level 2 (Child):   512 tokens — main retrieval units
        Level 3 (Leaf):    128 tokens — fine-grained matches

        The HierarchicalNodeParser creates parent-child relationships
        so auto-merging can reconstruct parent context at query time.
        """
        from llama_index.core.node_parser import (
            HierarchicalNodeParser,
            get_leaf_nodes,
        )

        chunk_sizes = [
            self._rag_config.chunking.parent_chunk_size,
            self._rag_config.chunking.child_chunk_size,
            self._rag_config.chunking.leaf_chunk_size,
        ]

        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes,
            chunk_overlap=self._rag_config.chunking.chunk_overlap,
        )

        all_nodes = parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(all_nodes)

        # Count by level
        parent_count = sum(
            1 for n in all_nodes
            if not hasattr(n, "parent_node") or n.parent_node is None
        )
        child_count = len(all_nodes) - parent_count - len(leaf_nodes)
        if child_count < 0:
            child_count = 0

        return {
            "all_nodes": all_nodes,
            "leaf_nodes": leaf_nodes,
            "total": len(all_nodes),
            "parent_count": parent_count,
            "child_count": child_count,
            "leaf_count": len(leaf_nodes),
        }

    def _get_embed_model(self):
        """Lazy-initialize the embedding model (CPU, keeps VRAM free)."""
        if self._embed_model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            self._embed_model = HuggingFaceEmbedding(
                model_name=self._rag_config.embedding.model_name,
                device=self._rag_config.embedding.device,
                embed_batch_size=self._rag_config.embedding.embed_batch_size,
                cache_folder=self._rag_config.embedding.cache_dir,
            )
            logger.info(
                "Embedding model loaded: %s (device=%s)",
                self._rag_config.embedding.model_name,
                self._rag_config.embedding.device,
            )

        return self._embed_model

    def _get_chroma_collection(self):
        """Get or create the ChromaDB collection."""
        import chromadb

        client = chromadb.PersistentClient(path=str(self._persist_dir))
        collection = client.get_or_create_collection(
            name=self._rag_config.vector_store.collection_name,
            metadata={"hnsw:space": self._rag_config.vector_store.distance_metric},
        )
        return collection

    def _build_vector_index(self, nodes: list) -> Any:
        """Build a VectorStoreIndex with ChromaDB backend."""
        from llama_index.core import (
            Settings as LlamaSettings,
            StorageContext,
            VectorStoreIndex,
        )
        from llama_index.vector_stores.chroma import ChromaVectorStore

        # Configure LlamaIndex settings
        LlamaSettings.embed_model = self._get_embed_model()
        LlamaSettings.chunk_size = self._rag_config.chunking.child_chunk_size
        LlamaSettings.chunk_overlap = self._rag_config.chunking.chunk_overlap

        # Set up ChromaDB vector store
        chroma_collection = self._get_chroma_collection()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build index from nodes
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )

        logger.info(
            "Vector index built with %d nodes in ChromaDB collection '%s'.",
            len(nodes),
            self._rag_config.vector_store.collection_name,
        )

        return index

    def _add_to_index(self, nodes: list) -> None:
        """Add new nodes to the existing index."""
        from llama_index.core import Settings as LlamaSettings

        LlamaSettings.embed_model = self._get_embed_model()

        chroma_collection = self._get_chroma_collection()
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import StorageContext, VectorStoreIndex

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
        )
        logger.info("Added %d nodes to existing index.", len(nodes))

    def _remove_from_index(self, deleted_paths: list[str]) -> None:
        """Remove nodes associated with deleted files from the index."""
        logger.info("Removing nodes for %d deleted files.", len(deleted_paths))
        # ChromaDB supports deletion by metadata filter
        try:
            chroma = self._get_chroma_collection()
            for path in deleted_paths:
                chroma.delete(where={"source_path": path})
                logger.debug("Removed nodes for: %s", path)
        except Exception as e:
            logger.error("Failed to remove nodes: %s", e)

    def get_retrieval_index(self):
        """
        Load the persisted index for retrieval (used by retriever.py).

        Returns a VectorStoreIndex connected to the persisted ChromaDB.
        """
        from llama_index.core import (
            Settings as LlamaSettings,
            StorageContext,
            VectorStoreIndex,
        )
        from llama_index.vector_stores.chroma import ChromaVectorStore

        LlamaSettings.embed_model = self._get_embed_model()

        chroma_collection = self._get_chroma_collection()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
        )

        return index
