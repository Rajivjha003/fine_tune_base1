"""
MerchFine Master Orchestration CLI.

Unified entry point for all pipeline operations.

Usage:
    python orchestrate.py --mode full          # Full pipeline: data → train → eval → deploy
    python orchestrate.py --mode eval-only     # Just run evaluation suite
    python orchestrate.py --mode retrain       # Data prep + training only
    python orchestrate.py --mode health        # System health check
    python orchestrate.py --mode feedback-loop # Process feedback → augment → retrain
    python orchestrate.py --mode validate      # Run config validation (_validate.py)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging for CLI operation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


async def run_full(args: argparse.Namespace) -> int:
    """Full pipeline: data → train → eval → deploy."""
    from orchestrator.pipeline import PipelineOrchestrator
    from core.system_init import SystemInitializer
    
    await SystemInitializer().full_bootstrap()

    orchestrator = PipelineOrchestrator()
    result = await orchestrator.run_full_pipeline()
    
    _print_result("Full Pipeline", result)
    return 0 if result.get("status") == "success" else 1


async def run_eval_only(args: argparse.Namespace) -> int:
    """Run only the evaluation suite."""
    from orchestrator.pipeline import PipelineOrchestrator
    from core.system_init import SystemInitializer
    
    await SystemInitializer().full_bootstrap()

    orchestrator = PipelineOrchestrator()
    result = await orchestrator.run_eval_only()
    
    _print_result("Evaluation", result)
    return 0 if result.get("status") == "success" else 1


async def run_retrain(args: argparse.Namespace) -> int:
    """Run data preparation + training only."""
    from orchestrator.pipeline import PipelineOrchestrator
    from core.system_init import SystemInitializer

    await SystemInitializer().full_bootstrap()

    orchestrator = PipelineOrchestrator()
    result = await orchestrator.run_retrain()
    
    _print_result("Retrain", result)
    return 0 if result.get("status") == "success" else 1


async def run_health(args: argparse.Namespace) -> int:
    """System health check."""
    from orchestrator.pipeline import PipelineOrchestrator
    from core.system_init import SystemInitializer

    await SystemInitializer().full_bootstrap()

    orchestrator = PipelineOrchestrator()
    result = await orchestrator.run_health_check()
    
    return 0 if result.get("status") == "healthy" else 1


async def run_feedback_loop(args: argparse.Namespace) -> int:
    """Process feedback → augment → retrain."""
    from orchestrator.pipeline import PipelineOrchestrator
    from core.system_init import SystemInitializer

    await SystemInitializer().full_bootstrap()

    orchestrator = PipelineOrchestrator()
    result = await orchestrator.run_feedback_loop()
    
    _print_result("Feedback Loop", result)
    return 0 if result.get("status") in ("success", "skipped") else 1


async def run_validate(args: argparse.Namespace) -> int:
    """Run configuration validation."""
    try:
        from core.config import get_settings
        settings = get_settings()
        
        print("=" * 50)
        print("  MerchFine Configuration Validation")
        print("=" * 50)
        
        checks = [
            ("Models config", bool(settings.models.models)),
            ("Training profiles", bool(settings.training.profiles)),
            ("Evaluation gates", bool(settings.evaluation.gates)),
            ("RAG config", bool(settings.rag)),
            ("Guardrails config", bool(settings.guardrails)),
        ]
        
        all_pass = True
        for name, passed in checks:
            status = "[OK]" if passed else "[FAIL]"
            print(f"  {status} {name}")
            if not passed:
                all_pass = False

        # Check test cases
        test_cases_path = settings.project_root / "evaluation" / "test_cases" / "domain_qa.jsonl"
        tc_count = 0
        if test_cases_path.exists():
            with open(test_cases_path, "r") as f:
                tc_count = sum(1 for line in f if line.strip())
        tc_status = "[OK]" if tc_count >= 10 else "[WARN]"
        print(f"  {tc_status} Test cases: {tc_count} loaded (min: 10)")

        # Check knowledge base
        kb_dir = settings.data_dir / "knowledge_base"
        kb_files = list(kb_dir.glob("*")) if kb_dir.exists() else []
        kb_status = "[OK]" if len(kb_files) >= 1 else "[WARN]"
        print(f"  {kb_status} Knowledge base: {len(kb_files)} files")

        print("=" * 50)
        overall = "PASS" if all_pass else "FAIL"
        print(f"  Result: {overall}")
        print("=" * 50)
        
        return 0 if all_pass else 1
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return 1


def _print_result(stage: str, result: dict) -> None:
    """Pretty-print pipeline stage results."""
    status = result.get("status", "unknown")
    icon = "[OK]" if status == "success" else "[FAIL]" if "fail" in status else "[->]"
    
    print(f"\n{'='*50}")
    print(f"  {icon} {stage}: {status.upper()}")
    print(f"{'='*50}")
    
    if "error" in result:
        print(f"  Error: {result['error']}")
    
    if "stages" in result:
        for stage_name, stage_data in result["stages"].items():
            if isinstance(stage_data, dict):
                s = stage_data.get("status", "?")
                print(f"    {stage_name}: {s}")
    
    if result.get("metrics"):
        print("\n  Metrics:")
        for name, data in result["metrics"].items():
            if isinstance(data, dict):
                score = data.get("score", "?")
                passed = "[OK]" if data.get("passed") else "[FAIL]"
                print(f"    {passed} {name}: {score}")
    
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MerchFine LLMOps Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrate.py --mode health        # Check system health
  python orchestrate.py --mode validate      # Validate configs
  python orchestrate.py --mode eval-only     # Run evaluation suite
  python orchestrate.py --mode full          # Full pipeline
  python orchestrate.py --mode full -v       # Full pipeline with debug logging
        """,
    )
    parser.add_argument(
        "--mode", "-m",
        required=True,
        choices=["full", "eval-only", "retrain", "health", "feedback-loop", "validate"],
        help="Pipeline mode to execute",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger("orchestrate")
    logger.info("MerchFine Orchestrator starting in '%s' mode...", args.mode)

    # Map mode to async function
    mode_map = {
        "full": run_full,
        "eval-only": run_eval_only,
        "retrain": run_retrain,
        "health": run_health,
        "feedback-loop": run_feedback_loop,
        "validate": run_validate,
    }

    handler = mode_map[args.mode]
    exit_code = asyncio.run(handler(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
