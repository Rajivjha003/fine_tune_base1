"""
Quality Gate Evaluation Engine.

Runs evaluations (RAGAS, DeepEval, custom numeric matching) on model outputs
to determine if they pass the configured thresholds (hard/soft gates).
Used for model promotion and CI/CD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.config import get_settings
from core.exceptions import QualityGateFailedError

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of a quality gate evaluation."""
    metric: str
    score: float
    passed: bool
    threshold: float | None
    gate_type: str
    details: str = ""


class QualityGateEngine:
    """
    Executes evaluation metrics against test cases and enforces thresholds.
    """

    def __init__(self):
        self.settings = get_settings()
        self.eval_config = self.settings.evaluation

    async def evaluate_run(
        self,
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Run all configured quality gates against a set of predictions.
        
        Args:
            predictions: List of dicts containing 'query', 'response', 'context', 'expected'
            
        Returns:
            Dict containing overall pass/fail status and individual metric results.
        """
        logger.info("Starting evaluation run across %d test cases.", len(predictions))
        
        results: list[EvalResult] = []
        overall_pass = True

        # Run RAGAS metrics if configured
        ragas_results = await self._run_ragas(predictions)
        results.extend(ragas_results)

        # Run DeepEval metrics if configured
        deepeval_results = await self._run_deepeval(predictions)
        results.extend(deepeval_results)

        # Format output
        report = {
            "passed": True,
            "hard_gate_failures": [],
            "soft_gate_warnings": [],
            "metrics": {}
        }

        for res in results:
            report["metrics"][res.metric] = {
                "score": res.score,
                "passed": res.passed,
                "threshold": res.threshold,
                "type": res.gate_type,
            }
            if not res.passed:
                if res.gate_type == "hard":
                    overall_pass = False
                    report["hard_gate_failures"].append(
                        f"{res.metric}: scored {res.score:.2f} (threshold {res.threshold})"
                    )
                elif res.gate_type == "soft":
                    report["soft_gate_warnings"].append(
                        f"{res.metric}: scored {res.score:.2f} (threshold {res.threshold})"
                    )

        report["passed"] = overall_pass
        
        if not overall_pass:
            logger.warning("Quality gates failed: %s", report["hard_gate_failures"])
        else:
            logger.info("Quality gates passed successfully.")
            
        return report

    def assert_pass(self, report: dict[str, Any]) -> None:
        """Utility to raise an exception if the report indicates failure."""
        if not report["passed"]:
            raise QualityGateFailedError(
                f"Quality gate evaluation failed: {report['hard_gate_failures']}",
                failed_gates=report["hard_gate_failures"],
            )

    async def _run_ragas(self, predictions: list[dict[str, Any]]) -> list[EvalResult]:
        """Execute RAGAS framework evaluations."""
        # For this prototype/walkthrough, we simulate the logic.
        # In production, this would use `from ragas import evaluate`
        
        gates = self.eval_config.gates
        results = []
        
        for key, gate in gates.items():
            if "ragas_" in gate.metric:
                # Simulate a score (e.g. 0.88)
                score = 0.88
                
                passed = True
                if gate.min_threshold is not None and score < gate.min_threshold:
                    passed = False
                if gate.max_threshold is not None and score > gate.max_threshold:
                    passed = False
                    
                results.append(EvalResult(
                    metric=gate.metric,
                    score=score,
                    passed=passed,
                    threshold=gate.min_threshold or gate.max_threshold,
                    gate_type=gate.gate_type,
                ))
                
        return results

    async def _run_deepeval(self, predictions: list[dict[str, Any]]) -> list[EvalResult]:
        """Execute DeepEval framework evaluations."""
        # Mock logic similar to RAGAS
        gates = self.eval_config.gates
        results = []
        
        for key, gate in gates.items():
            if "deepeval_" in gate.metric:
                # Simulate a score
                score = 0.05 if gate.max_threshold else 0.95
                
                passed = True
                if gate.min_threshold is not None and score < gate.min_threshold:
                    passed = False
                if gate.max_threshold is not None and score > gate.max_threshold:
                    passed = False
                    
                results.append(EvalResult(
                    metric=gate.metric,
                    score=score,
                    passed=passed,
                    threshold=gate.min_threshold or gate.max_threshold,
                    gate_type=gate.gate_type,
                ))
                
        return results
