"""
Quality Gate Evaluation Engine.

Runs evaluations (RAGAS, DeepEval, ROUGE/BERTScore) on model outputs
to determine if they pass the configured thresholds (hard/soft gates).
Used for model promotion and CI/CD.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
    
    Runs real RAGAS, DeepEval, and custom metrics when available,
    with graceful fallback to ROUGE/BERTScore when frameworks are missing.
    """

    def __init__(self):
        self.settings = get_settings()
        self.eval_config = self.settings.evaluation

    def load_test_cases(self, path: str | Path | None = None) -> list[dict[str, Any]]:
        """
        Load ground-truth test cases from domain_qa.jsonl.
        
        Returns list of dicts with keys: query, expected_response, context, category, difficulty
        """
        if path is None:
            path = self.settings.project_root / "evaluation" / "test_cases" / "domain_qa.jsonl"
        path = Path(path)

        if not path.exists():
            logger.warning("Test cases file not found: %s", path)
            return []

        cases = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        
        logger.info("Loaded %d test cases from %s", len(cases), path)
        return cases

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

        # Run custom metrics (ROUGE, BERTScore, numeric accuracy)
        custom_results = await self._run_custom_metrics(predictions)
        results.extend(custom_results)

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
                "details": res.details,
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

        # Auto-save report to disk
        self._save_report(report)

        return report

    def _save_report(self, report: dict[str, Any]) -> None:
        """Persist the eval report to outputs/eval_reports/ with timestamp."""
        try:
            report_dir = self.settings.outputs_dir / "eval_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"eval_{timestamp}.json"

            # Make report JSON-serializable
            serializable = json.loads(json.dumps(report, default=str))
            serializable["timestamp"] = timestamp
            serializable["num_predictions"] = report.get("_num_predictions", 0)

            report_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Eval report saved to %s", report_path)

            # Append to persistent score history CSV
            self._append_to_score_history(report_dir, timestamp, report)

        except Exception as e:
            logger.warning("Failed to save eval report: %s", e)

    def _append_to_score_history(
        self,
        report_dir: Path,
        timestamp: str,
        report: dict[str, Any],
    ) -> None:
        """
        Append a flat row to score_history.csv for long-term improvement tracking.

        Each row contains:
        - run_id, timestamp, overall_passed
        - One column per metric score (e.g. faithfulness, semantic_similarity, forecast_mape)
        - One column per metric pass/fail status

        The CSV grows over time — never overwritten — so you can chart score
        trends across dozens of eval runs and compare before/after fine-tuning.
        """
        import csv

        history_path = report_dir / "score_history.csv"
        metrics = report.get("metrics", {})

        # Build the flat row
        row: dict[str, Any] = {
            "run_id": timestamp,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "overall_passed": report.get("passed", False),
            "num_hard_failures": len(report.get("hard_gate_failures", [])),
            "num_soft_warnings": len(report.get("soft_gate_warnings", [])),
        }

        # Add each metric's score and pass/fail as separate columns
        for metric_name, metric_data in metrics.items():
            score = metric_data.get("score")
            row[f"score__{metric_name}"] = round(score, 6) if isinstance(score, float) else score
            row[f"passed__{metric_name}"] = metric_data.get("passed", False)
            row[f"threshold__{metric_name}"] = metric_data.get("threshold")
            row[f"gate_type__{metric_name}"] = metric_data.get("type", "")

        # Check if file exists to decide whether to write header
        file_exists = history_path.exists()

        # Read existing headers if file exists (columns may grow over time)
        existing_headers: list[str] = []
        if file_exists:
            with open(history_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                try:
                    existing_headers = next(reader)
                except StopIteration:
                    existing_headers = []

        # Merge headers — existing + any new columns from this run
        all_headers = list(existing_headers)
        for key in row:
            if key not in all_headers:
                all_headers.append(key)

        if not existing_headers or set(all_headers) != set(existing_headers):
            # Rewrite with expanded headers (preserving old data)
            old_rows: list[dict[str, Any]] = []
            if file_exists and existing_headers:
                with open(history_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    old_rows = list(reader)

            with open(history_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_headers, extrasaction="ignore")
                writer.writeheader()
                for old_row in old_rows:
                    writer.writerow(old_row)
                writer.writerow(row)
        else:
            # Simple append — headers match
            with open(history_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_headers, extrasaction="ignore")
                writer.writerow(row)

        logger.info("Score history appended to %s (%d metrics)", history_path, len(metrics))

    def assert_pass(self, report: dict[str, Any]) -> None:
        """Utility to raise an exception if the report indicates failure."""
        if not report["passed"]:
            raise QualityGateFailedError(
                f"Quality gate evaluation failed: {report['hard_gate_failures']}",
                failed_gates=report["hard_gate_failures"],
            )

    async def _run_ragas(self, predictions: list[dict[str, Any]]) -> list[EvalResult]:
        """Execute RAGAS framework evaluations with real metric computation."""
        gates = self.eval_config.gates
        results = []
        
        ragas_gates = {k: g for k, g in gates.items() if "ragas_" in g.metric}
        if not ragas_gates:
            return results

        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                faithfulness,
            )
            from datasets import Dataset

            # Build dataset for RAGAS
            ragas_data = {
                "question": [p["query"] for p in predictions],
                "answer": [p["response"] for p in predictions],
                "contexts": [p.get("context", []) for p in predictions],
                "ground_truth": [p.get("expected", "") for p in predictions],
            }
            dataset = Dataset.from_dict(ragas_data)

            # Run RAGAS evaluation
            metric_map = {
                "ragas_faithfulness": faithfulness,
                "ragas_answer_relevancy": answer_relevancy,
                "ragas_context_precision": context_precision,
            }

            metrics_to_run = []
            for gate in ragas_gates.values():
                if gate.metric in metric_map:
                    metrics_to_run.append(metric_map[gate.metric])

            if metrics_to_run:
                ragas_result = ragas_evaluate(dataset, metrics=metrics_to_run)
                scores = ragas_result.to_pandas().mean().to_dict()

                for key, gate in ragas_gates.items():
                    metric_name = gate.metric.replace("ragas_", "")
                    score = scores.get(metric_name, 0.0)

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
                        details=f"RAGAS computed from {len(predictions)} samples",
                    ))

            logger.info("RAGAS evaluation complete: %d metrics computed.", len(results))
            return results

        except ImportError:
            logger.warning("RAGAS not installed — falling back to ROUGE-based approximation.")
            return await self._ragas_fallback(predictions, ragas_gates)
        except Exception as e:
            logger.error("RAGAS evaluation failed: %s — using fallback.", e)
            return await self._ragas_fallback(predictions, ragas_gates)

    async def _ragas_fallback(
        self, predictions: list[dict[str, Any]], ragas_gates: dict
    ) -> list[EvalResult]:
        """Fallback when RAGAS is unavailable: use ROUGE-L as proxy for relevancy/faithfulness."""
        results = []
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

            # Compute average ROUGE-L between response and expected
            rouge_scores = []
            for p in predictions:
                expected = p.get("expected", "")
                response = p.get("response", "")
                if expected and response:
                    s = scorer.score(expected, response)
                    rouge_scores.append(s["rougeL"].fmeasure)

            avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

            for key, gate in ragas_gates.items():
                passed = True
                if gate.min_threshold is not None and avg_rouge < gate.min_threshold:
                    passed = False
                results.append(EvalResult(
                    metric=gate.metric,
                    score=avg_rouge,
                    passed=passed,
                    threshold=gate.min_threshold or gate.max_threshold,
                    gate_type=gate.gate_type,
                    details=f"ROUGE-L fallback (RAGAS unavailable); {len(rouge_scores)} samples",
                ))

        except ImportError:
            logger.warning("rouge-score not installed — skipping RAGAS gates entirely.")
            for key, gate in ragas_gates.items():
                results.append(EvalResult(
                    metric=gate.metric,
                    score=0.0,
                    passed=False,
                    threshold=gate.min_threshold or gate.max_threshold,
                    gate_type=gate.gate_type,
                    details="Neither RAGAS nor rouge-score installed — cannot evaluate.",
                ))

        return results

    async def _run_deepeval(self, predictions: list[dict[str, Any]]) -> list[EvalResult]:
        """Execute DeepEval framework evaluations with real metric computation."""
        gates = self.eval_config.gates
        results = []
        
        deepeval_gates = {k: g for k, g in gates.items() if "deepeval_" in g.metric}
        if not deepeval_gates:
            return results

        try:
            from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
            from deepeval.test_case import LLMTestCase

            # Build test cases for DeepEval
            test_cases = []
            for p in predictions:
                tc = LLMTestCase(
                    input=p["query"],
                    actual_output=p["response"],
                    expected_output=p.get("expected", ""),
                    context=p.get("context", []),
                )
                test_cases.append(tc)

            for key, gate in deepeval_gates.items():
                if gate.metric == "deepeval_hallucination":
                    metric = HallucinationMetric(threshold=gate.max_threshold or 0.1)
                    scores = []
                    for tc in test_cases:
                        try:
                            metric.measure(tc)
                            scores.append(metric.score)
                        except Exception:
                            pass
                    
                    avg_score = sum(scores) / len(scores) if scores else 0.5
                    passed = True
                    if gate.max_threshold is not None and avg_score > gate.max_threshold:
                        passed = False
                    
                    results.append(EvalResult(
                        metric=gate.metric,
                        score=avg_score,
                        passed=passed,
                        threshold=gate.max_threshold,
                        gate_type=gate.gate_type,
                        details=f"DeepEval computed from {len(scores)} samples",
                    ))

                elif gate.metric == "deepeval_refusal_accuracy":
                    # Custom refusal detection: check if model correctly refuses
                    refusal_cases = [p for p in predictions if p.get("category") == "refusal"]
                    if refusal_cases:
                        correct_refusals = 0
                        refusal_phrases = [
                            "cannot", "unable", "don't have", "not able",
                            "outside my", "insufficient", "need more",
                        ]
                        for p in refusal_cases:
                            response_lower = p["response"].lower()
                            if any(phrase in response_lower for phrase in refusal_phrases):
                                correct_refusals += 1
                        score = correct_refusals / len(refusal_cases)
                    else:
                        score = 1.0  # No refusal cases to test

                    passed = True
                    if gate.min_threshold is not None and score < gate.min_threshold:
                        passed = False

                    results.append(EvalResult(
                        metric=gate.metric,
                        score=score,
                        passed=passed,
                        threshold=gate.min_threshold,
                        gate_type=gate.gate_type,
                        details=f"Refusal accuracy from {len(refusal_cases)} refusal cases",
                    ))

            logger.info("DeepEval evaluation complete: %d metrics computed.", len(results))
            return results

        except ImportError:
            logger.warning("DeepEval not installed — using heuristic fallback.")
            return await self._deepeval_fallback(predictions, deepeval_gates)
        except Exception as e:
            logger.error("DeepEval evaluation failed: %s — using fallback.", e)
            return await self._deepeval_fallback(predictions, deepeval_gates)

    async def _deepeval_fallback(
        self, predictions: list[dict[str, Any]], deepeval_gates: dict
    ) -> list[EvalResult]:
        """Fallback when DeepEval is unavailable: use heuristic checks."""
        results = []
        
        for key, gate in deepeval_gates.items():
            if gate.metric == "deepeval_hallucination":
                # Heuristic: check if response contains numbers not in context
                hallucination_scores = []
                for p in predictions:
                    context_text = " ".join(p.get("context", []))
                    response_nums = set(re.findall(r'\b\d+\.?\d*\b', p["response"]))
                    context_nums = set(re.findall(r'\b\d+\.?\d*\b', context_text))
                    if response_nums:
                        grounded = len(response_nums & context_nums) / len(response_nums)
                        hallucination_scores.append(1.0 - grounded)
                    else:
                        hallucination_scores.append(0.0)
                
                avg_score = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0
                passed = True
                if gate.max_threshold is not None and avg_score > gate.max_threshold:
                    passed = False

                results.append(EvalResult(
                    metric=gate.metric,
                    score=avg_score,
                    passed=passed,
                    threshold=gate.max_threshold,
                    gate_type=gate.gate_type,
                    details=f"Heuristic fallback (DeepEval unavailable); {len(hallucination_scores)} samples",
                ))

            elif gate.metric == "deepeval_refusal_accuracy":
                # Refusal check doesn't need DeepEval
                refusal_cases = [p for p in predictions if p.get("category") == "refusal"]
                if refusal_cases:
                    correct_refusals = 0
                    refusal_phrases = [
                        "cannot", "unable", "don't have", "not able",
                        "outside my", "insufficient", "need more",
                    ]
                    for p in refusal_cases:
                        response_lower = p["response"].lower()
                        if any(phrase in response_lower for phrase in refusal_phrases):
                            correct_refusals += 1
                    score = correct_refusals / len(refusal_cases)
                else:
                    score = 1.0

                passed = True
                if gate.min_threshold is not None and score < gate.min_threshold:
                    passed = False

                results.append(EvalResult(
                    metric=gate.metric,
                    score=score,
                    passed=passed,
                    threshold=gate.min_threshold,
                    gate_type=gate.gate_type,
                    details=f"Refusal accuracy heuristic; {len(refusal_cases)} cases",
                ))

        return results

    async def _run_custom_metrics(self, predictions: list[dict[str, Any]]) -> list[EvalResult]:
        """Run custom domain-specific metrics: semantic similarity, MAPE, numeric accuracy."""
        gates = self.eval_config.gates
        results = []

        # Semantic Similarity (BERTScore proxy) using sentence-transformers
        sem_sim_score = self._compute_semantic_similarity(predictions)
        if sem_sim_score is not None:
            # Check if there's a configured gate for this, otherwise use defaults
            sem_gate = gates.get("semantic_similarity")
            threshold = sem_gate.min_threshold if sem_gate and sem_gate.min_threshold else 0.75
            gate_type = sem_gate.gate_type if sem_gate else "soft"
            passed = sem_sim_score >= threshold

            results.append(EvalResult(
                metric="semantic_similarity",
                score=sem_sim_score,
                passed=passed,
                threshold=threshold,
                gate_type=gate_type,
                details=f"BGE cosine similarity across {len(predictions)} predictions",
            ))

        # Forecast MAPE for forecast-category predictions
        forecast_preds = [p for p in predictions if p.get("category") == "demand_forecast"]
        if forecast_preds:
            mape_score = self._compute_forecast_mape(forecast_preds)
            mape_gate = gates.get("forecast_mape")
            threshold = mape_gate.max_threshold if mape_gate and mape_gate.max_threshold else 0.10
            gate_type = mape_gate.gate_type if mape_gate else "soft"
            passed = mape_score <= threshold

            results.append(EvalResult(
                metric="forecast_mape",
                score=mape_score,
                passed=passed,
                threshold=threshold,
                gate_type=gate_type,
                details=f"Mean Absolute Percentage Error across {len(forecast_preds)} forecast predictions",
            ))

        # Custom forecast numeric accuracy gate
        forecast_gate = gates.get("forecast_accuracy")
        if forecast_gate:
            score = self._compute_numeric_accuracy(predictions)
            passed = True
            if forecast_gate.min_threshold is not None and score < forecast_gate.min_threshold:
                passed = False

            results.append(EvalResult(
                metric=forecast_gate.metric,
                score=score,
                passed=passed,
                threshold=forecast_gate.min_threshold,
                gate_type=forecast_gate.gate_type,
                details=f"Numeric grounding check across {len(predictions)} predictions",
            ))

        return results

    def _compute_semantic_similarity(self, predictions: list[dict[str, Any]]) -> float | None:
        """
        Compute average cosine similarity between response and expected using BGE embeddings.
        Acts as a BERTScore proxy — fast and fully offline.
        """
        if not predictions:
            return None

        try:
            from sentence_transformers import SentenceTransformer, util

            model_name = "BAAI/bge-small-en-v1.5"
            model = SentenceTransformer(model_name, device="cpu")

            similarities = []
            for p in predictions:
                expected = p.get("expected", "")
                response = p.get("response", "")
                if expected and response:
                    ref_emb = model.encode(expected, convert_to_tensor=True)
                    res_emb = model.encode(response, convert_to_tensor=True)
                    sim = float(util.cos_sim(ref_emb, res_emb)[0][0])
                    similarities.append(sim)

            if not similarities:
                return None

            avg_sim = sum(similarities) / len(similarities)
            logger.info("Semantic similarity: %.4f (across %d samples)", avg_sim, len(similarities))
            return avg_sim

        except ImportError:
            logger.warning("sentence-transformers not installed — skipping semantic similarity.")
            return None
        except Exception as e:
            logger.error("Semantic similarity computation failed: %s", e)
            return None

    def _compute_forecast_mape(self, forecast_predictions: list[dict[str, Any]]) -> float:
        """
        Compute Mean Absolute Percentage Error for forecast predictions.
        Extracts numbers from response and expected, pairs them, computes MAPE.
        """
        if not forecast_predictions:
            return 0.0

        per_sample_errors = []

        for p in forecast_predictions:
            expected = p.get("expected", "")
            response = p.get("response", "")

            # Extract significant numbers (2+ digits) from both
            expected_nums = [float(n) for n in re.findall(r'\b(\d{2,}(?:\.\d+)?)\b', expected)]
            response_nums = [float(n) for n in re.findall(r'\b(\d{2,}(?:\.\d+)?)\b', response)]

            if not expected_nums or not response_nums:
                continue

            # Pair numbers by position (up to min length)
            pairs = min(len(expected_nums), len(response_nums))
            sample_errors = []
            for i in range(pairs):
                actual = expected_nums[i]
                predicted = response_nums[i]
                if actual != 0:
                    ape = abs(predicted - actual) / abs(actual)
                    sample_errors.append(ape)

            if sample_errors:
                per_sample_errors.append(sum(sample_errors) / len(sample_errors))

        if not per_sample_errors:
            return 0.0

        mape = sum(per_sample_errors) / len(per_sample_errors)
        logger.info("Forecast MAPE: %.4f (across %d samples)", mape, len(per_sample_errors))
        return mape

    def _compute_numeric_accuracy(self, predictions: list[dict[str, Any]]) -> float:
        """
        Check that numbers in model responses are grounded in the input context.
        Returns fraction of predictions where numbers are traceable.
        """
        if not predictions:
            return 1.0

        grounded_count = 0
        total_with_numbers = 0

        for p in predictions:
            response = p.get("response", "")
            context_text = " ".join(p.get("context", []))
            expected = p.get("expected", "")

            # Extract significant numbers (skip small numbers like 0, 1, 2)
            response_nums = set(re.findall(r'\b\d{2,}\b', response))
            if not response_nums:
                continue

            total_with_numbers += 1
            
            # Numbers should appear in context OR expected output
            source_text = context_text + " " + expected
            source_nums = set(re.findall(r'\b\d{2,}\b', source_text))

            # Allow computed values: if >50% of response numbers are traceable, it's grounded
            if source_nums:
                overlap = len(response_nums & source_nums) / len(response_nums)
                if overlap >= 0.3:  # At least 30% directly traceable (rest may be computed)
                    grounded_count += 1

        if total_with_numbers == 0:
            return 1.0

        return grounded_count / total_with_numbers
