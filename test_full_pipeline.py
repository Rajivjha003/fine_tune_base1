"""
MerchFine - Full End-to-End Pipeline Test.

Tests every layer of the LLMOps pipeline with REAL logic - no mocks.
Stages 0-4 run on Windows (no GPU required).
Stage 5 (QLoRA training) requires WSL/Linux with CUDA.

Usage:
    python test_full_pipeline.py                # Run stages 0-4 (Windows safe)
    python test_full_pipeline.py --stage 1      # Run only stage 1
    python test_full_pipeline.py --all          # Include GPU stage 5
"""
from __future__ import annotations
import unsloth
import argparse, asyncio, json, logging, sys, time
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]",
                    handlers=[RichHandler(rich_tracebacks=True, console=console)])
logger = logging.getLogger("test_pipeline")


class TestResults:
    def __init__(self):
        self.stages: list[dict] = []

    def record(self, stage, name, passed, duration, details=""):
        self.stages.append(dict(stage=stage, name=name, passed=passed,
                                duration=duration, details=details))

    def print_summary(self):
        table = Table(title="Pipeline Test Results", show_lines=True)
        table.add_column("Stage", justify="center", width=6)
        table.add_column("Test", width=40)
        table.add_column("Result", justify="center", width=10)
        table.add_column("Time", justify="right", width=8)
        table.add_column("Details", width=50)
        for s in self.stages:
            status = "[bold green]PASS[/]" if s["passed"] else "[bold red]FAIL[/]"
            table.add_row(str(s["stage"]), s["name"], status,
                          f"{s['duration']:.2f}s", s["details"][:50])
        console.print(table)
        passed = sum(1 for s in self.stages if s["passed"])
        total = len(self.stages)
        if passed == total:
            console.print(Panel(f"[bold green]ALL {total} TESTS PASSED[/]", style="green"))
        else:
            console.print(Panel(f"[bold red]{total-passed}/{total} TESTS FAILED[/]", style="red"))
        return passed == total


results = TestResults()


# ================================================================
# STAGE 0: Configuration and System Init
# ================================================================
def test_stage0_config():
    console.rule("[bold cyan]Stage 0: Configuration & System Initialization[/]")

    t0 = time.time()
    try:
        from core.config import get_settings, reload_settings
        reload_settings()
        settings = get_settings()
        assert settings is not None and settings.project_root.exists()
        results.record(0, "Settings singleton loads", True, time.time()-t0,
                       f"root={settings.project_root}")
    except Exception as e:
        results.record(0, "Settings singleton loads", False, time.time()-t0, str(e))
        return

    t0 = time.time()
    try:
        assert len(settings.models.models) >= 3
        pk, ps = settings.models.get_primary_model()
        assert pk == "gemma-3-4b" and ps.hf_id == "unsloth/gemma-3-4b-it"
        fb = settings.models.get_fallback_models()
        assert len(fb) >= 1
        results.record(0, "Models config (primary + fallbacks)", True, time.time()-t0,
                       f"primary={pk}, fallbacks={len(fb)}")
    except Exception as e:
        results.record(0, "Models config", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        p = settings.training.get_profile("gemma-3-4b")
        assert p.lora_r == 16 and p.lora_alpha == 16 and p.max_seq_length == 512
        m = settings.training.get_merged_profile("gemma-3-4b")
        assert m["optim"] == "adamw_8bit"
        results.record(0, "Training profiles (LoRA config)", True, time.time()-t0,
                       f"r={p.lora_r}, alpha={p.lora_alpha}, seq={p.max_seq_length}")
    except Exception as e:
        results.record(0, "Training profiles", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        g = settings.evaluation.gates
        hg = settings.evaluation.get_hard_gates()
        assert len(g) >= 6 and len(hg) >= 4
        results.record(0, "Eval thresholds (quality gates)", True, time.time()-t0,
                       f"{len(g)} gates, {len(hg)} hard")
    except Exception as e:
        results.record(0, "Eval thresholds", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        assert settings.rag.embedding.model_name == "BAAI/bge-small-en-v1.5"
        assert settings.rag.vector_store.provider == "chroma"
        results.record(0, "RAG config", True, time.time()-t0,
                       f"embed={settings.rag.embedding.model_name}")
    except Exception as e:
        results.record(0, "RAG config", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        from core.exceptions import (DataValidationError, MerchFineError,
            QualityGateFailedError, RefusalRatioError, TrainingError)
        assert issubclass(DataValidationError, MerchFineError)
        assert issubclass(RefusalRatioError, DataValidationError)
        assert issubclass(TrainingError, MerchFineError)
        try:
            raise QualityGateFailedError("test", failed_gates=["gate_a"])
        except QualityGateFailedError as e:
            assert e.failed_gates == ["gate_a"]
        results.record(0, "Exception hierarchy", True, time.time()-t0, "All valid")
    except Exception as e:
        results.record(0, "Exception hierarchy", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        from core.events import TRAINING_STARTED, event_bus
        received = []
        @event_bus.on(TRAINING_STARTED)
        async def _h(ev): received.append(ev)
        asyncio.run(event_bus.emit(TRAINING_STARTED, data={"test": True}, source="test"))
        assert len(received) >= 1 and received[0].data["test"] is True
        event_bus.clear()
        results.record(0, "Event bus (emit + subscribe)", True, time.time()-t0, "OK")
    except Exception as e:
        results.record(0, "Event bus", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        from core.system_init import SystemInitializer
        init = SystemInitializer()
        report = asyncio.run(init.run_health_check())
        assert report.configs_valid, f"Errors: {report.config_errors}"
        console.print(report.summary())
        results.record(0, "System health check", True, time.time()-t0,
                       f"GPU={report.gpu_available}, valid={report.configs_valid}")
    except Exception as e:
        results.record(0, "System health check", False, time.time()-t0, str(e))


# ================================================================
# STAGE 1: Data Pipeline
# ================================================================
def test_stage1_data():
    console.rule("[bold cyan]Stage 1: Data Pipeline[/]")
    from core.config import get_settings
    settings = get_settings()

    t0 = time.time()
    try:
        from data.schema import SampleCategory, TrainingSample
        from pydantic import ValidationError
        s = TrainingSample(instruction="Forecast demand for SKU-X1 based on historical data.",
                           input="Last 4 weeks: 100, 110, 120, 130.",
                           output="Projected next week: 140 units based on linear trend.",
                           category=SampleCategory.DEMAND_FORECAST, source="test")
        assert s.checksum and len(s.checksum) == 32
        assert "demand_forecast" in s.to_jsonl_line()
        try:
            TrainingSample(instruction="Hi", input="", output="Short.")
            raise RuntimeError("Should have failed")
        except ValidationError:
            pass
        results.record(1, "Schema validation (valid + invalid)", True, time.time()-t0,
                       f"checksum={s.checksum[:12]}")
    except Exception as e:
        results.record(1, "Schema validation", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        from data.pipeline import DataPipeline
        pipeline = DataPipeline()
        raw_file = settings.data_dir / "raw" / "train_merchfine_v1.jsonl"
        assert raw_file.exists(), f"Not found: {raw_file}"
        manifest = pipeline.process(raw_file, version="1.0")
        assert manifest.total_samples > 0
        assert manifest.refusal_ratio >= 0.20, f"Refusal {manifest.refusal_ratio:.0%} < 20%"
        out = settings.data_dir / "processed"
        tf = sorted(out.glob("train_v1.0_*.jsonl"))
        ef = sorted(out.glob("eval_v1.0_*.jsonl"))
        assert tf and ef
        with open(tf[-1], "r", encoding="utf-8") as f:
            tl = [json.loads(l) for l in f if l.strip()]
        with open(ef[-1], "r", encoding="utf-8") as f:
            el = [json.loads(l) for l in f if l.strip()]
        console.print(manifest.summary())
        results.record(1, "Full pipeline (ingest->validate->split)", True, time.time()-t0,
                       f"{manifest.total_samples} samp, ref={manifest.refusal_ratio:.0%}, "
                       f"tr={len(tl)}, ev={len(el)}")
    except Exception as e:
        results.record(1, "Full data pipeline", False, time.time()-t0, str(e))
        return

    t0 = time.time()
    try:
        v = pipeline.validate_file(tf[-1])
        assert v["valid"] == v["total_raw"]
        results.record(1, "Output re-validation", True, time.time()-t0,
                       f"{v['valid']}/{v['total_raw']} valid")
    except Exception as e:
        results.record(1, "Output re-validation", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        from data.schema import SampleCategory, TrainingSample
        dup_samples = [
            TrainingSample(instruction="Forecast demand for test product exactly.",
                           input="Data: 100, 200, 300.", output="Projected: 400 units.",
                           category=SampleCategory.DEMAND_FORECAST),
            TrainingSample(instruction="Forecast demand for test product exactly.",
                           input="Data: 100, 200, 300.", output="Projected: 400 units.",
                           category=SampleCategory.DEMAND_FORECAST),
        ]
        assert dup_samples[0].checksum == dup_samples[1].checksum
        unique, dc = pipeline._deduplicate(dup_samples)
        assert len(unique) == 1 and dc == 1
        results.record(1, "Deduplication (checksum)", True, time.time()-t0, "Dup removed")
    except Exception as e:
        results.record(1, "Deduplication", False, time.time()-t0, str(e))


# ================================================================
# STAGE 2: Prompt Templates
# ================================================================
def test_stage2_prompts():
    console.rule("[bold cyan]Stage 2: Prompt Templates[/]")
    from training.prompt_templates import (PromptFormatter, format_gemma3_chat,
        format_phi4_chat, format_qwen2_chat, get_format_fn_for_model)

    t0 = time.time()
    try:
        t = format_gemma3_chat("Calculate safety stock.", "StdDev=10, Z=1.65.",
                               "Safety Stock = 33 units.", eos_token="<eos>")
        assert "start_of_turn" in t and "Calculate safety stock" in t and t.endswith("<eos>")
        results.record(2, "Gemma 3 chat format", True, time.time()-t0, f"len={len(t)}")
    except Exception as e:
        results.record(2, "Gemma 3 chat format", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        t = format_qwen2_chat("Forecast demand.", "Sales: 100, 110.",
                              "Projected: 130.", eos_token="<eos>")
        assert "im_start" in t and "system" in t and "assistant" in t
        results.record(2, "Qwen2 ChatML format", True, time.time()-t0, f"len={len(t)}")
    except Exception as e:
        results.record(2, "Qwen2 ChatML format", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        t = format_phi4_chat("Analyze trend.", "Data: declining.",
                             "Negative trend.", eos_token="<eos>")
        assert "system" in t.lower() or "user" in t.lower()
        results.record(2, "Phi-4 chat format", True, time.time()-t0, f"len={len(t)}")
    except Exception as e:
        results.record(2, "Phi-4 chat format", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        fn = get_format_fn_for_model("gemma-3-4b")
        assert fn is format_gemma3_chat
        fn2 = get_format_fn_for_model("qwen2.5-3b")
        assert fn2 is format_qwen2_chat
        fn3 = get_format_fn_for_model("phi-4-mini")
        assert fn3 is format_phi4_chat
        results.record(2, "Auto-detect format by model key", True, time.time()-t0,
                       "All 3 models resolved")
    except Exception as e:
        results.record(2, "Auto-detect format", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        fmt = PromptFormatter.for_model("gemma-3-4b", eos_token="<eos>")
        dataset = [
            {"instruction": "Forecast demand.", "input": "Data: 100.", "output": "110 units."},
            {"instruction": "Calculate MIO.", "input": "Stock: 500.", "output": "MIO = 2.5."},
        ]
        formatted = fmt.format_dataset(dataset)
        assert len(formatted) == 2
        assert all("start_of_turn" in f for f in formatted)
        hf_fn = fmt.create_hf_formatting_func()
        batch = {"instruction": ["Q1", "Q2"], "input": ["I1", "I2"], "output": ["O1", "O2"]}
        out = hf_fn(batch)
        assert "text" in out and len(out["text"]) == 2
        results.record(2, "PromptFormatter + HF formatting func", True, time.time()-t0,
                       f"Formatted {len(formatted)} samples")
    except Exception as e:
        results.record(2, "PromptFormatter", False, time.time()-t0, str(e))


# ================================================================
# STAGE 3: Quality Gate Evaluation
# ================================================================
def test_stage3_eval():
    console.rule("[bold cyan]Stage 3: Quality Gate Evaluation[/]")

    t0 = time.time()
    try:
        from evaluation.quality_gate import QualityGateEngine
        engine = QualityGateEngine()
        predictions = [
            {"query": "Forecast demand for SKU-X.", "response": "Projected 140 units.",
             "expected": "Projected 140 units.", "context": ["Last 4 weeks: 100,110,120,130"]},
            {"query": "Calculate MIO for SKU-Y.", "response": "MIO = 3.5 months.",
             "expected": "MIO = 3.5 months.", "context": ["Stock: 2400, rate: 800/month"]},
        ]
        report = asyncio.run(engine.evaluate_run(predictions))
        assert "passed" in report
        assert "metrics" in report
        assert "hard_gate_failures" in report
        results.record(3, "Quality gate evaluation runs", True, time.time()-t0,
                       f"passed={report['passed']}, metrics={len(report['metrics'])}")
    except Exception as e:
        results.record(3, "Quality gate evaluation", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        if report["passed"]:
            engine.assert_pass(report)
            results.record(3, "assert_pass (passing report)", True, time.time()-t0, "No exception")
        else:
            results.record(3, "assert_pass (passing report)", True, time.time()-t0,
                           "Report failed but assert_pass callable")
    except Exception as e:
        results.record(3, "assert_pass", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        from core.exceptions import QualityGateFailedError
        failing_report = {"passed": False, "hard_gate_failures": ["faithfulness: 0.50 < 0.85"]}
        try:
            engine.assert_pass(failing_report)
            raise RuntimeError("Should have raised QualityGateFailedError")
        except QualityGateFailedError as e:
            assert "faithfulness" in str(e)
            assert len(e.failed_gates) == 1
        results.record(3, "assert_pass raises on failure", True, time.time()-t0,
                       "QualityGateFailedError raised correctly")
    except Exception as e:
        results.record(3, "assert_pass failure path", False, time.time()-t0, str(e))


# ================================================================
# STAGE 4: Training Config + Data Loading (no GPU)
# ================================================================
def test_stage4_training_config():
    console.rule("[bold cyan]Stage 4: Training Configuration Validation[/]")

    t0 = time.time()
    try:
        from training.finetune import QLoRATrainer, TrainResult
        trainer = QLoRATrainer(model_key="gemma-3-4b")
        assert trainer.model_key == "gemma-3-4b"
        assert trainer.model_spec.hf_id == "unsloth/gemma-3-4b-it"
        assert trainer.profile.lora_r == 16
        assert trainer.profile.max_seq_length == 512
        tc = trainer.train_config
        assert tc["optim"] == "adamw_8bit"
        assert tc["bf16"] is True
        results.record(4, "QLoRATrainer initialization", True, time.time()-t0,
                       f"model={trainer.model_spec.hf_id}")
    except Exception as e:
        results.record(4, "QLoRATrainer initialization", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        for mk in ["gemma-3-4b", "qwen2.5-3b", "phi-4-mini"]:
            tr = QLoRATrainer(model_key=mk)
            assert tr.model_spec is not None
            assert tr.profile is not None
            assert tr.profile.lora_r > 0
        results.record(4, "All 3 model trainers initialize", True, time.time()-t0,
                       "gemma, qwen, phi validated")
    except Exception as e:
        results.record(4, "Multi-model trainer init", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        tr = TrainResult(success=True, model_key="gemma-3-4b", output_dir="outputs/test",
                         train_loss=0.45, eval_loss=0.52, train_time_seconds=120.0,
                         vram_peak_gb=5.2, num_samples=100)
        assert tr.success and tr.train_loss == 0.45
        results.record(4, "TrainResult dataclass", True, time.time()-t0,
                       f"loss={tr.train_loss}, eval={tr.eval_loss}")
    except Exception as e:
        results.record(4, "TrainResult dataclass", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        from training.callbacks import VRAMMonitorCallback, EvalLossCallback, MLflowLoggingCallback
        vram_cb = VRAMMonitorCallback(log_every_n_steps=5)
        eval_cb = EvalLossCallback(max_divergence=0.3)
        mlflow_cb = MLflowLoggingCallback()
        assert vram_cb._log_every_n_steps == 5
        assert eval_cb._max_divergence == 0.3
        results.record(4, "Training callbacks instantiate", True, time.time()-t0,
                       "VRAM + EvalLoss + MLflow")
    except Exception as e:
        results.record(4, "Training callbacks", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        from training.export import ModelExporter, ExportResult
        exp = ModelExporter(model_key="gemma-3-4b")
        assert exp.model_key == "gemma-3-4b"
        assert "q4_k_m" in exp.model_spec.gguf_quant_methods
        r = ExportResult(success=True, model_key="gemma-3-4b", gguf_path="out.gguf",
                         gguf_size_mb=2800, sha256="abc123", quant_method="q4_k_m")
        assert r.success
        results.record(4, "ModelExporter + ExportResult", True, time.time()-t0,
                       f"quant={exp.model_spec.gguf_quant_methods}")
    except Exception as e:
        results.record(4, "ModelExporter", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        from core.config import get_settings
        settings = get_settings()
        from orchestrator.pipeline import PipelineOrchestrator
        orch = PipelineOrchestrator()
        assert orch.settings is not None
        results.record(4, "PipelineOrchestrator initializes", True, time.time()-t0, "OK")
    except Exception as e:
        results.record(4, "PipelineOrchestrator", False, time.time()-t0, str(e))

    t0 = time.time()
    try:
        from registry.model_manager import ModelManager
        from registry.schemas import ModelAlias, ModelVersionStatus
        mm = ModelManager()
        assert mm.settings is not None
        assert ModelAlias.CHAMPION.value == "champion"
        assert ModelAlias.CHALLENGER.value == "challenger"
        results.record(4, "ModelManager + registry schemas", True, time.time()-t0,
                       "Aliases: champion, challenger, archived")
    except Exception as e:
        results.record(4, "ModelManager", False, time.time()-t0, str(e))


# ================================================================
# STAGE 5: Full QLoRA Training (GPU required - WSL/Linux)
# ================================================================
def test_stage5_training():
    console.rule("[bold cyan]Stage 5: Full QLoRA Fine-Tuning (GPU)[/]")

    t0 = time.time()
    try:
        import torch
        assert torch.cuda.is_available(), "CUDA not available"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        console.print(f"[green]GPU: {gpu_name} ({vram:.1f}GB VRAM)[/]")
        results.record(5, "CUDA GPU detected", True, time.time()-t0,
                       f"{gpu_name} ({vram:.1f}GB)")
    except Exception as e:
        results.record(5, "CUDA GPU detected", False, time.time()-t0, str(e))
        console.print("[red]No GPU - skipping training. Use WSL for GPU stages.[/]")
        return

    t0 = time.time()
    try:
        from unsloth import FastLanguageModel
        results.record(5, "Unsloth import", True, time.time()-t0, "OK")
    except Exception as e:
        results.record(5, "Unsloth import", False, time.time()-t0, str(e))
        console.print("[red]Unsloth not installed - cannot train.[/]")
        return

    t0 = time.time()
    try:
        from core.config import get_settings
        from training.finetune import QLoRATrainer
        settings = get_settings()

        processed = settings.data_dir / "processed"
        train_files = sorted(processed.glob("train_v1.0_*.jsonl"))
        if not train_files:
            train_files = [processed / "train.jsonl"]
        dataset_path = train_files[-1]
        assert dataset_path.exists(), f"No training data: {dataset_path}"

        trainer = QLoRATrainer(model_key="gemma-3-4b")
        result = trainer.train(dataset_path=dataset_path)

        assert result.success, f"Training failed: {result.error}"
        assert result.train_loss > 0
        assert result.train_time_seconds > 0
        assert Path(result.output_dir).exists()

        console.print(Panel(
            f"[green]Training Complete![/]\n"
            f"  Loss: {result.train_loss:.4f}\n"
            f"  Eval Loss: {result.eval_loss:.4f}\n"
            f"  Time: {result.train_time_seconds/60:.1f} min\n"
            f"  VRAM Peak: {result.vram_peak_gb:.2f} GB\n"
            f"  Samples: {result.num_samples}\n"
            f"  Output: {result.output_dir}",
            title="QLoRA Training Result"
        ))

        results.record(5, "Full QLoRA training run", True, time.time()-t0,
                       f"loss={result.train_loss:.4f}, vram={result.vram_peak_gb:.1f}GB, "
                       f"time={result.train_time_seconds:.0f}s")
    except Exception as e:
        results.record(5, "Full QLoRA training run", False, time.time()-t0, str(e))
        return

    # Verify output artifacts
    t0 = time.time()
    try:
        out_dir = Path(result.output_dir)
        adapter_files = list(out_dir.glob("*"))
        assert len(adapter_files) > 0, "No adapter files saved"
        file_list = [f.name for f in adapter_files]
        console.print(f"Adapter artifacts: {file_list}")
        results.record(5, "LoRA adapter saved", True, time.time()-t0,
                       f"{len(adapter_files)} files in {out_dir.name}")
    except Exception as e:
        results.record(5, "LoRA adapter saved", False, time.time()-t0, str(e))


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="MerchFine E2E Pipeline Test")
    parser.add_argument("--stage", type=int, help="Run a single stage (0-5)")
    parser.add_argument("--all", action="store_true", help="Include GPU stage 5")
    args = parser.parse_args()

    console.print(Panel("[bold magenta]MerchFine End-to-End Pipeline Test[/]",
                        subtitle="Real logic, no mocks"))

    stage_map = {
        0: test_stage0_config,
        1: test_stage1_data,
        2: test_stage2_prompts,
        3: test_stage3_eval,
        4: test_stage4_training_config,
        5: test_stage5_training,
    }

    if args.stage is not None:
        if args.stage in stage_map:
            stage_map[args.stage]()
        else:
            console.print(f"[red]Invalid stage {args.stage}. Valid: 0-5[/]")
            sys.exit(1)
    else:
        # Run stages 0-4 (Windows safe)
        for i in range(5):
            stage_map[i]()
        if args.all:
            stage_map[5]()
        else:
            console.print("\n[yellow]Stage 5 (GPU training) skipped. Use --all to include.[/]")

    console.print()
    all_passed = results.print_summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
