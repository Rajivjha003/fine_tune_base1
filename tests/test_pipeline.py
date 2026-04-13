import pytest
from unittest.mock import AsyncMock, patch

from orchestrator.pipeline import PipelineOrchestrator
from core.exceptions import QualityGateFailedError


@pytest.fixture
def orchestrator():
    return PipelineOrchestrator()


@pytest.mark.asyncio
async def test_run_full_pipeline_success(orchestrator):
    # Mock all internal stages to succeed
    orchestrator._run_data_stage = AsyncMock(return_value={"status": "success", "records": 100})
    orchestrator._run_training_stage = AsyncMock(return_value={"status": "success", "loss": 1.2})
    orchestrator._run_evaluation_stage = AsyncMock(return_value={"status": "success"})
    orchestrator._run_registry_stage = AsyncMock(return_value={"status": "success"})
    orchestrator._run_deploy_stage = AsyncMock(return_value={"status": "success"})
    
    result = await orchestrator.run_full_pipeline()
    
    # Assert pipeline cascaded through all 5 stages safely
    assert result["status"] == "success"
    assert "data" in result["stages"]
    
    orchestrator._run_data_stage.assert_called_once()
    orchestrator._run_training_stage.assert_called_once()
    orchestrator._run_evaluation_stage.assert_called_once()
    orchestrator._run_registry_stage.assert_called_once()
    orchestrator._run_deploy_stage.assert_called_once()


@pytest.mark.asyncio
async def test_run_full_pipeline_quality_gate_failure(orchestrator):
    # Mock data & training to succeed
    orchestrator._run_data_stage = AsyncMock(return_value={"status": "success"})
    orchestrator._run_training_stage = AsyncMock(return_value={"status": "success"})
    
    # Mock evaluation to throw a Gate Error (e.g. Hallucination limit exceeded)
    orchestrator._run_evaluation_stage = AsyncMock(
        side_effect=QualityGateFailedError("Failed eval", failed_gates=["hallucination"])
    )
    # The deploy stage should never be reached
    orchestrator._run_deploy_stage = AsyncMock()
    
    # Patch event bus to capture the rollback emission
    with patch("core.events.event_bus.emit") as mock_emit:
        result = await orchestrator.run_full_pipeline()
        
        # Pipeline should safely halt and not crash
        assert result["status"] == "failed_quality_gate"
        
        # Ensure 'deploy' was strictly bypassed
        orchestrator._run_deploy_stage.assert_not_called()
        
        # Ensure the 'eval_gate_failed' event was emitted to trigger auto-rollback hooks
        mock_emit.assert_called_once()
        assert mock_emit.call_args[0][0] == "eval_gate_failed"


@pytest.mark.asyncio
async def test_run_eval_only_success(orchestrator):
    orchestrator._run_evaluation_stage = AsyncMock(return_value={"status": "success"})
    result = await orchestrator.run_eval_only()
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_run_feedback_loop_skip_if_no_metrics(orchestrator, tmp_path):
    orchestrator.settings.data_dir = tmp_path
    
    result = await orchestrator.run_feedback_loop()
    
    # Needs to skip cleanly instead of crashing if the log file doesn't exist
    assert result["status"] == "skipped"
