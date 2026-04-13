import pytest
from unittest.mock import patch, MagicMock
from core.system_init import SystemInitializer
from core.config import get_settings


@pytest.fixture
def initializer():
    return SystemInitializer()


def test_check_gpu(initializer):
    # torch is imported locally inside check_gpu() to prevent global loading delays.
    # We test the system natively without deeply mocking C++ bindings.
    initializer.check_gpu()
    
    if initializer.report.gpu_name != "N/A":
        assert initializer.report.vram_total_gb > 0
        assert initializer.report.vram_free_gb >= 0


@pytest.mark.asyncio
async def test_check_mlflow_fallback(initializer, tmp_path):
    settings = get_settings()
    settings.project_root = tmp_path
    
    with patch("httpx.AsyncClient.get", side_effect=Exception("Connection refused")):
        # MLflow fallback hook should also be patched so we don't start the real MLflow core here
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_registry_uri"):
            await initializer.check_mlflow()
        
        # Test that unreachable server cleanly falls back to SQLite
        assert initializer.report.mlflow_reachable is True
        assert "sqlite:///" in str(initializer.settings.mlflow_tracking_uri)
        assert (tmp_path / "mlruns").exists()


@pytest.mark.asyncio
async def test_check_redis_fallback(initializer):
    # If redis fails to connect, it should not crash but mark reachable=False
    with patch("redis.from_url", side_effect=Exception("Timeout")):
        await initializer.check_redis()
        
        assert initializer.report.redis_reachable is False
