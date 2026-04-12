"""
MLflow experiment tracking wrapper.

Simplifies MLflow run creation, metric/param logging, and artifact management
with a clean, typed interface.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import get_settings
from registry.schemas import RunMetadata

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Wrapper around MLflow client for experiment tracking.

    Handles:
    - Experiment creation/selection
    - Run lifecycle (start, log, end)
    - Metric, parameter, and tag logging
    - Artifact upload
    """

    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._active_run_id: str | None = None

    @property
    def client(self):
        """Lazy-initialized MLflow client."""
        if self._client is None:
            import mlflow

            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            self._client = mlflow.tracking.MlflowClient(self.settings.mlflow_tracking_uri)
        return self._client

    def ensure_experiment(self, name: str | None = None) -> str:
        """Create or get the experiment. Returns experiment ID."""
        import mlflow

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)

        exp_name = name or self.settings.mlflow_experiment_name
        experiment = mlflow.get_experiment_by_name(exp_name)

        if experiment is None:
            exp_id = mlflow.create_experiment(exp_name)
            logger.info("Created MLflow experiment '%s' (id=%s)", exp_name, exp_id)
        else:
            exp_id = experiment.experiment_id

        mlflow.set_experiment(exp_name)
        return exp_id

    def start_run(
        self,
        run_name: str,
        *,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> RunMetadata:
        """Start a new MLflow run. Returns RunMetadata."""
        import mlflow

        self.ensure_experiment()

        run = mlflow.start_run(run_name=run_name, nested=nested)
        self._active_run_id = run.info.run_id

        if tags:
            mlflow.set_tags(tags)

        logger.info("Started MLflow run: %s (id=%s)", run_name, run.info.run_id)

        return RunMetadata(
            run_id=run.info.run_id,
            experiment_name=self.settings.mlflow_experiment_name,
            model_key=tags.get("model_key", "") if tags else "",
            status="RUNNING",
            tags=tags or {},
        )

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current active MLflow run."""
        import mlflow

        mlflow.end_run(status=status)
        logger.info("Ended MLflow run: %s (status=%s)", self._active_run_id, status)
        self._active_run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to the active run."""
        import mlflow

        # MLflow params must be strings
        str_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(str_params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to the active run."""
        import mlflow

        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric."""
        import mlflow

        mlflow.log_metric(key, value, step=step)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the active run."""
        import mlflow

        mlflow.set_tag(key, value)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a local file as an artifact."""
        import mlflow

        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.debug("Logged artifact: %s", local_path)

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        """Log all files in a directory as artifacts."""
        import mlflow

        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
        logger.debug("Logged artifacts dir: %s", local_dir)

    def get_run(self, run_id: str) -> RunMetadata:
        """Get metadata for a specific run."""
        run = self.client.get_run(run_id)

        return RunMetadata(
            run_id=run.info.run_id,
            experiment_name=self.settings.mlflow_experiment_name,
            model_key=run.data.tags.get("model_key", ""),
            status=run.info.status,
            params=dict(run.data.params),
            metrics=dict(run.data.metrics),
            tags=dict(run.data.tags),
        )

    def search_runs(
        self,
        filter_string: str = "",
        order_by: list[str] | None = None,
        max_results: int = 10,
    ) -> list[RunMetadata]:
        """Search for runs matching a filter."""
        import mlflow

        self.ensure_experiment()
        experiment = mlflow.get_experiment_by_name(self.settings.mlflow_experiment_name)
        if experiment is None:
            return []

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["metrics.eval_loss ASC"],
            max_results=max_results,
        )

        return [
            RunMetadata(
                run_id=r.info.run_id,
                experiment_name=self.settings.mlflow_experiment_name,
                model_key=r.data.tags.get("model_key", ""),
                status=r.info.status,
                params=dict(r.data.params),
                metrics=dict(r.data.metrics),
                tags=dict(r.data.tags),
            )
            for r in runs
        ]
