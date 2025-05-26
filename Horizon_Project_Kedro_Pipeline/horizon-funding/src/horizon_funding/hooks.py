

import mlflow
import pandas as pd
from pathlib import Path
from kedro.framework.hooks import hook_impl

class ProjectHooks:
    @hook_impl
    def after_context_created(self, context, **kwargs):
        
        
        project_root = Path.cwd()
        mlruns_uri = f"file://{project_root / 'mlruns'}"
        mlflow.set_tracking_uri(mlruns_uri)
       
        mlflow.set_experiment("horizon-funding")

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog, **kwargs):
        """Start a fresh MLflow run before the pipeline kicks off."""
        mlflow.start_run()

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog, **kwargs):
        """End the MLflow run after the pipeline completes."""
        mlflow.end_run()

    @hook_impl
    def before_node_run(self, node, catalog, inputs, is_async, session_id, **kwargs):
        """
        Log any parameters that Kedro passes in.
        Your nodes consume `params:...`, so we pick them up here.
        """
        for name, val in inputs.items():
            if name.startswith("params:"):
                param_name = name.split("params:")[-1]
                mlflow.log_param(param_name, val)
            elif name == "params":
               
                for p, v in val.items():
                    mlflow.log_param(p, v)

    @hook_impl
    def after_dataset_saved(self, dataset_name: str, data, is_batch: bool, **kwargs):
        """
        Whenever a dataset is saved:
        - If it's a DataFrame/Series, dump it to CSV and log as an MLflow artifact.
        - If it's your final model or metrics, log them specially.
        """
      
        if isinstance(data, (pd.DataFrame, pd.Series)):
            out_path = Path("data") / f"{dataset_name}.csv"
            data.to_csv(out_path, index=False)
            mlflow.log_artifact(str(out_path))

     
        if dataset_name == "best_model":
          
            mlflow.sklearn.log_model(
                sk_model=data,
                artifact_path="model"
            )

       
        if dataset_name == "all_metrics" and isinstance(data, dict):
            for metric_name, metric_val in data.items():
                mlflow.log_metric(metric_name, metric_val)


