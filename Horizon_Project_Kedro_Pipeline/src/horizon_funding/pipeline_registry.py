from typing import Dict
from kedro.pipeline import Pipeline
from horizon_funding.pipelines import data_processing as dp, model_training as mt

def register_pipelines() -> dict[str, Pipeline]:
    dp_pipeline = dp.create_pipeline()
    mt_pipeline = mt.create_pipeline()

    return {
        "__default__": dp_pipeline + mt_pipeline,    
        "data_processing": dp_pipeline,
        "model_training":  mt_pipeline,
    }
