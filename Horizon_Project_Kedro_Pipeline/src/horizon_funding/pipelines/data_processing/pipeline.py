# src/<your_package>/pipelines/data_processing/pipeline.py

from kedro.pipeline import node, Pipeline
from .nodes import (
    load_and_merge,
    cast_types,
    drop_zero_totalcost,
    drop_na,
    assign_funding_class,
    add_duration,
    one_hot_encode,
    split_data,
    remove_outliers,
    standardize_numeric_columns,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_and_merge,
                inputs=["params:project_filepath", "params:organization_filepath"],
                outputs="processed_data",
                name="dp_load_and_merge",
            ),
            node(
                func=cast_types,
                inputs="processed_data",
                outputs="typed_data",
                name="dp_cast_types",
            ),
            node(
                func=drop_zero_totalcost,
                inputs="typed_data",
                outputs="filtered_data",
                name="dp_drop_zero_totalcost",
            ),
            node(
                func=drop_na,
                inputs="filtered_data",
                outputs="no_na_data",
                name="dp_drop_na",
            ),
            node(
                func=assign_funding_class,
                inputs="no_na_data",
                outputs="labeled_data",
                name="dp_assign_funding_class",
            ),
            node(
                func=add_duration,
                inputs="labeled_data",
                outputs="durated_data",
                name="dp_add_duration",
            ),
            node(
                func=one_hot_encode,
                inputs="durated_data",
                outputs="encoded_data",
                name="dp_one_hot_encode",
            ),
            node(
                func=split_data,
                inputs=["encoded_data", "params:target_column"],
                outputs=["X_train_raw", "X_test", "y_train_raw", "y_test"],
                name="dp_split_data",
            ),
            node(
                func=remove_outliers,
                inputs=["X_train_raw", "y_train_raw", "params:contamination"],
                outputs=["X_train_clean", "y_train_clean"],
                name="dp_remove_outliers",
            ),
            node(
                func=standardize_numeric_columns,
                inputs=["X_train_clean", "X_test", "params:numeric_columns"],
                outputs=["X_train_scaled", "X_test_scaled"],
                name="mt_standardize",
            ),  
        ]
    )

