from kedro.pipeline import Pipeline, node
from .nodes import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    evaluate_model,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_logistic_regression,
                inputs=["X_train_scaled", "y_train_clean"],
                outputs=["logistic_model", "logistic_cv_scores"],
                name="mt_train_logistic",
            ),
            node(
                func=train_random_forest,
                inputs=["X_train_clean", "y_train_clean"],
                outputs=["rf_model", "rf_cv_scores"],
                name="mt_train_rf",
            ),
            node(
                func=train_xgboost,
                inputs=["X_train_clean", "y_train_clean"],
                outputs=["xgb_model", "xgb_cv_scores"],
                name="mt_train_xgb",
            ),
        ]
    )

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
           
            node(
                func=train_random_forest,
                inputs=["X_train_clean", "y_train_clean"],
                outputs=["rf_model", "rf_cv_scores"],
                name="mt_train_rf",
            ),
            node(
                func=train_xgboost,
                inputs=["X_train_clean", "y_train_clean"],
                outputs=["xgb_model", "xgb_cv_scores"],
                name="mt_train_xgb",
            ),
            node(
                func=train_logistic_regression,
                inputs=["X_train_scaled", "y_train_clean"],
                outputs=["logistic_model", "logistic_cv_scores"],
                name="mt_train_logistic",
            ),

           
            node(
                func=evaluate_model,
                inputs=["rf_model", "X_test", "y_test"],
                outputs="rf_test_metrics",
                name="mt_evaluate_rf",
            ),
            node(
                func=evaluate_model,
                inputs=["xgb_model", "X_test", "y_test"],
                outputs="xgb_test_metrics",
                name="mt_evaluate_xgb",
            ),
            node(
                func=evaluate_model,
                inputs=["logistic_model", "X_test_scaled", "y_test"],
                outputs="logistic_test_metrics",
                name="mt_evaluate_logistic",
            ),
        ]
    )