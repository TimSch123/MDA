import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from typing import Tuple, Any, Dict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import numpy as np

def train_logistic_regression(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Tuple[Any, pd.Series]:
   
    model = LogisticRegression(max_iter=1_000)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train_scaled, y_train)
    return model, scores.tolist()


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Tuple[Any, pd.Series]:
   
    model = RandomForestClassifier(random_state=76)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train, y_train)
    return model, scores.tolist()


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Tuple[Any, pd.Series]:
  
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        enable_categorical=True,
        random_state=76
    )
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train, y_train)
    return model, scores.tolist()


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, average: str = "macro"
) -> Dict[str, Any]:
 
    y_pred = model.predict(X_test)

   
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)
    cm = confusion_matrix(y_test, y_pred).tolist()  

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }