import pandas as pd
from typing import Tuple
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




def load_and_merge(
    project_filepath: str, organization_filepath: str
) -> pd.DataFrame:

    project = pd.read_excel(project_filepath)
    organization = pd.read_excel(organization_filepath)

    
    project = project[[
        "id", "title", "ecMaxContribution", "totalCost",
        "startDate", "endDate", "fundingScheme"
    ]]
    organization = organization[[
        "projectID", "organisationID", "country", "SME", "order"
    ]]

  
    org_counts = (
        organization
        .groupby("projectID")["organisationID"]
        .nunique()
        .rename("org_count")
        .reset_index()
    )

    
    org_country = (
        organization
        .loc[organization["order"] == 1, ["projectID", "country"]]
        .drop_duplicates("projectID")
        .rename(columns={"country": "organiser_country"})
    )

   
    df = pd.merge(
        project,
        organization,
        left_on="id",
        right_on="projectID",
        how="left"
    )

   
    df = df.merge(
        org_counts, left_on="id", right_on="projectID", how="left"
    )
    df = df.merge(
        org_country, left_on="id", right_on="projectID", how="left"
    )

   
    df = df.set_index("id")

    
    df = df.drop(columns=[
        "projectID_x", "projectID_y", "title",
        "organisationID", "country", "order"
    ])

   
    df = (
        df
        .sort_index()             
        .groupby(level=0)
        .first()                  
    )

    return df



def cast_types(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
    for col in ["ecMaxContribution", "totalCost"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
    df["endDate"] = pd.to_datetime(df["endDate"], errors="coerce")
    df["startmonth"] = df["startDate"].dt.month

    return df


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()
   
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
    df["endDate"]   = pd.to_datetime(df["endDate"],   errors="coerce")
    
    df["duration_days"] = (df["endDate"] - df["startDate"]).dt.days
    df = df.drop(columns=["startDate", "endDate"])
    return df


def drop_zero_totalcost(df: pd.DataFrame) -> pd.DataFrame:
    
    return df[df["totalCost"] != 0].reset_index(drop=True)


def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    
    return df.dropna().reset_index(drop=True)


def assign_funding_class(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    bins = [
        df["ecMaxContribution"].min() - 1,
        2_000_000,
        4_000_000,
        df["ecMaxContribution"].max() + 1,
    ]
    labels = [0, 1, 2]
    df["funding_class"] = pd.cut(
        df["ecMaxContribution"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    df = df.drop(columns=["ecMaxContribution])
    return df





def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:

    df["SME"] = df["SME"].astype("category")
    df["fundingScheme"] = df["fundingScheme"].astype("category")
    df["organiser_country"] = df["organiser_country"].astype("category")


    return pd.get_dummies(
        df,
        columns=["fundingScheme", "organiser_country"],
        drop_first=True,
    )


def split_data(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(
        X, y, test_size=0.2, random_state=56, stratify=y
    )


def remove_outliers(
    X_train: pd.DataFrame, y_train: pd.Series, contamination: float
) -> Tuple[pd.DataFrame, pd.Series]:
   
    iso = IsolationForest(contamination=contamination, random_state=78)
    # only numeric cols
    num = X_train.select_dtypes(include="number")
    mask = iso.fit_predict(num) == 1
    return (
        X_train.loc[mask].reset_index(drop=True),
        y_train.loc[mask].reset_index(drop=True),
    )

def standardize_numeric_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_columns: list[str] = ["totalCost", "org_count", "duration"]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    
    X_train_scaled[numeric_columns] = scaler.fit_transform(
        X_train[numeric_columns]
    )
    X_test_scaled[numeric_columns] = scaler.transform(
        X_test[numeric_columns]
    )

    return X_train_scaled, X_test_scaled
