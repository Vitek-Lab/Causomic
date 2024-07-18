import pandas as pd

def normalize(df: pd.DataFrame) -> pd.DataFrame:

    df.loc[:, "LogIntensities"] = (df.loc[:, "LogIntensities"] - df.loc[:, "LogIntensities"].mean()) / df.loc[:, "LogIntensities"].std()

    return df