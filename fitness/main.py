import pandas as pd


def read_csv(file: str) -> pd.DataFrame:
    return pd.read_csv(file)


def cast_date_columns(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df[column_name] = pd.to_datetime(df[column_name])
    df[column_name] = df[column_name].dt.date
    return df


def calculate_mean_weight_per_day(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Date")["Weight"].mean().reset_index()


def insert_missing_days(df: pd.DataFrame) -> pd.DataFrame:
    date_range = pd.date_range(
        start=df["Date"].min(),
        end=df["Date"].max(),
    )
    df_imputed = df.set_index("Date").reindex(date_range).rename_axis("Date").reset_index()
    return df_imputed


def interpolate_missing_weights(df: pd.DataFrame) -> pd.DataFrame:
    return df.interpolate(method="linear", limit_direction="forward", axis=0)


# %%

df_fitbit = read_csv("data_files/fitness_agg.csv").pipe(cast_date_columns, column_name="Date")
df_fitbit = df_fitbit[["Date", "Weight"]]

df_withings = read_csv("data_files/weight.csv").pipe(cast_date_columns, column_name="Date")
df_withings = df_withings[["Date", "Weight (kg)"]]
df_withings.rename(columns={"Weight (kg)": "Weight"}, inplace=True)

df = pd.concat([df_fitbit, df_withings], ignore_index=True)
del (df_fitbit, df_withings)

# %%

df = (
    df.sort_values("Date")
    .pipe(calculate_mean_weight_per_day)
    .pipe(insert_missing_days)
    .pipe(interpolate_missing_weights)
    .pipe(cast_date_columns, column_name="Date")
    .sort_values("Date", ascending=False)
)


df.to_csv("data_files/weight_interpolated.csv", index=False)
