from helpers import load_withings_data
import pandas as pd
from pathlib import Path

##load the data
root_dir = Path.home() / "data" / "withings-data"
data = load_withings_data(root_dir)

## convert the data to a dataframe
df = pd.DataFrame(data).sort_values(by='date').dropna()
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert("America/Chicago")

## pivot the data to a wide, column for each type of weight measuremen
pvt = pd.pivot_table(
    df,
    index='date',
    columns='type',
    values='value',
    aggfunc='mean'
)
pvt.columns.name = None

## save raw data to csv
pvt.to_csv('withings-data.csv')

## smooth data and resample to daily
pvt = pvt.rolling(window='7D').mean()\
    .resample('D').mean()\
        .interpolate("linear")\
            .dropna()

## fit Holt-Winters exponential smoothing models to each column
## forcast 180 days out in to the future
from statsmodels.tsa.holtwinters import ExponentialSmoothing
models = {}
for col in pvt.columns:
    model = ExponentialSmoothing(pvt[col], trend="add", seasonal=None).fit()
    df1 = model.fittedvalues.to_frame(name=col)
    df2 = model.forecast(180).to_frame(name=col)
    df = pd.concat([df1, df2]).sort_index()
    df.index.name = 'date'
    models[col] = df

## concatentate the models in to the same format as the raw data
## and export to CSV
model_df = pd.concat(models.values(), axis=1).reset_index()
model_df.to_csv('withings-model.csv', index=False)
            