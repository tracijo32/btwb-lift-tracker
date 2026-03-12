import pandas as pd
from pathlib import Path
from helpers import load_withings_data, load_btwb_workout_data
from parse import parse_weightlifting_sets_from_btwb_workout_data
from kalman import (
    get_single_movements_max_daily,
    get_provisional_strength_curve,
    get_squats_for_forecast,
    compute_variance_by_squat,
    run_strength_filter,
    forecast_strength_with_uncertainty
)

def process_withings_data(path_to_files: Path) -> pd.DataFrame:
    data = load_withings_data(path_to_files)

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

    data_df = pvt.reset_index()

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
    model_df = pd.concat(models.values(), axis=1).reset_index()

    return data_df, model_df

def process_squats(path_to_files: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_btwb_workout_data(path_to_files)

    ## parse the weightlifting sets from the workout data
    metadata, sets = parse_weightlifting_sets_from_btwb_workout_data(data)

    ## get the max weight lifted for each # of reps of movement that day
    max_daily = get_single_movements_max_daily(metadata, sets)

    ## get the provisional strength curve: a rolling average of
    ## the 1 rep max back squat equivalent
    psc = get_provisional_strength_curve(
        max_daily, 
        start_date=max_daily['date'].min(),
        end_date=max_daily['date'].max())

    ## get the squats for the forecast:
    ## infers the 1RM back squat equivalent from the max weight lifted
    ## based on the squat type (back, front, overhead),
    ## the variation (normal, pause, tempo, etc.),
    ## and the number of repetitions (1, 3, 5, etc.)
    squat_df = get_squats_for_forecast(
        max_daily,
        max_repetitions=5
    )

    ## compute the observation variance for each squat type
    ## from the strength curve we computed earlier
    ## gives us an idea of the noise in the observations
    obs_var = compute_variance_by_squat(psc, squat_df)

    ## run the strength filter
    ## this returns a dataframe with the estimated latent strength
    ## and the estimated rate of improvement each day
    filtered_df, x, P = run_strength_filter(squat_df, obs_var)

    ## forecast the strength for the next 180 days
    forecast_df = forecast_strength_with_uncertainty(
        last_date=filtered_df['date'].iloc[-1],
        last_state=x,
        last_cov=P,
        days=180
    )

    ## save the forecast to pickle
    return squat_df, filtered_df, forecast_df

if __name__ == "__main__":
    import pickle
    root_dir = Path.home() / "data"

    path_to_withings_data = root_dir / "withings-data"
    withings_data, withings_model = process_withings_data(path_to_withings_data)
    pickle.dump(withings_data, open("withings_data.pkl", "wb"))
    pickle.dump(withings_model, open("withings_model.pkl", "wb"))   

    path_to_btwb_data = root_dir / "btwb" / "workout-events-2"
    squat_df, filtered_df, forecast_df = process_squats(path_to_btwb_data)
    pickle.dump(squat_df, open("squat_data.pkl", "wb"))
    pickle.dump(filtered_df, open("squat_filtered.pkl", "wb"))
    pickle.dump(forecast_df, open("squat_forecast.pkl", "wb"))