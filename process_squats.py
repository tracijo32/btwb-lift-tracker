from pathlib import Path
import sys

from parse import parse_weightlifting_sets_from_btwb_workout_data
from preprocessing import (
    get_single_movements_max_daily, 
    get_squats_for_forecast,
    get_provisional_strength_curve,
    compute_variance_by_squat
)
from kalman import run_strength_filter
from helpers import load_btwb_workout_data

## read in data
root_dir = Path.home() / "data" / "btwb" / "workout-events-2"
data = load_btwb_workout_data(root_dir)

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

## save the squats for the forecast to csv
squat_df.to_csv("squats.csv", index=False)

## compute the observation variance for each squat type
## from the strength curve we computed earlier
## gives us an idea of the noise in the observations
obs_var = compute_variance_by_squat(psc, squat_df)

## run the strength filter
## this returns a dataframe with the estimated latent strength
## and the estimated rate of improvement each day
results = run_strength_filter(squat_df, obs_var)

## save the results to csv
results.to_csv("strength_filter.csv", index=False)