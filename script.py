from process import process_withings_data, process_squats
from plotting import (
    create_bwbs_forecast_figure,
    create_squat_strength_figure,
    create_withings_figure,
)

withings_data, withings_model = process_withings_data('gs://withings-data')
squat_data, squat_model = process_squats('gs://btwb-workouts')

create_bwbs_forecast_figure(
    withings_data, 
    squat_data,
    withings_model,
    squat_model
).write_html('bwbs_forecast.html')

create_squat_strength_figure(
    squat_data, 
    squat_model
).write_html('squat_strength.html')

create_withings_figure(
    withings_data,
    withings_model
).write_html('withings.html')
