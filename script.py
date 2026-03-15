from process import process_withings_data, process_squats
from plotting import (
    create_bwbs_tracker_figure,
    create_squat_strength_figure,
    create_body_weight_figure,
    create_body_composition_figure,
)
from helpers import save_plotly_figure

withings_data, withings_model = process_withings_data('gs://withings-data')
squat_data, squat_model = process_squats('gs://btwb-workouts')

save_plotly_figure(
    create_bwbs_tracker_figure(
        withings_data, 
        squat_data, 
        withings_model, 
        squat_model
    ), 
    "gs://box.crossvalidatedfitness.com/figures/bwbs_tracker.html",
    project="crossvalidatedfitness-prod"
)

save_plotly_figure(
    create_squat_strength_figure(
        squat_data, 
        squat_model
    ), 
    "gs://box.crossvalidatedfitness.com/figures/squat_strength.html",
    project="crossvalidatedfitness-prod"
)

save_plotly_figure(
    create_body_weight_figure(
        withings_data, 
        withings_model
    ), 
    "gs://box.crossvalidatedfitness.com/figures/body_weight.html",
    project="crossvalidatedfitness-prod"
)

save_plotly_figure(
    create_body_composition_figure(
        withings_data, 
        withings_model
    ), 
    "gs://box.crossvalidatedfitness.com/figures/body_composition.html",
    project="crossvalidatedfitness-prod"
)