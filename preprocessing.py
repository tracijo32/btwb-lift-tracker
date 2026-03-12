import pandas as pd
from helpers import _validate_data_frame

def get_single_movements_max_daily(
    event_metadata_frame: pd.DataFrame,
    lifting_sets_frame: pd.DataFrame
) -> pd.DataFrame:

    events = _validate_data_frame(event_metadata_frame, {
        'event_id': 'string',
        'date': 'datetime64[ns]',
    })
    lifts = _validate_data_frame(lifting_sets_frame, {
        'event_id': 'string',
        'weight_lbs': 'float64',
        'repetitions': 'int64',
        'movements': 'list[string]',
        'is_complex': 'bool',
    })

    ## merge metadata to get date
    df = pd.merge(events, lifts, on='event_id')
    df['date'] = pd.to_datetime(df['date'])

    ## filter to single movement (no complexes)
    ## get the max weight lifted for each # of reps of movement that day
    df = df[
        ~df['is_complex'] &
        df['movements'].apply(len).eq(1)
    ].assign(movement = lambda x: x['movements'].apply(lambda x: x[0]))\
        .groupby(['date','movement','repetitions'])\
            ['weight_lbs'].max()\
                .reset_index()

    return df

def get_provisional_strength_curve(
    data_frame: pd.DataFrame,
    movement: str = 'back squat',
    repetitions: int = 1,
    window_days: int = 180,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    
    df = _validate_data_frame(data_frame, 
            dtypes = {
                'date': 'datetime64[ns]',
                'weight_lbs': 'float64',
                'repetitions': 'int64',
                'movement': 'string',
            }
    )

    if not start_date:
        start_date = df['date'].min()
    if not end_date:
        end_date = df['date'].max()

    df = df[df['movement'].eq(movement) & df['repetitions'].eq(repetitions)]
    df = df.set_index('date').sort_index().reindex(columns=['weight_lbs'])
    df['bs_1rm_est'] = df['weight_lbs'].rolling(window=f'{window_days}D').mean()

    interp_dates = pd.date_range(start=start_date, end=end_date, freq='D')\
        .to_series(name='date')

    df = pd.merge(df, interp_dates, left_index=True, right_index=True, how='outer')
    df['bs_1rm_est'] = df['bs_1rm_est'].interpolate(method='linear').bfill().ffill()

    return df['bs_1rm_est'].reset_index()

def get_squats_for_forecast(
    data_frame: pd.DataFrame,
    max_repetitions: int | None = None,
    variations_to_include: list[str] | None = None,
    squats_to_include: list[str] | None = None,
    repetitions_to_include: list[int] | None = None,
) -> pd.DataFrame:

    df = _validate_data_frame(data_frame, 
            dtypes = {
                'date': 'datetime64[ns]',
                'weight_lbs': 'float64',
                'repetitions': 'int64',
                'movement': 'string',
            }
    )

    from modifiers import BASE_SQUAT_MODIFIERS, VARIATION_MODIFIERS, epley_multiplier
    move_df = pd.DataFrame([
        {
            'movement': f"{'' if v == 'normal' else v+' '}{s} squat",
            'squat': s,
            'variation': v,
            'multiplier': 1 / (x * y)
        }
            for s, x in BASE_SQUAT_MODIFIERS.items()
            for v, y in VARIATION_MODIFIERS.items()
    ])

    ## merge the lift data with the modifiers to get the
    ## equivalent 1RM back squat
    df = pd.merge(move_df, df, on='movement')
    df['epley'] = df['repetitions'].apply(epley_multiplier)
    df['bs_1rm_est'] = df[['weight_lbs','multiplier','epley']].prod(axis=1)

    ## filter down the data frame to the desired repetitions, variations, and squats
    if max_repetitions is not None:
        df = df[df['repetitions'].le(max_repetitions)]
    if repetitions_to_include is not None:
        df = df[df['repetitions'].isin(repetitions_to_include)]
    if variations_to_include is not None:
        df = df[df['variation'].isin(variations_to_include)]
    if squats_to_include is not None:
        df = df[df['squat'].isin(squats_to_include)]

    return df.reindex(columns=['date','squat','variation','repetitions','bs_1rm_est'])

def compute_variance_by_squat(
    strength_curve: pd.DataFrame,
    lift_data: pd.DataFrame
):
    curve = _validate_data_frame(strength_curve, {
        'date': 'datetime64[ns]',
        'bs_1rm_est': 'float64',
    })
    lifts = _validate_data_frame(lift_data, {
        'date': 'datetime64[ns]',
        'squat': 'string',
        'variation': 'string',
        'repetitions': 'int64',
        'bs_1rm_est': 'float64',
    })

    res = pd.merge(lifts, curve, on='date',suffixes=['_modified','_actual'])
    res['residual'] = res['bs_1rm_est_modified'].sub(res['bs_1rm_est_actual'])

    return res.groupby(['squat'])['residual'].var().to_dict()