import numpy as np
import pandas as pd
from pykalman import KalmanFilter

from helpers import lookup_nested_obs_var, _validate_data_frame

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

    return df.reindex(columns=['date','squat','variation','repetitions','weight_lbs','bs_1rm_est'])

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

def run_strength_filter(
    df: pd.DataFrame,
    obs_var: dict,
    obs_var_default: float = 100,
    strength_process_var: float = 0.25,
    rate_process_var: float = 1e-4,
    initial_state_covariance: float = 10.0,
    initial_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Estimate latent (BS-equivalent) strength from noisy BS-equivalent observations.

    2-state model:
        state[0] = latent strength
        state[1] = latent daily rate of improvement

    Expected columns in df:
        - date
        - squat: back, front, overhead
        - variation: normal, pause, other, ...
        - repetitions: 1, 3, 5, ...
        - bs_1rm_est: estimated 1RM back squat equivalent (observation)

    Observation variance (obs_var)
    ------------------------------
    obs_var is a nested dictionary. Each level can either be a dict (more specific)
    or a scalar variance (applies to all remaining lower levels).

    Special keys:
        - 'all': applies to everything at that level
        - 'other': applies to any key not explicitly assigned at that level

    If a level is missing (e.g. no reps dict), the scalar at the higher level
    is treated as applying to all lower levels.
    """

    if "date" not in df.columns:
        raise ValueError("df must include a 'date' column")
    if "bs_1rm_est" not in df.columns:
        raise ValueError("df must include a 'bs_1rm_est' column")

    if df.empty:
        out = df.copy()
        out["strength_estimate"] = []
        out["strength_sd"] = []
        out["rate_estimate"] = []
        out["rate_sd"] = []
        return out

    # 1) Sort chronologically
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    observations = df["bs_1rm_est"].astype(float).to_numpy()
    if not np.isfinite(observations).all():
        bad = int(np.size(observations) - np.isfinite(observations).sum())
        raise ValueError(
            f"bs_1rm_est contains {bad} non-finite values; "
            "drop or impute missing observations before filtering"
        )

    # 2) Observation variance by lift type
    n = len(df)
    squat_series = df["squat"] if "squat" in df.columns else pd.Series([None] * n, index=df.index)
    variation_series = df["variation"] if "variation" in df.columns else pd.Series([None] * n, index=df.index)
    reps_series = df["repetitions"] if "repetitions" in df.columns else pd.Series([None] * n, index=df.index)

    observation_covariance = np.array(
        [
            lookup_nested_obs_var(
                obs_var,
                s if pd.notna(s) else None,
                v if pd.notna(v) else None,
                int(r) if pd.notna(r) else None,
                default=obs_var_default,
            )
            for s, v, r in zip(squat_series, variation_series, reps_series)
        ],
        dtype=float,
    )

    # 3) Time gaps between observations (in days)
    #    These are what make the 2-state model useful for forecasting.
    dt_days = df["date"].diff().dt.days.fillna(0).astype(float).to_numpy()

    # 4) Create a 2-state Kalman filter
    #    state = [strength, daily_rate]
    #    We will override transition_matrix and transition_covariance
    #    at each step because dt can change.
    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=np.array([[1.0, 0.0]]),  # we only observe strength, not rate directly
        transition_covariance=np.eye(2),
        initial_state_mean=np.array([observations[0], float(initial_rate)]),
        initial_state_covariance=np.array(
            [
                [float(initial_state_covariance), 0.0],
                [0.0, 0.01],  # initial uncertainty on daily rate
            ]
        ),
    )

    # Containers
    strength_means = []
    strength_vars = []
    rate_means = []
    rate_vars = []

    # 5) Initialize current state
    x = np.array([observations[0], float(initial_rate)], dtype=float)
    P = np.array(
        [
            [float(initial_state_covariance), 0.0],
            [0.0, 0.01],
        ],
        dtype=float,
    )

    # 6) Sequential updates
    for dt, y, R in zip(dt_days, observations, observation_covariance):
        # Transition matrix for this time gap:
        # strength_t = strength_{t-1} + dt * rate_{t-1}
        # rate_t = rate_{t-1}
        A = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])

        # Process noise for this time gap.
        # Strength can wiggle a bit; rate can also drift slowly.
        # Scaling by max(dt, 1.0) avoids zero process noise on same-day entries.
        q_scale = max(dt, 1.0)
        Q = np.array([
            [float(strength_process_var) * q_scale, 0.0],
            [0.0, float(rate_process_var) * q_scale],
        ])

        # Observation covariance must be 1x1 for a 1D observation.
        R_mat = np.array([[float(R)]])

        x, P = kf.filter_update(
            filtered_state_mean=x,
            filtered_state_covariance=P,
            observation=np.array([float(y)]),
            transition_matrix=A,
            transition_covariance=Q,
            observation_covariance=R_mat,
        )

        x = np.asarray(x, dtype=float).reshape(-1)
        P = np.asarray(P, dtype=float)

        strength_means.append(float(x[0]))
        strength_vars.append(float(P[0, 0]))
        rate_means.append(float(x[1]))
        rate_vars.append(float(P[1, 1]))

    # 7) Save outputs
    df["strength_estimate"] = strength_means
    df["strength_sd"] = np.sqrt(np.asarray(strength_vars, dtype=float))
    df["rate_estimate"] = rate_means
    df["rate_sd"] = np.sqrt(np.asarray(rate_vars, dtype=float))

    ## report just the filtered columns by date
    df = df.reindex(columns=['date','strength_estimate','strength_sd','rate_estimate','rate_sd'])

    return df, x, P

def forecast_strength_with_uncertainty(
    last_date: pd.Timestamp,
    last_state: np.ndarray,
    last_cov: np.ndarray,
    days: int = 180,
    strength_process_var: float = 0.25,
    rate_process_var: float = 1e-4,
) -> pd.DataFrame:
    """
    Forecast latent strength and uncertainty forward from the last filtered state.

    Parameters
    ----------
    last_date : pd.Timestamp
        Date of the last observation.
    last_state : np.ndarray
        Final filtered state, shape (2,)
        [strength, daily_rate]
    last_cov : np.ndarray
        Final filtered covariance, shape (2, 2)
    days : int
        Number of days to forecast
    strength_process_var : float
        Process variance for strength
    rate_process_var : float
        Process variance for rate

    Returns
    -------
    pd.DataFrame
        Columns:
        - date
        - forecast_strength
        - forecast_sd
        - lower_95
        - upper_95
    """
    x = np.asarray(last_state, dtype=float).reshape(2)
    P = np.asarray(last_cov, dtype=float).reshape(2, 2)

    rows = []

    for h in range(1, days + 1):
        dt = 1.0

        A = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])

        Q = np.array([
            [strength_process_var * dt, 0.0],
            [0.0, rate_process_var * dt],
        ])

        # Predict next state and covariance
        x = A @ x
        P = A @ P @ A.T + Q

        strength_mean = x[0]
        strength_sd = np.sqrt(P[0, 0])

        rows.append({
            "date": last_date + pd.Timedelta(days=h),
            "strength_estimate": strength_mean,
            "strength_sd": strength_sd,
        })

    return pd.DataFrame(rows)