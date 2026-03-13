import pandas as pd
import plotly.graph_objects as go
from plotly_theme import register_chunk_template, ChunkTheme
from helpers import _validate_data_frame, hex_to_rgba

COLORS = ChunkTheme()

### withings #####################
def reformat_withings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat the withings data frame to the correct columns and types for plotting.
    """
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'],)
    df = _validate_data_frame(df, 
                dtypes = {
                    'date': 'datetime64[us, America/Chicago]',
                    'bone': 'float64',
                    'fat': 'float64',
                    'muscle': 'float64',
                    'water': 'float64',
                }
            ).dropna()
    df['total'] = df[['bone','fat','muscle']].sum(axis=1)
    return df

def get_withings_traces(
    withings_data_frame: pd.DataFrame,
    withings_model_frame: pd.DataFrame,
    marker_color: str = COLORS.text,
    model_line_color: str = COLORS.grid,
    n_days_forecast: int = 30
) -> dict[str, go.Scatter]:

    """
    Get the plotting traces for the Withings data and model of bodyweight timeseries.

    Parameters
    ----------
    withings_data_frame: pd.DataFrame
        The withings data frame to plot.
    withings_model_frame: pd.DataFrame
        The withings model frame to plot.
    marker_color: str
        The color of the marker for the measurements.
    model_line_color: str
        The color of the line for the model fit.

    Returns
    -------
    dict[str, go.Scatter]
        The traces for the withings data frame.
    """
    ## reformat frames, column type validation happens in the function
    meas = reformat_withings(withings_data_frame)
    model = reformat_withings(withings_model_frame)

    ## get the date range of the actual measurements    
    dt_min, dt_max = meas['date'].min(), meas['date'].max()
    dt_max_forecast = dt_max + pd.Timedelta(days=n_days_forecast)

    ## get the model fit, which is the model values spanning the date range of the actual measurements
    model_fit = model[model['date'].between(dt_min, dt_max)]

    ## get the model forecast, which is the model values for the date range beyond data
    model_forecast = model[model['date'].between(dt_max, dt_max_forecast,inclusive='right')]

    ## add the last day of the model fit to the model forecast for continuity
    model_forecast = pd.concat([model_fit.iloc[-1:], model_forecast])
    model_forecast = model_forecast.sort_values('date')

    traces = [
        ## withings total weight actual measurements
        go.Scatter(
            x=meas['date'], 
            y=meas['total'], 
            mode='markers', 
            name='measurements',
            marker=dict(color=marker_color),
            showlegend=True
        ),
        ## withings total weight Holt-Winters Exponential Smoothing model
        ## spanning the date range of the actual measurements
        go.Scatter(
            x=model_fit['date'], 
            y=model_fit['total'], 
            mode='lines', 
            legendgroup='bodyweight',
            legendgrouptitle=dict(text='body weight'),
            name='model fit',
            line=dict(color=model_line_color),
            showlegend=True
        ),
        ## withings total weight Holt-Winters Exponential Smoothing model
        ## forecasted values for the date range after the actual measurements
        go.Scatter(
            x=model_forecast['date'], 
            y=model_forecast['total'], 
            legendgroup='bodyweight',
            legendgrouptitle=dict(text='body weight'),
            mode='lines', 
            name='model forecast',
            line=dict(color=model_line_color,dash='dot'),
            showlegend=True
        ),
    ]
    return traces

def create_squat_strength_figure(
    withings_data_frame: pd.DataFrame,
    withings_model_frame: pd.DataFrame
) -> go.Figure:
    withings_traces = get_withings_traces(
        withings_data_frame,
        withings_model_frame
    )
    fig = go.Figure(withings_traces)
    fig.update_layout(
        title='Weight',
        xaxis_title='Date',
        yaxis_title='Weight (lbs)'
    )
    return fig

### squats #####################
def create_squat_movement_label(row: pd.Series) -> str:
    f""" 
    Create a label for the squat movement.
    e.g. squat=back, variation=normal -> back squat
         squat=front, variation=pause -> pause front squat
    """
    v = row['variation']
    u = '' if v == 'normal' else ' '+v
    s = row['squat']
    return f'{s} {u} squat'

def create_hover_text(row: pd.Series) -> str:
    """
    Create a hover text for the squat data.

    e.g.
    date: 2026-03-13
    movement: pause front squat
    3 reps @ 100 lbs
    1RM: 100 lbs
    """
    m = create_squat_movement_label(row)
    d = row['date'].strftime('%Y-%m-%d')
    r = row['repetitions']
    w = row['weight_lbs']
    b = row['bs_1rm_est']
    
    text = '<br>'.join(
        [
            f'{d}',
            f'{m}',
            f'{r} reps @ {w:.1f} lbs',
            f'~1RM BS: {b:.1f} lbs'
        ]
    )
    return text

def reformat_squat_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat the squat data frame to the correct columns and types for plotting.
    """
    df = _validate_data_frame(df, 
        dtypes={
            'date': 'datetime64[ns]',
            'squat': 'string',
            'variation': 'string',
            'weight_lbs': 'float64',
            'repetitions': 'int64',
            'bs_1rm_est': 'float64'
        }
    )

    df['hovertext'] = df.apply(create_hover_text, axis=1)
    return df

def create_squat_measurement_traces(
    squat_data_frame: pd.DataFrame,
    squat_colors: dict[str, str],
    variation_markers: dict[str, str],
    shape_indicator_color: str,
    marker_size: int = 10,
) -> list[go.Scatter]:
    """
    Create the measurement traces for the squat data.
    """

    df = reformat_squat_data(squat_data_frame)

    ## create the traces of the data
    marker_traces = []
    for s, c in squat_colors.items():
        for v, m  in variation_markers.items():
            subset = df[
                (df['squat'] == s) &
                (df['variation'] == v)
            ]
            if len(subset) == 0:
                continue
            t = go.Scatter(
                x=subset['date'],
                y=subset['bs_1rm_est'],
                mode='markers',
                hovertext=subset['hovertext'],
                hoverinfo='text',
                marker=dict(
                    color=c, 
                    symbol=m, 
                    size=6
                ),
                showlegend=False
            )
            marker_traces.append(t)

    legend_traces = [
        go.Scatter(
            x=[pd.NA],
            y=[pd.NA],
            mode='markers',
            marker=dict(
                color=c, 
                symbol='circle', 
                size=marker_size
            ),
            name=s,
            legendgroup='squat',
            legendgrouptitle=dict(text='squat'),
            showlegend=True
        )
        for s, c in squat_colors.items()
    ] + [
        go.Scatter(
            x=[pd.NA],
            y=[pd.NA],
            mode='markers',
            marker=dict(
                symbol=f'{m}-open', 
                size=marker_size,
                color=shape_indicator_color,
                line=dict(width=1)
            ),
            name=v,
            legendgroup='variation',
            legendgrouptitle=dict(text='variation'),
            showlegend=True
        )
        for v, m in variation_markers.items()
    ]

    return marker_traces+legend_traces

def reformat_squat_model(
    squat_model_frame: pd.DataFrame
) -> pd.DataFrame:
    df = _validate_data_frame(
        squat_model_frame,
        dtypes={
            'date': 'datetime64[ns]',
            'strength_estimate': 'float64',
            'strength_sd': 'float64'
        }
    ).rename(columns={
        'strength_estimate': 'value',
        'strength_sd': 'error'
    })

    return df

def get_squat_model_error_band_traces(
    squat_model_frame: pd.DataFrame,
    fill_color: str | None = None,
    opacity: float = 0.2,
    legendgroup: str = "band",
    legend_title: str = "1 RM BS<br>Estimate",
    name: str = "Kalman<br>Filter ±1σ",
) -> list[go.Scatter]:
    """
    Get the error band plotting traces for the Kalman filter.
    """
    if fill_color is None:
        fill_color = COLORS.text

    df = reformat_squat_model(squat_model_frame)

    traces = [
        go.Scatter(
            x=df['date'],
            y=df['value']+df['error'],
            mode='lines',
            line=dict(width=0),
            legendgroup=legendgroup,
            showlegend=False,
            hoverinfo='none'
        ),
        go.Scatter(
            name=name,
            x=df['date'],
            y=df['value']-df['error'],
            line=dict(width=0),
            mode='lines',
            fillcolor=hex_to_rgba(fill_color, opacity),
            fill='tonexty',
            legendgroup=legendgroup,
            legendgrouptitle=dict(text=legend_title),
            showlegend=True,
            hoverinfo='none'
        )
    ]
    return traces

def get_strength_plotting_traces(
    squat_data_frame: pd.DataFrame,
    squat_model_frame: pd.DataFrame,
) -> list[go.Scatter]:
    """
    Get the strength plotting traces for the squat data.
    """
        
    squat_colors = {
        'back': COLORS.blue,
        'front': COLORS.orange,
        'overhead': COLORS.green
    }

    missing_squats = set(squat_data_frame['squat'].unique()) - set(squat_colors.keys())
    assert len(missing_squats) == 0, f"Missing assigned colors for squats: {missing_squats}"

    variation_markers = {
        'normal': 'circle',
        'pause': 'square',
        'tempo': 'triangle-up',
        'in-the-hole': 'triangle-down',
    }

    missing_variations = set(squat_data_frame['variation'].unique()) - set(variation_markers.keys())
    assert len(missing_variations) == 0, f"Missing assigned markers for variations: {missing_variations}"

    meas_traces = create_squat_measurement_traces(
        squat_data_frame, 
        squat_colors, 
        variation_markers, 
        shape_indicator_color=COLORS.text
    )
    
    error_traces = get_squat_model_error_band_traces(squat_model_frame)

    return meas_traces+error_traces

def create_squat_strength_figure(
    squat_data_frame: pd.DataFrame,
    squat_model_frame: pd.DataFrame
) -> go.Figure:
    squat_traces = get_strength_plotting_traces(
        squat_data_frame,
        squat_model_frame
    )
    fig = go.Figure(squat_traces)
    fig.update_layout(hovermode='closest')
    fig.update_layout(
        title='Squat Strength',
        xaxis_title='Date',
        yaxis_title='One Rep Max Back Squat Equivalent (lbs)'
    )
    return fig