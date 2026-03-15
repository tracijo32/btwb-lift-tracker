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
    df['% fat'] = df['fat'] / df['total'] * 100
    return df

def create_withings_traces(
    withings_data_frame: pd.DataFrame,
    withings_model_frame: pd.DataFrame,
    marker_color: str = COLORS.text,
    line_color: str = COLORS.grid,
    column_name: str = 'total'
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

    ## get the model fit, which is the model values spanning the date range of the actual measurements
    model_fit = model[model['date'].between(dt_min, dt_max)]

    traces = [
        ## withings total weight actual measurements
        go.Scatter(
            x=meas['date'], 
            y=meas[column_name], 
            mode='markers', 
            name='measurements',
            marker=dict(color=marker_color),
            showlegend=False
        ),
        ## withings total weight Holt-Winters Exponential Smoothing model
        ## spanning the date range of the actual measurements
        go.Scatter(
            x=model_fit['date'], 
            y=model_fit[column_name], 
            mode='lines', 
            name='7 day rolling average',
            line=dict(color=line_color),
            showlegend=False
        )
    ]
    return traces


def create_body_weight_figure(
    withings_data: pd.DataFrame,
    withings_model: pd.DataFrame
) -> go.Figure:

    traces = create_withings_traces(
        withings_data,
        withings_model,
        column_name='total',
        marker_color=COLORS.purple,
        line_color=COLORS.purple
    )
    fig = go.Figure(traces)

    fig.update_layout(
        title='Body weight',
        xaxis_title='Date',
        yaxis_title='Weight (lb)',
    )
    return fig

def create_body_composition_figure(
    withings_data: pd.DataFrame,
    withings_model: pd.DataFrame
) -> go.Figure:

    traces = create_withings_traces(
        withings_data,
        withings_model,
        column_name='% fat',
        marker_color=COLORS.sea_green,
        line_color=COLORS.sea_green
    )
    fig = go.Figure(traces)

    fig.update_layout(
        title='Body Composition',
        xaxis_title='Date',
        yaxis_title='% body fat',
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

def create_squat_model_error_band_traces(
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
    
    error_traces = create_squat_model_error_band_traces(squat_model_frame)

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

### both withings and squat strength #####################
def reformat_goal_frame(
    goal_frame: pd.DataFrame,
    weight_column_name: str = 'weight'
):
    g = goal_frame.copy()
    g['date'] = pd.to_datetime(g['date'].dt.date)

    assert weight_column_name in g.columns, \
        f"weight_column_name {weight_column_name} not in {g.columns}"
    g = g.rename(columns={weight_column_name:'weight'})\
        .groupby('date')['weight'].mean().reset_index()

    return g

def get_body_weight_back_squat_prediction(
    combined_weight_frame: pd.DataFrame,
):

    df = _validate_data_frame(
        combined_weight_frame,
        dtypes = {
        'date': 'datetime64[ns]',
        'weight_body': 'float64',
        'weight_squat': 'float64',
        }
    )
    goal_met = df['weight_squat'].ge(df['weight_body'])

    if goal_met.sum() == 0:
        last_date = df['date'].max()
        return last_date, None

    instance_met = df.loc[df.loc[goal_met, 'date'].idxmin()]
    date_met = instance_met['date']
    weight_met = instance_met[['weight_squat', 'weight_body']].mean()
    return date_met, weight_met

def create_bwbs_tracker_figure(
    withings_data_frame: pd.DataFrame,
    squat_data_frame: pd.DataFrame,
    withings_model_frame: pd.DataFrame,
    squat_model_frame: pd.DataFrame
):

    ## reformat the data frames to have a 'date' column and a 'weight' column
    ## then merge them together on date. all of the forecasts are sampled daily,
    ## so the dates will be the same.
    b = reformat_goal_frame(withings_model_frame, 'total')
    s = reformat_goal_frame(squat_model_frame, 'strength_estimate')
    combined_weights = pd.merge(b, s, on='date',suffixes=('_body', '_squat'))

    ## separate the models into past and future:
    ## past are fitted to the data
    ## future are forecasts
    squat_date_max = squat_data_frame['date'].max().date()
    withings_date_max = withings_data_frame['date'].max().date()
    last_data_date = pd.to_datetime(max(squat_date_max, withings_date_max))

    combined_weights_past = combined_weights[combined_weights['date'].le(last_data_date)]
    combined_weights_future = combined_weights[combined_weights['date'].gt(last_data_date)]

    ## plot the data as lines
    ## body weight is dark blue, back squat is green
    ## fits are solid, forecasts are dashed
    traces = [
        go.Scatter(
            x=combined_weights_past['date'],
            y=combined_weights_past['weight_body'],
            mode='lines',
            line=dict(color=COLORS.blue),
            name='body weight (7 day rolling average)',
        ),
        go.Scatter(
            x=combined_weights_past['date'],
            y=combined_weights_past['weight_squat'],
            mode='lines',
            line=dict(color=COLORS.orange),
            name='back squat (Kalman filter)',
        ),
            go.Scatter(
            x=combined_weights_future['date'],
            y=combined_weights_future['weight_body'],
            mode='lines',
            line=dict(color=COLORS.blue, dash='dash'),
            name='body weight (forecast)',
        ),
        go.Scatter(
            x=combined_weights_future['date'],
            y=combined_weights_future['weight_squat'],
            mode='lines',
            line=dict(color=COLORS.orange, dash='dash'),
            name='back squat (forecast)',
        )
    ]

    ## plot the models
    fig = go.Figure(data=traces)

    ## using the forecasts for both models, find the date and weight
    ## where the back squat first exceeds the body weight
    ## if that never happens, then the goal is the last date of the forecast
    ## the weight returne is None
    goal_date, goal_weight = get_body_weight_back_squat_prediction(combined_weights_future)

    ## if the goal is reached, then plot the goal as a star
    ## the text is formatted to show the date and weight
    if goal_weight is not None:
        text = '<br>'.join([
            'A body-weight back squat',
            'is predicted to be reached',
            f'around {goal_date.strftime("%B %d, %Y")}',
            f'at a weight of {goal_weight:.0f} lbs.'
        ])
        fig.add_trace(
            go.Scatter(
                x=[goal_date],
                y=[goal_weight],
                mode='markers',
                marker=dict(
                    color=COLORS.green,
                    symbol='star',
                    size=30
                ),
                name='goal',
                showlegend=False
            )
        )
        fig.add_annotation(
            x=goal_date,
            y=goal_weight,
            text=text,
            font=dict(size=12),
            showarrow=True,
            ax=0,
            ay=100,
            standoff=15,
            arrowhead=2,
            arrowwidth=2,
            arrowcolor=COLORS.green,
        )
    else:
        ## if the goal is not reached, then place an annotation
        ## saying that the goal is to occur before the last date
        ## of the forecast
        txt_idx = combined_weights_future['date'].idxmin()
        x = combined_weights_future.loc[txt_idx,'date']
        y = combined_weights_future.loc[txt_idx,['weight_body','weight_squat']].mean()

        last_model_date = combined_weights_future['date'].max()

        text = '<br>'.join([
            'A body-weight back squat',
            'is not predicted to be reached',
            f'before {last_model_date.month_name()} {last_model_date.year}.'
        ])
        fig.add_annotation(x=x,y=y,text=text,showarrow=False)  

    ## add true 1RM back squat data points to the figure
    min_date = combined_weights['date'].min()
    bs_1rm = squat_data_frame[
        squat_data_frame['date'].ge(min_date) & 
        squat_data_frame['squat'].eq('back') &
        squat_data_frame['variation'].eq('normal') &
        squat_data_frame['repetitions'].eq(1)
    ].rename(columns={'weight_lbs':'weight'})

    fig.add_trace(
        go.Scatter(
            x=bs_1rm['date'],
            y=bs_1rm['weight'],
            mode='markers',
            marker=dict(
                color=COLORS.orange,
                symbol='diamond',
                size=15
            ),
            name='actual back squat 1RM',
            showlegend=False
        )
    )

    ## update all of the layout elements
    fig.update_layout(
        title='Body-weight back squat<br>goal tracker',
        xaxis_title='Date',
        yaxis_title='Weight (lb)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    return fig

if __name__ == '__main__':
    import pickle
    squat_data = pickle.load(open('squat_data.pkl','rb'))
    squat_model = pickle.load(open('squat_model.pkl','rb'))
    withings_data = pickle.load(open('withings_data.pkl','rb'))
    withings_model = pickle.load(open('withings_model.pkl','rb'))

    from plotly_theme import register_chunk_template
    _ = register_chunk_template(transparent=True, set_default=True)

    create_bwbs_tracker_figure(withings_model, squat_model)\
        .write_html('bwbs_forecast.html')

    create_squat_strength_figure(squat_data, squat_model)\
        .write_html('squat_strength.html')

    create_withings_figure(withings_data, withings_model)\
        .write_html('withings.html')
