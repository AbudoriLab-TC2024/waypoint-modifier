import numpy as np
import pandas as pd
import plotly.graph_objects as go


def curvature(
    df: pd.DataFrame, window=20, x_col="x", y_col="y", min_r=0.1
) -> pd.Series:
    """
    Compute the curvatures of a trajectory.
    """
    ret = []
    for i in range(len(df)):
        if i < window or i >= len(df) - window:
            ret.append(0.0)
        else:
            x = df[x_col].values[i - window : i + window + 1]
            y = df[y_col].values[i - window : i + window + 1]

            A = np.vstack((x, y, np.ones(len(x)))).T
            v = -(x**2 + y**2)
            u, *_ = np.linalg.lstsq(A, v, rcond=None)

            cx_pred = u[0] / -2
            cy_pred = u[1] / -2
            r_pred = np.sqrt(cx_pred**2 + cy_pred**2 - u[2])
            r_pred = max(r_pred, min_r)
            ret.append(1.0 / r_pred)

    return pd.Series(ret, index=df.index)


def set_curvature(
    df: pd.DataFrame,
    x_col="x",
    y_col="y",
    curvature_col="curvature",
    window=20,
    min_r=0.1,
) -> pd.DataFrame:
    """
    Add curvature to the dataframe.
    """
    df[curvature_col] = curvature(
        df, window=window, x_col=x_col, y_col=y_col, min_r=min_r
    )
    return df


def set_distance(
    df: pd.DataFrame, x_col="x", y_col="y", distance_col="distance"
) -> pd.DataFrame:
    """
    Add distance to the dataframe.
    """
    df[distance_col] = np.sqrt(df[x_col].diff() ** 2 + df[y_col].diff() ** 2).fillna(0)
    return df


def do_sample(
    df: pd.DataFrame,
    max_d: float,
    max_c: float,
    x_col="x",
    y_col="y",
    distance_col="distance",
    curvature_col="curvature",
) -> pd.DataFrame:
    """
    Sample waypoints from the trajectory.
    """
    dist = 0.0
    curv = 0.0

    ret = []
    for i in range(len(df)):
        dist += df[distance_col].values[i]
        curv += df[curvature_col].values[i]

        if dist > max_d or curv > max_c:
            ret.append(df.iloc[i])
            dist = 0.0
            curv = 0.0

    ret_df = pd.DataFrame(ret).copy()
    ret_df.reset_index(drop=True, inplace=True)
    set_distance(ret_df, x_col, y_col, distance_col)

    return ret_df


def resample(
    df: pd.DataFrame,
    max_d: float,
    max_c: float,
    window=20,
    min_r=0.1,
    x_col="x",
    y_col="y",
    distance_col="distance",
    curvature_col="curvature",
) -> pd.DataFrame:
    """
    Add distance and curvaute to the input df, and resample it.
    """
    set_distance(df, x_col, y_col, distance_col)
    set_curvature(df, x_col, y_col, curvature_col, window, min_r)

    return do_sample(df, max_d, max_c, x_col, y_col, distance_col, curvature_col)


def create_figure(title=None, height=600) -> go.Figure:
    fig = go.Figure()
    fig.update_xaxes(title_text="x", scaleanchor="y", scaleratio=1)
    fig.update_xaxes(title_text="y")
    fig.update_layout(dragmode="pan", height=height, title=title)
    return fig


def plot_trajectory(
    df: pd.DataFrame,
    name: str,
    x_col="x",
    y_col="y",
    distance_col="distance",
    fig: go.Figure = None,
    marker_size=2,
    **kwargs
) -> go.Figure:
    if fig == None:
        fig = create_figure()

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            text=df.index.astype(str) + "<br>dist=" + df[distance_col].astype(str),
            marker=dict(color=df.index, size=marker_size),
            mode="markers",
            name=name,
            **kwargs
        )
    )

    return fig
