from dash import Dash, dcc, html, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go
import dash
import numpy as np
import argparse

import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("initial_file", type=str, help="Initial file to load")
parser.add_argument("--base-csv", type=str, help="Base pose list csv file")
parser.add_argument(
    "--simple-xy-col-for-base-csv",
    action="store_true",
    help="Use 'x', 'y' for the base csv columns name instead of '/pcl_pose/pose/pose/positon/x,y'",
)

args = parser.parse_args()
initial_file = args.initial_file
base_file = args.base_csv

base_df = pd.read_csv(base_file) if base_file else None


# csv format
X = "x"
Y = "y"
ACTION = "action"
YAW = "yaw"
BASE_X, BASE_Y = (
    ("x", "y")
    if args.simple_xy_col_for_base_csv
    else ("/pcl_pose/pose/pose/position/x", "/pcl_pose/pose/pose/position/y")
)

# for visualization
COLOR = {
    "continue": "blue",
    "stop": "red",
}


app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Label(
                    children="Waypoint Viewer",
                    style={"flex": "1", "text-align": "left", "align-self": "center"},
                ),
                html.Div(
                    [
                        dcc.Input(
                            id="filename-input",
                            value=initial_file,
                            type="text",
                            style={"width": "100%"},
                        ),
                        html.Button("Change", id="filename-change-button", n_clicks=0),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "flex-end",
                        "flex": "1",
                    },
                ),
            ],
            style={"display": "flex", "justifyContent": "space-between"},
        ),
        dcc.Graph(
            id="graph-content",
            style={"flex": "1", "align-items": "stretch"},
            config={"scrollZoom": True},
        ),
        dcc.Interval(id="file-check-interval", interval=500, n_intervals=0),
        dcc.Store(id="new-filename"),
        dcc.Store(id="filename"),
        dcc.Store(id="file-timestamp"),
        dcc.Store(id="reload-signal"),
        dcc.Store(id="data-frame"),
        dcc.Store(id="graph-view"),
        dcc.Store(id="graph-data"),
    ],
    style={
        "display": "flex",
        "flex-flow": "column",
        "height": "100vh",
    },
)


@callback(
    Output("new-filename", "data"),
    State("filename-input", "value"),
    Input("filename-change-button", "n_clicks"),
)
def change_filename(value, _):
    return value


@callback(
    Output("reload-signal", "data"),
    Output("filename", "data"),
    Output("file-timestamp", "data"),
    Input("file-check-interval", "n_intervals"),
    State("new-filename", "data"),
    State("filename", "data"),
    State("file-timestamp", "data"),
)
def update_file_timestamp(_, new_filename, filename, file_timestamp):
    if new_filename is None:
        return dash.no_update, dash.no_update, dash.no_update

    if new_filename != filename or file_timestamp is None:
        file_timestamp = 0

    new_timestamp = os.path.getmtime(new_filename)
    if new_timestamp > file_timestamp:
        return True, new_filename, new_timestamp

    return False, new_filename, new_timestamp


@callback(Output("graph-view", "data"), Input("graph-content", "relayoutData"))
def update_graph_view(relayoutData):
    if relayoutData is None:
        return dash.no_update
    ret = {}
    if "xaxis.range[0]" in relayoutData:
        ret["xaxis"] = [relayoutData["xaxis.range[0]"], relayoutData["xaxis.range[1]"]]
    if "yaxis.range[0]" in relayoutData:
        ret["yaxis"] = [relayoutData["yaxis.range[0]"], relayoutData["yaxis.range[1]"]]
    if "autosize" in relayoutData:
        ret["autosize"] = relayoutData["autosize"]
    return ret

def compute_yaw(df: pd.DataFrame) -> pd.DataFrame:
    yaw = 0.0

    has_yaw_deg = "yaw_deg" in df.columns

    def set_if_nan(i, yaw):
        if not np.isnan(df.at[df.index[i], YAW]):
            return
        if has_yaw_deg and not np.isnan(df.at[df.index[i], "yaw_deg"]):
            yaw = np.radians(df.at[df.index[i], "yaw_deg"])

        df.at[df.index[i], YAW] = yaw

    for i in range(len(df) - 1):
        dx = df.at[df.index[i + 1], X] - df.at[df.index[i], X]
        dy = df.at[df.index[i + 1], Y] - df.at[df.index[i], Y]
        yaw = np.arctan2(dy, dx)
        set_if_nan(i, yaw)

    set_if_nan(len(df) - 1, yaw)

    return df



@callback(
    Output("graph-content", "figure"),
    Input("reload-signal", "data"),
    State("filename", "data"),
    State("graph-view", "data"),
)
def load_and_update_graph(reload_signal, filename, graph_view):
    if not reload_signal or filename is None:
        return dash.no_update
    df = pd.read_csv(filename)

    df.loc[df[ACTION].isna(), [ACTION]] = ""
    if df.at[df.index[-1], ACTION] == "":
        df.at[df.index[-1], ACTION] = "stop"

    # tolerance
    if "left_tolerance" not in df.columns:
        df["left_tolerance"] = 0.0
    df.loc[df["left_tolerance"].isna(), ["left_tolerance"]] = 0.0
    if "right_tolerance" not in df.columns:
        df["right_tolerance"] = 0.0
    df.loc[df["right_tolerance"].isna(), ["right_tolerance"]] = 0.0


    if YAW not in df.columns:
        df[YAW] = np.nan

    compute_yaw(df)
    df["yaw_for_arrow"] = 90 - np.degrees(df[YAW])
    df["color"] = df[ACTION].apply(lambda x: COLOR.get(x, "green"))
    df["text"] = df[[ACTION, YAW]].apply(
        lambda x: f"{int(x.name) + 1: >4}: yaw={x[YAW]:.4f}{'<br>action=' + x[ACTION] if x[ACTION] else ''}",
        axis=1,
    )

    fig = go.Figure()

    if base_df is not None:
        fig.add_trace(
            go.Scatter(
                x=base_df[BASE_X],
                y=base_df[BASE_Y],
                text=df.index.astype(str),
                mode="markers",
                marker=dict(symbol="circle", color=base_df.index),
                name="base",
                opacity=0.4,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df[X],
            y=df[Y],
            text=df["text"],
            mode="markers+lines",
            marker_color=df["color"],
            line_color="gray",
            marker=dict(symbol="arrow", angle=df["yaw_for_arrow"], size=15),
        )
    )

    # add tolerance
    tole_x = []
    tole_y = []
    for i, row in df.iterrows():
        # left
        tole_x.append(row[X] + row["left_tolerance"] * np.cos(row[YAW] + np.pi / 2))
        tole_y.append(row[Y] + row["left_tolerance"] * np.sin(row[YAW] + np.pi / 2))
        # right
        tole_x.append(row[X] + row["right_tolerance"] * np.cos(row[YAW] - np.pi / 2))
        tole_y.append(row[Y] + row["right_tolerance"] * np.sin(row[YAW] - np.pi / 2))
        # insert nan
        tole_x.append(np.nan)
        tole_y.append(np.nan)

    fig.add_trace(
        go.Scatter(
            x=tole_x,
            y=tole_y,
            mode="lines",
            line_color="blue",
            name="tolerance",
        )
    )




    fig.update_layout(dragmode="pan", width=None, height=None, autosize=True)
    fig.update_xaxes(scaleanchor=Y, scaleratio=1, title_text=X)
    fig.update_yaxes(title_text=Y)
    if graph_view is None:
        graph_view = {}
    if "xaxis" in graph_view:
        fig.update_xaxes(range=graph_view["xaxis"])
    if "yaxis" in graph_view:
        fig.update_yaxes(range=graph_view["yaxis"])
    if "autosize" in graph_view:
        fig.update_layout(autosize=graph_view["autosize"])

    return fig


def main():
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
