from dash import Dash, dcc, html, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go
import dash
import numpy as np
import sys

import pandas as pd
import os

initial_file = sys.argv[1]


# csv format
X = "x"
Y = "y"
ACTION = "action"
YAW = "yaw"

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
    df[YAW] = np.arctan2(df[Y].diff().fillna(0), df[X].diff().fillna(0))
    df["yaw_for_arrow"] = 90 - np.degrees(df[YAW])
    df["color"] = df[ACTION].apply(lambda x: COLOR.get(x, "green"))
    df["text"] = df[[ACTION, YAW]].apply(
        lambda x: f"{int(x.name) + 1: >4}: yaw={x[YAW]:.4f}{'<br>action=' + x[ACTION] if x[ACTION] else ''}",
        axis=1,
    )

    fig = go.Figure()

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


# @callback(
#     Output("graph-data", "data"),
#     Output("file-timestamp", "data"),
#     Input("file-check-interval", "n_intervals"),
#     State("file-timestamp", "data"),
# )
# def reload_data(n_intervals, filename, file_timestamp):
#     if filename is None:
#         return dash.no_update
#     last_modified_time = file_timestamp

#     # Get the current modified time of the file
#     current_modified_time = os.path.getmtime(filename)

#     # Check if the file has been modified since the last check
#     if current_modified_time == last_modified_time:
#         return dash.no_update

#     # Read the file and return the data
#     df = pd.read_csv(filename)
#     return {
#         "columns": [{"name": i, "id": i} for i in df.columns],
#         "values": df.to_dict("records"),
#     }, current_modified_time


# @callback(Output("graph-view", "data"), Input("graph-content", "relayoutData"))
# def update_graph_view(relayoutData):
#     if relayoutData is None:
#         return dash.no_update
#     ret = {}
#     if "xaxis.range[0]" in relayoutData:
#         ret["xaxis"] = [relayoutData["xaxis.range[0]"], relayoutData["xaxis.range[1]"]]
#     if "yaxis.range[0]" in relayoutData:
#         ret["yaxis"] = [relayoutData["yaxis.range[0]"], relayoutData["yaxis.range[1]"]]
#     if "autosize" in relayoutData:
#         ret["autosize"] = relayoutData["autosize"]
#     return ret


# @callback(
#     Output("graph-content", "figure"),
#     Input("graph-data", "data"),
#     State("graph-view", "data"),
# )
# def update_graph(data, graph_view):
#     fig = go.Figure()
#     fig.update_layout(dragmode="pan", width=None, height=None, autosize=True)
#     fig.update_xaxes(scaleanchor=Y, scaleratio=1, title_text=X)
#     fig.update_yaxes(title_text=Y)
#     df = data.get("values", [])

#     yaws = [0]
#     for i in range(1, len(df)):
#         dx = df[i][X] - df[i - 1][X]
#         dy = df[i][Y] - df[i - 1][Y]
#         yaws.append(math.atan2(dy, dx))

#     yaws_for_arrow = [90 - math.degrees(yaw) for yaw in yaws]

#     text = [
#         f"{i+1}, yaw: {yaw}{'<br>action: ' + elem['action'] if elem['action'] else ''}"
#         for i, (yaw, elem) in enumerate(zip(yaws, df))
#     ]
#     color = [COLOR.get(elem[ACTION], "green") for elem in df]

#     fig.add_trace(
#         go.Scatter(
#             x=[d[X] for d in df],
#             y=[d[Y] for d in df],
#             text=text,
#             marker_color=color,
#             mode="markers+lines",
#             line_color="gray",
#             marker=dict(symbol="arrow", angle=yaws_for_arrow, size=15),
#         )
#     )
#     if graph_view:
#         if "xaxis" in graph_view:
#             fig.update_xaxes(range=graph_view["xaxis"])
#         if "yaxis" in graph_view:
#             fig.update_yaxes(range=graph_view["yaxis"])
#         if "autosize" in graph_view:
#             fig.update_layout(autosize=graph_view["autosize"])
#     return fig


def main():
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
