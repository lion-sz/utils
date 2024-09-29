import polars as pl
from polars import col
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px


from app import app, state


def build_app(config):
    dat = pl.read_parquet("trace.parquet")

    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    dimensions = list(config.space.keys())
    rs = dat.select(col("r").unique().sort())["r"].to_list()

    title = dbc.Row(dbc.Col(html.H1("Hyperparameter Tuning")))
    settings = [
        dbc.Row(
            [
                dbc.Col(dbc.Label("3d Plot"), width=2),
                dbc.Col(dbc.Switch(id="select-3d", label="", value=False)),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Label("Log scale"), width=2),
                dbc.Col(dbc.Switch(id="select-log", label="", value=False)),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Label("X axis"), width=2),
                dbc.Col(
                    dbc.RadioItems(
                        options=dimensions,
                        value=dimensions[0],
                        id="var-select-x",
                        inline=True,
                    )
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Label("Y axis"), width=2),
                dbc.Col(
                    dbc.RadioItems(
                        options=dimensions,
                        value=dimensions[1],
                        id="var-select-y",
                        inline=True,
                    )
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Label("Z axis"), width=2),
                dbc.Col(
                    dbc.RadioItems(
                        options=dimensions,
                        value=dimensions[2],
                        id="var-select-z",
                        inline=True,
                    )
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Label("Min Resources Used"), width=2),
                dbc.Col(
                    dbc.RadioItems(options=rs, value=rs[0], id="select-r", inline=True)
                ),
            ]
        ),
    ]
    plot = dbc.Row(dbc.Col(dcc.Graph(figure={}, id="plot", style={"height": 800})))

    app.layout = dbc.Container([html.Div([title, *settings, plot])])

    # Add controls to build the interaction
    @callback(
        Output(component_id="plot", component_property="figure"),
        Input(component_id="select-3d", component_property="value"),
        Input(component_id="select-log", component_property="value"),
        Input(component_id="var-select-x", component_property="value"),
        Input(component_id="var-select-y", component_property="value"),
        Input(component_id="var-select-z", component_property="value"),
        Input(component_id="select-r", component_property="value"),
    )
    def update_graph(show3d: bool, log_loss: bool, x, y, z, min_r):
        tmp = dat.filter(col("r").ge(min_r)).filter(
            col("r") == col("r").max().over("id")
        )
        if log_loss:
            tmp = tmp.with_columns(col("loss").log())
        tmp = tmp.to_pandas()
        if show3d:
            fig = px.scatter_3d(tmp, x=x, y=y, z=z, color="loss")
        else:
            fig = px.scatter(tmp, x=x, y=y, color="loss")
        return fig

    return app


@app.command("visualize")
def visualize():
    config = state["config"]
    dash_app = build_app(config.tune)
    dash_app.run(debug=True, host="0.0.0.0")
    return
