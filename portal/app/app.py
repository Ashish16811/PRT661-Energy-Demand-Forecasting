from __future__ import annotations

import os
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, callback, clientside_callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

from pipeline.config import (
    PORTAL_VERSION,
    REGION_COORDS,
    REGION_LABELS,
    REGION_MARKET,
    REGION_TZ,
    REGIONS,
)
from pipeline.database import (
    init_db,
    read_df,
    sql_half_hour,
    sql_minutes_between,
    sql_month,
    sql_month_start,
    sql_year,
    utcnow_naive,
)
from pipeline import insights as ins
from pipeline.feature_contract import FEATURE_SET_A, FEATURE_SET_B, WEATHER_FEATURES
from pipeline.model_evidence import (
    CORE_MODELS,
    FEATURE_SET_LABEL,
    evidence,
    load_manifest,
    model_feature_set,
    production_config,
)
from pipeline.region_prefs import default_region, resolve_region
from pipeline.timezones import (
    local_now as region_local_now,
    market_now as region_market_now_aware,
    market_now_naive as region_market_now_naive,
    next_market_half_hour as next_region_market_half_hour,
    market_naive_to_local,
    region_clock_label,
    market_clock_label,
)

init_db()

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="PRT661 Energy Demand Forecasting Portal",
    update_title=None,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "theme-color", "content": "#08162C"},
        {"name": "description", "content": "PRT661 Australian electricity demand forecasting portal: live demand, next-30-minute forecasts, verification and AEMO comparison."},
    ],
)
server = app.server


@server.get("/healthz")
def healthz():
    """Railway deployment health check: app import + database reachability."""
    try:
        probe = read_df("SELECT 1 AS ok")
        if probe.empty:
            raise RuntimeError("database probe returned no rows")
    except Exception:
        return {"status": "unhealthy", "portal_version": PORTAL_VERSION}, 503
    return {"status": "ok", "portal_version": PORTAL_VERSION}, 200


# ---------------------------------------------------------------------------
# v3: production-model helpers. The served model per region comes from the
# evidence-driven resolver (selected_model.json + existing portal bundle), never
# from list order. Cached for five minutes inside production_config().
# ---------------------------------------------------------------------------
def _prod_cfg():
    return production_config()


def _default_model(region: str) -> str:
    return _prod_cfg().default_model.get(region, "RandomForest_A")


def _prod_models(region: str) -> list:
    models = _prod_cfg().production_models.get(region) or []
    return models or [_default_model(region)]


def _serving(region: str):
    return _prod_cfg().status.get(region)


def _fs_label(model: str) -> str:
    fs = model_feature_set(model)
    return FEATURE_SET_LABEL.get(fs, "") if fs else ""

# ---------------------------------------------------------------------------
# Design tokens — "golden hour energy infrastructure".
# Kept in sync with the custom properties at the top of assets/dashboard.css.
#   navy  = technical credibility     warm white = clarity
#   gold  = energy identity (accent)  blue = forecasting / data
#   green = verification / status
# ---------------------------------------------------------------------------
NAVY = "#08162C"            # Deep midnight navy
NAVY_2 = "#10264D"          # Energy navy
GOLD = "#F2B544"            # Golden amber (accent only)
SUNSET = "#E98A36"          # Sunset orange
IVORY = "#F7F4ED"
SURFACE = "#FFFDF8"         # Warm white
TEXT = "#102039"
MUTED = "#657187"
GRIDLINE = "#EDF0F5"
AXIS = "#D5DBE5"
BORDER = "#E4E1D8"
FORECAST_BLUE = "#2675F5"
VERIFIED_GREEN = "#15976F"

# Semantic chart colours, shared by every Plotly figure in the portal.
ACTUAL_COLOR = "#7C3AED"    # observed AEMO demand — vivid violet
FORECAST_COLOR = "#2F7BFF"  # project forecast — bright forecasting blue
AEMO_COLOR = "#FF9F1C"      # AEMO pre-dispatch benchmark — bright amber, dashed
ERROR_COLOR = "#12B886"     # verification metrics — bright emerald
SMAPE_COLOR = "#E6399B"     # percentage error line — bright magenta
ACCENT = FORECAST_BLUE      # legacy alias used by loaders / fallbacks
AMBER = AEMO_COLOR          # legacy alias
SUCCESS = VERIFIED_GREEN    # legacy alias
FONT_STACK = "Manrope, Inter, 'Segoe UI', ui-sans-serif, system-ui, -apple-system, Arial, sans-serif"
PORTAL_TIMEZONE = os.getenv("PORTAL_TIMEZONE", "Australia/Adelaide")

# One restrained identity colour per region, shared by map, cards and share bar.
REGION_COLORS = {
    "NSW1": "#2F7BFF",
    "QLD1": "#FF9F1C",
    "VIC1": "#8B5CF6",
    "SA1": "#EC4899",
    "TAS1": "#12B886",
    "WA": "#06B6D4",
}

# Regional year comparison: earlier years recede along a navy ramp and the
# most recent selected year is drawn in sunset orange so it reads first.
YEAR_RAMP = ["#06B6D4", "#8B5CF6", "#12B886", "#EC4899", "#2F7BFF", "#A3E635"]
LATEST_YEAR_COLOR = "#FF7A1A"

# Lucide-style outline icons are drawn by CSS masks (see dashboard.css), so the
# portal stays pure Dash with no extra icon package.
def icon(name: str, extra: str = ""):
    return html.Span(className=f"ico ico-{name} {extra}".strip(), **{"aria-hidden": "true"})


STATUS_LABELS = {
    "live": "Live",
    "verified": "Verified",
    "rolling": "Rolling",
    "waiting": "Waiting",
    "stale": "Stale",
    "nodata": "No data",
    "error": "Error",
}


def status_chip(state: str, text: str | None = None, title: str | None = None):
    """Consistent status label: dot + text + subtle tint (never colour alone)."""
    kwargs = {"title": title} if title else {}
    return html.Span(
        [html.Span(className="status-dot", **{"aria-hidden": "true"}), html.Span(text or STATUS_LABELS.get(state, state))],
        className=f"status status--{state}",
        **kwargs,
    )


def metric_card(title, value_id, subtitle="", variant="", subtitle_id=None):
    """KPI tile: small label, strong tabular number, quiet supporting line."""
    extra = f" metric--{variant}" if variant else ""
    sub_kwargs = {"id": subtitle_id} if subtitle_id else {}
    return html.Div(
        [
            html.Div(title, className="metric-label"),
            html.Div("—", id=value_id, className="metric-value"),
            html.Div(subtitle, className="metric-sub", **sub_kwargs),
        ],
        className=f"metric metric-{value_id.replace('_', '-')}{extra}",
    )


def small_badge(text, tone="neutral"):
    return html.Span(text, className=f"badge badge-{tone}")


def section_head(title, subtitle="", icon_name=None, aside=None, level=2):
    heading = html.H2 if level == 2 else html.H3
    return html.Div(
        [
            html.Div(
                [
                    html.Div([icon(icon_name, "section-ico") if icon_name else None, heading(title, className="section-title")], className="section-title-row"),
                    html.P(subtitle, className="section-sub") if subtitle else None,
                ]
            ),
            aside,
        ],
        className="section-head",
    )


def chart_card(title, subtitle, graph, className="", title_id=None, subtitle_id=None):
    """HTML title above a clean Plotly canvas (no duplicated Plotly titles)."""
    title_kwargs = {"id": title_id} if title_id else {}
    subtitle_kwargs = {"id": subtitle_id} if subtitle_id else {}
    return html.Div(
        [
            html.Div(
                [
                    html.H3(title, className="chart-card-title", **title_kwargs),
                    html.P(subtitle, className="chart-card-subtitle", **subtitle_kwargs),
                ],
                className="chart-card-head",
            ),
            graph,
        ],
        className=f"chart-card {className}".strip(),
    )


def chart_style(height=480):
    # CSV downloads are provided separately, so Plotly's floating mode bar is
    # hidden to keep a clean presentation surface.
    return {
        "displayModeBar": False,
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": False,
    }


def style_figure(fig: go.Figure, title: str = "", height: int = 480):
    """Single chart design system shared by every figure in the portal."""
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": TEXT}, "x": 0.01, "xanchor": "left"} if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_STACK, "color": "#3A4861", "size": 12},
        margin=dict(l=6, r=14, t=44 if not title else 64, b=6),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=NAVY_2,
            bordercolor=NAVY_2,
            font={"family": FONT_STACK, "size": 12, "color": "#FFFDF8"},
            namelength=-1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font={"size": 12, "color": "#3A4861"},
            title_font={"size": 12, "color": MUTED},
            itemwidth=40,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        height=height,
        separators=".,",
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, automargin=True, showline=True,
        linecolor=AXIS, linewidth=1, ticks="outside", ticklen=4, tickcolor=AXIS,
        tickfont={"size": 11, "color": MUTED}, title_font={"size": 12, "color": MUTED},
        showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#9AA6BA",
        spikethickness=1, spikedash="dot",
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRIDLINE, gridwidth=1, zeroline=False, automargin=True,
        tickfont={"size": 11, "color": MUTED}, title_font={"size": 12, "color": MUTED},
        separatethousands=True, ticks="", showline=False,
    )
    return fig


def empty_figure(message: str, height: int = 320, title: str | None = None):
    """Honest empty state: a short heading plus what will make data appear."""
    fig = go.Figure()
    style_figure(fig, "", height)
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False, showline=False, showspikes=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    heading = title or "No data to display yet"
    fig.add_annotation(
        text=f"<b>{heading}</b><br><span style='font-size:12px;color:{MUTED}'>{message}</span>",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, align="center",
        font={"size": 14, "color": TEXT, "family": FONT_STACK},
    )
    fig.update_layout(hovermode=False, margin=dict(l=24, r=24, t=24, b=24))
    return fig


def _marker_line(fig: go.Figure, x, color: str, label: str = "", dash: str = "dot"):
    """Vertical marker drawn as a shape + annotation.

    Equivalent to fig.add_vline(annotation_text=...) but avoids the Plotly
    datetime-annotation arithmetic issue, and keeps labels clear of the legend.
    """
    fig.add_shape(
        type="line", xref="x", yref="paper", x0=x, x1=x, y0=0, y1=1,
        line={"color": color, "width": 1.2, "dash": dash}, layer="below",
    )
    if label:
        fig.add_annotation(
            x=x, xref="x", y=1, yref="paper", text=label, showarrow=False,
            xanchor="left", yanchor="top", xshift=6, yshift=-2,
            font={"size": 11, "color": color, "family": FONT_STACK},
            bgcolor="rgba(255,253,248,0.9)", borderpad=2,
        )


def _portal_now() -> pd.Timestamp:
    """Current portal wall-clock time used for the public 'next interval' view."""
    return pd.Timestamp.now(tz=ZoneInfo(PORTAL_TIMEZONE)).tz_localize(None)


def _next_half_hour(ts: pd.Timestamp | None = None) -> pd.Timestamp:
    ts = pd.Timestamp(ts if ts is not None else _portal_now())
    return ts.floor("30min") + pd.Timedelta(minutes=30)


def _region_now(region: str) -> pd.Timestamp:
    """Region market-clock timestamp used by stored demand/forecast tables."""
    return region_market_now_naive(region)


def _next_region_half_hour(region: str) -> pd.Timestamp:
    return next_region_market_half_hour(region)


def _local_display_time(region: str, market_timestamp) -> pd.Timestamp:
    return market_naive_to_local(region, market_timestamp)


def _fresh_age_minutes(value) -> float | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return max(0.0, (pd.Timestamp(utcnow_naive()) - pd.Timestamp(ts)).total_seconds() / 60.0)


def _forecast_export_end(next_target: pd.Timestamp, horizon: str) -> pd.Timestamp:
    """Return the final timestamp included in a rolling-outlook export.

    4M follows the portal definition: the remainder of the current calendar
    month plus the next three complete calendar months.
    """
    next_target = pd.Timestamp(next_target)
    if horizon == "48H":
        return next_target + pd.Timedelta(hours=48)
    if horizon == "7D":
        return next_target + pd.Timedelta(days=7)
    if horizon == "30D":
        return next_target + pd.Timedelta(days=30)
    month_start = _portal_now().normalize().replace(day=1)
    return pd.Timestamp(month_start + pd.DateOffset(months=4) - pd.Timedelta(microseconds=1))


def australia_map(selected_region: str = "NSW1", height: int = 360):
    """Australia-only region selector: warm neutral land, ringed selected region."""
    text_positions = {
        "NSW1": "middle left",
        "QLD1": "middle left",
        "VIC1": "bottom center",
        "SA1": "middle left",
        "TAS1": "middle right",
        "WA": "middle right",
    }
    fig = go.Figure()

    # Soft ring behind the selected region. It carries customdata too, so a
    # click on the ring selects the same region as a click on the marker.
    if selected_region in REGION_COORDS:
        lat, lon = REGION_COORDS[selected_region]
        color = REGION_COLORS.get(selected_region, ACCENT)
        fig.add_trace(
            go.Scattergeo(
                lat=[lat], lon=[lon], customdata=[[selected_region]],
                mode="markers", hoverinfo="skip", showlegend=False,
                marker={"size": 44, "color": color, "opacity": 0.16, "line": {"width": 0}},
                name="selected-halo",
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lat=[lat], lon=[lon], customdata=[[selected_region]],
                mode="markers", hoverinfo="skip", showlegend=False,
                marker={"size": 32, "color": "rgba(0,0,0,0)", "line": {"width": 1.6, "color": color}},
                name="selected-ring",
            )
        )

    for region in REGIONS:
        lat, lon = REGION_COORDS[region]
        selected = region == selected_region
        fig.add_trace(
            go.Scattergeo(
                lat=[lat],
                lon=[lon],
                text=[f"<b>{region}</b>" if selected else region],
                customdata=[[region]],
                hovertemplate=f"<b>{region}</b> · {REGION_LABELS[region]}<br>{REGION_MARKET[region]} market<br>Select to analyse<extra></extra>",
                mode="markers+text",
                textposition=text_positions.get(region, "top center"),
                textfont={
                    "size": 13 if selected else 11,
                    "color": NAVY if selected else "#5A6478",
                    "family": FONT_STACK,
                },
                marker={
                    "size": 18 if selected else 13,
                    "color": REGION_COLORS.get(region, ACCENT),
                    "line": {"width": 3 if selected else 2, "color": "#FFFDF8"},
                    "opacity": 1.0 if selected else 0.85,
                },
                showlegend=False,
                name=region,
            )
        )
    fig.update_geos(
        projection_type="mercator",
        center={"lat": -27.5, "lon": 133.5},
        lataxis_range=[-45.5, -9.5],
        lonaxis_range=[110, 157],
        showland=True,
        landcolor="#EDE7DA",
        showocean=True,
        oceancolor="#FFFDF8",
        showcountries=False,
        showcoastlines=True,
        coastlinecolor="#C7BCA8",
        coastlinewidth=0.8,
        showlakes=False,
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
        resolution=50,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode="event+select",
        dragmode=False,
        hoverlabel=dict(bgcolor=NAVY_2, bordercolor=NAVY_2, font={"family": FONT_STACK, "size": 12, "color": "#FFFDF8"}),
    )
    return fig


def _map_legend(selected_region: str = "NSW1", where: str = "regional"):
    """Region list beside a map. Each row is a button that selects that region."""
    return [
        html.Button(
            [
                html.Span(className="map-legend-dot", **{"aria-hidden": "true"}),
                html.Span(r, className="map-legend-code"),
                html.Span(REGION_LABELS[r], className="map-legend-name"),
                html.Span(REGION_MARKET[r], className="map-legend-market"),
            ],
            id={"type": "region-pick", "region": r, "where": where},
            n_clicks=0,
            type="button",
            className="map-legend-row is-selected" if r == selected_region else "map-legend-row",
            style={"--region-color": REGION_COLORS.get(r, ACCENT)},
            **{"aria-pressed": "true" if r == selected_region else "false", "aria-label": f"Select {REGION_LABELS[r]}"},
        )
        for r in REGIONS
    ]


def _loading(children):
    """dcc.Loading that dims (rather than blanks) content during refreshes.

    delay_show / overlay_style exist in newer Dash releases only, so they are
    applied when supported and silently skipped on older versions.
    """
    try:
        return dcc.Loading(
            children, type="dot", color=FORECAST_BLUE, delay_show=450,
            overlay_style={"visibility": "visible", "opacity": 0.55},
        )
    except TypeError:
        return dcc.Loading(children, type="dot", color=FORECAST_BLUE)


def _graph(graph_id, height, min_height=280):
    return _loading(
        dcc.Graph(
            id=graph_id,
            config=chart_style(height),
            style={"height": f"clamp({min_height}px, 62vw, {height}px)"},
            className="chart-graph",
        )
    )


# ---------------------------------------------------------------------------
# Navigation — Dashboard now owns the complete live operational experience.
# Forecast Performance, AEMO Comparison and Regional Analysis remain separate.
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("dashboard", "Dashboard", "dashboard"),
    ("performance", "Forecast Performance", "target"),
    ("aemo", "AEMO Comparison", "arrows"),
    ("history", "Regional Analysis", "map"),   # value kept as "history" (existing view key)
]
DEFAULT_TAB = "dashboard"
NAV_KEYS = [v for v, _, _ in NAV_ITEMS]

VIEW_COPY = {
    "performance": ("Forecast performance", "Live verification of stored forecasts and historical model evidence."),
    "aemo": ("AEMO comparison", "Project forecast against AEMO pre-dispatch on common timestamps only."),
    "history": ("Regional analysis", "Demand history, seasonal pattern and year-on-year comparison."),
}

CAPABILITIES = [
    ("activity", "Actual demand", "30-minute demand, six regions"),
    ("target", "Forecast verification", "Scored as actuals arrive"),
    ("arrows", "AEMO comparison", "Benchmarked against pre-dispatch"),
    ("calendar", "Four-month outlook", "Rest of this month plus three more"),
]

TEAM = [
    ("Ashish Shrestha", "Forecasting & Project Lead"),
    ("Bishal Dahal", "Model Verification & Dashboard"),
    ("Sudip Lamichhane", "Data Acquisition & Automation"),
    ("Suraj Raut", "Data Engineering & Pipeline"),
]


def _initials(name: str) -> str:
    return "".join(p[0] for p in name.split()[:2]).upper()


def _hero():
    return html.Section(
        [
            html.Div(
                [
                    html.Div("PRT661", className="hero-code"),
                    html.H1(["Energy demand", html.Br(), "forecasting portal"], className="hero-title"),
                    html.P(
                        "Live demand • Forecast verification • AEMO comparison • Four-month outlook",
                        className="hero-subtitle",
                    ),
                    html.Div(id="dashboard-portal-clock", className="hero-clock", **{"aria-live": "off"}),
                ],
                className="hero-copy",
            ),
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url("hero_transmission.svg"),
                        alt="",
                        className="hero-art-img",
                        **{"aria-hidden": "true"},
                    ),
                    html.Div(className="hero-art-overlay"),
                    html.Ul(
                        [html.Li("NEM + WEM"), html.Li("6 regions"), html.Li("30-minute grid")],
                        className="hero-tags",
                        **{"aria-label": "Portal coverage"},
                    ),
                ],
                className="hero-art",
            ),
        ],
        className="hero",
        **{"aria-label": "Portal overview"},
    )


def _capability_strip():
    return html.Section(
        html.Ul(
            [
                html.Li(
                    [
                        html.Span(icon(ic), className=f"capability-ico capability-ico--{i}"),
                        html.Div([html.Div(title, className="capability-title"), html.Div(copy, className="capability-copy")]),
                    ],
                    className="capability",
                )
                for i, (ic, title, copy) in enumerate(CAPABILITIES, start=1)
            ],
            className="capability-list",
        ),
        className="capability-strip",
        **{"aria-label": "Forecasting capability"},
    )


def _download_row(icon_name, title, copy, fmt, button_id, button_label, aria_label, scope):
    return html.Li(
        [
            html.Span(icon(icon_name), className="export-ico"),
            html.Div(
                [
                    html.Div([html.H4(title, className="export-title"), html.Span(fmt, className="file-badge")], className="export-title-row"),
                    html.P(copy, className="export-copy"),
                    html.Div(scope, className="export-scope"),
                ],
                className="export-body",
            ),
            html.Button(
                [icon("download", "btn-ico"), html.Span(button_label)],
                id=button_id,
                n_clicks=0,
                className="btn btn-secondary",
                **{"aria-label": aria_label},
            ),
        ],
        className="export-row",
    )


def _downloads_panel():
    """Evidence & export centre — every existing dcc.Download is preserved."""
    return html.Div(
        [
            section_head(
                "Evidence & export centre",
                "Download forecasting outputs, verification evidence and regional analysis files.",
                "download",
                level=3,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Region", htmlFor="export-region", className="field-label"),
                            dcc.Dropdown(
                                id="export-region",
                                options=[{"label": f"{r} — {REGION_LABELS[r]}", "value": r} for r in REGIONS],
                                value=default_region(),
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                        className="field",
                    ),
                    html.Div(
                        [
                            html.Div("Model", className="field-label", id="export-model-label"),
                            dcc.RadioItems(id="export-model", inline=True, className="segmented"),
                        ],
                        className="field field--auto",
                    ),
                    html.Div(
                        [
                            html.Label("Export month", htmlFor="download-month", className="field-label"),
                            dcc.Dropdown(id="download-month", clearable=False, searchable=False, placeholder="No scored month yet"),
                        ],
                        className="field",
                    ),
                    html.Div(
                        [
                            html.Label("Forecast export horizon", htmlFor="download-forecast-horizon", className="field-label"),
                            dcc.Dropdown(
                                id="download-forecast-horizon",
                                options=[
                                    {"label": "48H — next 48 hours", "value": "48H"},
                                    {"label": "7D — next 7 days", "value": "7D"},
                                    {"label": "30D — next 30 days", "value": "30D"},
                                    {"label": "4M — rest of this month + next 3 months", "value": "4M"},
                                ],
                                value="4M",
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                        className="field field--wide",
                    ),
                ],
                className="export-filters",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(icon("package"), className="pack-ico"),
                            html.Div(
                                [
                                    html.Div([html.H4("Complete region pack", className="pack-title"), html.Span("ZIP", className="file-badge file-badge--light")], className="export-title-row"),
                                    html.P(
                                        "Forecast vs actual, accuracy archive, next-interval tracking, AEMO comparison, "
                                        "rolling outlook and demand history for the selected region, with a README summary.",
                                        className="pack-copy",
                                    ),
                                    html.Div("Scope: export month + forecast horizon", className="pack-scope"),
                                ]
                            ),
                        ],
                        className="pack-body",
                    ),
                    html.Button(
                        [icon("download", "btn-ico"), html.Span("Download ZIP")],
                        id="btn-download-region-pack",
                        n_clicks=0,
                        className="btn btn-gold",
                        **{"aria-label": "Download complete region analysis ZIP"},
                    ),
                ],
                className="pack-card",
            ),
            html.Ul(
                [
                    _download_row("target", "Forecast vs actual", "Stored next-interval forecasts matched with actual demand.", "CSV",
                                  "btn-download-monthly-comparison", "CSV", "Download forecast vs actual CSV", "Scope: export month"),
                    _download_row("archive", "Monthly accuracy archive", "MAE, RMSE, sMAPE, bias and coverage by month.", "CSV",
                                  "btn-download-monthly-summary", "CSV", "Download monthly accuracy archive CSV", "Scope: all scored months"),
                    _download_row("arrows", "AEMO comparison", "Project vs AEMO pre-dispatch on common timestamps.", "CSV",
                                  "btn-download-aemo", "CSV", "Download AEMO comparison CSV", "Scope: latest common window"),
                    _download_row("trending", "Upcoming forecast", "Rolling outlook for the selected export horizon.", "CSV",
                                  "btn-download-upcoming", "CSV", "Download upcoming forecast CSV for the selected horizon", "Scope: forecast export horizon"),
                    _download_row("history", "Rolling tracking", "Next-interval history for the latest month.", "CSV",
                                  "btn-download-tracking", "CSV", "Download rolling tracking CSV", "Scope: latest month"),
                    _download_row("shield", "Production model card", "Selected vs served model, bundle version, selection evidence and known limitations.", "MD",
                                  "btn-download-model-card", "Markdown", "Download production model card", "Scope: export region"),
                    _download_row("layers", "Model evidence", "Ten-model historical evidence for the export region (2022–2025 evidence window).", "CSV",
                                  "btn-download-model-evidence", "CSV", "Download model evidence CSV", "Scope: export region"),
                ],
                className="export-list",
            ),
            html.P("Files download only when stored data exists for the selected scope.", className="export-note"),
        ],
        className="utility-panel-inner",
    )


def _feature_chip_list(features, cls="feature-chip"):
    return html.Ul([html.Li(f, className=cls) for f in features], className="feature-chips")


MODEL_JOURNEY = [
    ("Naive baselines", "Yesterday / last-week repetition — the reference every model must beat."),
    ("Random Forest · XGBoost", "Tree ensembles on Feature Set A (core) and B (weather-enhanced)."),
    ("SARIMA · SARIMAX", "Classical statistical time-series models."),
    ("Lecturer feedback", "Explore stronger, neural approaches."),
    ("MLP · GRU", "Neural network and deep sequence model, added and tested on the same periods."),
    ("Evidence-based selection", "Lowest four-month error over matched historical backtests; 2022–2025 evidence retained, with the full-history years used for selection."),
    ("Random Forest retained", "More complex models were tested fairly and kept only if they improved unseen historical forecasts."),
]


def _model_journey():
    return html.Ol(
        [html.Li([html.Div(t, className="journey-title"), html.Div(c, className="journey-copy")],
                 className="journey-step" + (" journey-step--feedback" if t == "Lecturer feedback" else "")
                 + (" journey-step--final" if t == "Random Forest retained" else ""))
         for t, c in MODEL_JOURNEY],
        className="journey",
        **{"aria-label": "Model development journey"},
    )


def _methodology_panel():
    steps = [
        ("database", "Data sources", "AEMO NEM and WEM demand, weather, public holidays and AEMO pre-dispatch."),
        ("shield", "Validation & cleaning", "Quality checks and harmonisation to a 30-minute grid."),
        ("layers", "Feature engineering", "Forecast-safe features on the market clock and each region's local time."),
        ("network", "Modelling", "Ten candidate models compared on matched historical periods; Random Forest serves production."),
        ("trending", "Forecast generation", "Stored next-30-minute forecast plus a rolling four-month outlook; Model B uses genuine forecast weather only where available, then forecast-safe climatology."),
        ("check", "Verification", "Each stored next-interval forecast is matched with actual demand on arrival."),
        ("dashboard", "Portal & automation", "Scheduled refresh, live views, accuracy history and evidence exports."),
    ]
    lifecycle = [
        ("clock", "Every 30 minutes", "Stores the next-interval forecast."),
        ("refresh", "Every hour", "Refreshes the rolling four-month outlook."),
        ("history", "Every month", "Retrains the production Random Forest bundles on the latest completed data."),
        ("globe", "Timezone aware", "NEM market time is kept separate from each region's DST-aware local time; WA uses AWST."),
    ]
    core_groups = ["Recent demand & lags", "Same-time averages", "Calendar & season", "Public holidays",
                   "Historical monthly profiles", "Trend"]
    return html.Div(
        [
            section_head("Methodology", "How demand data becomes a verified forecast.", "workflow", level=3),
            html.Ol(
                [
                    html.Li(
                        [
                            html.Div([html.Span(f"{i:02d}", className="flow-no"), icon(ic, "flow-ico")], className="flow-top"),
                            html.Div(title, className="flow-title"),
                            html.Div(copy, className="flow-copy"),
                        ],
                        className="flow-step",
                    )
                    for i, (ic, title, copy) in enumerate(steps, start=1)
                ],
                className="method-flow",
                **{"aria-label": "Forecasting pipeline"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Feature sets", className="method-sub"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div([html.Span("A", className="model-mark"), html.Div([html.Div("A · core", className="model-name"), html.Div(f"RandomForest_A · {len(FEATURE_SET_A)} features · no weather", className="model-code")])], className="model-head"),
                                            html.Ul([html.Li(x) for x in core_groups], className="model-tags"),
                                        ],
                                        className="model-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div([html.Span("B", className="model-mark model-mark--b"), html.Div([html.Div("B · weather-enhanced", className="model-name"), html.Div(f"RandomForest_B · {len(FEATURE_SET_B)} features", className="model-code")])], className="model-head"),
                                            html.Ul([html.Li("Everything in A"), html.Li(f"+ {len(WEATHER_FEATURES)} weather/context features: temperature, humidity, rainfall, daily max/min/mean, temperature history, climatology, rain probability, heating/cooling degree days")], className="model-tags"),
                                        ],
                                        className="model-card",
                                    ),
                                ],
                                className="model-grid",
                            ),
                            html.P(id="method-region-note", className="method-note"),
                            html.Details(
                                [html.Summary("Exact feature lists"),
                                 html.H5("Feature Set A (core)", className="method-mini"), _feature_chip_list(FEATURE_SET_A),
                                 html.H5("Added in Feature Set B (weather/context)", className="method-mini"), _feature_chip_list(WEATHER_FEATURES)],
                                className="disclosure",
                            ),
                        ],
                        className="method-col",
                    ),
                    html.Div(
                        [
                            html.H4("Automatic lifecycle", className="method-sub"),
                            html.Ul(
                                [
                                    html.Li([icon(ic, "life-ico"), html.Div([html.Div(t, className="life-title"), html.Div(c, className="life-copy")])], className="life-item")
                                    for ic, t, c in lifecycle
                                ],
                                className="life-list",
                            ),
                            html.Div(
                                [
                                    html.P([html.Strong("Next-interval scoring. "), "Live accuracy uses stored next-interval forecasts only."]),
                                    html.P([html.Strong("Replay vs live. "), "Live-verified forecasts take priority over historical replay for the same interval."]),
                                    html.P([html.Strong("Research vs production. "), "MLP, GRU, XGBoost, SARIMA/SARIMAX and the baselines are research evidence only; the live portal serves one selected Random Forest model per region."]),
                                    html.P([html.Strong("Weather horizon. "), "Feature Set B uses genuine forecast weather only for the available near-term weather horizon; later 4-month values use forecast-safe climatology and recursive demand lags."]),
                                ],
                                className="method-rules",
                            ),
                        ],
                        className="method-col",
                    ),
                ],
                className="method-split",
            ),
            html.H4("Model journey", className="method-sub"),
            _model_journey(),
            html.P(id="evidence-version-method", className="method-note"),
        ],
        className="utility-panel-inner",
    )


def _utility_button(button_id, panel_id, icon_name, title, copy, action):
    return html.Button(
        [
            html.Span(icon(icon_name), className="utility-ico"),
            html.Span(
                [html.Span(title, className="utility-title"), html.Span(copy, className="utility-copy")],
                className="utility-text",
            ),
            html.Span([html.Span(action, className="utility-action-text"), icon("chevron", "utility-chevron")], className="utility-action"),
        ],
        id=button_id,
        n_clicks=0,
        className="utility-card",
        **{"aria-controls": panel_id, "aria-expanded": "false"},
    )


def _team_strip():
    return html.Section(
        [
            html.H2("Project team", className="team-heading"),
            html.Ul(
                [
                    html.Li(
                        [html.Span(_initials(name), className="team-initials", **{"aria-hidden": "true"}),
                         html.Div([html.Div(name, className="team-name"), html.Div(role, className="team-role")])],
                        className="team-member",
                    )
                    for name, role in TEAM
                ],
                className="team-list",
            ),
        ],
        className="team-strip",
        **{"aria-label": "Project team"},
    )


def _live_utilities():
    """Former Dashboard utilities (downloads + methodology), unchanged ids."""
    return html.Section(
        [
            html.Div(
                [
                    _utility_button("btn-toggle-downloads", "panel-downloads", "download", "Downloads & evidence",
                                    "Export forecasts, verification, accuracy, model card and regional evidence.", "View downloads"),
                    _utility_button("btn-toggle-methodology", "panel-methodology", "workflow", "Methodology",
                                    "Data, feature sets, model journey and verification.", "View methodology"),
                ],
                className="utility-grid",
            ),
            html.Div(html.Div(_downloads_panel(), className="utility-panel-clip"), id="panel-downloads", className="utility-panel", role="region", **{"aria-label": "Downloads and evidence"}),
            html.Div(html.Div(_methodology_panel(), className="utility-panel-clip"), id="panel-methodology", className="utility-panel", role="region", **{"aria-label": "Methodology"}),
        ],
        className="dash-section dash-section--utilities",
        **{"aria-label": "Downloads and methodology"},
    )


def _dashboard_layout():
    """v2 Dashboard shell with the complete former Live experience embedded.

    The legacy Regional Status card grid is intentionally not mounted. The
    standalone Live navigation tab has also been removed. Dashboard now owns
    the Regional Analysis map/current-state card plus all live forecast content.
    """
    return [
        _hero(),
        _capability_strip(),
        # The Regional Analysis map/current-state card now belongs on Dashboard.
        # It is the primary selected-region control and shows the latest stored
        # operational state before the detailed live forecast story.
        html.Section(
            [
                section_head(
                    "Live regional status",
                    "Select a region on the Australian map or from the list to update the live view below.",
                    "map",
                ),
                _regional_state_panel(),
            ],
            className="dash-section dashboard-region-state",
            **{"aria-label": "Live regional status"},
        ),
        # Clean live outlook: chart + always-visible upcoming intervals.
        # Region/time/current-state context comes from the hero and map above.
        _live_layout(include_team=False),
        html.Section(
            [
                html.Div(
                    [
                        _utility_button("btn-toggle-downloads", "panel-downloads", "download", "Downloads & evidence",
                                        "Export forecasts, verification, accuracy and regional evidence.", "View downloads"),
                        _utility_button("btn-toggle-methodology", "panel-methodology", "workflow", "Methodology",
                                        "Data preparation, modelling and verification.", "View methodology"),
                    ],
                    className="utility-grid",
                ),
                html.Div(html.Div(_downloads_panel(), className="utility-panel-clip"), id="panel-downloads", className="utility-panel", role="region", **{"aria-label": "Downloads and evidence"}),
                html.Div(html.Div(_methodology_panel(), className="utility-panel-clip"), id="panel-methodology", className="utility-panel", role="region", **{"aria-label": "Methodology"}),
            ],
            className="dash-section dash-section--utilities",
            **{"aria-label": "Downloads and methodology"},
        ),
        _team_strip(),
    ]

def _context_bar():
    """Shared region/model context for Live, Performance, AEMO and Regional views.

    The controls keep their original ids (region-dropdown, model-switch) so every
    existing callback continues to read the same component state.
    """
    return html.Section(
        [
            html.Div(
                [html.H2("", id="view-title", className="view-title"), html.P("", id="view-subtitle", className="view-subtitle")],
                className="context-heading",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Region", htmlFor="region-dropdown", className="field-label"),
                            dcc.Dropdown(
                                id="region-dropdown",
                                options=[{"label": f"{r} — {REGION_LABELS[r]}", "value": r} for r in REGIONS],
                                value=default_region(),
                                clearable=False,
                                searchable=False,
                                className="context-dropdown",
                            ),
                        ],
                        className="field field--region",
                        id="region-field",
                    ),
                    html.Div(
                        [
                            html.Div("Production model", className="field-label", id="model-switch-label"),
                            dcc.RadioItems(id="model-switch", inline=True, className="segmented model-switch"),
                            html.Div(id="model-switch-static", className="model-static"),
                        ],
                        className="field field--auto",
                        id="model-switch-field",
                    ),
                ],
                className="context-controls",
            ),
            html.Div(id="market-label", className="context-meta"),
        ],
        id="context-bar",
        className="context-bar is-hidden",
        **{"aria-label": "Region and model"},
    )


def _live_region_picker():
    return html.Div(
        [
            html.Div(
                [html.H3("Region", className="chart-card-title"),
                 html.P("Select on the map or from the list.", className="chart-card-subtitle")],
                className="chart-card-head",
            ),
            dcc.Graph(
                id="live-region-map",
                figure=australia_map(default_region(), 300),
                config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "responsive": True},
                className="region-map-graph",
                style={"height": "300px"},
            ),
            html.Div(_map_legend(default_region(), "live"), id="live-map-legend", className="map-legend map-legend--chips"),
        ],
        className="chart-card live-map-card",
    )


def _live_layout(include_team: bool = True):
    """Clean live forecast workspace embedded in Dashboard.

    Region selection/current state is provided by the map card above.  The
    duplicate dark live header, repeated KPI block, model-selection narrative,
    month-to-date block and dashboard region tiles are intentionally omitted.
    The rolling outlook is kept visible beside the chart (v2 pattern) and can
    show up to 50 upcoming intervals without opening a disclosure.
    """
    return html.Div(
        [
            # Keep freshness visible because it is operationally important, but
            # avoid another large header or duplicated current-demand cards.
            html.Div(id="live-freshness", className="freshness"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [html.H3("Demand and forecast", className="chart-card-title"),
                                         html.P("48H / 7D at 30 minutes · 30D / 4M as daily means · 4M is a long-range demand outlook", className="chart-card-subtitle")]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Horizon", className="field-label sr-only", id="horizon-label"),
                                            dcc.RadioItems(
                                                id="horizon",
                                                options=[{"label": h, "value": h} for h in ("48H", "7D", "30D", "4M")],
                                                value="48H",
                                                inline=True,
                                                className="segmented horizon-switch",
                                            ),
                                            dcc.Checklist(
                                                id="error-band-toggle",
                                                options=[{"label": " Typical error range", "value": "band"}],
                                                value=["band"],
                                                className="band-toggle",
                                            ),
                                        ],
                                        className="chart-tools",
                                        role="group",
                                        **{"aria-labelledby": "horizon-label"},
                                    ),
                                ],
                                className="chart-card-head chart-card-head--tools",
                            ),
                            _graph("live-chart", 460, 300),
                            html.P(id="live-chart-note", className="chart-note"),
                            html.P(id="live-chart-sr", className="sr-only"),
                        ],
                        className="chart-card chart-card--live",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [html.H3("Forecast outlook", className="chart-card-title"),
                                         html.P("Upcoming 30-minute intervals · market time", className="chart-card-subtitle")]
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Rows", htmlFor="upcoming-count", className="field-label field-label--inline"),
                                            dcc.Dropdown(
                                                id="upcoming-count",
                                                options=[{"label": str(n), "value": n} for n in (10, 20, 30, 40, 50)],
                                                value=20,
                                                clearable=False,
                                                searchable=False,
                                                className="compact-dropdown",
                                            ),
                                        ],
                                        className="chart-tools",
                                    ),
                                ],
                                className="chart-card-head chart-card-head--tools",
                            ),
                            html.Div(id="forecast-table", className="forecast-table-wrap", tabIndex="0", **{"aria-label": "Upcoming forecast intervals"}),
                        ],
                        className="chart-card table-card",
                    ),
                ],
                className="live-main-grid dashboard-outlook-grid",
            ),
            html.Details(
                [
                    html.Summary([html.Span("System status", className="disclosure-title"),
                                  html.Span("Data feeds and scheduled jobs", className="disclosure-sub")]),
                    html.Div(id="operational-status", className="ops-status"),
                ],
                className="disclosure disclosure--card dashboard-system-status",
                open=True,
            ),
            # Compatibility targets for callbacks that still calculate the
            # underlying live metrics/evidence.  They stay out of the visual UI.
            html.Div(
                [
                    html.Div(id="portal-clock"),
                    html.Div(id="live-region-name"),
                    html.Div(id="live-region-code"),
                    html.Div(id="region-source"),
                    html.Button("Use my timezone", id="btn-use-timezone", n_clicks=0),
                    html.Div(id="m-actual"), html.Div(id="m-next"), html.Div(id="m-mae"),
                    html.Div(id="m-smape"), html.Div(id="m-peak"), html.Div(id="m-updated"),
                    html.Div(id="tracking-coverage"), html.Div(id="m-actual-sub"), html.Div(id="m-change"),
                    html.Div(id="live-insight"), html.Div(id="live-model-line"),
                    html.Button("View performance", id="btn-view-performance", n_clicks=0),
                    html.Button("Why this model?", id="btn-why-model", n_clicks=0),
                ],
                className="callback-compat",
                style={"display": "none"},
                **{"aria-hidden": "true"},
            ),
            *([_team_strip()] if include_team else []),
        ],
        className="view view-live view-live--embedded",
    )


def _regional_state_panel():
    """Australian region selector + selected-region current-state summary.

    This panel was previously the top of Regional Analysis. It now lives on
    Dashboard because it is operational context: region selection, current
    demand, next 30-minute forecast and local/market time.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.H3("Australia", className="chart-card-title"),
                         html.P("Select a region on the map or in the list.", className="chart-card-subtitle")],
                        className="chart-card-head",
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Graph(
                                    id="region-map",
                                    figure=australia_map(default_region()),
                                    config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "responsive": True},
                                    className="region-map-graph",
                                    style={"height": "360px"},
                                ),
                                className="map-canvas",
                            ),
                            html.Div(_map_legend(default_region()), id="map-legend", className="map-legend"),
                        ],
                        className="map-body",
                    ),
                ],
                className="chart-card map-card",
            ),
            html.Div(id="region-profile", className="chart-card region-profile"),
        ],
        className="regional-top",
    )


def _history_layout():
    """Historical/seasonal analysis only; operational region state is on Dashboard."""
    years = list(range(2022, datetime.now().year + 1))
    return html.Div(
        [
            html.Section(
                [
                    section_head(
                        "National energy pulse",
                        "Six-region demand and the next 30-minute forecast, shown here as context for regional analysis.",
                        "zap",
                    ),
                    html.Div(id="national-overview-metrics", className="pulse-kpis"),
                    html.Div(id="region-share", className="region-share"),
                ],
                className="regional-pulse",
                **{"aria-label": "National energy pulse"},
            ),
            html.Section(
                [
                    section_head("Other regions", "Current demand and direction. Select a region to analyse it.", "map", level=3),
                    html.Div(id="other-regions-strip", className="region-strip"),
                ],
                className="dash-section regional-other-regions",
                **{"aria-label": "Other regions"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Years shown", htmlFor="history-years", className="field-label"),
                            dcc.Dropdown(
                                id="history-years",
                                options=[{"label": y, "value": y} for y in years],
                                value=years,
                                multi=True,
                                className="years-dropdown",
                            ),
                        ],
                        className="field field--grow",
                    ),
                    html.Div(id="history-status", className="toolbar-note"),
                ],
                className="toolbar",
            ),
            html.Div(
                [
                    chart_card(
                        "Seasonal pattern by year",
                        "Monthly mean demand, January to December.",
                        _graph("monthly-history", 420, 300),
                    ),
                    chart_card(
                        "Typical day by year",
                        "Average 30-minute demand profile across all days.",
                        _graph("daily-profile", 420, 300),
                    ),
                ],
                className="two-chart-grid",
            ),
            chart_card(
                "Forecast difficulty by region",
                "Selected production model, four-month error as % of typical demand (2022–2025 evidence window). Descriptive only.",
                _graph("region-difficulty", 300, 240),
            ),
            html.P(id="region-difficulty-note", className="info-line info-line--quiet"),
        ],
        className="view view-regional",
    )

def _performance_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Mode", className="field-label sr-only", id="performance-mode-label"),
                    dcc.RadioItems(
                        id="performance-mode",
                        options=[{"label": "Live verification", "value": "verification"},
                                 {"label": "Model evidence", "value": "comparison"}],
                        value="verification",
                        inline=True,
                        className="segmented mode-switch",
                    ),
                ],
                className="mode-bar",
                role="group",
                **{"aria-labelledby": "performance-mode-label"},
            ),
            html.Div(id="perf-comparison", className="perf-mode is-hidden", children=[
                dcc.Loading(html.Div(id="mc-content", className="mc-grid"), type="dot", color=FORECAST_BLUE),
            ]),
            html.Div(id="perf-verification", className="perf-mode", children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("View", className="field-label", id="performance-view-label"),
                            dcc.RadioItems(
                                id="performance-view",
                                options=[
                                    {"label": "Month", "value": "month"},
                                    {"label": "Year", "value": "year"},
                                ],
                                value="month",
                                inline=True,
                                className="segmented performance-view-switch",
                            ),
                        ],
                        className="field field--auto",
                    ),
                    html.Div(
                        [
                            html.Label("Year", htmlFor="performance-year", className="field-label"),
                            dcc.Dropdown(id="performance-year", clearable=False, searchable=False, className="compact-dropdown", placeholder="—"),
                        ],
                        className="field field--narrow",
                    ),
                    html.Div(
                        [
                            html.Label("Month", htmlFor="performance-month", className="field-label"),
                            dcc.Dropdown(id="performance-month", clearable=False, searchable=False, className="month-dropdown", placeholder="—"),
                        ],
                        id="performance-month-control",
                        className="field field--month",
                    ),
                    html.Div(
                        [
                            html.Span([html.Span(className="prov-dot prov-dot--live", **{"aria-hidden": "true"}), "Live verified"], className="prov-item",
                                      title="Forecast stored before the target interval, then matched with actual demand."),
                            html.Span([html.Span(className="prov-dot prov-dot--replay", **{"aria-hidden": "true"}), "Historical replay"], className="prov-item",
                                      title="Leakage-safe replay used for intervals before live verification began."),
                        ],
                        className="provenance-key",
                        **{"aria-label": "Evidence types"},
                    ),
                ],
                className="toolbar",
            ),
            html.Div(id="performance-window", className="info-line"),
            html.Div(
                [
                    metric_card("MAE", "p-mae", "MW · average size of the miss"),
                    metric_card("sMAPE", "p-smape", "Average % miss"),
                    metric_card("Coverage", "p-count", "Matched 30-min intervals"),
                ],
                className="kpi-row kpi-row--three",
            ),
            html.P(id="p-mae-plain", className="info-line info-line--quiet"),
            html.Details(
                [
                    html.Summary([html.Span("More metrics", className="disclosure-title"),
                                  html.Span("RMSE and bias", className="disclosure-sub")]),
                    html.Div(
                        [
                            metric_card("RMSE", "p-rmse", "MW · penalises large misses"),
                            metric_card("Bias", "p-bias", "MW · forecast − actual (negative = too low)"),
                        ],
                        className="kpi-row kpi-row--two",
                    ),
                ],
                className="disclosure",
            ),
            html.Div(id="performance-coverage", className="info-line info-line--quiet"),
            chart_card(
                "Forecast vs actual",
                "Hourly display; metrics use exact 30-minute matches.",
                _graph("performance-chart", 440, 300),
                className="chart-card--primary",
                title_id="performance-main-title",
                subtitle_id="performance-main-subtitle",
            ),
            chart_card(
                "Daily MAE",
                "Daily mean absolute error.",
                _graph("error-chart", 300, 240),
                title_id="performance-secondary-title",
                subtitle_id="performance-secondary-subtitle",
            ),
            ]),
        ],
        className="view view-performance",
    )


def _aemo_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span([html.Span(className="legend-line legend-line--project", **{"aria-hidden": "true"}), "Project forecast"], className="versus-side"),
                            html.Span("vs", className="versus-vs"),
                            html.Span([html.Span(className="legend-line legend-line--aemo", **{"aria-hidden": "true"}), "AEMO pre-dispatch (POE50)"], className="versus-side"),
                        ],
                        className="versus-line",
                    ),
                    html.Ul(
                        [
                            html.Li([icon("check", "rule-ico"), "Exact common timestamps only"]),
                            html.Li([icon("check", "rule-ico"), "AEMO horizon is never extended"]),
                            html.Li([icon("check", "rule-ico"), "Forecast-to-forecast gap, not accuracy against actual"]),
                        ],
                        className="benchmark-rules",
                    ),
                ],
                className="benchmark-callout",
            ),
            html.Div(
                [
                    metric_card("Common timestamps", "a-common", "Same target intervals"),
                    metric_card("Mean absolute gap", "a-mae-gap", "MW · how far apart the two forecasts are"),
                ],
                className="kpi-row kpi-row--two",
            ),
            html.Details(
                [
                    html.Summary([html.Span("More detail", className="disclosure-title"),
                                  html.Span("Signed gap and overlapping period", className="disclosure-sub")]),
                    html.Div(
                        [
                            metric_card("Mean signed gap", "a-diff", "MW · project − AEMO"),
                            metric_card("Coverage period", "a-window", "Overlapping span"),
                        ],
                        className="kpi-row kpi-row--two",
                    ),
                ],
                className="disclosure",
            ),
            html.Div(id="aemo-coverage", className="info-line info-line--quiet"),
            chart_card(
                "Project vs AEMO forecast",
                "Stored overlapping forecast window only.",
                _graph("aemo-chart", 460, 300),
            ),
        ],
        className="view view-aemo",
    )


def _pane(value, children):
    return html.Div(
        children,
        id=f"pane-{value}",
        className="pane is-active" if value == DEFAULT_TAB else "pane",
        role="tabpanel",
        **{"aria-label": dict((v, l) for v, l, _ in NAV_ITEMS)[value]},
    )


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="selected-region", data=default_region()),
        dcc.Store(id="region-preference", storage_type="local"),
        dcc.Store(id="browser-tz"),
        html.A("Skip to content", href="#tab-content", className="skip-link"),
        # Primary navigation -------------------------------------------------
        html.Header(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(icon("zap"), className="brand-mark"),
                            html.Span([html.Span("PRT661", className="brand-code"), html.Span("Energy Demand Forecasting", className="brand-name")], className="brand-text"),
                        ],
                        className="brand",
                    ),
                    dcc.Tabs(
                        id="tabs",
                        value=DEFAULT_TAB,
                        className="nav-tabs",
                        parent_className="nav-tabs-parent",
                        children=[
                            dcc.Tab(label=label, value=value, className=f"nav-tab ico-{ic}", selected_className="nav-tab--active")
                            for value, label, ic in NAV_ITEMS
                        ],
                    ),
                ],
                className="topbar-inner",
            ),
            className="topbar",
        ),
        # Views ---------------------------------------------------------------
        html.Main(
            [
                _context_bar(),
                _pane("dashboard", _dashboard_layout()),
                _pane("performance", _performance_layout()),
                _pane("aemo", _aemo_layout()),
                _pane("history", _history_layout()),
            ],
            id="tab-content",
            className="content",
            tabIndex="-1",
        ),
        html.Footer(
            [
                html.Span("Charles Darwin University · PRT661 Energy Demand Forecasting"),
                html.Span("AEMO NEM & WEM demand · 30-minute forecasting portal"),
                html.Span(id="evidence-footer", className="footer-evidence"),
            ],
            className="portal-footer",
        ),
        dcc.Download(id="download-tracking"),
        dcc.Download(id="download-upcoming"),
        dcc.Download(id="download-monthly-comparison"),
        dcc.Download(id="download-monthly-summary"),
        dcc.Download(id="download-aemo-comparison"),
        dcc.Download(id="download-region-pack"),
        dcc.Download(id="download-model-card"),
        dcc.Download(id="download-model-evidence"),
        dcc.Interval(id="refresh-ui", interval=60 * 1000, n_intervals=0),
    ],
    className="portal-shell",
)


# ---------------------------------------------------------------------------
# Navigation behaviour (client-side: instant, no server round trip)
# ---------------------------------------------------------------------------
clientside_callback(
    """
    function(tab) {
        var keys = %s;
        var copy = %s;
        tab = keys.indexOf(tab) >= 0 ? tab : "dashboard";
        var classes = keys.map(function (k) { return k === tab ? "pane is-active" : "pane"; });
        // Dashboard owns the live region controls directly; analytical views use the shared context bar.
        var ctxClass = tab === "dashboard" ? "context-bar is-hidden" : "context-bar";
        var c = copy[tab] || ["", ""];
        if (window.__prtLastTab && window.__prtLastTab !== tab) {
            var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
            window.scrollTo({top: 0, behavior: "auto"});
        }
        window.__prtLastTab = tab;
        return classes.concat([ctxClass, c[0], c[1]]);
    }
    """ % (__import__("json").dumps(NAV_KEYS), __import__("json").dumps({k: list(v) for k, v in VIEW_COPY.items()})),
    [Output(f"pane-{v}", "className") for v, _, _ in NAV_ITEMS]
    + [Output("context-bar", "className"), Output("view-title", "children"), Output("view-subtitle", "children")],
    Input("tabs", "value"),
)

clientside_callback(
    """
    function(nDownloads, nMethod) {
        function state(n, id, btn) {
            var open = (n || 0) % 2 === 1;
            var b = document.getElementById(btn);
            if (b) {
                b.setAttribute("aria-expanded", open ? "true" : "false");
                var t = b.querySelector(".utility-action-text");
                if (t) { t.textContent = open ? "Hide" : (btn === "btn-toggle-downloads" ? "View downloads" : "View methodology"); }
            }
            return open ? "utility-panel is-open" : "utility-panel";
        }
        var trig = (window.dash_clientside.callback_context.triggered || [])[0];
        var d = state(nDownloads, "panel-downloads", "btn-toggle-downloads");
        var m = state(nMethod, "panel-methodology", "btn-toggle-methodology");
        if (trig && trig.value) {
            var target = trig.prop_id.indexOf("downloads") >= 0 ? "panel-downloads" : "panel-methodology";
            var isOpen = (trig.prop_id.indexOf("downloads") >= 0 ? d : m).indexOf("is-open") >= 0;
            if (isOpen) {
                var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
                setTimeout(function () {
                    var el = document.getElementById(target);
                    if (el) { el.scrollIntoView({behavior: reduce ? "auto" : "smooth", block: "start"}); }
                }, 60);
            }
        }
        return [d, m];
    }
    """,
    Output("panel-downloads", "className"),
    Output("panel-methodology", "className"),
    Input("btn-toggle-downloads", "n_clicks"),
    Input("btn-toggle-methodology", "n_clicks"),
)


# ---------------------------------------------------------------------------
# v3: browser timezone (UI only) and deep links
# ---------------------------------------------------------------------------
clientside_callback(
    """
    function(_path) {
        try { return Intl.DateTimeFormat().resolvedOptions().timeZone || ""; }
        catch (e) { return ""; }
    }
    """,
    Output("browser-tz", "data"),
    Input("url", "pathname"),
)

clientside_callback(
    """
    function(mode) {
        var cmp = mode === "comparison";
        return [cmp ? "perf-mode" : "perf-mode is-hidden", cmp ? "perf-mode is-hidden" : "perf-mode"];
    }
    """,
    Output("perf-comparison", "className"),
    Output("perf-verification", "className"),
    Input("performance-mode", "value"),
)


def _query_params(search: str | None) -> dict:
    from urllib.parse import parse_qs
    q = parse_qs((search or "").lstrip("?"))
    return {k: v[0] for k, v in q.items() if v}


@callback(
    Output("tabs", "value"),
    Output("performance-mode", "value"),
    Input("url", "search"),
    Input("btn-view-performance", "n_clicks"),
    Input("btn-why-model", "n_clicks"),
    State("performance-mode", "value"),
)
def navigate(search, _perf, _why, mode):
    """Deep links plus Dashboard live links into Forecast Performance."""
    trig = ctx.triggered_id
    if trig == "btn-view-performance":
        return "performance", "verification"
    if trig == "btn-why-model":
        return "performance", "comparison"
    q = _query_params(search)
    view = q.get("view")
    if view == "live":
        view = "dashboard"
    view = "history" if view in ("regional", "analysis") else view
    new_mode = q.get("mode") if q.get("mode") in ("verification", "comparison") else (mode or "verification")
    if view in NAV_KEYS:
        return view, new_mode
    if q.get("mode") in ("verification", "comparison"):
        return "performance", new_mode
    raise PreventUpdate


@callback(
    Output("selected-region", "data"),
    Output("region-dropdown", "value"),
    Output("export-region", "value"),
    Output("region-preference", "data"),
    Output("region-source", "children"),
    Input("region-map", "clickData"),
    Input("region-dropdown", "value"),
    Input("export-region", "value"),
    Input({"type": "region-pick", "region": ALL, "where": ALL}, "n_clicks"),
    Input({"type": "region-tile", "region": ALL}, "n_clicks"),
    Input("browser-tz", "data"),
    Input("btn-use-timezone", "n_clicks"),
    Input("url", "search"),
    State("region-preference", "data"),
    State("selected-region", "data"),
    prevent_initial_call=True,
)
def choose_region(click, dropdown, export_region, _pick_clicks, _tile_clicks,
                  browser_tz, _tz_reset, search, pref, current):
    """Single source of truth for the selected region.

    Manual actions (maps, dropdowns, region tiles, export region) are stored as a
    manual preference that timezone detection never overwrites. Automatic events
    (page load / timezone / URL) follow: URL > manual > timezone > stored > default.
    """
    trig = ctx.triggered_id
    clicked = None
    if trig == "region-map" and click:
        clicked = click["points"][0]["customdata"][0]
    elif isinstance(trig, dict) and trig.get("type") in ("region-pick", "region-tile"):
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            raise PreventUpdate
        clicked = trig.get("region")
    elif trig == "export-region":
        clicked = export_region
    elif trig == "region-dropdown":
        clicked = dropdown

    if clicked is not None:
        if clicked == current and (pref or {}).get("selection_source") == "manual":
            raise PreventUpdate
        region, new_pref, msg = resolve_region("manual", pref=pref, browser_tz=browser_tz, clicked_region=clicked)
    elif trig == "btn-use-timezone":
        region, new_pref, msg = resolve_region("reset", pref=pref, browser_tz=browser_tz)
    else:
        url_region = _query_params(search).get("region")
        url_region = url_region.upper() if url_region else None
        region, new_pref, msg = resolve_region("auto", url_region=url_region, pref=pref, browser_tz=browser_tz)
    return region, region, region, new_pref, msg


@callback(
    Output("region-map", "figure"),
    Output("map-legend", "children"),
    Input("selected-region", "data"),
)
def update_region_map(region):
    region = region or "NSW1"
    return australia_map(region), _map_legend(region, "dashboard")


@callback(
    Output("model-switch", "options"),
    Output("model-switch", "value"),
    Output("model-switch", "style"),
    Output("model-switch-static", "children"),
    Output("model-switch-label", "children"),
    Output("export-model", "options"),
    Output("export-model", "value"),
    Output("market-label", "children"),
    Input("selected-region", "data"),
    Input("model-switch", "value"),
    Input("export-model", "value"),
)
def model_options(region, switch_value, export_value):
    """Production configuration per region.

    Every region now exposes one production model in the live portal. Historical
    A/B results remain available in Forecast Performance -> Model Comparison, but
    they are not selectable as alternate live-serving models.
    """
    region = region or default_region()
    models = _prod_models(region)
    served = _default_model(region)

    def label(m):
        tag = "production" if m == served else "comparison"
        return f"{_fs_label(m) or m} ({tag})"

    opts = [{"label": label(m), "value": m} for m in models]
    trig = ctx.triggered_id
    if trig == "model-switch" and switch_value in models:
        value = switch_value
    elif trig == "export-model" and export_value in models:
        value = export_value
    else:
        value = served
    single = len(models) <= 1
    static = [] if not single else [html.Span("Production model", className="model-static-label"),
                                    html.Strong(f"{served} · {_fs_label(served).split(' · ')[-1] if _fs_label(served) else ''}".rstrip(" ·"))]
    zone = REGION_TZ[region].split("/")[-1].replace("_", " ")
    summary = [
        html.Span(className="context-dot", style={"--region-color": REGION_COLORS.get(region, ACCENT)}, **{"aria-hidden": "true"}),
        html.Span(f"{region} · {REGION_LABELS[region]}", className="context-meta-strong"),
        html.Span(f"{REGION_MARKET[region]} market"),
        html.Span(f"Serving {served}"),
        html.Span(f"Local time: {zone}"),
    ]
    return (opts, value, {"display": "none"} if single else {}, static,
            "Production model" if single else "Compare A/B configuration", opts, value, summary)


@callback(
    Output("operational-status", "children"),
    Input("selected-region", "data"),
    Input("model-switch", "value"),
    Input("refresh-ui", "n_intervals"),
)
def operational_status(region, model, _):
    region = region or "NSW1"
    model = model or _default_model(region)

    model_row = read_df(
        """
        SELECT model_version, training_end, created_at
        FROM model_registry
        WHERE region=:r AND model_name=:m AND status='PRODUCTION'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        {"r": region, "m": model},
    )
    demand_row = read_df(
        "SELECT MAX(retrieved_at) AS ts FROM actual_demand WHERE region=:r", {"r": region}
    )
    weather_row = read_df(
        "SELECT MAX(retrieved_at) AS ts FROM weather_forecast WHERE region=:r", {"r": region}
    )
    aemo_row = read_df(
        "SELECT MAX(retrieved_at) AS ts FROM aemo_forecast WHERE region=:r", {"r": region}
    )
    fc_row = read_df(
        "SELECT MAX(created_at) AS ts FROM next_interval_forecast WHERE region=:r AND model_name=:m",
        {"r": region, "m": model},
    )
    run_row = read_df(
        """
        SELECT status, finished_at
        FROM pipeline_runs
        WHERE job_type='OPERATIONAL_30MIN'
        ORDER BY finished_at DESC
        LIMIT 1
        """
    )

    def feed(label, icon_name, frame, good_minutes, warn_minutes):
        value = frame.iloc[0, 0] if frame is not None and not frame.empty else None
        age = _fresh_age_minutes(value)
        if age is None:
            chip = status_chip("nodata", "No data")
        elif age <= good_minutes:
            chip = status_chip("live", "Current")
        elif age <= warn_minutes:
            chip = status_chip("rolling", f"{age:.0f} min old")
        else:
            chip = status_chip("stale", f"Stale · {age:.0f} min")
        return html.Li([html.Span([icon(icon_name, "feed-ico"), label], className="feed-label"), chip], className="feed")

    model_version = "—"
    training_end = "—"
    if not model_row.empty:
        model_version = str(model_row.iloc[0].get("model_version", "—"))
        te = pd.to_datetime(model_row.iloc[0].get("training_end"), errors="coerce")
        if pd.notna(te):
            training_end = te.strftime("%d %b %Y")

    if run_row.empty:
        pipeline_chip = status_chip("nodata", "No run")
    else:
        run_status = str(run_row.iloc[0].get("status", "—"))
        if run_status.upper() == "COMPLETE":
            pipeline_chip = status_chip("live", "Complete")
        elif run_status.upper() in {"FAILED", "ERROR"}:
            pipeline_chip = status_chip("error", run_status.title())
        else:
            pipeline_chip = status_chip("waiting", run_status.title())

    if region == "WA":
        # WA is in the WEM: AEMO NEM pre-dispatch is not part of its evidence.
        aemo_feed = html.Li([html.Span([icon("arrows", "feed-ico"), "AEMO pre-dispatch"], className="feed-label"),
                             status_chip("nodata", "Not used · WEM")], className="feed")
    else:
        aemo_feed = feed("AEMO pre-dispatch", "arrows", aemo_row, 90, 180)

    local = region_local_now(region)
    next_month = (local.normalize().replace(day=1) + pd.offsets.MonthBegin(1)).date()
    content = [
        html.Div(
            [
                html.Div(
                    [
                        html.Div([icon("gauge", "section-ico"), html.H2("System status", className="section-title")], className="section-title-row"),
                        html.P(
                            [
                                "Data feeds for ",
                                html.Strong(f"{region} · {REGION_LABELS[region]}", style={"--region-color": REGION_COLORS.get(region, ACCENT)}, className="ops-region"),
                                f" · {model.replace('RandomForest_', 'Model ')}",
                            ],
                            className="section-sub",
                        ),
                    ]
                ),
                html.Div(
                    [html.Span(f"{region_clock_label(region)} local"), html.Span(f"{market_clock_label(region)} market")],
                    className="ops-clock",
                ),
            ],
            className="section-head",
        ),
        html.Ul(
            [
                feed("Actual demand", "activity", demand_row, 35, 90),
                feed("Weather", "globe", weather_row, 90, 180),
                feed("Next forecast", "trending", fc_row, 45, 90),
                aemo_feed,
                html.Li([html.Span([icon("refresh", "feed-ico"), "30-min pipeline"], className="feed-label"), pipeline_chip], className="feed"),
            ],
            className="feed-list",
        ),
        html.Div(
            [
                html.Span([html.Span("Model version", className="life-k"), html.Span(model_version, className="life-v")]),
                html.Span([html.Span("Trained through", className="life-k"), html.Span(training_end, className="life-v")]),
                html.Span([html.Span("Next retrain", className="life-k"), html.Span(f"{pd.Timestamp(next_month):%d %b %Y}", className="life-v")]),
            ],
            className="model-lifecycle",
        ),
    ]

    return content


@callback(
    Output("portal-clock", "children"),
    Output("live-region-name", "children"),
    Output("live-region-code", "children"),
    Output("method-region-note", "children"),
    Input("refresh-ui", "n_intervals"),
    Input("selected-region", "data"),
)
def portal_clock(_, region):
    """Selected region's civil time, plus the market clock where it differs (v3)."""
    region = region or default_region()
    local = region_local_now(region)
    market = region_market_now_naive(region)
    nxt = _next_region_half_hour(region)
    nxt_local = _local_display_time(region, nxt)
    idx = int(market.hour * 2 + (1 if market.minute >= 30 else 0))
    rail = [
        html.Span(className="rail-cell " + ("is-now" if i == idx else ("is-next" if i == idx + 1 else ("is-past" if i < idx else ""))))
        for i in range(48)
    ]
    market_zone = "AEST (NEM market)" if REGION_MARKET[region] == "NEM" else "AWST (WEM market)"
    differs = local.utcoffset() != region_market_now_aware(region).utcoffset()
    blocks = [
        html.Div([html.Span(f"Local time · {REGION_LABELS[region]}", className="clock-label"),
                  html.Span(f"{local:%H:%M}", className="clock-time"),
                  html.Span(f"{local:%a %d %b} · {local.tzname()}", className="clock-date")], className="clock-block"),
    ]
    if differs:
        blocks.append(html.Div([html.Span("Market time", className="clock-label"),
                                html.Span(f"{market:%H:%M}", className="clock-time clock-time--market"),
                                html.Span(market_zone, className="clock-date")], className="clock-block"))
    blocks.append(html.Div([html.Span("Next interval", className="clock-label"),
                            html.Span(f"{nxt_local:%H:%M}", className="clock-time clock-time--next"),
                            html.Span(f"Starts in {max(0, int((nxt - market).total_seconds() // 60))} min", className="clock-date")],
                           className="clock-block clock-block--next"))
    clock = [
        html.Div(blocks + [html.Span(f"{REGION_LABELS[region]} local time {local:%H:%M}, next interval {nxt_local:%H:%M}", className="sr-only")],
                 className="clock-row"),
        html.Div(
            [html.Div(rail, className="interval-rail", **{"aria-hidden": "true"}),
             html.Div([html.Span("00:00"), html.Span(f"Trading interval {idx + 1} of 48 (market day)"), html.Span("24:00")],
                      className="interval-rail-scale")],
            className="interval-rail-wrap",
        ),
    ]
    sel = ins.selected_model_insight(region, evidence("regional_model_winners.csv"))
    note = ins.feature_set_insight(region, evidence("model_comparison_by_region.csv"), sel.get("feature_set")) if sel else ""
    return clock, REGION_LABELS[region], region, note


@callback(
    Output("dashboard-portal-clock", "children"),
    Input("refresh-ui", "n_intervals"),
    Input("selected-region", "data"),
)
def dashboard_portal_clock(_, region):
    """Selected region time in the original v2 Dashboard hero."""
    region = region or default_region()
    local = region_local_now(region)
    market = region_market_now_naive(region)
    nxt_market = _next_region_half_hour(region)
    nxt_local = _local_display_time(region, nxt_market)
    idx = int(market.hour * 2 + (1 if market.minute >= 30 else 0))
    rail = [
        html.Span(
            className="rail-cell " + ("is-now" if i == idx else ("is-next" if i == idx + 1 else ("is-past" if i < idx else ""))),
        )
        for i in range(48)
    ]
    local_zone = local.tzname() or REGION_TZ[region].split("/")[-1].replace("_", " ")
    market_zone = "AEST" if REGION_MARKET[region] == "NEM" else "AWST"
    local_market_note = f"{REGION_MARKET[region]} market · {market_zone}"
    return [
        html.Div(
            [
                html.Div(
                    [
                        html.Span(f"Local time · {REGION_LABELS[region]}", className="clock-label"),
                        html.Span(f"{local:%H:%M}", className="clock-time"),
                        html.Span(f"{local:%a %d %b %Y} · {local_zone}", className="clock-date"),
                    ],
                    className="clock-block",
                ),
                html.Div(
                    [
                        html.Span("Next interval", className="clock-label"),
                        html.Span(f"{nxt_local:%H:%M}", className="clock-time clock-time--next"),
                        html.Span(
                            f"Starts in {max(0, int((nxt_market - market).total_seconds() // 60))} min · {local_market_note}",
                            className="clock-date",
                        ),
                    ],
                    className="clock-block clock-block--next",
                ),
                html.Span(
                    f"{REGION_LABELS[region]} local time {local:%d %b %Y %H:%M}; next interval {nxt_local:%H:%M}",
                    className="sr-only",
                ),
            ],
            className="clock-row",
        ),
        html.Div(
            [
                html.Div(rail, className="interval-rail", **{"aria-hidden": "true"}),
                html.Div(
                    [html.Span("00:00"), html.Span(f"Trading interval {idx + 1} of 48 · {REGION_MARKET[region]} market day"), html.Span("24:00")],
                    className="interval-rail-scale",
                ),
            ],
            className="interval-rail-wrap",
        ),
    ]


def _mw_value(value, css="num"):
    """Big number with a quieter unit, or an honest dash."""
    if value is None or not np.isfinite(value):
        return html.Span("—", className=f"{css} is-empty")
    return html.Span([html.Span(f"{value:,.0f}"), html.Span(" MW", className="unit")], className=css)


@callback(
    Output("national-overview-metrics", "children"),
    Output("region-share", "children"),
    Input("refresh-ui", "n_intervals"),
)
def region_overview(_):
    """Render the national pulse and regional-share bar; no Regional Status grid."""
    target_map = {r: _next_region_half_hour(r) for r in REGIONS}
    min_target = min(target_map.values())
    max_target = max(target_map.values())

    actuals = read_df(
        """
        SELECT region, timestamp, demand_mw
        FROM (
            SELECT region, timestamp, demand_mw,
                   ROW_NUMBER() OVER (PARTITION BY region ORDER BY timestamp DESC) AS rn
            FROM actual_demand
        ) x
        WHERE rn=1
        """
    )

    # Primary source: a genuinely stored next-30-minute forecast.
    forecasts = read_df(
        """
        WITH ranked AS (
            SELECT region, model_name, forecast_origin, target_timestamp, forecast_mw, created_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY region, model_name, target_timestamp
                       ORDER BY forecast_origin DESC
                   ) AS rn
            FROM next_interval_forecast
            WHERE target_timestamp >= :min_target
              AND target_timestamp < :end_target
        )
        SELECT region, model_name, forecast_origin, target_timestamp, forecast_mw, created_at
        FROM ranked
        WHERE rn=1
        """,
        {
            "min_target": min_target.to_pydatetime(),
            "end_target": (max_target + pd.Timedelta(minutes=30)).to_pydatetime(),
        },
    )

    actual_map = {}
    if not actuals.empty:
        actuals["timestamp"] = pd.to_datetime(actuals["timestamp"], errors="coerce")
        actuals["demand_mw"] = pd.to_numeric(actuals["demand_mw"], errors="coerce")
        actual_map = {r["region"]: r for _, r in actuals.iterrows()}

    forecast_map = {}
    if not forecasts.empty:
        forecasts["target_timestamp"] = pd.to_datetime(forecasts["target_timestamp"], errors="coerce")
        forecasts["forecast_mw"] = pd.to_numeric(forecasts["forecast_mw"], errors="coerce")
        forecasts["created_at"] = pd.to_datetime(forecasts.get("created_at"), errors="coerce") if "created_at" in forecasts.columns else pd.NaT
        for _, r in forecasts.iterrows():
            region = r["region"]
            exact_target = pd.Timestamp(r["target_timestamp"]) == pd.Timestamp(target_map.get(region))
            age = _fresh_age_minutes(r.get("created_at"))
            fresh = age is None or age <= 90
            if r["model_name"] == _default_model(region) and exact_target and fresh:
                item = r.copy()
                item["forecast_source"] = "LIVE 30M"
                forecast_map[region] = item

    # Display fallback only: if the dedicated operational table has not yet stored
    # the next interval, use the latest rolling project forecast for that same target.
    # This prevents blank dashboard cards while preserving next_interval_forecast as
    # the authoritative source for operational accuracy scoring.
    missing_regions = [r for r in REGIONS if r not in forecast_map]
    if missing_regions:
        fallback = read_df(
            """
            WITH ranked AS (
                SELECT region, model_name, forecast_origin, target_timestamp, forecast_mw, created_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY region, model_name, target_timestamp
                           ORDER BY created_at DESC, forecast_origin DESC
                       ) AS rn
                FROM project_forecast
                WHERE target_timestamp >= :min_target
                  AND target_timestamp < :end_target
            )
            SELECT region, model_name, forecast_origin, target_timestamp, forecast_mw, created_at
            FROM ranked
            WHERE rn=1
            """,
            {
                "min_target": min_target.to_pydatetime(),
                "end_target": (max_target + pd.Timedelta(minutes=30)).to_pydatetime(),
            },
        )
        if not fallback.empty:
            fallback["target_timestamp"] = pd.to_datetime(fallback["target_timestamp"], errors="coerce")
            fallback["forecast_mw"] = pd.to_numeric(fallback["forecast_mw"], errors="coerce")
            fallback["created_at"] = pd.to_datetime(fallback.get("created_at"), errors="coerce") if "created_at" in fallback.columns else pd.NaT
            for _, r in fallback.iterrows():
                region = r["region"]
                exact_target = pd.Timestamp(r["target_timestamp"]) == pd.Timestamp(target_map.get(region))
                age = _fresh_age_minutes(r.get("created_at"))
                fresh = age is not None and age <= 120
                if (
                    region in missing_regions
                    and r["model_name"] == _default_model(region)
                    and region not in forecast_map
                    and exact_target
                    and fresh
                ):
                    item = r.copy()
                    item["forecast_source"] = "ROLLING"
                    forecast_map[region] = item

    cards, summaries = [], []
    for region in REGIONS:
        model = _default_model(region)
        a = actual_map.get(region)
        f = forecast_map.get(region)

        latest_ts = pd.NaT if a is None else pd.Timestamp(a["timestamp"])
        current_value = np.nan if a is None else float(a["demand_mw"])

        forecast_ts = pd.NaT if f is None else pd.Timestamp(f["target_timestamp"])
        next_value = np.nan if f is None else float(f["forecast_mw"])
        source_text = "WAITING" if f is None else str(f.get("forecast_source", "FORECAST"))

        if pd.isna(latest_ts):
            actual_time = "Awaiting actual"
        else:
            local_actual = _local_display_time(region, latest_ts)
            actual_time = f"Actual {local_actual:%H:%M} {local_actual.tzname()}"
        if pd.isna(forecast_ts):
            forecast_time = None
        else:
            local_fc = _local_display_time(region, forecast_ts)
            forecast_time = f"for {local_fc:%H:%M} {local_fc.tzname()}"
        model_label = model.replace("RandomForest_", "Model ")

        # Display labels only; the internal source values are unchanged.
        if source_text == "LIVE 30M":
            source_chip = status_chip("live", "Live 30m", "Stored by the 30-minute operational cycle")
        elif source_text == "ROLLING":
            source_chip = status_chip("rolling", "Rolling fallback", "Display fallback from the latest rolling outlook — not scored as live-verified")
        else:
            source_chip = status_chip("waiting", "Waiting", "No forecast stored yet for the next interval")

        if np.isfinite(current_value) and np.isfinite(next_value) and current_value != 0:
            delta = next_value - current_value
            pct = 100.0 * delta / current_value
            direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
            change = html.Div(
                [
                    html.Span(icon("trend-" + direction if direction != "flat" else "minus", "change-ico")),
                    html.Span(f"{delta:+,.0f} MW", className="change-mw"),
                    html.Span(f"{pct:+.2f}%", className="change-pct"),
                ],
                className=f"rc-change-value is-{direction}",
            )
        else:
            change = html.Div("Waiting for next forecast", className="rc-change-value is-waiting")

        next_block = (
            [_mw_value(next_value, "num num--forecast"), html.Span(forecast_time, className="rc-when")]
            if np.isfinite(next_value)
            else [html.Span("—", className="num num--forecast is-empty"), html.Span("Awaiting forecast", className="rc-when")]
        )

        cards.append(html.Article([
            html.Div([
                html.Div([
                    html.Div([html.Span(className="region-dot", **{"aria-hidden": "true"}), html.Span(region, className="region-code")], className="region-code-row"),
                    html.Div(REGION_LABELS[region], className="region-name"),
                ]),
                html.Span(REGION_MARKET[region], className="market-pill"),
            ], className="rc-head"),
            html.Div([
                html.Div([html.Div("Current demand", className="rc-label"), _mw_value(current_value), html.Span(actual_time, className="rc-when")], className="rc-metric"),
                html.Div([html.Div("Next 30 min", className="rc-label"), *next_block], className="rc-metric rc-metric--next"),
            ], className="rc-metrics"),
            html.Div([html.Span("Change", className="rc-label"), change], className="rc-change"),
            html.Div([
                html.Span(model_label, className="model-pill"),
                source_chip,
                html.Button(
                    [html.Span("Live view"), icon("arrow-right", "btn-ico")],
                    id={"type": "region-open", "region": region},
                    n_clicks=0,
                    className="btn btn-quiet btn-sm rc-open",
                    **{"aria-label": f"Open live view for {REGION_LABELS[region]}"},
                ),
            ], className="rc-foot"),
        ], className=f"region-card region-{region.lower()}", style={"--region-color": REGION_COLORS.get(region, ACCENT)}))

        summaries.append({"region": region, "actual": current_value, "next": next_value, "actual_ts": latest_ts})

    valid_actual = [x for x in summaries if np.isfinite(x["actual"])]
    valid_next = [x for x in summaries if np.isfinite(x["next"])]
    total_actual = sum(x["actual"] for x in valid_actual) if valid_actual else np.nan
    total_next = sum(x["next"] for x in valid_next) if valid_next else np.nan
    highest = max(valid_actual, key=lambda x: x["actual"]) if valid_actual else None
    freshness = max((x["actual_ts"] for x in valid_actual if pd.notna(x["actual_ts"])), default=pd.NaT)

    def pulse(label, value_children, sub, css=""):
        return html.Div(
            [html.Div(label, className="pulse-label"), html.Div(value_children, className="pulse-value"), html.Div(sub, className="pulse-sub")],
            className=f"pulse {css}".strip(),
        )

    next_sub = (
        "No next-interval forecast stored yet"
        if not valid_next
        else f"Next local 30-min interval · {len(valid_next)} of {len(REGIONS)} regions"
    )
    national = [
        pulse("National demand", _mw_value(total_actual), f"Sum of {len(valid_actual)} regions" if valid_actual else "Awaiting actuals", "pulse--lead"),
        pulse(
            "Next interval forecast",
            _mw_value(total_next, "num num--forecast") if np.isfinite(total_next) else html.Span("Awaiting forecast", className="num num--wait"),
            next_sub,
        ),
        pulse(
            "Peak region",
            html.Span(highest["region"], className="num") if highest else html.Span("—", className="num is-empty"),
            f"{REGION_LABELS[highest['region']]} · {highest['actual']:,.0f} MW" if highest else "Awaiting actuals",
        ),
        pulse(
            "Latest actual",
            html.Span(freshness.strftime("%H:%M"), className="num") if pd.notna(freshness) else html.Span("—", className="num is-empty"),
            f"Latest regional actual · {freshness:%d %b}" if pd.notna(freshness) else "Awaiting actuals",
        ),
    ]

    if valid_actual and np.isfinite(total_actual) and total_actual > 0:
        ordered = sorted(valid_actual, key=lambda x: x["actual"], reverse=True)
        share = [
            html.Div(
                [
                    html.Span("Share of national demand", className="share-title"),
                    html.Span("Latest actual per region", className="share-note"),
                ],
                className="share-head",
            ),
            html.Div(
                [
                    html.Span(
                        className="share-seg",
                        style={"--region-color": REGION_COLORS.get(x["region"], ACCENT), "flexGrow": f"{x['actual']:.3f}"},
                        title=f"{x['region']}: {x['actual']:,.0f} MW ({100 * x['actual'] / total_actual:.1f}%)",
                    )
                    for x in ordered
                ],
                className="share-bar",
                role="img",
                **{"aria-label": "Share of national demand by region: " + ", ".join(f"{x['region']} {100 * x['actual'] / total_actual:.1f} percent" for x in ordered)},
            ),
            html.Ul(
                [
                    html.Li(
                        [
                            html.Span(className="share-dot", **{"aria-hidden": "true"}),
                            html.Span(x["region"], className="share-code"),
                            html.Span(f"{100 * x['actual'] / total_actual:.1f}%", className="share-pct"),
                        ],
                        style={"--region-color": REGION_COLORS.get(x["region"], ACCENT)},
                    )
                    for x in ordered
                ],
                className="share-legend",
            ),
        ]
    else:
        share = html.Div("Regional share appears once actual demand is stored.", className="empty-note")

    return national, share


def _sparkline(values, color):
    fig = go.Figure(go.Scatter(y=list(values), mode="lines", line={"width": 1.6, "color": color}, hoverinfo="skip"))
    fig.update_layout(height=36, margin={"l": 0, "r": 0, "t": 2, "b": 2}, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      xaxis={"visible": False, "fixedrange": True}, yaxis={"visible": False, "fixedrange": True})
    return dcc.Graph(figure=fig, config={"displayModeBar": False, "staticPlot": True}, className="spark", style={"height": "36px"})


@callback(
    Output("other-regions-strip", "children"),
    Input("refresh-ui", "n_intervals"),
    Input("selected-region", "data"),
    Input("tabs", "value"),
)
def other_regions_strip(_, selected, tab):
    """Compact cross-region snapshot for Regional Analysis."""
    if tab != "history":
        raise PreventUpdate
    selected = selected or default_region()
    others = [r for r in REGIONS if r != selected]
    recent = read_df(
        """
        SELECT region, timestamp, demand_mw FROM (
            SELECT region, timestamp, demand_mw,
                   ROW_NUMBER() OVER (PARTITION BY region ORDER BY timestamp DESC) AS rn
            FROM actual_demand
        ) x WHERE rn <= 48
        """
    )
    nxt = read_df(
        """
        SELECT region, model_name, target_timestamp, forecast_mw FROM (
            SELECT region, model_name, target_timestamp, forecast_mw,
                   ROW_NUMBER() OVER (PARTITION BY region, model_name ORDER BY target_timestamp DESC, forecast_origin DESC) AS rn
            FROM next_interval_forecast
        ) x WHERE rn = 1
        """
    )
    tiles = []
    for r in others:
        d = recent[recent["region"] == r].sort_values("timestamp") if not recent.empty else pd.DataFrame()
        cur = float(d["demand_mw"].iloc[-1]) if not d.empty else np.nan
        f = nxt[(nxt["region"] == r) & (nxt["model_name"] == _default_model(r))] if not nxt.empty else pd.DataFrame()
        fv = float(f["forecast_mw"].iloc[0]) if not f.empty else np.nan
        direction = ins.forecast_direction_insight(cur, fv)
        arrow = {"up": "trend-up", "down": "trend-down", "flat": "minus"}.get(direction["direction"], "minus")
        tiles.append(
            html.Button(
                [
                    html.Div([html.Span(className="context-dot", style={"--region-color": REGION_COLORS.get(r, ACCENT)}, **{"aria-hidden": "true"}),
                              html.Span(r, className="tile-code"), html.Span(REGION_LABELS[r].split(" / ")[0], className="tile-name")], className="tile-head"),
                    html.Div([_mw_value(cur), html.Span([icon(arrow, "tile-ico"), html.Span(direction["words"])], className=f"tile-dir is-{direction['direction']}")],
                             className="tile-body"),
                    _sparkline(d["demand_mw"].tolist(), REGION_COLORS.get(r, ACCENT)) if len(d) > 2 else html.Div("No recent data", className="tile-empty"),
                ],
                id={"type": "region-tile", "region": r},
                n_clicks=0,
                className="region-tile",
                title=f"Show {REGION_LABELS[r]}",
                **{"aria-label": f"Show {REGION_LABELS[r]}. {direction['words']}."},
            )
        )
    return tiles


@callback(
    Output("region-profile", "children"),
    Input("selected-region", "data"),
    Input("model-switch", "value"),
    Input("tabs", "value"),
    Input("refresh-ui", "n_intervals"),
)
def region_profile(region, model, tab, _):
    """Selected-region current-state summary beside the Dashboard map."""
    if tab != "dashboard":
        raise PreventUpdate
    region = region or "NSW1"
    model = model or _default_model(region)

    span = read_df(
        "SELECT MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts FROM actual_demand WHERE region=:r",
        {"r": region},
    )
    first_ts = pd.to_datetime(span.iloc[0]["first_ts"], errors="coerce") if len(span) else pd.NaT
    last_ts = pd.to_datetime(span.iloc[0]["last_ts"], errors="coerce") if len(span) else pd.NaT
    latest_value = np.nan
    if pd.notna(last_ts):
        v = read_df(
            "SELECT demand_mw FROM actual_demand WHERE region=:r AND timestamp=:t LIMIT 1",
            {"r": region, "t": pd.Timestamp(last_ts).to_pydatetime()},
        )
        if len(v):
            latest_value = float(pd.to_numeric(v.iloc[0, 0], errors="coerce"))

    next_point = _latest_next_interval(region, model)
    if next_point.empty:
        next_children = [html.Span("—", className="num num--forecast is-empty"), status_chip("waiting", "Awaiting forecast")]
    else:
        r = next_point.iloc[0]
        src = str(r.get("display_source", "LIVE 30M"))
        next_children = [
            _mw_value(float(r.forecast_mw), "num num--forecast"),
            status_chip("live", "Live 30m") if src == "LIVE 30M" else status_chip("rolling", "Rolling fallback"),
        ]

    if pd.notna(last_ts):
        local_actual = _local_display_time(region, last_ts)
        actual_when = f"Actual {local_actual:%d %b %H:%M} {local_actual.tzname()}"
    else:
        actual_when = "No actual demand stored"

    def row(label, value, note=None):
        return html.Div([html.Dt(label), html.Dd([value, html.Div(note, className="profile-note") if note else None])], className="profile-row")

    return [
        html.Div(
            [
                html.Span(region, className="profile-code"),
                html.Div([html.H3(REGION_LABELS[region], className="profile-name"),
                          html.Div(f"{REGION_MARKET[region]} market · {REGION_TZ[region]}", className="profile-meta")]),
            ],
            className="profile-head",
            style={"--region-color": REGION_COLORS.get(region, ACCENT)},
        ),
        html.Dl(
            [
                row("Current demand", _mw_value(latest_value), actual_when),
                row("Next 30-min forecast", html.Span(next_children, className="profile-inline")),
                row("Local time", html.Span(region_clock_label(region), className="profile-text"), f"Market clock {market_clock_label(region)}"),
                row("History stored", html.Span(
                    f"{first_ts:%b %Y} – {last_ts:%b %Y}" if pd.notna(first_ts) and pd.notna(last_ts) else "No history yet",
                    className="profile-text")),
            ],
            className="profile-list",
        ),
    ]


def _latest_origin(region, model):
    d = read_df(
        """
        SELECT MAX(forecast_origin) AS origin
        FROM project_forecast
        WHERE region=:r AND model_name=:m
        """,
        {"r": region, "m": model},
    )
    return pd.to_datetime(d.iloc[0, 0]) if len(d) and pd.notna(d.iloc[0, 0]) else pd.NaT


def _actuals(region: str, start=None, end=None) -> pd.DataFrame:
    """Read only the actual-demand window needed by the active view."""
    sql = "SELECT timestamp, demand_mw FROM actual_demand WHERE region=:r"
    params = {"r": region}
    if start is not None:
        sql += " AND timestamp >= :start"
        params["start"] = pd.Timestamp(start).to_pydatetime()
    if end is not None:
        sql += " AND timestamp <= :end"
        params["end"] = pd.Timestamp(end).to_pydatetime()
    sql += " ORDER BY timestamp"
    d = read_df(sql, params)
    if not d.empty:
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        d["demand_mw"] = pd.to_numeric(d["demand_mw"], errors="coerce")
        d = d.dropna(subset=["timestamp", "demand_mw"])
    return d


def _month_bounds(actual: pd.DataFrame):
    if actual.empty:
        return pd.NaT, pd.NaT
    latest = pd.Timestamp(actual["timestamp"].max())
    start = latest.normalize().replace(day=1)
    return start, latest


def _selected_month_bounds(actual: pd.DataFrame, month_value=None):
    if actual.empty:
        return pd.NaT, pd.NaT, pd.NaT
    latest = min(pd.Timestamp(actual["timestamp"].max()), _portal_now().floor("30min"))
    if month_value:
        month_start = pd.Timestamp(f"{month_value}-01")
    else:
        month_start = latest.normalize().replace(day=1)
    next_month = month_start + pd.offsets.MonthBegin(1)
    observed_end = min(latest, next_month - pd.Timedelta(microseconds=1))
    return month_start, next_month, observed_end


def _month_start_origin(region: str, model: str, month_start) -> pd.Timestamp:
    """Find the frozen project vintage captured near the start of a month.

    Only origins in the first 24 hours are eligible. This deliberately excludes
    ordinary rolling forecasts created later in the month.
    """
    if pd.isna(month_start):
        return pd.NaT
    month_start = pd.Timestamp(month_start)
    d = read_df(
        """
        SELECT MIN(forecast_origin) AS origin
        FROM project_forecast
        WHERE region=:r AND model_name=:m
          AND forecast_origin >= :start
          AND forecast_origin < :cutoff
        """,
        {
            "r": region,
            "m": model,
            "start": month_start.to_pydatetime(),
            "cutoff": (month_start + pd.Timedelta(days=1)).to_pydatetime(),
        },
    )
    return pd.to_datetime(d.iloc[0, 0]) if len(d) and pd.notna(d.iloc[0, 0]) else pd.NaT


def _monthly_frozen_tracking(region: str, model: str, month_start, observed_end) -> pd.DataFrame:
    """Compare the immutable month-start forecast against actual demand to date."""
    origin = _month_start_origin(region, model, month_start)
    if pd.isna(origin) or pd.isna(observed_end):
        return pd.DataFrame()

    d = read_df(
        """
        WITH ranked AS (
            SELECT f.target_timestamp, f.forecast_mw, f.forecast_origin,
                   f.model_version, f.created_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY f.target_timestamp
                       ORDER BY f.created_at ASC, f.model_version ASC
                   ) AS rn
            FROM project_forecast f
            WHERE f.region=:r AND f.model_name=:m AND f.forecast_origin=:origin
              AND f.target_timestamp >= :start AND f.target_timestamp <= :end
        )
        SELECT f.target_timestamp, f.forecast_mw, f.forecast_origin,
               f.model_version, a.demand_mw AS actual_mw
        FROM ranked f
        JOIN actual_demand a ON a.region=:r AND a.timestamp=f.target_timestamp
        WHERE f.rn=1
        ORDER BY f.target_timestamp
        """,
        {
            "r": region,
            "m": model,
            "origin": pd.Timestamp(origin).to_pydatetime(),
            "start": pd.Timestamp(month_start).to_pydatetime(),
            "end": pd.Timestamp(observed_end).to_pydatetime(),
        },
    )
    if d.empty:
        return d
    d["target_timestamp"] = pd.to_datetime(d["target_timestamp"], errors="coerce")
    d["forecast_origin"] = pd.to_datetime(d["forecast_origin"], errors="coerce")
    for c in ("forecast_mw", "actual_mw"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["target_timestamp", "forecast_mw", "actual_mw"])
    d["error_mw"] = d["forecast_mw"] - d["actual_mw"]
    d["abs_error_mw"] = d["error_mw"].abs()
    d["squared_error"] = d["error_mw"] ** 2
    denom = d["forecast_mw"].abs() + d["actual_mw"].abs()
    d["smape"] = np.where(denom >= 1.0, 200.0 * d["abs_error_mw"] / denom, np.nan)
    return d


def _operational_tracking(region: str, model: str, start, end) -> pd.DataFrame:
    """Continuous next-interval performance with explicit provenance.

    LIVE_VERIFIED rows come from predictions stored before the target interval.
    HISTORICAL_REPLAY rows fill earlier/gap intervals using the leakage-safe replay
    table. If both exist for a timestamp, LIVE_VERIFIED always wins.
    """
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()

    d = read_df(
        """
        WITH live_ranked AS (
            SELECT model_version, forecast_origin, target_timestamp,
                   forecast_mw, actual_mw, error_mw, abs_error_mw,
                   squared_error, smape_component AS smape,
                   'LIVE_VERIFIED' AS provenance,
                   2 AS provenance_priority,
                   ROW_NUMBER() OVER (
                       PARTITION BY target_timestamp
                       ORDER BY forecast_origin DESC
                   ) AS source_rn
            FROM next_interval_accuracy
            WHERE region=:r AND model_name=:m
              AND target_timestamp >= :start
              AND target_timestamp <= :end
        ),
        replay_ranked AS (
            SELECT model_version, forecast_origin, target_timestamp,
                   forecast_mw, actual_mw, error_mw, abs_error_mw,
                   squared_error, smape_component AS smape,
                   'HISTORICAL_REPLAY' AS provenance,
                   1 AS provenance_priority,
                   ROW_NUMBER() OVER (
                       PARTITION BY target_timestamp
                       ORDER BY forecast_origin DESC
                   ) AS source_rn
            FROM historical_replay_accuracy
            WHERE region=:r AND model_name=:m
              AND target_timestamp >= :start
              AND target_timestamp <= :end
        ),
        candidates AS (
            SELECT * FROM live_ranked WHERE source_rn=1
            UNION ALL
            SELECT * FROM replay_ranked WHERE source_rn=1
        ),
        final_ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY target_timestamp
                       ORDER BY provenance_priority DESC, forecast_origin DESC
                   ) AS final_rn
            FROM candidates
        )
        SELECT model_version, forecast_origin, target_timestamp,
               forecast_mw, actual_mw, error_mw, abs_error_mw,
               squared_error, smape, provenance
        FROM final_ranked
        WHERE final_rn=1
        ORDER BY target_timestamp
        """,
        {
            "r": region,
            "m": model,
            "start": pd.Timestamp(start).to_pydatetime(),
            "end": pd.Timestamp(end).to_pydatetime(),
        },
    )
    if d.empty:
        return d
    d["target_timestamp"] = pd.to_datetime(d["target_timestamp"], errors="coerce")
    d["forecast_origin"] = pd.to_datetime(d["forecast_origin"], errors="coerce")
    for c in ("forecast_mw", "actual_mw", "error_mw", "abs_error_mw", "squared_error", "smape"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["target_timestamp", "forecast_mw", "actual_mw"])

def _latest_next_interval(region: str, model: str, next_target=None) -> pd.DataFrame:
    """Return the next interval users should see on the Live page.

    Prefer a genuinely stored one-step operational forecast. If the 30-minute
    operational cycle has not captured that interval yet, use the latest rolling
    project forecast for the exact same target as a *display fallback only*.
    Accuracy scoring continues to use next_interval_forecast / replay tables and
    therefore never treats this fallback as live-verified evidence.
    """
    next_target = pd.Timestamp(next_target if next_target is not None else _next_region_half_hour(region))
    end_target = next_target + pd.Timedelta(minutes=30)

    d = read_df(
        """
        SELECT region, model_name, model_version, forecast_origin,
               target_timestamp, forecast_mw, lead_minutes, created_at
        FROM next_interval_forecast
        WHERE region=:r AND model_name=:m
          AND target_timestamp >= :target
          AND target_timestamp < :end_target
        ORDER BY target_timestamp ASC, forecast_origin DESC
        LIMIT 1
        """,
        {
            "r": region,
            "m": model,
            "target": next_target.to_pydatetime(),
            "end_target": end_target.to_pydatetime(),
        },
    )
    source = "LIVE 30M"
    if not d.empty and "created_at" in d.columns:
        age = _fresh_age_minutes(d.iloc[0].get("created_at"))
        if age is not None and age > 90:
            d = pd.DataFrame()

    if d.empty:
        # project_forecast.forecast_origin is the first forecast grid timestamp,
        # not necessarily the wall-clock creation time. Do not require it to be
        # <= now; created_at determines when the vintage was actually written.
        d = read_df(
            f"""
            SELECT region, model_name, model_version, forecast_origin,
                   target_timestamp, forecast_mw,
                   {sql_minutes_between("target_timestamp", "created_at")} AS lead_minutes,
                   created_at
            FROM project_forecast
            WHERE region=:r AND model_name=:m
              AND target_timestamp >= :target
              AND target_timestamp < :end_target
            ORDER BY created_at DESC, forecast_origin DESC
            LIMIT 1
            """,
            {
                "r": region,
                "m": model,
                "target": next_target.to_pydatetime(),
                "end_target": end_target.to_pydatetime(),
            },
        )
        source = "ROLLING FALLBACK"
        if not d.empty and "created_at" in d.columns:
            age = _fresh_age_minutes(d.iloc[0].get("created_at"))
            if age is None or age > 120:
                d = pd.DataFrame()

    if not d.empty:
        d["forecast_origin"] = pd.to_datetime(d["forecast_origin"], errors="coerce")
        d["target_timestamp"] = pd.to_datetime(d["target_timestamp"], errors="coerce")
        d["created_at"] = pd.to_datetime(d["created_at"], errors="coerce")
        d["forecast_mw"] = pd.to_numeric(d["forecast_mw"], errors="coerce")
        d["display_source"] = source
    return d


def _available_performance_months(region: str, year: int | None = None):
    d = read_df(
        f"""
        SELECT DISTINCT {sql_month_start("timestamp")} AS month_start
        FROM actual_demand
        WHERE region=:r AND timestamp >= '2026-09-01'
        ORDER BY month_start
        """,
        {"r": region},
    )
    if d.empty:
        return []
    vals = pd.to_datetime(d["month_start"], errors="coerce").dropna()
    if year is not None:
        vals = vals[vals.dt.year == int(year)]
    return [{"label": x.strftime("%B %Y"), "value": x.strftime("%Y-%m")} for x in vals]


def _available_performance_years(region: str):
    months = _available_performance_months(region)
    years = sorted({int(x["value"][:4]) for x in months})
    return [{"label": str(y), "value": y} for y in years]


def _monthly_accuracy_archive(region: str, model: str) -> pd.DataFrame:
    opts = _available_performance_months(region)
    rows = []
    for opt in opts:
        start = pd.Timestamp(f"{opt['value']}-01")
        next_month = start + pd.offsets.MonthBegin(1)
        latest_row = read_df(
            "SELECT MAX(timestamp) AS latest FROM actual_demand WHERE region=:r AND timestamp>=:start AND timestamp<:next_month",
            {"r": region, "start": start.to_pydatetime(), "next_month": next_month.to_pydatetime()},
        )
        latest = pd.to_datetime(latest_row.iloc[0, 0], errors="coerce") if len(latest_row) else pd.NaT
        if pd.isna(latest):
            continue
        end = min(pd.Timestamp(latest), next_month - pd.Timedelta(microseconds=1))
        if start.strftime("%Y-%m") == region_local_now(region).strftime("%Y-%m"):
            end = min(end, _region_now(region).floor("30min"))
        d = _operational_tracking(region, model, start, end)
        if d.empty:
            continue
        mae, rmse, smape, bias = _accuracy_metrics(d)
        live_n = int((d.get("provenance") == "LIVE_VERIFIED").sum()) if "provenance" in d.columns else len(d)
        replay_n = int((d.get("provenance") == "HISTORICAL_REPLAY").sum()) if "provenance" in d.columns else 0
        expected = max(1, int(((pd.Timestamp(end).floor("30min") - start).total_seconds() // 1800) + 1))
        rows.append({
            "month": start.strftime("%Y-%m"),
            "month_label": start.strftime("%b %Y"),
            "first_forecast_origin": d["forecast_origin"].min(),
            "last_forecast_origin": d["forecast_origin"].max(),
            "matched_intervals": len(d),
            "coverage_pct": 100.0 * len(d) / expected,
            "live_verified_intervals": live_n,
            "historical_replay_intervals": replay_n,
            "MAE_MW": mae,
            "RMSE_MW": rmse,
            "sMAPE_pct": smape,
            "Bias_MW": bias,
        })
    return pd.DataFrame(rows)


def _latest_aemo_comparison(region: str, model: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Use the newest stored AEMO pre-dispatch snapshot that has exact target overlap.

    AEMO is never extended to an artificial horizon. The function searches recent
    stored vintages because a transient AEMO timeout can leave the newest project
    and AEMO origins slightly asynchronous.
    """
    if region == "WA":
        return pd.DataFrame(), pd.NaT, pd.NaT

    a_orig = read_df(
        """SELECT DISTINCT forecast_origin FROM aemo_forecast
           WHERE region=:r AND source='AEMO_PREDISPATCH_30MIN'
           ORDER BY forecast_origin DESC LIMIT 8""",
        {"r": region},
    )
    p_orig = read_df(
        """SELECT DISTINCT forecast_origin FROM project_forecast
           WHERE region=:r AND model_name=:m
           ORDER BY forecast_origin DESC LIMIT 12""",
        {"r": region, "m": model},
    )
    if a_orig.empty or p_orig.empty:
        return pd.DataFrame(), pd.NaT, pd.NaT

    a_origins = [pd.Timestamp(x) for x in pd.to_datetime(a_orig["forecast_origin"], errors="coerce").dropna()]
    p_origins = [pd.Timestamp(x) for x in pd.to_datetime(p_orig["forecast_origin"], errors="coerce").dropna()]
    chosen = None

    for ao in a_origins:
        candidates = [po for po in p_origins if po <= ao]
        if not candidates:
            candidates = p_origins[:3]
        best_for_a = None
        for po in candidates:
            cnt = read_df(
                """SELECT COUNT(*) AS n
                   FROM project_forecast p
                   JOIN aemo_forecast a
                     ON a.region=p.region AND a.target_timestamp=p.target_timestamp
                   WHERE p.region=:r AND p.model_name=:m
                     AND p.forecast_origin=:po
                     AND a.forecast_origin=:ao
                     AND a.source='AEMO_PREDISPATCH_30MIN'""",
                {"r": region, "m": model, "po": po.to_pydatetime(), "ao": ao.to_pydatetime()},
            )
            common = int(cnt.iloc[0]["n"]) if len(cnt) else 0
            if common > 0 and (best_for_a is None or common > best_for_a[0]):
                best_for_a = (common, po, ao)
        if best_for_a is not None:
            chosen = best_for_a
            break

    if chosen is None:
        return pd.DataFrame(), pd.NaT, a_origins[0] if a_origins else pd.NaT

    _, project_origin, aemo_origin = chosen
    d = read_df(
        """SELECT p.target_timestamp, p.forecast_mw AS project_mw, a.poe50_mw AS aemo_mw
           FROM project_forecast p
           JOIN aemo_forecast a
             ON a.region=p.region AND a.target_timestamp=p.target_timestamp
           WHERE p.region=:r AND p.model_name=:m
             AND p.forecast_origin=:po
             AND a.forecast_origin=:ao
             AND a.source='AEMO_PREDISPATCH_30MIN'
           ORDER BY p.target_timestamp""",
        {"r": region, "m": model, "po": project_origin.to_pydatetime(), "ao": aemo_origin.to_pydatetime()},
    )
    if d.empty:
        return d, project_origin, aemo_origin
    d["target_timestamp"] = pd.to_datetime(d["target_timestamp"], errors="coerce")
    d["project_mw"] = pd.to_numeric(d["project_mw"], errors="coerce")
    d["aemo_mw"] = pd.to_numeric(d["aemo_mw"], errors="coerce")
    d = d.dropna(subset=["target_timestamp", "project_mw", "aemo_mw"]).drop_duplicates("target_timestamp")
    d["difference_mw"] = d["project_mw"] - d["aemo_mw"]
    d["absolute_difference_mw"] = d["difference_mw"].abs()
    return d.reset_index(drop=True), project_origin, aemo_origin


def _rolling_tracking(region: str, model: str, start, end) -> pd.DataFrame:
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()

    d = read_df(
        """
        WITH ranked AS (
            SELECT
                f.target_timestamp,
                f.forecast_mw,
                f.forecast_origin,
                f.model_version,
                f.created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY f.target_timestamp
                    ORDER BY f.forecast_origin DESC, f.created_at DESC
                ) AS rn
            FROM project_forecast f
            WHERE f.region=:r
              AND f.model_name=:m
              AND f.target_timestamp >= :start
              AND f.target_timestamp <= :end
              AND f.forecast_origin <= f.target_timestamp
        )
        SELECT
            x.target_timestamp,
            x.forecast_mw,
            x.forecast_origin,
            x.model_version,
            a.demand_mw AS actual_mw
        FROM ranked x
        JOIN actual_demand a
          ON a.region=:r
         AND a.timestamp=x.target_timestamp
        WHERE x.rn=1
        ORDER BY x.target_timestamp
        """,
        {
            "r": region,
            "m": model,
            "start": pd.Timestamp(start).to_pydatetime(),
            "end": pd.Timestamp(end).to_pydatetime(),
        },
    )
    if d.empty:
        return d

    d["target_timestamp"] = pd.to_datetime(d["target_timestamp"], errors="coerce")
    d["forecast_origin"] = pd.to_datetime(d["forecast_origin"], errors="coerce")
    for c in ("forecast_mw", "actual_mw"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["target_timestamp", "forecast_mw", "actual_mw"])
    d["error_mw"] = d["forecast_mw"] - d["actual_mw"]
    d["abs_error_mw"] = d["error_mw"].abs()
    d["squared_error"] = d["error_mw"] ** 2
    denom = d["forecast_mw"].abs() + d["actual_mw"].abs()
    d["smape"] = np.where(denom >= 1.0, 200.0 * d["abs_error_mw"] / denom, np.nan)
    return d


def _accuracy_metrics(tracking: pd.DataFrame):
    if tracking.empty:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(tracking["abs_error_mw"].mean()),
        float(np.sqrt(tracking["squared_error"].mean())),
        float(tracking["smape"].mean()),
        float(tracking["error_mw"].mean()),
    )


def _coverage_text(tracking: pd.DataFrame, start, end):
    if pd.isna(start) or pd.isna(end):
        return "No current-month actuals"
    expected = max(0, int(((pd.Timestamp(end).floor("30min") - pd.Timestamp(start)).total_seconds() // 1800) + 1))
    covered = len(tracking)
    pct = (100.0 * covered / expected) if expected else 0.0
    if covered:
        first = tracking["target_timestamp"].min()
        return f"{covered:,}/{expected:,} intervals ({pct:.1f}%) • first tracked {first:%d %b %H:%M}"
    return f"0/{expected:,} intervals tracked yet"


# ---------------------------------------------------------------------------
# v3 Live helpers: typical error range, provenance line, insight, freshness
# ---------------------------------------------------------------------------
def _add_error_band(fig, region, model, origin, shown, horizon, enabled) -> str:
    """Shade the historical p10–p90 residual range of the SELECTED model around the
    forecast. Hidden when the viewed/served model is not the evidence-selected one."""
    if not enabled or shown is None or shown.empty or pd.isna(origin):
        return ""
    st = _serving(region)
    if st is None or st.selected != model:
        return "Typical error range is shown only for the evidence-selected production model."
    daily = horizon in ("30D", "4M")
    bands = evidence("forecast_error_bands_daily.csv" if daily else "forecast_error_bands.csv")
    if bands.empty:
        return ""
    b = bands[(bands["region"] == region) & (bands["model"] == model)]
    if not daily:
        b = b[b["hour"].astype(str) != "ALL"].copy()
        b["hour"] = b["hour"].astype(int)
    s = shown.copy()
    s["lead_band"] = [ins.band_for_lead(pd.Timestamp(x) - pd.Timestamp(origin)) for x in s["target_timestamp"]]
    if daily:
        m = s.merge(b[["lead_band", "residual_p10", "residual_p90"]], on="lead_band", how="left")
    else:
        s["hour"] = [_local_display_time(region, x).hour for x in s["target_timestamp"]]
        m = s.merge(b[["lead_band", "hour", "residual_p10", "residual_p90"]], on=["lead_band", "hour"], how="left")
    m = m.dropna(subset=["residual_p10", "residual_p90"])
    if m.empty:
        return ""
    lower, upper = m["forecast_mw"] - m["residual_p90"], m["forecast_mw"] - m["residual_p10"]
    fig.add_trace(go.Scatter(x=m["target_timestamp"], y=upper, line={"width": 0}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=m["target_timestamp"], y=lower, line={"width": 0}, fill="tonexty",
                             fillcolor="rgba(47,123,255,0.14)", name="Typical error range", hoverinfo="skip"))
    return ("Shaded area: typical error range (10th–90th percentile of errors) from the retained historical backtests"
            + (" of daily means" if daily else "") + " — not a probability forecast.")


def _live_extras(region, model, fc, next_point, latest_ts, latest_actual_value, origin, horizon, shown, band_note):
    st = _serving(region)
    served = _default_model(region)
    # --- insight
    parts = []
    if not next_point.empty:
        parts.append(ins.forecast_direction_insight(latest_actual_value, float(next_point.iloc[0].forecast_mw))["sentence"])
    if shown is not None and not shown.empty:
        parts.append(ins.horizon_peak_insight(shown, horizon))
        thr = evidence("peak_thresholds.csv")
        thr = thr[thr["region"] == region].sort_values("backtest_origin") if not thr.empty else thr
        if not thr.empty and float(shown["forecast_mw"].max()) > float(thr["interval_threshold_MW"].iloc[-1]):
            pk = ins.peak_insight(region, evidence("peak_normal_vs_high_summary.csv"), model)
            chip = "High-demand period forecast."
            if pk.get("under_forecasts_peaks"):
                chip += f" In backtests this model tended to forecast too low at {region} peaks."
            parts.append(chip)
        rev = _forecast_revision(region, model, origin, shown)
        if rev:
            parts.append(rev)
    insight = [html.Span(icon("trending", "insight-ico")), html.Span(" ".join(p for p in parts if p) or
               "The insight appears once actual demand and stored forecasts are available.")]
    # --- production model line with provenance of the plotted forecast
    sel = ins.selected_model_insight(region, evidence("regional_model_winners.csv"))
    conf = sel.get("confidence", {}).get("label") if sel else None
    version = str(fc["model_version"].iloc[0]) if not fc.empty and "model_version" in fc else None
    line = [html.Span("Production model", className="model-line-label"),
            html.Strong(f"{served} · {_fs_label(served).split(' · ')[-1]}" if _fs_label(served) else served),
            html.Span("selected from the retained 2022–2025 evidence window" if st and st.selected == served else "fallback", className="model-line-meta")]
    if conf and st and st.selected == served:
        line.append(html.Span(conf, className=f"badge badge-{'good' if 'Clear' in conf else 'neutral'}",
                              title=sel["confidence"]["rule"]))
    if st and st.mismatch:
        line.append(status_chip("stale", f"Selected {st.selected} not yet available — serving {served}", st.message))
    if model != served:
        line.append(status_chip("rolling", f"Viewing comparison: {model}", "Comparison view only — scheduled production forecasts use the production model."))
    if version:
        run = f" · run {pd.Timestamp(origin):%d %b %H:%M}" if pd.notna(origin) else ""
        line.append(html.Span(f"Chart forecasts: {model} {version}{run}", className="model-line-prov"))
    # --- freshness
    latest_origin = origin
    fr = ins.freshness(latest_ts, latest_origin, _region_now(region))
    fresh = [] if not fr["messages"] else [status_chip("stale", " ".join(fr["messages"]))]
    sr = (f"{REGION_LABELS[region]} demand chart. " + " ".join(p for p in parts if p))
    return insight, line, fresh, band_note, sr


def _forecast_revision(region, model, origin, shown) -> str:
    """Latest hourly outlook vs the previous one over the same targets (words only)."""
    if pd.isna(origin) or shown is None or shown.empty:
        return ""
    prev = read_df(
        """
        SELECT AVG(p.forecast_mw) AS prev_mean FROM project_forecast p
        WHERE p.region=:r AND p.model_name=:m AND p.target_timestamp >= :a AND p.target_timestamp <= :b
          AND p.forecast_origin = (SELECT MAX(forecast_origin) FROM project_forecast
                                   WHERE region=:r AND model_name=:m AND forecast_origin < :o)
        """,
        {"r": region, "m": model, "a": pd.Timestamp(shown["target_timestamp"].min()).to_pydatetime(),
         "b": pd.Timestamp(shown["target_timestamp"].max()).to_pydatetime(), "o": pd.Timestamp(origin).to_pydatetime()},
    )
    if prev.empty or pd.isna(prev.iloc[0, 0]) or float(prev.iloc[0, 0]) == 0:
        return ""
    pct = 100 * (float(shown["forecast_mw"].mean()) - float(prev.iloc[0, 0])) / float(prev.iloc[0, 0])
    if abs(pct) < 0.5:
        return "The outlook is essentially unchanged since the previous hourly update."
    return f"Since the previous hourly update the outlook moved {'up' if pct > 0 else 'down'} {'slightly' if abs(pct) < 2 else 'noticeably'}."


@callback(
    Output("live-chart", "figure"),
    Output("m-actual", "children"),
    Output("m-next", "children"),
    Output("m-mae", "children"),
    Output("m-smape", "children"),
    Output("m-peak", "children"),
    Output("m-updated", "children"),
    Output("tracking-coverage", "children"),
    Output("forecast-table", "children"),
    Output("m-actual-sub", "children"),
    Output("m-change", "children"),
    Output("live-insight", "children"),
    Output("live-model-line", "children"),
    Output("live-freshness", "children"),
    Output("live-chart-note", "children"),
    Output("live-chart-sr", "children"),
    Input("selected-region", "data"),
    Input("model-switch", "value"),
    Input("horizon", "value"),
    Input("upcoming-count", "value"),
    Input("refresh-ui", "n_intervals"),
    Input("tabs", "value"),
    Input("error-band-toggle", "value"),
)
def update_live(region, model, horizon, upcoming_count, _, tab, band_toggle=None):
    # Live is embedded in Dashboard; query only while Dashboard is visible.
    if tab != "dashboard":
        raise PreventUpdate
    region = region or default_region()
    model = model if model in _prod_models(region) else _default_model(region)
    latest_row = read_df(
        "SELECT MAX(timestamp) AS latest FROM actual_demand WHERE region=:r",
        {"r": region},
    )
    latest_ts = pd.to_datetime(latest_row.iloc[0, 0], errors="coerce") if len(latest_row) else pd.NaT
    month_start = latest_ts.normalize().replace(day=1) if pd.notna(latest_ts) else pd.NaT
    actual = _actuals(region, start=month_start) if pd.notna(month_start) else pd.DataFrame()

    # Primary live-performance evidence: the point forecast that was stored for
    # each next 30-minute interval before its actual demand arrived.
    tracking = _operational_tracking(region, model, month_start, latest_ts)
    mae, _, smape, _ = _accuracy_metrics(tracking)

    # The next card uses the dedicated one-step forecast table, not the first row
    # of a multi-month forecast vintage.
    next_target = _next_region_half_hour(region)
    next_point = _latest_next_interval(region, model, next_target)

    # The main horizon chart and upcoming table continue to use the latest rolling
    # multi-month forecast, which is a different product from one-step accuracy.
    origin = _latest_origin(region, model)
    fc = pd.DataFrame()
    if pd.notna(origin):
        fc = read_df(
            """
            SELECT target_timestamp, forecast_mw, created_at, model_version, model_name
            FROM project_forecast
            WHERE region=:r AND model_name=:m AND forecast_origin=:o
              AND target_timestamp >= :next_target
            ORDER BY target_timestamp
            """,
            {
                "r": region,
                "m": model,
                "o": origin.to_pydatetime(),
                "next_target": next_target.to_pydatetime(),
            },
        )
        if not fc.empty:
            fc["target_timestamp"] = pd.to_datetime(fc["target_timestamp"], errors="coerce")
            fc["forecast_mw"] = pd.to_numeric(fc["forecast_mw"], errors="coerce")
            fc = fc.dropna(subset=["target_timestamp", "forecast_mw"])

    fig = go.Figure()
    current_month = actual.copy()
    latest_actual_value = np.nan if current_month.empty else float(current_month.iloc[-1].demand_mw)
    latest_actual = _mw_value(latest_actual_value)
    if pd.isna(latest_ts):
        actual_sub = "Awaiting actual demand"
    else:
        local_actual = _local_display_time(region, latest_ts)
        age_min = max(0, int((_region_now(region) - pd.Timestamp(latest_ts)).total_seconds() // 60))
        actual_sub = [
            html.Span(f"Latest actual {local_actual:%H:%M} {local_actual.tzname()}"),
            status_chip("stale", f"{age_min} min old") if age_min > 90 else html.Span(f" · {age_min} min ago", className="lead-age"),
        ]

    limits = {
        "48H": pd.Timedelta(hours=48),
        "7D": pd.Timedelta(days=7),
        "30D": pd.Timedelta(days=30),
        "4M": pd.Timedelta(days=124),
    }
    future_end = next_target + limits[horizon]

    if horizon in ("48H", "7D"):
        if not current_month.empty:
            lookback = pd.Timedelta(days=2 if horizon == "48H" else 7)
            recent = current_month[current_month["timestamp"] >= pd.Timestamp(latest_ts) - lookback].copy()
            fig.add_trace(go.Scatter(
                x=recent["timestamp"], y=recent["demand_mw"],
                name="Actual demand", line={"width": 2.2, "color": ACTUAL_COLOR},
                hovertemplate="%{y:,.0f} MW<extra>Actual</extra>",
            ))
        shown = fc[fc["target_timestamp"] <= future_end].copy() if not fc.empty else pd.DataFrame()
        if not shown.empty:
            fig.add_trace(go.Scatter(
                x=shown["target_timestamp"], y=shown["forecast_mw"],
                name="Forecast (rolling outlook)", line={"width": 2.4, "color": FORECAST_COLOR},
                hovertemplate="%{y:,.0f} MW<extra>Forecast</extra>",
            ))
    else:
        if not current_month.empty:
            daily_actual = current_month.set_index("timestamp")["demand_mw"].resample("D").mean().reset_index()
            fig.add_trace(go.Scatter(
                x=daily_actual["timestamp"], y=daily_actual["demand_mw"],
                name="Actual demand (daily mean)", mode="lines+markers", line={"width": 2.2, "color": ACTUAL_COLOR},
                marker={"size": 5, "color": ACTUAL_COLOR, "line": {"width": 1.2, "color": SURFACE}},
                hovertemplate="%{y:,.0f} MW<extra>Actual</extra>",
            ))
        shown = fc[fc["target_timestamp"] <= future_end].copy() if not fc.empty else pd.DataFrame()
        if not shown.empty:
            shown = shown.set_index("target_timestamp")["forecast_mw"].resample("D").mean().reset_index()
            fig.add_trace(go.Scatter(
                x=shown["target_timestamp"], y=shown["forecast_mw"],
                name="Forecast (daily mean)", mode="lines+markers", line={"width": 2.4, "color": FORECAST_COLOR},
                marker={"size": 5, "color": FORECAST_COLOR, "line": {"width": 1.2, "color": SURFACE}},
                hovertemplate="%{y:,.0f} MW<extra>Forecast</extra>",
            ))

    band_note = _add_error_band(fig, region, model, origin, shown, horizon, bool(band_toggle))
    if fig.data:
        style_figure(fig, "", 460)
        now_x = _region_now(region)
        # Forecast region: a barely-there tint from "now" to the end of the shown outlook.
        if not shown.empty:
            fig.add_shape(
                type="rect", xref="x", yref="paper", x0=now_x.to_pydatetime(),
                x1=pd.Timestamp(shown["target_timestamp"].max()).to_pydatetime(), y0=0, y1=1,
                fillcolor="rgba(47,123,255,0.07)", line={"width": 0}, layer="below",
            )
        if pd.notna(latest_ts):
            _marker_line(fig, pd.Timestamp(latest_ts).to_pydatetime(), "#A3ADBD", "", dash="dot")
        _marker_line(fig, now_x.to_pydatetime(), "#FF7A1A", "Now", dash="dash")
        fig.update_xaxes(title_text=None)
        fig.update_yaxes(title_text="Demand (MW)")
    else:
        fig = empty_figure("The rolling outlook appears after the hourly pipeline runs with a promoted model.", 460, "Forecast pending")

    if tracking.empty:
        coverage = "No completed next-interval forecasts have been matched to actual demand yet."
    else:
        month_expected = max(1, int(((pd.Timestamp(latest_ts).floor("30min") - month_start).total_seconds() // 1800) + 1))
        pct = 100.0 * len(tracking) / month_expected
        live_n = int((tracking["provenance"] == "LIVE_VERIFIED").sum()) if "provenance" in tracking.columns else len(tracking)
        replay_n = int((tracking["provenance"] == "HISTORICAL_REPLAY").sum()) if "provenance" in tracking.columns else 0
        coverage = (
            f"{len(tracking):,}/{month_expected:,} matches ({pct:.1f}%) • "
            f"Live {live_n:,} • Replay {replay_n:,}"
        )

    mae_text = f"{mae:,.1f} MW" if np.isfinite(mae) else "—"
    smape_text = f"{smape:.2f}%" if np.isfinite(smape) else "—"

    if next_point.empty:
        nextv = _mw_value(np.nan, "num num--forecast")
        one_step_updated = [status_chip("waiting", "Awaiting forecast"),
                            html.Span("Appears after the next scheduled 30-minute run", className="lead-age")]
        change_children = html.Span("Waiting for next forecast", className="lead-wait")
    else:
        r = next_point.iloc[0]
        next_value = float(r.forecast_mw)
        nextv = _mw_value(next_value, "num num--forecast")
        source = str(r.get("display_source", "LIVE 30M"))
        target_local = _local_display_time(region, pd.Timestamp(r.target_timestamp))
        one_step_updated = [
            html.Span(f"Target {target_local:%H:%M} {target_local.tzname()}", className="lead-target"),
            status_chip("live", "Live 30m", "Stored by the 30-minute operational cycle")
            if source == "LIVE 30M"
            else status_chip("rolling", "Rolling fallback", "Display fallback from the latest rolling outlook — not scored as live-verified"),
        ]
        if np.isfinite(latest_actual_value) and latest_actual_value != 0:
            delta = next_value - latest_actual_value
            pct = 100.0 * delta / latest_actual_value
            d_ins = ins.forecast_direction_insight(latest_actual_value, next_value)
            direction = d_ins["direction"]
            change_children = html.Span(
                [
                    icon("trend-" + direction if direction in ("up", "down") else "minus", "change-ico"),
                    html.Span(d_ins["words"], className="change-words"),
                ],
                className=f"lead-change is-{direction}",
                title=f"{delta:+,.0f} MW ({pct:+.2f}%) versus the latest actual",
            )
        else:
            change_children = html.Span("Waiting for actual demand", className="lead-wait")

    extras = _live_extras(region, model, fc, next_point, latest_ts, latest_actual_value, origin, horizon,
                          fc[fc["target_timestamp"] <= future_end] if not fc.empty else fc, band_note)
    if fc.empty:
        return (fig, latest_actual, nextv, mae_text, smape_text, "—", one_step_updated, coverage,
                html.Div([html.Strong("No upcoming intervals yet"), html.Span("Rows appear after the hourly rolling outlook is stored.")], className="empty-table-state"),
                actual_sub, change_children, *extras)

    peak_scope = fc[fc["target_timestamp"] <= future_end]
    if peak_scope.empty:
        peak = "—"
    else:
        peak_idx = peak_scope["forecast_mw"].idxmax()
        peak = [
            html.Span([f"{peak_scope.loc[peak_idx, 'forecast_mw']:,.0f}", html.Span(" MW", className="unit")], className="metric-main"),
            html.Span(f"{peak_scope.loc[peak_idx, 'target_timestamp']:%a %d %b %H:%M}", className="metric-when"),
        ]

    n = int(upcoming_count or 20)
    rows = fc.head(max(1, min(n, 50))).copy()
    table = html.Table(
        [
            html.Thead(html.Tr([
                html.Th("Interval", className="forecast-th left", scope="col"),
                html.Th("Forecast MW", className="forecast-th right", scope="col"),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td([html.Span(f"{r.target_timestamp:%d %b} "), html.Strong(f"{r.target_timestamp:%H:%M}")], className="forecast-td"),
                    html.Td(f"{r.forecast_mw:,.1f}", className="forecast-td value"),
                ])
                for _, r in rows.iterrows()
            ]),
        ],
        className="forecast-table",
    )
    return (fig, latest_actual, nextv, mae_text, smape_text, peak, one_step_updated, coverage, table, actual_sub,
            change_children, *extras)


@callback(
    Output("download-tracking", "data"),
    Input("btn-download-tracking", "n_clicks"),
    State("selected-region", "data"),
    State("model-switch", "value"),
    prevent_initial_call=True,
)
def download_tracking(n_clicks, region, model):
    if not n_clicks:
        return None
    actual = _actuals(region)
    start, end = _month_bounds(actual)
    d = _operational_tracking(region, model, start, end)
    if d.empty:
        return None
    export = d[["target_timestamp", "forecast_origin", "model_version", "forecast_mw", "actual_mw", "error_mw", "abs_error_mw", "smape", "provenance"]].copy()
    return dcc.send_data_frame(export.to_csv, f"{region}_{model}_next_interval_tracking.csv", index=False)

@callback(
    Output("download-upcoming", "data"),
    Input("btn-download-upcoming", "n_clicks"),
    State("selected-region", "data"),
    State("model-switch", "value"),
    State("download-forecast-horizon", "value"),
    prevent_initial_call=True,
)
def download_upcoming(n_clicks, region, model, horizon):
    if not n_clicks:
        return None
    origin = _latest_origin(region, model)
    if pd.isna(origin):
        return None
    horizon = horizon if horizon in {"48H", "7D", "30D", "4M"} else "4M"
    next_target = _next_region_half_hour(region)
    end_target = _forecast_export_end(next_target, horizon)
    d = read_df(
        """
        SELECT forecast_origin, target_timestamp, forecast_mw, model_name, model_version, created_at
        FROM project_forecast
        WHERE region=:r AND model_name=:m AND forecast_origin=:o
          AND target_timestamp >= :next_target
          AND target_timestamp <= :end_target
        ORDER BY target_timestamp
        """,
        {
            "r": region,
            "m": model,
            "o": origin.to_pydatetime(),
            "next_target": next_target.to_pydatetime(),
            "end_target": end_target.to_pydatetime(),
        },
    )
    if d.empty:
        return None
    d["target_timestamp"] = pd.to_datetime(d["target_timestamp"], errors="coerce")
    d.insert(0, "region", region)
    d.insert(1, "market", REGION_MARKET[region])
    d.insert(2, "export_horizon", horizon)
    d["exported_at"] = _portal_now()
    safe_model = model.replace("/", "-")
    return dcc.send_data_frame(
        d.to_csv,
        f"{region}_{safe_model}_{horizon}_rolling_forecast.csv",
        index=False,
    )


@callback(
    Output("monthly-history", "figure"),
    Output("daily-profile", "figure"),
    Output("history-status", "children"),
    Input("selected-region", "data"),
    Input("history-years", "value"),
    Input("tabs", "value"),
)
def history(region, years, tab="history"):
    if tab != "history":
        raise PreventUpdate
    years_clean = sorted({int(y) for y in (years or []) if str(y).isdigit() and 2022 <= int(y) <= 2100})
    if not years_clean:
        msg = "Select at least one year to display regional demand history."
        return empty_figure(msg, 420, "No years selected"), empty_figure(msg, 420, "No years selected"), msg

    year_sql = ",".join(str(y) for y in years_clean)
    monthly = read_df(
        f"""
        SELECT
            {sql_year("timestamp")} AS year,
            {sql_month("timestamp")} AS month,
            AVG(demand_mw) AS demand_mw,
            MAX(demand_mw) AS peak_mw,
            COUNT(*) AS intervals
        FROM actual_demand
        WHERE region=:r
          AND {sql_year("timestamp")} IN ({year_sql})
        GROUP BY 1,2
        ORDER BY 1,2
        """,
        {"r": region},
    )
    profile = read_df(
        f"""
        SELECT
            {sql_year("timestamp")} AS year,
            {sql_half_hour("timestamp")} AS half_hour,
            AVG(demand_mw) AS demand_mw,
            COUNT(*) AS intervals
        FROM actual_demand
        WHERE region=:r
          AND {sql_year("timestamp")} IN ({year_sql})
        GROUP BY 1,2
        ORDER BY 1,2
        """,
        {"r": region},
    )

    if monthly.empty or profile.empty:
        msg = f"No {region} actual-demand history is available for the selected years."
        return empty_figure(msg, 420, "No history stored"), empty_figure(msg, 420, "No history stored"), msg

    month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["year"] = pd.to_numeric(monthly["year"], errors="coerce").astype("Int64")
    monthly["month"] = pd.to_numeric(monthly["month"], errors="coerce").astype("Int64")
    monthly["demand_mw"] = pd.to_numeric(monthly["demand_mw"], errors="coerce")
    monthly = monthly.dropna(subset=["year", "month", "demand_mw"])
    monthly["year_label"] = monthly["year"].astype(str)
    monthly["month_label"] = monthly["month"].map(month_labels)

    year_order = [str(y) for y in years_clean]
    # Each earlier year gets its own bright colour; the latest selected year leads in bright orange.
    earlier = max(0, len(year_order) - 1)
    ramp = [YEAR_RAMP[i % len(YEAR_RAMP)] for i in range(earlier)]
    year_colors = {y: c for y, c in zip(year_order[:-1], ramp)}
    if year_order:
        year_colors[year_order[-1]] = LATEST_YEAR_COLOR
    f1 = px.line(
        monthly,
        x="month_label",
        y="demand_mw",
        color="year_label",
        markers=True,
        category_orders={"month_label": list(month_labels.values()), "year_label": year_order},
        color_discrete_map=year_colors,
        labels={"month_label":"Month", "demand_mw":"Demand (MW)", "year_label":"Year"},
    )
    style_figure(f1, "", 420)
    f1.update_traces(line={"width": 2.0}, marker={"size": 6, "line": {"width": 1.2, "color": SURFACE}},
                     hovertemplate="%{y:,.0f} MW")
    if year_order:
        f1.update_traces(selector={"name": year_order[-1]}, line={"width": 3.0}, marker={"size": 7, "line": {"width": 1.5, "color": SURFACE}})
    f1.update_layout(legend_title_text="Year")
    f1.update_xaxes(title_text="Month")
    f1.update_yaxes(title_text="Average demand (MW)")

    profile["year"] = pd.to_numeric(profile["year"], errors="coerce").astype("Int64")
    profile["half_hour"] = pd.to_numeric(profile["half_hour"], errors="coerce").astype("Int64")
    profile["demand_mw"] = pd.to_numeric(profile["demand_mw"], errors="coerce")
    profile = profile.dropna(subset=["year", "half_hour", "demand_mw"])
    profile["year_label"] = profile["year"].astype(str)
    f2 = px.line(
        profile,
        x="half_hour",
        y="demand_mw",
        color="year_label",
        category_orders={"year_label": year_order},
        color_discrete_map=year_colors,
        labels={"half_hour":"Time of day", "demand_mw":"Demand (MW)", "year_label":"Year"},
    )
    style_figure(f2, "", 420)
    f2.update_traces(line={"width": 2.0}, hovertemplate="%{y:,.0f} MW")
    if year_order:
        f2.update_traces(selector={"name": year_order[-1]}, line={"width": 3.0})
    f2.update_layout(legend_title_text="Year")
    tickvals = list(range(0, 48, 4))
    ticktext = [f"{v//2:02d}:00" for v in tickvals]
    f2.update_xaxes(title_text="Time of day", tickvals=tickvals, ticktext=ticktext)
    f2.update_yaxes(title_text="Average demand (MW)")

    interval_count = int(pd.to_numeric(monthly.get("intervals", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    status = (
        f"{REGION_LABELS[region]} • {years_clean[0]}–{years_clean[-1]} • "
        f"{len(years_clean)} year{'s' if len(years_clean) != 1 else ''} selected"
    )
    return f1, f2, status


def _perf_notes(coverage_text, d, mae):
    """Coverage line + a plain-English MAE translation + a mixed-model-version warning."""
    out = []
    if d is not None and not d.empty and np.isfinite(mae):
        typical = float(pd.to_numeric(d["actual_mw"], errors="coerce").mean())
        if typical > 0:
            out.append(html.Span(f"On average the forecast missed by about {mae:,.0f} MW — about {100 * mae / typical:.0f}% of typical demand in this period. ",
                                 className="plain-metric"))
    out.append(html.Span(coverage_text))
    if d is not None and not d.empty and "model_version" in d and d["model_version"].nunique() > 1:
        versions = ", ".join(sorted(map(str, d["model_version"].dropna().unique()))[:4])
        out.append(html.Span([" ", status_chip("stale", "Multiple production model versions exist in this selected period",
                                              f"Versions: {versions}. Metrics combine them and are not attributed to the latest version alone.")]))
    return out


@callback(
    Output("performance-year", "options"),
    Output("performance-year", "value"),
    Input("selected-region", "data"),
    Input("refresh-ui", "n_intervals"),
    State("performance-year", "value"),
)
def performance_year_options(region, _, current):
    opts = _available_performance_years(region)
    values = [o["value"] for o in opts]
    current_year = int(region_local_now(region).year)
    if current in values:
        value = current
    elif current_year in values:
        value = current_year
    else:
        value = values[-1] if values else None
    return opts, value


@callback(
    Output("performance-month", "options"),
    Output("performance-month", "value"),
    Input("selected-region", "data"),
    Input("performance-year", "value"),
    Input("refresh-ui", "n_intervals"),
    State("performance-month", "value"),
)
def performance_month_options(region, year, _, current):
    opts = _available_performance_months(region, year)
    values = [o["value"] for o in opts]
    if current in values:
        value = current
    else:
        value = values[-1] if values else None
    return opts, value


@callback(
    Output("performance-month-control", "style"),
    Input("performance-view", "value"),
)
def performance_month_visibility(view):
    return {"display": "none"} if view == "year" else {"display": "block"}


@callback(
    Output("download-month", "options"),
    Output("download-month", "value"),
    Input("selected-region", "data"),
    Input("refresh-ui", "n_intervals"),
    State("download-month", "value"),
)
def download_month_options(region, _, current):
    opts = _available_performance_months(region)
    values = [o["value"] for o in opts]
    value = current if current in values else (values[-1] if values else None)
    return opts, value


@callback(
    Output("performance-chart", "figure"),
    Output("error-chart", "figure"),
    Output("p-mae", "children"),
    Output("p-rmse", "children"),
    Output("p-smape", "children"),
    Output("p-bias", "children"),
    Output("p-count", "children"),
    Output("performance-coverage", "children"),
    Output("performance-window", "children"),
    Output("performance-main-title", "children"),
    Output("performance-main-subtitle", "children"),
    Output("performance-secondary-title", "children"),
    Output("performance-secondary-subtitle", "children"),
    Input("selected-region", "data"),
    Input("model-switch", "value"),
    Input("performance-view", "value"),
    Input("performance-year", "value"),
    Input("performance-month", "value"),
    Input("refresh-ui", "n_intervals"),
    Input("tabs", "value"),
)
def performance(region, model, view, year_value, month_value, _, tab="performance"):
    if tab != "performance":
        raise PreventUpdate
    if view == "year":
        if year_value is None:
            msg = "No performance year is available yet."
            return (
                empty_figure(msg, 440, "Verification pending"), empty_figure("Monthly sMAPE will appear as evidence accumulates.", 300, "Verification pending"),
                "—", "—", "—", "—", "0", msg, msg,
                "Monthly accuracy by year", "Each point summarizes the combined next-interval performance record for one month.",
                "Monthly sMAPE", "A compact month-by-month accuracy history.",
            )

        year = int(year_value)
        archive = _monthly_accuracy_archive(region, model)
        if not archive.empty:
            archive = archive[archive["month"].str.startswith(str(year))].copy()

        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(microseconds=1)
        latest_row = read_df(
            "SELECT MAX(timestamp) AS latest FROM actual_demand WHERE region=:r AND timestamp>=:start AND timestamp<=:end",
            {"r": region, "start": year_start.to_pydatetime(), "end": year_end.to_pydatetime()},
        )
        latest = pd.to_datetime(latest_row.iloc[0, 0], errors="coerce") if len(latest_row) else pd.NaT
        if pd.isna(latest):
            msg = f"No AEMO actual demand is stored for {year}."
            return (
                empty_figure(msg, 440, "No actual demand stored"), empty_figure("Monthly sMAPE is unavailable.", 300, "No actual demand stored"),
                "—", "—", "—", "—", "0", msg, str(year),
                "Monthly accuracy by year", "No stored evidence yet.", "Monthly sMAPE", "No stored evidence yet.",
            )

        end = min(pd.Timestamp(latest), year_end)
        if year == region_local_now(region).year:
            end = min(end, _region_now(region).floor("30min"))
        d = _operational_tracking(region, model, year_start, end)
        if d.empty or archive.empty:
            msg = f"No verified next-interval forecast evidence is available for {year} yet."
            return (
                empty_figure(msg, 440, "Verification pending"), empty_figure("Monthly sMAPE will appear as months are verified.", 300, "Verification pending"),
                "—", "—", "—", "—", "0", msg, f"{year} year view",
                "Monthly accuracy by year", "Previous months will remain available as the portal accumulates evidence.",
                "Monthly sMAPE", "Month-by-month accuracy history.",
            )

        mae, rmse, smape, bias = _accuracy_metrics(d)
        archive = archive.sort_values("month")
        # Earlier months in forecasting blue; the latest month in bright amber.
        bar_colors = [FORECAST_COLOR] * max(0, len(archive) - 1) + [AEMO_COLOR] if len(archive) else []
        fig = go.Figure([
            go.Bar(
                x=archive["month_label"], y=archive["MAE_MW"], name="Monthly MAE",
                marker={"color": bar_colors, "line": {"width": 0}},
                width=0.56, text=archive["MAE_MW"].round(0), texttemplate="%{text:,.0f}", textposition="outside",
                textfont={"size": 11, "color": TEXT},
                hovertemplate="%{x}<br>MAE: %{y:,.1f} MW<extra></extra>"
            )
        ])
        style_figure(fig, "", 440)
        fig.update_layout(hovermode="closest", bargap=0.28)
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="MAE (MW)")

        err = go.Figure([
            go.Scatter(
                x=archive["month_label"], y=archive["sMAPE_pct"], name="Monthly sMAPE",
                mode="lines+markers", line={"width": 2.6, "color": SMAPE_COLOR}, marker={"size": 8, "color": SMAPE_COLOR, "line": {"width": 1.6, "color": SURFACE}},
                hovertemplate="%{y:.2f}%<extra>sMAPE</extra>",
            )
        ])
        style_figure(err, "", 300)
        err.update_xaxes(title_text="Month")
        err.update_yaxes(title_text="sMAPE (%)")

        first = d["target_timestamp"].min()
        last = d["target_timestamp"].max()
        coverage = _perf_notes(f"{len(d):,} verified intervals • {len(archive)} month(s)", d, mae)
        window = f"{year} performance history • verified targets {first:%d %b %Y %H:%M} → {last:%d %b %Y %H:%M}"
        return (
            fig, err, f"{mae:,.1f}", f"{rmse:,.1f}", f"{smape:.2f}%", f"{bias:+,.1f}", f"{len(d):,}", coverage, window,
            f"Monthly MAE by month — {year}", "Next-30-minute MAE by calendar month; lower is better.",
            f"Monthly sMAPE by month — {year}", "Monthly sMAPE; lower is better.",
        )

    if not month_value:
        msg = "No performance month is available yet."
        return (
            empty_figure(msg, 440, "Verification pending"), empty_figure("Daily MAE will appear after matching intervals are scored.", 300, "Verification pending"),
            "—", "—", "—", "—", "0", msg, msg,
            "Forecast vs actual", "Exact stored next-interval evidence.", "Daily MAE", "Daily error summary.",
        )

    start = pd.Timestamp(f"{month_value}-01")
    next_month = start + pd.offsets.MonthBegin(1)
    latest_row = read_df(
        "SELECT MAX(timestamp) AS latest FROM actual_demand WHERE region=:r AND timestamp>=:start AND timestamp<:next_month",
        {"r": region, "start": start.to_pydatetime(), "next_month": next_month.to_pydatetime()},
    )
    latest = pd.to_datetime(latest_row.iloc[0, 0], errors="coerce") if len(latest_row) else pd.NaT
    if pd.isna(latest):
        msg = f"No AEMO actual demand is stored for {start:%B %Y}."
        return (
            empty_figure(msg, 440, "No actual demand stored"), empty_figure("Daily MAE is unavailable.", 300, "No actual demand stored"),
            "—", "—", "—", "—", "0", msg, msg,
            "Forecast vs actual", "Exact stored next-interval evidence.", "Daily MAE", "Daily error summary.",
        )

    observed_end = min(pd.Timestamp(latest), _region_now(region).floor("30min"), next_month - pd.Timedelta(microseconds=1))
    d = _operational_tracking(region, model, start, observed_end)
    month_name = start.strftime("%B %Y")

    if d.empty:
        msg = (
            f"No next-interval performance record is available for {month_name} yet. "
            "Run the historical replay once for earlier September data, then keep the 30-minute operational cycle running for live verification."
        )
        window = f"{month_name} • actual demand available through {observed_end:%d %b %Y %H:%M}"
        return (
            empty_figure(msg, 440, "Verification pending"), empty_figure("Daily MAE appears after next-interval forecasts are verified.", 300, "Verification pending"),
            "—", "—", "—", "—", "0", msg, window,
            "Forecast vs actual", "Displayed as hourly means; metrics use exact 30-minute matches.",
            "Daily MAE", "Absolute error summarized by day.",
        )

    mae, rmse, smape, bias = _accuracy_metrics(d)
    first_target = d["target_timestamp"].min()
    last_target = d["target_timestamp"].max()

    disp = d.set_index("target_timestamp")[["actual_mw", "forecast_mw"]].resample("1h").mean().dropna(how="all").reset_index()
    fig = go.Figure([
        go.Scatter(x=disp["target_timestamp"], y=disp["actual_mw"], name="Actual demand", line={"width": 2.0, "color": ACTUAL_COLOR},
                   hovertemplate="%{y:,.0f} MW<extra>Actual</extra>"),
        go.Scatter(x=disp["target_timestamp"], y=disp["forecast_mw"], name="Next-30-min forecast", line={"width": 2.0, "color": FORECAST_COLOR},
                   hovertemplate="%{y:,.0f} MW<extra>Forecast</extra>"),
    ])
    style_figure(fig, "", 440)
    fig.update_xaxes(title_text="Operational tracking window", tickformat="%d %b %H:%M", nticks=10)
    fig.update_yaxes(title_text="Demand (MW)")

    daily = d.set_index("target_timestamp").resample("D").agg(daily_mae=("abs_error_mw", "mean")).dropna().reset_index()
    err = go.Figure([go.Bar(
        x=daily["target_timestamp"], y=daily["daily_mae"], name="Daily MAE",
        marker={"color": ERROR_COLOR, "line": {"width": 0}},
        hovertemplate="%{x|%d %b}<br>Daily MAE: %{y:,.1f} MW<extra></extra>"
    )])
    style_figure(err, "", 300)
    err.update_layout(hovermode="closest", showlegend=False, bargap=0.24)
    err.update_xaxes(title_text="Day", tickformat="%d %b", nticks=10)
    err.update_yaxes(title_text="MAE (MW)")

    month_expected = max(1, int(((pd.Timestamp(observed_end).floor("30min") - start).total_seconds() // 1800) + 1))
    month_pct = 100.0 * len(d) / month_expected
    live_n = int((d["provenance"] == "LIVE_VERIFIED").sum()) if "provenance" in d.columns else len(d)
    replay_n = int((d["provenance"] == "HISTORICAL_REPLAY").sum()) if "provenance" in d.columns else 0
    live_rows = d[d["provenance"] == "LIVE_VERIFIED"] if "provenance" in d.columns else pd.DataFrame()
    first_live = live_rows["target_timestamp"].min() if not live_rows.empty else pd.NaT
    if pd.notna(first_live):
        _marker_line(fig, pd.Timestamp(first_live).to_pydatetime(), ERROR_COLOR, "Live verified from here")
    coverage = _perf_notes(
        f"{len(d):,}/{month_expected:,} matches ({month_pct:.1f}%) • "
        f"Replay {replay_n:,} • Live {live_n:,}", d, mae
    )
    provenance_window = (
        f" • live verified from {pd.Timestamp(first_live):%d %b %H:%M}" if pd.notna(first_live) else " • historical replay only so far"
    )
    window = (
        f"{month_name} • {first_target:%d %b %H:%M} → {last_target:%d %b %H:%M}"
        f"{provenance_window}"
    )
    return (
        fig, err, f"{mae:,.1f}", f"{rmse:,.1f}", f"{smape:.2f}%", f"{bias:+,.1f}", f"{len(d):,}", coverage, window,
        f"Next-30-minute forecast vs actual — {month_name}", "Hourly display; metrics use exact 30-minute matches.",
        f"Daily MAE — {month_name}", "Daily next-interval MAE; lower is better.",
    )

@callback(
    Output("download-monthly-comparison", "data"),
    Input("btn-download-monthly-comparison", "n_clicks"),
    State("selected-region", "data"),
    State("model-switch", "value"),
    State("download-month", "value"),
    prevent_initial_call=True,
)
def download_monthly_comparison(n_clicks, region, model, month_value):
    if not n_clicks or not month_value:
        return None
    start = pd.Timestamp(f"{month_value}-01")
    next_month = start + pd.offsets.MonthBegin(1)
    latest_row = read_df(
        "SELECT MAX(timestamp) AS latest FROM actual_demand WHERE region=:r AND timestamp>=:start AND timestamp<:next_month",
        {"r": region, "start": start.to_pydatetime(), "next_month": next_month.to_pydatetime()},
    )
    end = pd.to_datetime(latest_row.iloc[0, 0], errors="coerce") if len(latest_row) else pd.NaT
    if pd.isna(end):
        return None
    d = _operational_tracking(region, model, start, min(pd.Timestamp(end), _region_now(region).floor("30min")))
    if d.empty:
        return None
    export = d[["target_timestamp", "forecast_origin", "model_version", "forecast_mw", "actual_mw", "error_mw", "abs_error_mw", "smape", "provenance"]].copy()
    return dcc.send_data_frame(export.to_csv, f"{region}_{model}_{month_value}_next30min_forecast_vs_actual.csv", index=False)

@callback(
    Output("download-monthly-summary", "data"),
    Input("btn-download-monthly-summary", "n_clicks"),
    State("selected-region", "data"),
    State("model-switch", "value"),
    prevent_initial_call=True,
)
def download_monthly_summary(n_clicks, region, model):
    if not n_clicks:
        return None
    d = _monthly_accuracy_archive(region, model)
    if d.empty:
        return None
    return dcc.send_data_frame(d.to_csv, f"{region}_{model}_monthly_accuracy_archive.csv", index=False)



@callback(
    Output("download-region-pack", "data"),
    Input("btn-download-region-pack", "n_clicks"),
    State("selected-region", "data"),
    State("model-switch", "value"),
    State("download-month", "value"),
    State("download-forecast-horizon", "value"),
    prevent_initial_call=True,
)
def download_region_pack(n_clicks, region, model, month_value, forecast_horizon):
    if not n_clicks:
        return None

    month_value = month_value or region_local_now(region).strftime("%Y-%m")
    start = pd.Timestamp(f"{month_value}-01")
    next_month = start + pd.offsets.MonthBegin(1)
    latest_row = read_df(
        "SELECT MAX(timestamp) AS latest FROM actual_demand WHERE region=:r AND timestamp>=:start AND timestamp<:next_month",
        {"r": region, "start": start.to_pydatetime(), "next_month": next_month.to_pydatetime()},
    )
    observed_end = pd.to_datetime(latest_row.iloc[0, 0], errors="coerce") if len(latest_row) else pd.NaT
    if pd.notna(observed_end):
        observed_end = min(pd.Timestamp(observed_end), _region_now(region).floor("30min"), next_month - pd.Timedelta(microseconds=1))

    comparison = _operational_tracking(region, model, start, observed_end) if pd.notna(observed_end) else pd.DataFrame()
    archive = _monthly_accuracy_archive(region, model)
    aemo_df, project_origin, aemo_origin = _latest_aemo_comparison(region, model)
    rolling = _operational_tracking(region, model, start, observed_end) if pd.notna(observed_end) else pd.DataFrame()

    origin = _latest_origin(region, model)
    forecast_horizon = forecast_horizon if forecast_horizon in {"48H", "7D", "30D", "4M"} else "4M"
    next_target = _next_region_half_hour(region)
    end_target = _forecast_export_end(next_target, forecast_horizon)
    forecast_outlook = pd.DataFrame()
    if pd.notna(origin):
        forecast_outlook = read_df(
            """
            SELECT forecast_origin, target_timestamp, forecast_mw, model_name, model_version, created_at
            FROM project_forecast
            WHERE region=:r AND model_name=:m AND forecast_origin=:o
              AND target_timestamp >= :next_target
              AND target_timestamp <= :end_target
            ORDER BY target_timestamp
            """,
            {
                "r": region,
                "m": model,
                "o": origin.to_pydatetime(),
                "next_target": next_target.to_pydatetime(),
                "end_target": end_target.to_pydatetime(),
            },
        )

    history = _actuals(region)
    monthly_demand = pd.DataFrame()
    daily_profile = pd.DataFrame()
    if not history.empty:
        h = history.copy()
        h["year"] = h["timestamp"].dt.year
        h["month"] = h["timestamp"].dt.month
        monthly_demand = h.groupby(["year", "month"], as_index=False).agg(
            mean_demand_mw=("demand_mw", "mean"),
            peak_demand_mw=("demand_mw", "max"),
            intervals=("demand_mw", "size"),
        )
        h["half_hour"] = h["timestamp"].dt.hour * 2 + (h["timestamp"].dt.minute >= 30).astype(int)
        daily_profile = h.groupby(["year", "half_hour"], as_index=False).agg(mean_demand_mw=("demand_mw", "mean"))

    mae, rmse, smape, bias = _accuracy_metrics(comparison)
    summary_lines = [
        "PRT661 Energy Demand Forecasting - Region Analysis Pack",
        f"Region: {region} - {REGION_LABELS[region]}",
        f"Market: {REGION_MARKET[region]}",
        f"Model: {model}",
        f"Export month: {month_value}",
        f"Portal export time: {_portal_now():%Y-%m-%d %H:%M}",
        f"Rolling forecast export horizon: {forecast_horizon}",
        "",
        "Operational next-interval forecast performance",
        f"Matched intervals: {len(comparison):,}",
        f"MAE MW: {mae:.3f}" if np.isfinite(mae) else "MAE MW: unavailable",
        f"RMSE MW: {rmse:.3f}" if np.isfinite(rmse) else "RMSE MW: unavailable",
        f"sMAPE %: {smape:.3f}" if np.isfinite(smape) else "sMAPE %: unavailable",
        f"Bias MW: {bias:.3f}" if np.isfinite(bias) else "Bias MW: unavailable",
        "",
        "Files in this ZIP are the underlying data used by the portal for further analysis. Operational accuracy uses stored next-30-minute predictions only.",
    ]

    def writer(buf):
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README.txt", "\n".join(summary_lines))
            if not comparison.empty:
                zf.writestr("forecast_vs_actual.csv", comparison.to_csv(index=False))
            if not archive.empty:
                zf.writestr("monthly_accuracy_archive.csv", archive.to_csv(index=False))
            if not rolling.empty:
                zf.writestr("next_interval_tracking.csv", rolling.to_csv(index=False))
            if not aemo_df.empty:
                ae = aemo_df.copy()
                ae.insert(0, "project_forecast_origin", project_origin)
                ae.insert(1, "aemo_forecast_origin", aemo_origin)
                zf.writestr("project_vs_AEMO_predispatch.csv", ae.to_csv(index=False))
            if not forecast_outlook.empty:
                zf.writestr(f"rolling_forecast_{forecast_horizon}.csv", forecast_outlook.to_csv(index=False))
            if not monthly_demand.empty:
                zf.writestr("monthly_actual_demand.csv", monthly_demand.to_csv(index=False))
            if not daily_profile.empty:
                zf.writestr("average_24h_profile.csv", daily_profile.to_csv(index=False))

    safe_model = model.replace("/", "-")
    return dcc.send_bytes(writer, f"{region}_{safe_model}_{month_value}_analysis_pack.zip")


@callback(
    Output("aemo-chart", "figure"),
    Output("a-common", "children"),
    Output("a-mae-gap", "children"),
    Output("a-diff", "children"),
    Output("a-window", "children"),
    Output("aemo-coverage", "children"),
    Input("selected-region", "data"),
    Input("model-switch", "value"),
    Input("refresh-ui", "n_intervals"),
    Input("tabs", "value"),
)
def aemo(region, model, _, tab="aemo"):
    if tab != "aemo":
        raise PreventUpdate
    d, project_origin, aemo_origin = _latest_aemo_comparison(region, model)
    fig = go.Figure()

    if d.empty:
        msg = "A common project/AEMO pre-dispatch forecast window is not currently available for this region. Run the hourly refresh to store the latest AEMO snapshot." if region != "WA" else "AEMO NEM pre-dispatch comparison is not used for WA in this portal."
        fig = empty_figure(
            "No overlapping pre-dispatch observations are currently stored." if region != "WA"
            else "WA is part of the WEM, so NEM pre-dispatch does not apply.",
            460,
            "AEMO comparison unavailable" if region != "WA" else "Not applicable for WA",
        )
        return fig, "0", "—", "—", "—", msg

    fig.add_trace(go.Scatter(x=d["target_timestamp"], y=d["project_mw"], name="Project forecast", line={"width": 2.4, "color": FORECAST_COLOR},
                             hovertemplate="%{y:,.0f} MW<extra>Project</extra>"))
    fig.add_trace(go.Scatter(x=d["target_timestamp"], y=d["aemo_mw"], name="AEMO pre-dispatch (POE50)", line={"width": 2.4, "color": AEMO_COLOR, "dash": "dash"},
                             hovertemplate="%{y:,.0f} MW<extra>AEMO</extra>"))
    style_figure(fig, "", 460)
    fig.update_xaxes(title_text="Forecast target", tickformat="%d %b %H:%M", nticks=10)
    fig.update_yaxes(title_text="Demand forecast (MW)")

    mean_abs_gap = float(d["absolute_difference_mw"].mean())
    mean_diff = float(d["difference_mw"].mean())
    window_hours = (d["target_timestamp"].max() - d["target_timestamp"].min()).total_seconds() / 3600 if len(d) > 1 else 0
    coverage = [
        html.Span([html.Span("AEMO origin", className="info-k"), f"{aemo_origin:%d %b %H:%M}"]),
        html.Span([html.Span("Project origin", className="info-k"), f"{project_origin:%d %b %H:%M}"]),
        html.Span([html.Span("Common window", className="info-k"), f"{d['target_timestamp'].min():%d %b %H:%M} → {d['target_timestamp'].max():%d %b %H:%M}"]),
        html.Span([html.Span("Region", className="info-k"), f"{region} · {REGION_LABELS[region]}"]),
    ]
    return fig, f"{len(d):,}", f"{mean_abs_gap:,.1f}", f"{mean_diff:+,.1f}", f"{window_hours:.0f} h", coverage


@callback(
    Output("download-aemo-comparison", "data"),
    Input("btn-download-aemo", "n_clicks"),
    State("selected-region", "data"),
    State("model-switch", "value"),
    prevent_initial_call=True,
)
def download_aemo_comparison(n_clicks, region, model):
    if not n_clicks:
        return None
    d, project_origin, aemo_origin = _latest_aemo_comparison(region, model)
    if d.empty:
        return None
    export = d.copy()
    export.insert(0, "project_forecast_origin", project_origin)
    export.insert(1, "aemo_forecast_origin", aemo_origin)
    return dcc.send_data_frame(export.to_csv, f"{region}_{model}_project_vs_AEMO_predispatch.csv", index=False)




# =============================================================================
# v3: Forecast Performance -> Model comparison (evidence-driven, research vs production)
# =============================================================================
FAMILY_COLORS = {
    "Baseline": "#A3ADBD", "Tree ensemble": "#15976F", "Gradient boosting": "#2F7BFF",
    "Statistical": "#E98A36", "Neural network": "#7C3AED", "Deep sequence": "#E6399B",
}
FAMILY_PATTERN = {"Baseline": "", "Tree ensemble": "", "Gradient boosting": "", "Statistical": ".",
                  "Neural network": "/", "Deep sequence": "x"}


def _mc_card(question, answer, body=None, detail=None, badge=None, wide=False):
    return html.Section(
        [
            html.Div([html.H3(question, className="mc-q"), badge], className="mc-head"),
            html.P(answer, className="mc-a"),
            body,
            html.Details([html.Summary("Technical detail"), detail], className="disclosure mc-detail") if detail is not None else None,
        ],
        className="mc-card" + (" mc-card--wide" if wide else ""),
    )


def _mini_fig(height=220):
    fig = go.Figure()
    fig.update_layout(height=height, margin={"l": 8, "r": 12, "t": 8, "b": 8}, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"family": FONT_STACK, "size": 12, "color": TEXT},
                      showlegend=False)
    fig.update_xaxes(gridcolor=GRIDLINE, zeroline=False)
    fig.update_yaxes(gridcolor=GRIDLINE, zeroline=False)
    return fig


def _g(fig, sr_text):
    return html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True}, className="mc-graph"),
                     html.P(sr_text, className="sr-only")])


def _simple_table(df, cols, fmt=None):
    fmt = fmt or {}
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(c, scope="col") for c in cols])),
        html.Tbody([html.Tr([html.Td(fmt.get(c, lambda v: v)(r[c])) for c in cols]) for _, r in df.iterrows()]),
    ], className="evidence-table"), className="table-scroll", tabIndex="0")


def _f1(v):
    return f"{v:,.1f}" if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else "—"


def _model_comparison_cards(region, metric="mean_MAE"):
    _, st = load_manifest()
    if st.state == "EVIDENCE_MISSING":
        return [html.Div([html.Strong("Model evidence is not available."),
                          html.Span(" The curated evidence folder data/model_evidence/current/ was not found.")], className="empty-table-state")]
    winners = evidence("regional_model_winners.csv")
    by_region = evidence("model_comparison_by_region.csv")
    sel = ins.selected_model_insight(region, winners)
    if not sel:
        return [html.Div("No selection evidence for this region.", className="empty-table-state")]
    model = sel["model"]
    serving = _serving(region)
    cards = []
    if st.state != "OK":
        cards.append(html.Div(status_chip("stale", "Evidence files could not be fully verified against the manifest",
                                          "; ".join(st.problems[:5])), className="mc-warning"))

    # 1. selected model --------------------------------------------------------
    badges = html.Div([
        html.Span("Selected for production", className="badge badge-good"),
        html.Span("Previous work" if sel.get("development_period") == "PREVIOUS_WORK" else "New after feedback", className="badge badge-neutral"),
        html.Span(sel["confidence"]["label"], className="badge badge-" + ("good" if sel["confidence"]["clear"] else "neutral"),
                  title=sel["confidence"]["rule"]),
    ], className="badge-row")
    serve_note = None
    if serving and serving.mismatch:
        serve_note = html.P([status_chip("stale", f"Currently serving {serving.served}"), " ", serving.message], className="mc-serving")
    cards.append(html.Section([
        html.Div([html.Span("Selected model", className="mc-kicker"),
                  html.H3(f"{model} · {FEATURE_SET_LABEL.get(sel['feature_set'], '').split(' · ')[-1]}", className="mc-title")]),
        badges, html.P(sel["sentence"], className="mc-a"), serve_note,
    ], className="mc-card mc-card--wide mc-card--hero"))

    # 2. ranking ----------------------------------------------------------------
    rank = ins.ranking_frame(region, by_region, evidence("model_failures.csv"), model)
    if not rank.empty:
        r = rank.iloc[::-1]
        val_col = "mean_nMAE" if metric == "mean_nMAE" else "mean_MAE"
        labels = [f"{m}{'  ★ new' if n else ''}{'  (incomplete)' if not c else ''}" for m, n, c in zip(r["model"], r["is_new"], r["complete"])]
        fig = _mini_fig(max(300, 30 * len(r)))
        fig.add_trace(go.Bar(
            x=r[val_col], y=labels, orientation="h",
            marker={"color": [FAMILY_COLORS.get(f, "#999") if c else "#D5DBE5" for f, c in zip(r["family_label"], r["complete"])],
                    "line": {"color": [NAVY if s else "rgba(0,0,0,0)" for s in r["is_selected"]], "width": [3 if s else 0 for s in r["is_selected"]]},
                    "pattern": {"shape": [FAMILY_PATTERN.get(f, "") if c else "/" for f, c in zip(r["family_label"], r["complete"])]}},
            customdata=np.stack([r["mean_RMSE"], r["mean_sMAPE"], r["mean_nMAE"], r["mean_Bias"], r["development_period"], r["family_label"], r["failed_runs"]], axis=1),
            hovertemplate=("<b>%{y}</b><br>%{customdata[5]} · %{customdata[4]}<br>MAE %{x:,.1f}" if val_col == "mean_MAE" else "<b>%{y}</b><br>nMAE %{x:.2f}%")
                          + "<br>RMSE %{customdata[0]:,.1f} MW<br>sMAPE %{customdata[1]:.2f}%<br>nMAE %{customdata[2]:.2f}%<br>Bias %{customdata[3]:,.1f} MW<br>Failed runs %{customdata[6]}<extra></extra>",
        ))
        fig.update_xaxes(title_text="Mean four-month MAE (MW) — lower is better" if val_col == "mean_MAE" else "nMAE (% of typical demand) — lower is better")
        legend = html.Ul([html.Li([html.Span(className="fam-dot", style={"background": c}), f]) for f, c in FAMILY_COLORS.items()],
                         className="fam-legend")
        cards.append(_mc_card(
            "Which models had the lowest historical error?", ins.ranking_sentence(rank, region),
            html.Div([
                dcc.RadioItems(id="mc-metric", options=[{"label": "MAE (MW)", "value": "mean_MAE"}, {"label": "nMAE (%)", "value": "mean_nMAE"}],
                               value=metric, inline=True, className="segmented mc-metric-switch"),
                _g(fig, ins.ranking_sentence(rank, region)), legend,
                html.P("★ new = added after lecturer feedback (research candidate, not deployed). Outlined bar = selected for production. "
                       "Grey hatched = incomplete: some historical runs failed, so it is not comparable.", className="mc-note"),
            ]),
            _simple_table(rank, ["model", "family_label", "development_period", "mean_MAE", "mean_RMSE", "mean_sMAPE", "mean_nMAE", "mean_Bias", "failed_runs"],
                          {c: _f1 for c in ["mean_MAE", "mean_RMSE", "mean_sMAPE", "mean_nMAE", "mean_Bias"]}),
            wide=True))

    # Additional analysis 1: accuracy versus systematic bias for every candidate.
    bias_frame = by_region[by_region["region"] == region].copy() if not by_region.empty else pd.DataFrame()
    if not bias_frame.empty and {"mean_MAE", "mean_Bias", "model"}.issubset(bias_frame.columns):
        bias_frame["mean_MAE"] = pd.to_numeric(bias_frame["mean_MAE"], errors="coerce")
        bias_frame["abs_Bias"] = pd.to_numeric(bias_frame["mean_Bias"], errors="coerce").abs()
        bias_frame = bias_frame.dropna(subset=["mean_MAE", "abs_Bias"])
        if not bias_frame.empty:
            fig = _mini_fig(250)
            fig.add_trace(go.Scatter(
                x=bias_frame["mean_MAE"], y=bias_frame["abs_Bias"], mode="markers+text",
                text=bias_frame["model"], textposition="top center",
                marker={
                    "size": [18 if m == model else 11 for m in bias_frame["model"]],
                    "color": [VERIFIED_GREEN if m == model else FORECAST_BLUE for m in bias_frame["model"]],
                    "line": {"color": [NAVY if m == model else "#FFFFFF" for m in bias_frame["model"]],
                             "width": [2 if m == model else 1 for m in bias_frame["model"]]},
                },
                customdata=np.stack([bias_frame["mean_Bias"]], axis=1),
                hovertemplate="<b>%{text}</b><br>MAE %{x:,.1f} MW<br>|Bias| %{y:,.1f} MW<br>Signed bias %{customdata[0]:+,.1f} MW<extra></extra>",
            ))
            fig.update_xaxes(title_text="Average error — MAE (MW) → lower is better")
            fig.update_yaxes(title_text="Systematic bias — |Bias| (MW) → lower is better")
            cards.append(_mc_card(
                "Accuracy and bias together",
                "Models closest to the lower-left combine smaller average errors with less systematic over- or under-forecasting. The selected production model is highlighted.",
                _g(fig, "Accuracy versus bias for candidate models; lower-left is better."),
            ))

    # Additional analysis 2: selected model robustness across the six regions.
    cross = by_region[by_region["model"] == model].copy() if not by_region.empty else pd.DataFrame()
    if not cross.empty and {"region", "mean_nMAE"}.issubset(cross.columns):
        cross["mean_nMAE"] = pd.to_numeric(cross["mean_nMAE"], errors="coerce")
        cross = cross.dropna(subset=["mean_nMAE"])
        if not cross.empty:
            ordered = [r for r in REGIONS if r in set(cross["region"])]
            cross["region"] = pd.Categorical(cross["region"], categories=ordered, ordered=True)
            cross = cross.sort_values("region")
            fig = _mini_fig(250)
            fig.add_trace(go.Bar(
                x=cross["region"].astype(str), y=cross["mean_nMAE"],
                marker={"color": [NAVY if r == region else REGION_COLORS.get(str(r), FORECAST_BLUE) for r in cross["region"]]},
                text=[f"{v:.1f}%" for v in cross["mean_nMAE"]], textposition="outside",
                hovertemplate="%{x}: %{y:.2f}% of typical demand<extra></extra>",
            ))
            fig.update_yaxes(title_text="nMAE (% of typical demand)")
            cards.append(_mc_card(
                "How robust is this model across regions?",
                "Normalising error by typical demand makes regions easier to compare. Lower bars indicate a smaller error relative to each region's demand scale.",
                _g(fig, "Selected model normalised error across available regions."),
            ))

    # 3. neural ------------------------------------------------------------------
    nv = ins.neural_comparison_insight(region, evidence("neural_vs_previous_comparison.csv"))
    if nv:
        fig = _mini_fig(150)
        fig.add_trace(go.Bar(x=[x["change_pct"] for x in nv["rows"]], y=[x["model"] for x in nv["rows"]], orientation="h",
                             marker={"color": ["#7C3AED", "#E6399B"], "pattern": {"shape": ["/", "x"]}},
                             text=[x["words"] for x in nv["rows"]], textposition="auto",
                             hovertemplate="%{y}: %{text}<extra></extra>"))
        fig.add_vline(x=0, line={"color": NAVY, "width": 1})
        fig.update_xaxes(title_text=f"← lower error · error vs {nv['champion']} (%) · higher error →")
        cards.append(_mc_card("Did deep learning help?", nv["sentence"], _g(fig, nv["sentence"]),
                              _simple_table(pd.DataFrame(nv["rows"]), ["model", "words", "material_improvement", "p_value"],
                                            {"p_value": lambda v: f"{v:.3f}" if pd.notna(v) else "—"}),
                              badge=html.Span("MLP · GRU added after feedback — research only", className="badge badge-neutral")))

    # 4. horizon --------------------------------------------------------------------
    hz = ins.horizon_insight(region, evidence("model_comparison_by_horizon.csv"), model)
    if "pivot" in hz:
        fig = _mini_fig(230)
        for m, color, dash_ in ((model, VERIFIED_GREEN, "solid"), ("MLP", "#7C3AED", "dot"), ("GRU", "#E6399B", "dash")):
            if m in hz["pivot"]:
                fig.add_trace(go.Scatter(x=["Days 1–7", "Days 8–30", "Months 2–4"], y=hz["pivot"][m], name=m, mode="lines+markers",
                                         line={"color": color, "dash": dash_, "width": 2.4}, hovertemplate=m + ": %{y:,.0f} MW<extra></extra>"))
        fig.update_layout(showlegend=True, legend={"orientation": "h", "y": -0.25})
        fig.update_yaxes(title_text="MAE (MW)")
        cards.append(_mc_card("Does performance change further ahead?", hz["sentence"], _g(fig, hz["sentence"])))

    # 5. stability -------------------------------------------------------------------
    stb = ins.stability_insight(region, evidence("model_comparison_by_year.csv"), model)
    f = stb.get("frame", pd.DataFrame())
    if not f.empty:
        fig = _mini_fig(200)
        fig.add_trace(go.Scatter(x=f["year"].astype(str), y=f["rank"], mode="markers+text",
                                 marker={"size": 16, "color": [VERIFIED_GREEN if s else "rgba(0,0,0,0)" for s in f["selection_year"]],
                                         "line": {"color": VERIFIED_GREEN, "width": 2}},
                                 text=[f"#{r}" for r in f["rank"]], textposition="top center",
                                 customdata=np.stack([f["gap_pct"], f["origin_type"]], axis=1),
                                 hovertemplate="%{x}: rank %{y} · %{customdata[0]:.1f}% behind best · %{customdata[1]}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", title_text="Rank (1 = best)", dtick=1)
        cards.append(_mc_card("Was it consistent across years?", stb["sentence"], _g(fig, stb["sentence"]),
                              html.P("Hollow marker: 2022 limited-history backtest (no earlier Sep–Dec history) — reported, not used for selection.", className="mc-note")))

    # 6. weather -----------------------------------------------------------------------
    wx = ins.weather_insight(region, evidence("weather_ablation_metrics.csv"), by_region, sel["feature_set"])
    if wx["bands"]:
        fig = _mini_fig(180)
        fig.add_trace(go.Bar(x=[ins.BAND_LABEL[b].capitalize() for b, _, _ in wx["bands"]], y=[p for _, p, _ in wx["bands"]],
                             marker={"color": [VERIFIED_GREEN if p < 0 else SUNSET for _, p, _ in wx["bands"]]},
                             hovertemplate="%{x}: %{y:+.1f}% error change<extra></extra>"))
        fig.add_hline(y=0, line={"color": NAVY, "width": 1})
        fig.update_yaxes(title_text="Error change with weather (%)<br>below 0 = helped")
        cards.append(_mc_card("Did weather improve the forecast?", wx["sentence"],
                              html.Div([_g(fig, wx["sentence"]), html.P(wx["feature_set_sentence"], className="mc-a"),
                                        html.P(wx["caveat"], className="mc-note")])))

    # 7. drivers ---------------------------------------------------------------------------
    dr = ins.drivers_insight(region, evidence("feature_importance_grouped.csv"))
    if not dr["frame"].empty:
        d = dr["frame"].iloc[::-1]
        fig = _mini_fig(200)
        fig.add_trace(go.Bar(x=100 * d["importance_share"], y=d["driver_group"], orientation="h", marker={"color": FORECAST_BLUE},
                             hovertemplate="%{y}: %{x:.0f}% of importance<extra></extra>"))
        fig.update_xaxes(title_text="Share of model importance (%)")
        top = evidence("feature_importance_top10_selected.csv")
        top = top[top["region"] == region].head(8)
        cards.append(_mc_card("What drives the forecast?", dr["sentence"],
                              html.Div([_g(fig, dr["sentence"]), html.P(dr["caveat"], className="mc-note")]),
                              _simple_table(top, ["feature", "driver_group", "importance"], {"importance": lambda v: f"{v:.3f}"})))

    # 8. peaks -------------------------------------------------------------------------------
    pk = ins.peak_insight(region, evidence("peak_normal_vs_high_summary.csv"), model)
    if not pk["frame"].empty:
        p = pk["frame"]
        fig = _mini_fig(170)
        fig.add_trace(go.Bar(x=["Normal demand", "Highest-demand periods"], y=p["MAE"], marker={"color": [FORECAST_BLUE, SUNSET]},
                             hovertemplate="%{x}: %{y:,.0f} MW average miss<extra></extra>"))
        fig.update_yaxes(title_text="Average miss (MW)")
        tech = evidence("peak_demand_metrics.csv")
        tech = tech[(tech["region"] == region) & (tech["model"] == model) & tech["used_for_selection"].astype(str).str.lower().eq("true")]
        cards.append(_mc_card("Are high-demand periods harder?", pk["sentence"], _g(fig, pk["sentence"]),
                              html.Div([html.P("Technical peak diagnostics — secondary event-detection metrics, not forecast accuracy.", className="mc-note"),
                                        _simple_table(tech, ["backtest_year", "peak_MAE", "peak_Bias", "peak_day_precision", "peak_day_recall", "peak_day_F1"],
                                                      {c: (lambda v: f"{v:.2f}") for c in ["peak_day_precision", "peak_day_recall", "peak_day_F1"]}
                                                      | {"peak_MAE": _f1, "peak_Bias": _f1})])))

    # 9. compact technical evidence -----------------------------------------------------------
    # Keep statistical support and declared sensitivity experiments, but remove the
    # project-journey card and the failure log from the presentation-facing portal.
    stats = evidence("statistical_comparison.csv")
    stats = stats[stats["region"] == region]
    extra = ins.additional_experiments(by_region, region)
    cards.append(html.Details([
        html.Summary([html.Span("Technical evidence", className="disclosure-title"),
                      html.Span("Statistical tests and sensitivity experiments", className="disclosure-sub")]),
        html.H4("Diebold–Mariano tests (selected model vs each other model)", className="method-mini"),
        _simple_table(stats, ["compared_model", "n_days", "mean_diff", "p_value"], {"mean_diff": _f1, "p_value": lambda v: f"{v:.3f}"}),
        html.P("Negative mean difference = the selected model had lower daily error. p-values support the evidence; they do not replace the practical selection rule.", className="mc-note"),
        html.H4("Additional experiments (not production candidates)", className="method-mini"),
        _simple_table(extra, ["model", "role", "mean_MAE"], {"mean_MAE": _f1}) if not extra.empty else html.P("None.", className="mc-note"),
        html.P("This section is intentionally compact for presentation use. Detailed run logs remain in the exported evidence rather than the main portal view.", className="mc-note"),
    ], className="disclosure disclosure--card mc-card--wide"))
    return cards


@callback(
    Output("mc-content", "children"),
    Input("selected-region", "data"),
    Input("performance-mode", "value"),
    Input("tabs", "value"),
)
def model_comparison(region, mode, tab):
    """Evidence-driven comparison. Not tied to refresh-ui: evidence is static and cached."""
    if tab != "performance" or mode != "comparison":
        raise PreventUpdate
    return _model_comparison_cards(region or default_region())


@callback(
    Output("mc-content", "children", allow_duplicate=True),
    Input("mc-metric", "value"),
    State("selected-region", "data"),
    prevent_initial_call=True,
)
def model_comparison_metric(metric, region):
    return _model_comparison_cards(region or default_region(), metric or "mean_MAE")


@callback(
    Output("region-difficulty", "figure"),
    Output("region-difficulty-note", "children"),
    Input("tabs", "value"),
    Input("selected-region", "data"),
)
def region_difficulty(tab, region):
    if tab != "history":
        raise PreventUpdate
    rd = ins.region_difficulty(evidence("regional_model_winners.csv"))
    f = rd["frame"]
    if f.empty:
        return empty_figure("Model evidence not available.", 300), ""
    fig = _mini_fig(300)
    fig.add_trace(go.Bar(x=f["region"], y=f["nMAE_pct"], text=f["best_model"],
                         marker={"color": [NAVY if r == region else REGION_COLORS.get(r, ACCENT) for r in f["region"]]},
                         hovertemplate="%{x}: %{y:.1f}% of typical demand (%{text})<extra></extra>"))
    fig.update_yaxes(title_text="Four-month error, % of typical demand")
    return fig, rd["sentence"] + " This describes forecast difficulty; it does not explain its causes."


@callback(
    Output("evidence-footer", "children"),
    Output("evidence-version-method", "children"),
    Input("url", "pathname"),
)
def evidence_footer(_):
    _, st = load_manifest()
    return f"Portal v{PORTAL_VERSION} · {st.footer()}", st.footer()


def _model_card_markdown(region: str) -> str:
    _, st = load_manifest()
    sel = ins.selected_model_insight(region, evidence("regional_model_winners.csv"))
    serving = _serving(region)
    reg = read_df("SELECT model_name, model_version, training_start, training_end, status, created_at FROM model_registry WHERE region=:r ORDER BY created_at DESC",
                  {"r": region})
    served = serving.served if serving else _default_model(region)
    rr = reg[reg["model_name"] == served].head(1) if not reg.empty else reg
    pk = ins.peak_insight(region, evidence("peak_normal_vs_high_summary.csv"), sel.get("model", served))
    acc = read_df("SELECT AVG(abs_error_mw) AS mae, COUNT(*) AS n, MAX(target_timestamp) AS last_ts FROM next_interval_accuracy WHERE region=:r AND model_name=:m",
                  {"r": region, "m": served})
    lines = [
        f"# Production model card — {region} ({REGION_LABELS[region]})", "",
        f"Portal v{PORTAL_VERSION} · generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC", "",
        "## Model",
        f"- Selected model (evidence): **{sel.get('model', 'not available')}** — feature set {FEATURE_SET_LABEL.get(sel.get('feature_set'), '—')}",
        f"- Served model (live jobs): **{served}**" + (f" — {serving.message}" if serving and serving.mismatch else ""),
        f"- Bundle version: {rr['model_version'].iloc[0] if not rr.empty else 'not recorded in model_registry'}",
        f"- Training window: {rr['training_start'].iloc[0] if not rr.empty else '—'} → {rr['training_end'].iloc[0] if not rr.empty else '—'}", "",
        "## Intended use",
        "Half-hourly operational demand outlook for the current month plus three months; not a peak-warning or dispatch tool.", "",
        "## Selection evidence",
        sel.get("sentence", "Not available."),
        f"Decision confidence: {sel.get('confidence', {}).get('label', '—')} ({sel.get('confidence', {}).get('rule', '')})", "",
        "## Known limitation",
        pk.get("sentence", "Not available."),
        "Historical tests used climatological weather for the forecast period, not realised weather.", "",
        "## Latest verification",
        (f"Next-interval MAE {float(acc['mae'].iloc[0]):,.1f} MW over {int(acc['n'].iloc[0]):,} verified intervals (latest {acc['last_ts'].iloc[0]})."
         if not acc.empty and pd.notna(acc['mae'].iloc[0]) else "No verified next-interval forecasts stored yet."), "",
        "## Evidence version",
        st.footer(),
    ]
    return "\n".join(lines) + "\n"


@callback(
    Output("download-model-card", "data"),
    Input("btn-download-model-card", "n_clicks"),
    State("export-region", "value"),
    prevent_initial_call=True,
)
def download_model_card(n, region):
    if not n:
        raise PreventUpdate
    region = region or default_region()
    return dict(content=_model_card_markdown(region), filename=f"{region}_production_model_card.md")


@callback(
    Output("download-model-evidence", "data"),
    Input("btn-download-model-evidence", "n_clicks"),
    State("export-region", "value"),
    prevent_initial_call=True,
)
def download_model_evidence(n, region):
    if not n:
        raise PreventUpdate
    region = region or default_region()
    d = evidence("model_comparison_by_region.csv")
    d = d[(d["region"] == region) & d["model"].isin(CORE_MODELS)]
    if d.empty:
        raise PreventUpdate
    return dcc.send_data_frame(d.to_csv, f"{region}_ten_model_evidence_2022_2025.csv", index=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)
