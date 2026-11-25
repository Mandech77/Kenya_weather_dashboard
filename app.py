"""Kenya Meteorological Dashboard - built with Gestalt principles of perception.

Run locally:   streamlit run app.py
"""
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import (CONDITION_COLOURS, CONDITION_ORDER, MUTED, STATION_COLOURS,
                        load_clean)

st.set_page_config(page_title="Kenya Weather Dashboard", page_icon="🌦️", layout="wide")

# ----------------------------------------------------------------------------
# Global styling
#   * Figure-ground: a very light grey page (ground) with white cards (figure)
#   * Common region: every chart lives inside a bordered, rounded card
#   * Proximity: consistent, tight spacing inside a card, bigger gaps between cards
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background-color: #F4F6F9; }
      div[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-radius: 12px; }
      h1 { font-size: 1.9rem !important; margin-bottom: 0 !important; }
      h3 { font-size: 1.05rem !important; margin: 0 0 .1rem 0 !important; color: #1F2937; }
      .sub { color: #6B7280; font-size: .9rem; margin: 0 0 .6rem 0; }
      .finding { font-size: .95rem; line-height: 1.5; margin: .25rem 0; color: #1F2937; }
      .finding b { color: #0B3C5D; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    raw, df, log, missing_before = load_clean()
    return raw, df, log, missing_before


raw, df_all, clean_log, missing_before = get_data()
ALL_STATIONS = sorted(df_all["Station"].unique())

# ----------------------------------------------------------------------------
# Sidebar filters (grouped together = proximity / common region)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    chosen = st.multiselect("Stations", ALL_STATIONS, default=ALL_STATIONS)
    dmin, dmax = df_all["Date"].min().date(), df_all["Date"].max().date()
    date_range = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    focus = st.selectbox(
        "Highlight one station",
        ["None"] + (chosen or ALL_STATIONS),
        help="The chosen station stays in colour; the others fade to grey (figure-ground).",
    )
    st.caption("Each station keeps the same colour in every chart.")

if not chosen:
    st.warning("Select at least one station in the sidebar.")
    st.stop()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    d0, d1 = date_range
else:
    d0 = d1 = date_range[0] if isinstance(date_range, (tuple, list)) else date_range

df = df_all[(df_all["Station"].isin(chosen)) &
            (df_all["Date"] >= pd.Timestamp(d0)) & (df_all["Date"] <= pd.Timestamp(d1))]
if df.empty:
    st.warning("No observations in this selection.")
    st.stop()


def colours(stations):
    """Station -> colour. With a focus station, everyone else becomes 'ground' grey."""
    if focus != "None" and focus in stations:
        return {s: (STATION_COLOURS[s] if s == focus else MUTED) for s in stations}
    return {s: STATION_COLOURS[s] for s in stations}


def style(fig, height=330, legend=False):
    """One look for every chart (similarity + Pragnanz: minimal, no clutter)."""
    fig.update_layout(
        template="plotly_white", height=height, showlegend=legend,
        margin=dict(l=10, r=10, t=10, b=10), font=dict(size=12, color="#1F2937"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#9CA3AF")
    fig.update_yaxes(gridcolor="#E5E7EB", zeroline=False)
    return fig


@contextmanager
def card(col, title, subtitle):
    """A bordered card (COMMON REGION) placed in a column."""
    box = col.container(border=True)
    box.markdown(f"### {title}")
    box.markdown(f"<p class='sub'>{subtitle}</p>", unsafe_allow_html=True)
    yield box


# ----------------------------------------------------------------------------
# Station-level numbers used by KPIs and the findings
# ----------------------------------------------------------------------------
by_st = df.groupby("Station").agg(
    temp=("Mean_Temperature_C", "mean"), tmin=("Minimum_Temperature_C", "mean"),
    tmax=("Maximum_Temperature_C", "mean"), rain_total=("Rainfall_mm", "sum"),
    rain_mean=("Rainfall_mm", "mean"), rainy_pct=("Rainy_Day", "mean"),
    hum=("Relative_Humidity_pct", "mean"), wind=("Wind_Speed_kmh", "mean"),
    wind_max=("Wind_Speed_kmh", "max"),
)
order_temp = by_st.sort_values("temp", ascending=False).index.tolist()
hot, wet, windy = by_st["temp"].idxmax(), by_st["rain_total"].idxmax(), by_st["wind"].idxmax()
r_all = df["Relative_Humidity_pct"].corr(df["Rainfall_mm"])
r_by = {s: g["Relative_Humidity_pct"].corr(g["Rainfall_mm"]) for s, g in df.groupby("Station")}

tab_dash, tab_data, tab_gestalt = st.tabs(["📊 Dashboard", "🧹 Data preparation", "🧠 Gestalt design notes"])

# ============================================================================
# TAB 1 - DASHBOARD
# ============================================================================
with tab_dash:
    st.title("Kenya Weather Dashboard")
    st.markdown(
        f"<p class='sub'>{len(chosen)} station(s) · {pd.Timestamp(d0):%d %b %Y} – {pd.Timestamp(d1):%d %b %Y} · "
        f"{len(df):,} daily observations</p>", unsafe_allow_html=True)

    # --- KPI row: the top of the visual hierarchy ---------------------------------
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, note in [
        (k1, "Hottest station", hot, f"{by_st.loc[hot, 'temp']:.1f} °C average"),
        (k2, "Wettest station", wet, f"{by_st.loc[wet, 'rain_total']:,.0f} mm total"),
        (k3, "Windiest station", windy, f"{by_st.loc[windy, 'wind']:.1f} km/h average"),
        (k4, "Humidity ↔ rainfall", f"r = {r_all:.2f}", "daily correlation, all selected"),
    ]:
        with col.container(border=True):
            st.caption(label)
            st.markdown(f"<span style='font-size:1.6rem;font-weight:700;color:#0B3C5D'>{value}</span>",
                        unsafe_allow_html=True)
            st.caption(note)

    # --- Row 1: temperature comparison + monthly trend ----------------------------
    c1, c2 = st.columns([1, 1.25])
    with card(c1, "Temperature by station", "Daily mean temperature (°C); diamond = average. Sorted hottest → coolest.") as box:
        cm = colours(order_temp)
        fig = go.Figure()
        for s in order_temp:
            sub = df[df["Station"] == s]
            fig.add_trace(go.Box(y=sub["Mean_Temperature_C"], name=s, boxmean=True,
                                 marker_color=cm[s], line=dict(color=cm[s]),
                                 fillcolor=cm[s], opacity=0.85 if cm[s] != MUTED else 0.6,
                                 boxpoints=False))
        fig.update_yaxes(title="°C")
        box.plotly_chart(style(fig), width="stretch")

    with card(c2, "Monthly temperature trend", "Monthly average of daily mean temperature. The shaded band is the "
              "average daily min–max range of the highlighted station.") as box:
        monthly = (df.groupby(["Station", "Month"])
                     .agg(t=("Mean_Temperature_C", "mean"), lo=("Minimum_Temperature_C", "mean"),
                          hi=("Maximum_Temperature_C", "mean")).reset_index())
        cm = colours(sorted(chosen))
        fig = go.Figure()
        # grey (ground) lines first so coloured (figure) lines draw on top
        for s in sorted(chosen, key=lambda x: x == focus):
            g = monthly[monthly["Station"] == s]
            is_focus = focus == s
            if is_focus:  # CLOSURE: filled min-max band reads as one closed shape
                fig.add_trace(go.Scatter(x=list(g["Month"]) + list(g["Month"])[::-1],
                                         y=list(g["hi"]) + list(g["lo"])[::-1], fill="toself",
                                         fillcolor="rgba(0,0,0,0.08)", line=dict(width=0),
                                         hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=g["Month"], y=g["t"], mode="lines+markers", name=s,   # CONTINUITY + CONNECTEDNESS
                line=dict(color=cm[s], width=3.5 if is_focus else 2.2, shape="spline", smoothing=0.6),
                marker=dict(size=6, color=cm[s])))
        # direct labels at the line ends (fewer eye movements than a legend), nudged apart so they never overlap
        ends = monthly.sort_values("Month").groupby("Station")["t"].last().sort_values()
        pos, last = {}, -1e9
        for s_, y_ in ends.items():
            last = max(y_, last + 0.75)
            pos[s_] = last
        for s_, y_ in ends.items():
            fig.add_annotation(x=monthly["Month"].max(), y=pos[s_], text=f"  {s_}", showarrow=False, xanchor="left",
                               font=dict(color=cm[s_] if cm[s_] != MUTED else "#6B7280", size=11))
        months = sorted(monthly["Month"].unique())
        fig.update_xaxes(tickvals=months, ticktext=[pd.Timestamp(m).strftime("%b") for m in months],
                         range=[pd.Timestamp(months[0]) - pd.Timedelta(days=10),
                                pd.Timestamp(months[-1]) + pd.Timedelta(days=12)])
        fig.update_yaxes(title="°C")
        fig = style(fig)
        fig.update_layout(margin=dict(l=10, r=80, t=10, b=10))
        box.plotly_chart(fig, width="stretch")

    # --- Row 2: rainfall distribution ----------------------------------------------
    c3, c4 = st.columns([1.25, 1])
    with card(c3, "Rainfall distribution", "Rainfall on rainy days only (mm). Width = how often that amount occurs.") as box:
        cm = colours(order_temp)
        fig = go.Figure()
        for s in sorted(chosen, key=lambda x: by_st.loc[x, "rain_total"], reverse=True):
            sub = df[(df["Station"] == s) & (df["Rainy_Day"])]
            if sub.empty:
                continue
            fig.add_trace(go.Violin(y=sub["Rainfall_mm"], name=s, box_visible=True, meanline_visible=True,
                                    line_color=cm[s], fillcolor=cm[s], opacity=0.75, spanmode="hard",
                                    points=False))
        fig.update_yaxes(title="mm / day")
        box.plotly_chart(style(fig), width="stretch")

    with card(c4, "Share of days by weather condition", "Same station order as the temperature chart; "
              "condition colours go from sand (dry) to deep blue (heavy rain).") as box:
        cond = (df.groupby(["Station", "Weather_Condition"]).size().rename("n").reset_index())
        cond["pct"] = cond["n"] / cond.groupby("Station")["n"].transform("sum") * 100
        fig = px.bar(cond, y="Station", x="pct", color="Weather_Condition", orientation="h",
                     category_orders={"Station": order_temp[::-1], "Weather_Condition": CONDITION_ORDER},
                     color_discrete_map=CONDITION_COLOURS, text=[f"{p:.0f}%" if p >= 6 else "" for p in cond["pct"]])
        fig.update_traces(textposition="inside", textfont_size=10, insidetextanchor="middle")
        fig.update_xaxes(title="% of days", range=[0, 100])
        fig.update_yaxes(title=None)
        fig = style(fig, legend=True, height=340)
        fig.update_layout(bargap=0.25, legend=dict(orientation="h", y=-0.3, x=0, title=None, font=dict(size=11)))
        box.plotly_chart(fig, width="stretch")

    # --- Row 3: humidity & rainfall ---------------------------------------------------
    c5, c6 = st.columns([1.25, 1])
    with card(c5, "Humidity and rainfall", "Each dot is one day. Lines are each station's straight-line trend "
              "(connectedness: they link scattered dots into one pattern).") as box:
        cm = colours(sorted(chosen))
        fig = go.Figure()
        for s in sorted(chosen, key=lambda x: x == focus):
            g = df[df["Station"] == s]
            fig.add_trace(go.Scatter(x=g["Relative_Humidity_pct"], y=g["Rainfall_mm"], mode="markers", name=s,
                                     marker=dict(color=cm[s], size=5, opacity=0.35 if focus == "None" else 0.5)))
            if len(g) > 2:
                m, b = np.polyfit(g["Relative_Humidity_pct"], g["Rainfall_mm"], 1)
                xs = np.array([g["Relative_Humidity_pct"].min(), g["Relative_Humidity_pct"].max()])
                fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", showlegend=False, hoverinfo="skip",
                                         line=dict(color=cm[s], width=3.5 if focus == s else 2.5)))
        fig.update_xaxes(title="Relative humidity (%)")
        fig.update_yaxes(title="Rainfall (mm / day)")
        box.plotly_chart(style(fig, legend=True), width="stretch")

    with card(c6, "Rainfall by humidity band", "Average rainfall and chance of a rainy day, grouped into "
              "10-point humidity bands.") as box:
        bins = [0, 50, 60, 70, 80, 90, 101]
        labels = ["<50", "50–60", "60–70", "70–80", "80–90", "90+"]
        d2 = df.assign(Band=pd.cut(df["Relative_Humidity_pct"], bins=bins, labels=labels, right=False))
        band = d2.groupby("Band", observed=True).agg(rain=("Rainfall_mm", "mean"), rainy=("Rainy_Day", "mean"),
                                                      n=("Rainfall_mm", "size")).reset_index()
        fig = go.Figure(go.Bar(x=band["Band"].astype(str), y=band["rain"], marker_color="#2B6CB0",
                               text=[f"{p:.0%} rainy" for p in band["rainy"]], textposition="outside",
                               customdata=band["n"], hovertemplate="%{x}% humidity<br>%{y:.1f} mm avg<br>%{customdata} days<extra></extra>"))
        fig.update_xaxes(title="Relative humidity band (%)")
        fig.update_yaxes(title="Average rainfall (mm)", range=[0, max(band["rain"].max() * 1.25, 1)])
        box.plotly_chart(style(fig), width="stretch")

    # --- Row 4: wind + findings ----------------------------------------------------------
    c7, c8 = st.columns([1.25, 1])
    with card(c7, "Wind speed by station", "Daily wind speed (km/h); diamond = average. Sorted windiest → calmest.") as box:
        order_wind = by_st.sort_values("wind", ascending=False).index.tolist()
        cm = colours(order_wind)
        fig = go.Figure()
        for s in order_wind:
            sub = df[df["Station"] == s]
            fig.add_trace(go.Box(y=sub["Wind_Speed_kmh"], name=s, boxmean=True, marker_color=cm[s],
                                 line=dict(color=cm[s]), fillcolor=cm[s], opacity=0.85 if cm[s] != MUTED else 0.6,
                                 boxpoints=False))
        fig.update_yaxes(title="km/h")
        box.plotly_chart(style(fig), width="stretch")

    with card(c8, "What the data says", "Updates automatically with your filters.") as box:
        cool = by_st["temp"].idxmin()
        if abs(r_all) < 0.1:
            link = (f"💧 Humidity and rainfall show <b>almost no linear relationship</b> (r = {r_all:.2f}): "
                    f"rainy days occur at every humidity level, so humidity alone does not predict rain here.")
        else:
            strength = "strong" if abs(r_all) >= .5 else "moderate" if abs(r_all) >= .3 else "weak"
            top_r = max(r_by, key=lambda s: abs(r_by[s]))
            link = (f"💧 Humidity and rainfall show a <b>{strength} {'positive' if r_all > 0 else 'negative'}</b> "
                    f"link (r = {r_all:.2f}); it is strongest at <b>{top_r}</b> (r = {r_by[top_r]:.2f}).")
        box.markdown(
            f"<p class='finding'>🌡️ <b>{hot}</b> is the hottest station at <b>{by_st.loc[hot,'temp']:.1f} °C</b> "
            f"on average; <b>{cool}</b> is the coolest at {by_st.loc[cool,'temp']:.1f} °C.</p>"
            f"<p class='finding'>🌧️ <b>{wet}</b> received the most rain "
            f"(<b>{by_st.loc[wet,'rain_total']:,.0f} mm</b>, rain on {by_st.loc[wet,'rainy_pct']:.0%} of days).</p>"
            f"<p class='finding'>{link}</p>"
            f"<p class='finding'>💨 <b>{windy}</b> is the windiest ({by_st.loc[windy,'wind']:.1f} km/h average).</p>",
            unsafe_allow_html=True)

# ============================================================================
# TAB 2 - DATA PREPARATION
# ============================================================================
with tab_data:
    st.subheader("a) Data preparation and exploration")
    st.markdown("**1. Import** – `pd.read_csv(..., parse_dates=['Date'])`")
    st.write(f"Raw shape: **{raw.shape[0]:,} rows × {raw.shape[1]} columns** "
             f"({raw['Date'].min():%d %b %Y} – {raw['Date'].max():%d %b %Y}); "
             f"{raw['Station'].nunique()} stations × 365 days.")
    a, b = st.columns(2)
    with a:
        st.markdown("**2. Structure and data types**")
        st.dataframe(pd.DataFrame({"dtype": raw.dtypes.astype(str), "non-null": raw.notna().sum(),
                                   "missing": raw.isna().sum()}), width="stretch")
    with b:
        st.markdown("**3. Missing values and cleaning**")
        for line in clean_log:
            st.markdown(f"- {line}")
    st.markdown("**Descriptive statistics (after cleaning, all stations)**")
    st.dataframe(df_all.select_dtypes("number").drop(columns=["Latitude", "Longitude"])
                 .describe().T.round(2), width="stretch")
    st.markdown("**4. Station summary (current filter)**")
    summ = by_st.rename(columns={
        "temp": "Mean temp °C", "tmin": "Avg min °C", "tmax": "Avg max °C", "rain_total": "Total rain mm",
        "rain_mean": "Rain mm/day", "rainy_pct": "Rainy days", "hum": "Humidity %", "wind": "Wind km/h",
        "wind_max": "Max wind km/h"}).round(2)
    summ["Rainy days"] = (summ["Rainy days"] * 100).round(1).astype(str) + "%"
    st.dataframe(summ.sort_values("Mean temp °C", ascending=False), width="stretch")

# ============================================================================
# TAB 3 - GESTALT NOTES
# ============================================================================
with tab_gestalt:
    st.subheader("b) How each Gestalt principle is used in this dashboard")
    notes = pd.DataFrame([
        ("Proximity", "KPI cards sit together at the top; each chart sits beside its caption; the filters are grouped in one sidebar.",
         "Related things are read as one group, so the page is scanned in a clear order."),
        ("Similarity", "One fixed colour per station in every chart; one colour ramp for weather conditions; all charts share fonts and styling.",
         "A viewer learns the colour once and can follow a station across charts."),
        ("Continuity", "Smooth monthly lines with direct end labels; regression lines through the scatter.",
         "The eye follows a line and perceives a seasonal trend instead of 12 separate numbers."),
        ("Closure", "The shaded min–max band, box outlines and violin shapes are read as complete forms.",
         "The mind completes the shape, so ranges and spread are seen at a glance."),
        ("Figure-Ground", "White cards on a pale-grey page; 'Highlight one station' keeps one station coloured and fades the rest to grey.",
         "Attention goes to the station being studied while context stays visible."),
        ("Common Region", "Each chart is enclosed in a rounded bordered card with its title.",
         "A boundary tells the viewer which title, chart and caption belong together."),
        ("Connectedness", "Lines join monthly markers and scattered daily dots (trend lines).",
         "Linked elements are perceived as one series, which shows relationships such as humidity ↔ rainfall."),
        ("Prägnanz (Simplicity)", "Plain white theme, light grid, no 3D, no legends where direct labels work, only five station colours.",
         "Simple forms are processed fastest, so the key message is not buried in decoration."),
    ], columns=["Principle", "Where it is applied", "Why it helps interpretation"])
    st.dataframe(notes, width="stretch", hide_index=True)
