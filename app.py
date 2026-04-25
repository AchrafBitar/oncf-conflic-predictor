"""
ONCF — Traffic & Power Conflict Predictor (LGV Tanger ↔ Kenitra)
MVP "Virtual Real-Time" simulation built with Streamlit + Plotly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Static data: LGV Tanger ↔ Kenitra
# ---------------------------------------------------------------------------

LINE_START_PK = 0.0       # Tanger
LINE_END_PK = 126.8       # Kenitra
TRAIN_LENGTH_KM = 0.2     # ~200 m for a US (single) Al Boraq
UM_LENGTH_KM = 0.4        # ~400 m for a UM (coupled)

# Reference stations along the LGV (PK in km).
STATIONS = [
    {"name": "Tanger Ville",    "pk": 0.0},
    {"name": "Asilah",          "pk": 35.0},
    {"name": "Sidi El Yamani",  "pk": 63.0},
    {"name": "Souk El Arbaa",   "pk": 85.0},
    {"name": "Sidi Slimane",    "pk": 100.0},
    {"name": "Kenitra",         "pk": 126.8},
]

# Electrical substation sectors (approximate boundaries on this MVP).
# Each sector has a power ceiling (KVA) used to flag risk when two
# trains overlap inside it.
SUBSTATIONS = [
    {"name": "El Aouama SST1",   "pk_start": 0.0,  "pk_end": 45.0,  "kva_limit": 14500},
    {"name": "Oulad Slama SST2", "pk_start": 45.0, "pk_end": 90.0,  "kva_limit": 14500},
    {"name": "Sidi Allal SST3",  "pk_start": 90.0, "pk_end": 126.8, "kva_limit": 14500},
]

# Power draw model (very rough, for didactic visualisation).
POWER_PER_TRAIN_KVA = {
    ("US", "Normal"): 7800,
    ("US", "Eco"):    6500,
    ("UM", "Normal"): 13500,
    ("UM", "Eco"):    11200,
}

SPEED_KMH = {"Normal": 320.0, "Eco": 290.0}


# ---------------------------------------------------------------------------
# Domain logic
# ---------------------------------------------------------------------------

@dataclass
class Train:
    name: str
    origin_pk: float          # PK at departure
    direction: int            # +1 toward Kenitra, -1 toward Tanger
    departure: datetime
    mode: str                 # "Normal" | "Eco"
    composition: str          # "US" | "UM"
    delay_min: float = 0.0    # reported delay in minutes

    @property
    def speed(self) -> float:
        return SPEED_KMH[self.mode]

    @property
    def length_km(self) -> float:
        return UM_LENGTH_KM if self.composition == "UM" else TRAIN_LENGTH_KM

    def position_at(self, t: datetime, extra_delay_min: float = 0.0) -> float | None:
        """PK of the train head at time t. None if not departed or terminus reached."""
        effective_dep = self.departure + timedelta(minutes=self.delay_min + extra_delay_min)
        if t < effective_dep:
            return None
        elapsed_h = (t - effective_dep).total_seconds() / 3600.0
        pk = self.origin_pk + self.direction * self.speed * elapsed_h
        if pk < LINE_START_PK or pk > LINE_END_PK:
            return None
        return pk


def meeting_pk(a: Train, b: Train, scenario: str = "optimistic") -> tuple[float, datetime] | None:
    """
    Solve for the time t when trains A and B occupy the same PK.
    scenario adjusts the assumed delays:
      - optimistic: both on time (use only reported delay_min as-is)
      - pessimistic: A +2 min, B -2 min (worst case crossing earlier/later)
    Returns (PK, datetime) or None if they never meet on this segment.
    """
    if a.direction == b.direction:
        return None

    delta_a, delta_b = 0.0, 0.0
    if scenario == "pessimistic":
        delta_a, delta_b = +2.0, -2.0

    dep_a = a.departure + timedelta(minutes=a.delay_min + delta_a)
    dep_b = b.departure + timedelta(minutes=b.delay_min + delta_b)

    # Position(t) = origin + dir * speed * (t - dep)
    # Solve a.pos(t) == b.pos(t)  ->  t in hours from a common reference.
    ref = min(dep_a, dep_b)
    ta0 = (dep_a - ref).total_seconds() / 3600.0
    tb0 = (dep_b - ref).total_seconds() / 3600.0

    # a.origin + a.dir*a.speed*(t-ta0) = b.origin + b.dir*b.speed*(t-tb0)
    num = (b.origin_pk - a.origin_pk) - a.direction * a.speed * ta0 + b.direction * b.speed * tb0
    den = b.direction * b.speed - a.direction * a.speed
    if den == 0:
        return None
    t_h = -num / den  # hours from ref

    pk = a.origin_pk + a.direction * a.speed * (t_h - ta0)
    if pk < LINE_START_PK or pk > LINE_END_PK:
        return None

    when = ref + timedelta(hours=t_h)
    if when < dep_a or when < dep_b:
        return None
    return pk, when


def substation_for(pk: float) -> dict | None:
    for s in SUBSTATIONS:
        if s["pk_start"] <= pk <= s["pk_end"]:
            return s
    return None


def power_draw(train: Train) -> int:
    return POWER_PER_TRAIN_KVA.get((train.composition, train.mode), 7000)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def build_map(
    trains: list[Train],
    now: datetime,
    intersection_range: tuple[float, float] | None,
    meeting_point: float | None,
) -> go.Figure:
    fig = go.Figure()

    # Substations as colored bands.
    band_colors = ["rgba(46,134,193,0.18)", "rgba(241,196,15,0.18)", "rgba(155,89,182,0.18)"]
    for i, sst in enumerate(SUBSTATIONS):
        fig.add_shape(
            type="rect",
            x0=sst["pk_start"], x1=sst["pk_end"], y0=-0.6, y1=0.6,
            fillcolor=band_colors[i % len(band_colors)], line_width=0, layer="below",
        )
        fig.add_annotation(
            x=(sst["pk_start"] + sst["pk_end"]) / 2, y=0.55,
            text=f"<b>{sst['name']}</b><br>{sst['kva_limit']} KVA",
            showarrow=False, font=dict(size=11, color="#333"),
        )

    # Track line.
    fig.add_trace(go.Scatter(
        x=[LINE_START_PK, LINE_END_PK], y=[0, 0],
        mode="lines", line=dict(color="#2c3e50", width=4),
        hoverinfo="skip", showlegend=False,
    ))

    # Stations.
    fig.add_trace(go.Scatter(
        x=[s["pk"] for s in STATIONS], y=[0] * len(STATIONS),
        mode="markers+text",
        marker=dict(symbol="square", size=12, color="#34495e"),
        text=[s["name"] for s in STATIONS],
        textposition="bottom center",
        textfont=dict(size=10),
        hovertemplate="%{text}<br>PK %{x:.1f}<extra></extra>",
        showlegend=False,
    ))

    # Danger zone (intersection range).
    if intersection_range is not None:
        lo, hi = intersection_range
        fig.add_shape(
            type="rect",
            x0=lo, x1=hi, y0=-0.35, y1=0.35,
            fillcolor="rgba(231,76,60,0.35)", line=dict(color="#c0392b", width=2),
            layer="below",
        )
        fig.add_annotation(
            x=(lo + hi) / 2, y=0.3,
            text=f"⚠ Danger Zone<br>PK {lo:.1f} – {hi:.1f}",
            showarrow=False, font=dict(size=11, color="#c0392b"),
        )

    if meeting_point is not None:
        fig.add_trace(go.Scatter(
            x=[meeting_point], y=[0],
            mode="markers",
            marker=dict(symbol="x", size=18, color="#c0392b", line=dict(width=2)),
            name="Meeting point",
            hovertemplate=f"Meeting @ PK {meeting_point:.2f}<extra></extra>",
        ))

    # Trains.
    train_colors = {"Train A": "#27ae60", "Train B": "#2980b9"}
    for tr in trains:
        pk = tr.position_at(now)
        if pk is None:
            continue
        y = 0.15 if tr.direction == +1 else -0.15
        fig.add_trace(go.Scatter(
            x=[pk], y=[y],
            mode="markers+text",
            marker=dict(symbol="triangle-right" if tr.direction == 1 else "triangle-left",
                        size=22, color=train_colors.get(tr.name, "#16a085")),
            text=[f"{tr.name} ({tr.composition}/{tr.mode})"],
            textposition="top center" if tr.direction == 1 else "bottom center",
            textfont=dict(size=11, color=train_colors.get(tr.name, "#16a085")),
            hovertemplate=f"{tr.name}<br>PK %{{x:.2f}}<br>{tr.speed:.0f} km/h<extra></extra>",
            showlegend=False,
        ))

    fig.update_xaxes(title="Kilometric Point (PK)", range=[-3, LINE_END_PK + 3], showgrid=True)
    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_layout(
        height=380, margin=dict(l=20, r=20, t=20, b=40),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ONCF Conflict Predictor", layout="wide", page_icon="🚆")

st.title("🚆 ONCF — Traffic & Power Conflict Predictor")
st.caption("LGV Al Boraq · Tanger ↔ Kenitra · Predictive simulation MVP")

with st.sidebar:
    st.header("⚙️ Scenario inputs")

    today = datetime.now().replace(second=0, microsecond=0)

    st.subheader("🟢 Train A — Tanger → Kenitra")
    dep_a_time = st.time_input("Departure A", value=today.time(), key="dep_a")
    mode_a = st.selectbox("Driving mode A", ["Normal", "Eco"], key="mode_a")
    comp_a = st.selectbox("Composition A", ["US", "UM"], key="comp_a")
    delay_a = st.slider("Reported delay A (min)", -5.0, 15.0, 0.0, 0.5, key="delay_a")

    st.markdown("---")

    st.subheader("🔵 Train B — Kenitra → Tanger")
    dep_b_time = st.time_input("Departure B", value=today.time(), key="dep_b")
    mode_b = st.selectbox("Driving mode B", ["Normal", "Eco"], key="mode_b")
    comp_b = st.selectbox("Composition B", ["US", "UM"], key="comp_b")
    delay_b = st.slider("Reported delay B (min)", -5.0, 15.0, 0.0, 0.5, key="delay_b")

    st.markdown("---")
    st.subheader("▶️ Simulation")
    sim_speed = st.slider("Sim speed (×real time)", 1, 240, 60)
    run_sim = st.toggle("Run live simulation", value=False)
    if st.button("Reset clock"):
        st.session_state.pop("sim_clock", None)
        st.session_state.pop("sim_anchor", None)


dep_a_dt = datetime.combine(today.date(), dep_a_time)
dep_b_dt = datetime.combine(today.date(), dep_b_time)

train_a = Train("Train A", origin_pk=LINE_START_PK, direction=+1,
                departure=dep_a_dt, mode=mode_a, composition=comp_a, delay_min=delay_a)
train_b = Train("Train B", origin_pk=LINE_END_PK, direction=-1,
                departure=dep_b_dt, mode=mode_b, composition=comp_b, delay_min=delay_b)

# --- Intersection scenarios ----------------------------------------------
opt = meeting_pk(train_a, train_b, "optimistic")
pes = meeting_pk(train_a, train_b, "pessimistic")

intersection_range = None
meeting_main = None
if opt and pes:
    pks = sorted([opt[0], pes[0]])
    intersection_range = (pks[0], pks[1])
    meeting_main = opt[0]
elif opt:
    meeting_main = opt[0]
    intersection_range = (opt[0] - 2, opt[0] + 2)

# --- Simulation clock -----------------------------------------------------
earliest_dep = min(dep_a_dt, dep_b_dt)
if "sim_clock" not in st.session_state:
    st.session_state.sim_clock = earliest_dep
if "sim_anchor" not in st.session_state:
    st.session_state.sim_anchor = (time.time(), earliest_dep)

if run_sim:
    real_now = time.time()
    real_anchor, sim_anchor = st.session_state.sim_anchor
    elapsed_real = real_now - real_anchor
    st.session_state.sim_clock = sim_anchor + timedelta(seconds=elapsed_real * sim_speed)
else:
    # When paused, advance only when user changes inputs; keep stable.
    st.session_state.sim_anchor = (time.time(), st.session_state.sim_clock)

now = st.session_state.sim_clock

# --- Top KPIs -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sim clock", now.strftime("%H:%M:%S"))
if meeting_main is not None:
    sst = substation_for(meeting_main)
    c2.metric("Predicted meeting PK", f"{meeting_main:.2f}")
    c3.metric("Sector", sst["name"] if sst else "—")
    total_kva = power_draw(train_a) + power_draw(train_b)
    limit = sst["kva_limit"] if sst else 14500
    c4.metric("Combined draw", f"{total_kva} KVA",
              delta=f"{total_kva - limit:+d} vs limit", delta_color="inverse")
else:
    c2.metric("Predicted meeting PK", "—")
    c3.metric("Sector", "—")
    c4.metric("Combined draw", "—")

# --- Map ------------------------------------------------------------------
fig = build_map([train_a, train_b], now, intersection_range, meeting_main)
st.plotly_chart(fig, use_container_width=True)

# --- Warning banner -------------------------------------------------------
if meeting_main is not None:
    sst = substation_for(meeting_main)
    total_kva = power_draw(train_a) + power_draw(train_b)
    if sst and total_kva > sst["kva_limit"]:
        st.error(
            f"⚠ **Power conflict risk** — meeting near PK {meeting_main:.1f} "
            f"inside **{sst['name']}**. Combined draw {total_kva} KVA exceeds "
            f"{sst['kva_limit']} KVA. Suggest *Marche sur l'erre* on Train A."
        )
    elif sst:
        st.success(
            f"✅ Meeting near PK {meeting_main:.1f} in **{sst['name']}** — "
            f"draw {total_kva} KVA within {sst['kva_limit']} KVA limit."
        )
else:
    st.info("Trains do not meet on this segment with the current inputs.")

# --- Scenario table -------------------------------------------------------
st.subheader("📊 Intersection scenarios")
rows = []
for label, res in [("Optimistic (on time)", opt), ("Pessimistic (A +2, B −2)", pes)]:
    if res is None:
        rows.append({"Scenario": label, "Meeting PK": "—", "ETA": "—", "Sector": "—"})
        continue
    pk, when = res
    sst = substation_for(pk)
    rows.append({
        "Scenario": label,
        "Meeting PK": f"{pk:.2f}",
        "ETA": when.strftime("%H:%M:%S"),
        "Sector": sst["name"] if sst else "Outside sector",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- Live refresh ---------------------------------------------------------
if run_sim:
    time.sleep(1.0)
    st.rerun()
