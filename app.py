"""
ONCF — Traffic & Power Conflict Predictor
LGV Al Boraq · Tanger Ville ↔ Kenitra (≈193 km)

Models the real eco-conduite speed profile from the EPGV report (v01, 01/09/2022)
and the two LGV substations AOUAMA SST1 (sector Nord) and OULAD SLAMA SST2
(sector Sud), to predict where two opposing Al Boraq trains will meet and whether
their combined power draw risks exceeding the puissance souscrite.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# LGV reference data (extracted from the EPGV eco-conduite report and the
# ONCF "Carte Réseau à jour" map).
# ---------------------------------------------------------------------------

LGV_LENGTH_KM = 193.0            # Tanger Ville (PK 0) → Kenitra (PK ≈193)
US_LENGTH_KM = 0.200             # 200 m — RGV M single rame
UM_LENGTH_KM = 0.400             # 400 m — two coupled rames

# PCV / sectioning waypoints (from Annexe 1 of the report).
WAYPOINTS = [
    {"name": "Tanger Ville",          "pk": 0.0},
    {"name": "Zone séparation",       "pk": 11.0},
    {"name": "PCV Aquass Briech",     "pk": 29.5},
    {"name": "PCV Sidi El Yamani",    "pk": 55.2},
    {"name": "PCVE Laaouamra",        "pk": 85.7},
    {"name": "Zone sectionnement",    "pk": 98.4},
    {"name": "PCV Chouaafa",          "pk": 111.4},
    {"name": "PCV Bahara Ouled Ayad", "pk": 136.9},
    {"name": "PCV Benmansour",        "pk": 163.5},
    {"name": "Kenitra",               "pk": 193.0},
]

# Two LGV-feeding substations.
# Sector boundary set at the PK 98+404 sectioning zone.
SUBSTATIONS = [
    {
        "name": "AOUAMA SST1 (Nord)",
        "pk_start": 0.0,
        "pk_end": 98.4,
        "kva_souscrite": 12000,
        "mva_installed": 40,
    },
    {
        "name": "OULAD SLAMA SST2 (Sud)",
        "pk_start": 98.4,
        "pk_end": LGV_LENGTH_KM,
        "kva_souscrite": 14500,
        "mva_installed": 40,
    },
]

# ---------------------------------------------------------------------------
# Speed/power profile model.
#
# A profile is a list of segments, ordered along the train's direction of
# travel.  Each segment carries:
#   pk_a / pk_b : start/end PK in track coordinates (a < b for Pair, a > b for
#                 Impair direction)
#   v_kmh       : cruise speed for the segment, or None for marche sur l'erre
#   phase       : "accel290" | "cruise290" | "accel320" | "cruise320" | "coast"
#
# The eco-conduite profiles below come straight from §VII.2 of the report
# (mise à jour 15/07/2022).  "Marche normale" is the original RED3 / 320 km/h
# profile used as the worst-case baseline.
# ---------------------------------------------------------------------------

ECO_PAIR = [   # Tanger → Kenitra
    (0.0,  11.0,  "accel290"),   # accel from start
    (11.0, 60.0,  "cruise290"),
    (60.0, 98.4,  "cruise320"),  # incl. accel to 320 just after PK 60
    (98.4, 164.0, "cruise290"),
    (164.0, 193.0, "coast"),     # marche sur l'erre to Kenitra
]

ECO_IMPAIR = [  # Kenitra → Tanger (PK decreasing)
    (193.0, 183.5, "accel290"),
    (183.5, 111.0, "cruise290"),
    (111.0, 70.0,  "cruise320"),
    (70.0,  25.0,  "cruise290"),
    (25.0,  0.0,   "coast"),
]

NORMAL_PAIR = [(0.0, 193.0, "cruise320")]
NORMAL_IMPAIR = [(193.0, 0.0, "cruise320")]

PHASE_SPEED_KMH = {
    "accel290":  220.0,   # average speed during accel ramp (rough)
    "cruise290": 290.0,
    "accel320":  305.0,
    "cruise320": 320.0,
    "coast":     230.0,   # average decay during marche sur l'erre
}

# Per-rame electrical power (kW) absorbed by the traction chain.
# Cruise figures from §III.3 of the report (η = 0.86).
PHASE_POWER_KW_PER_RAME = {
    "accel290":  8500,    # near Pmax during ramp-up
    "cruise290": 4843,
    "accel320":  9000,
    "cruise320": 6377,
    "coast":     150,     # auxiliaries only — traction cut
}


# ---------------------------------------------------------------------------
# Train model
# ---------------------------------------------------------------------------

@dataclass
class Train:
    name: str
    direction: int          # +1 = Pair (T→K), -1 = Impair (K→T)
    departure: datetime
    profile_name: str       # "Eco" | "Normale"
    composition: str        # "US" | "UM"
    delay_min: float = 0.0

    @property
    def length_km(self) -> float:
        return UM_LENGTH_KM if self.composition == "UM" else US_LENGTH_KM

    @property
    def n_rames(self) -> int:
        return 2 if self.composition == "UM" else 1

    @property
    def origin_pk(self) -> float:
        return 0.0 if self.direction == +1 else LGV_LENGTH_KM

    def profile(self) -> list[tuple[float, float, str]]:
        if self.profile_name == "Normale":
            return NORMAL_PAIR if self.direction == +1 else NORMAL_IMPAIR
        return ECO_PAIR if self.direction == +1 else ECO_IMPAIR

    def effective_departure(self, extra_min: float = 0.0) -> datetime:
        return self.departure + timedelta(minutes=self.delay_min + extra_min)

    def position_at(self, t: datetime, extra_min: float = 0.0) -> tuple[float, str] | None:
        """
        Integrate the speed profile to find where the train head is at time t.
        Returns (PK, phase) or None if the train hasn't departed or has reached
        its terminus.
        """
        dep = self.effective_departure(extra_min)
        if t < dep:
            return None
        elapsed_h = (t - dep).total_seconds() / 3600.0

        for pk_a, pk_b, phase in self.profile():
            seg_len_km = abs(pk_b - pk_a)
            v = PHASE_SPEED_KMH[phase]
            seg_dur_h = seg_len_km / v
            if elapsed_h <= seg_dur_h:
                pk = pk_a + math.copysign(elapsed_h * v, pk_b - pk_a)
                return pk, phase
            elapsed_h -= seg_dur_h

        return None  # past terminus

    def total_trip_minutes(self) -> float:
        h = 0.0
        for pk_a, pk_b, phase in self.profile():
            h += abs(pk_b - pk_a) / PHASE_SPEED_KMH[phase]
        return h * 60.0

    def power_kw_at(self, t: datetime) -> int:
        pos = self.position_at(t)
        if pos is None:
            return 0
        _, phase = pos
        return PHASE_POWER_KW_PER_RAME[phase] * self.n_rames


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def find_meeting(a: Train, b: Train, scenario: str = "optimistic",
                 step_s: float = 5.0) -> tuple[float, datetime, str, str] | None:
    """
    Numerically scan a 3 h window for the moment trains A and B share the
    same PK.  Profiles are piecewise so an analytical solve is brittle —
    a fine time scan is more robust and still quick.

    scenario:
      - "optimistic"  : both trains run with their declared delays
      - "pessimistic" : A is +2 min late, B is -2 min early (or vice-versa) —
                        we pick the combination that maximises crossing PK
                        spread relative to the optimistic case.
    """
    if a.direction == b.direction:
        return None

    if scenario == "optimistic":
        delays = [(0.0, 0.0)]
    else:
        delays = [(+2.0, -2.0), (-2.0, +2.0)]

    best = None
    for da, db in delays:
        result = _scan_meeting(a, b, da, db, step_s)
        if result is None:
            continue
        if best is None:
            best = result
            continue
        # Pick the case furthest from the optimistic crossing.
        if abs(result[0] - best[0]) > 0:
            best = result
    return best


def _scan_meeting(a: Train, b: Train, da: float, db: float, step_s: float):
    earliest = min(a.effective_departure(da), b.effective_departure(db))
    horizon_s = 3 * 3600
    n = int(horizon_s / step_s)
    prev_diff = None
    prev_t = earliest
    for i in range(n + 1):
        t = earliest + timedelta(seconds=i * step_s)
        pa = a.position_at(t, extra_min=da)
        pb = b.position_at(t, extra_min=db)
        if pa is None or pb is None:
            prev_diff = None
            prev_t = t
            continue
        diff = pa[0] - pb[0]
        if prev_diff is not None and prev_diff * diff <= 0:
            # Linear interpolation to refine.
            frac = prev_diff / (prev_diff - diff) if (prev_diff - diff) != 0 else 0.5
            pk = (pa[0] + pb[0]) / 2
            t_cross = prev_t + (t - prev_t) * frac
            return pk, t_cross, pa[1], pb[1]
        prev_diff = diff
        prev_t = t
    return None


def substation_for(pk: float) -> dict | None:
    for sst in SUBSTATIONS:
        if sst["pk_start"] <= pk <= sst["pk_end"]:
            return sst
    return None


def substation_load(trains: list[Train], t: datetime) -> dict[str, dict]:
    """Sum live kW draw of trains within each substation's PK window."""
    out = {sst["name"]: {"kw": 0, "trains": []} for sst in SUBSTATIONS}
    for tr in trains:
        pos = tr.position_at(t)
        if pos is None:
            continue
        pk, phase = pos
        sst = substation_for(pk)
        if sst is None:
            continue
        kw = PHASE_POWER_KW_PER_RAME[phase] * tr.n_rames
        out[sst["name"]]["kw"] += kw
        out[sst["name"]]["trains"].append(f"{tr.name}:{phase}({kw} kW)")
    return out


def kw_to_kva(kw: int, pf: float = 0.95) -> int:
    """Rough apparent-power conversion. The report shows reactive ≈ 12% of active."""
    if pf <= 0:
        return kw
    return int(round(kw / pf))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

TRAIN_COLOR = {"Train A (Pair)": "#27ae60", "Train B (Impair)": "#2980b9"}


def build_track_map(trains, now, intersection_range, meeting):
    fig = go.Figure()

    sst_colors = ["rgba(46,134,193,0.22)", "rgba(241,196,15,0.22)"]
    for i, sst in enumerate(SUBSTATIONS):
        fig.add_shape(type="rect",
                      x0=sst["pk_start"], x1=sst["pk_end"], y0=-0.7, y1=0.7,
                      fillcolor=sst_colors[i], line_width=0, layer="below")
        fig.add_annotation(x=(sst["pk_start"] + sst["pk_end"]) / 2, y=0.6,
                           text=f"<b>{sst['name']}</b><br>PS {sst['kva_souscrite']} KVA",
                           showarrow=False, font=dict(size=11, color="#222"))

    fig.add_trace(go.Scatter(
        x=[0, LGV_LENGTH_KM], y=[0, 0],
        mode="lines", line=dict(color="#2c3e50", width=4),
        showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=[w["pk"] for w in WAYPOINTS], y=[0] * len(WAYPOINTS),
        mode="markers+text",
        marker=dict(symbol="line-ns", size=14, color="#34495e", line=dict(width=2)),
        text=[w["name"] for w in WAYPOINTS],
        textposition="bottom center", textfont=dict(size=9),
        hovertemplate="%{text}<br>PK %{x:.1f}<extra></extra>",
        showlegend=False,
    ))

    if intersection_range:
        lo, hi = intersection_range
        fig.add_shape(type="rect",
                      x0=lo, x1=hi, y0=-0.4, y1=0.4,
                      fillcolor="rgba(231,76,60,0.35)",
                      line=dict(color="#c0392b", width=2), layer="below")
        fig.add_annotation(x=(lo + hi) / 2, y=0.45,
                           text=f"⚠ Danger Zone PK {lo:.1f}–{hi:.1f}",
                           showarrow=False, font=dict(size=11, color="#c0392b"))

    if meeting is not None:
        fig.add_trace(go.Scatter(
            x=[meeting], y=[0], mode="markers",
            marker=dict(symbol="x", size=18, color="#c0392b", line=dict(width=2)),
            name="Meeting point",
            hovertemplate=f"Crossing @ PK {meeting:.2f}<extra></extra>",
        ))

    for tr in trains:
        pos = tr.position_at(now)
        if pos is None:
            continue
        pk, phase = pos
        y = 0.18 if tr.direction == 1 else -0.18
        sym = "triangle-right" if tr.direction == 1 else "triangle-left"
        fig.add_trace(go.Scatter(
            x=[pk], y=[y], mode="markers+text",
            marker=dict(symbol=sym, size=24,
                        color=TRAIN_COLOR.get(tr.name, "#16a085")),
            text=[f"{tr.name}<br>{tr.composition} · {phase}"],
            textposition="top center" if tr.direction == 1 else "bottom center",
            textfont=dict(size=10, color=TRAIN_COLOR.get(tr.name, "#16a085")),
            hovertemplate=f"{tr.name}<br>PK %{{x:.2f}}<br>{phase}<extra></extra>",
            showlegend=False,
        ))

    fig.update_xaxes(title="PK (km)", range=[-3, LGV_LENGTH_KM + 3], showgrid=True)
    fig.update_yaxes(visible=False, range=[-1.0, 1.0])
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=40),
                      plot_bgcolor="#fafafa", paper_bgcolor="white")
    return fig


def build_marche_graphique(trains, now, meeting_pt, meeting_t):
    """
    Time-distance diagram (graphique de circulation): time on X, PK on Y.
    Each train is a polyline; their crossing is the meeting point.
    """
    fig = go.Figure()

    # Substation horizontal bands.
    sst_colors = ["rgba(46,134,193,0.18)", "rgba(241,196,15,0.18)"]
    earliest = min(t.effective_departure() for t in trains)
    latest = max(t.effective_departure() + timedelta(minutes=t.total_trip_minutes() + 5)
                 for t in trains)
    for i, sst in enumerate(SUBSTATIONS):
        fig.add_shape(type="rect",
                      x0=earliest, x1=latest,
                      y0=sst["pk_start"], y1=sst["pk_end"],
                      fillcolor=sst_colors[i], line_width=0, layer="below")

    # Train trajectories — sample profile.
    for tr in trains:
        xs, ys, phases = [], [], []
        dep = tr.effective_departure()
        elapsed = 0.0
        xs.append(dep)
        ys.append(tr.origin_pk)
        phases.append("start")
        for pk_a, pk_b, phase in tr.profile():
            seg_h = abs(pk_b - pk_a) / PHASE_SPEED_KMH[phase]
            elapsed += seg_h
            xs.append(dep + timedelta(hours=elapsed))
            ys.append(pk_b)
            phases.append(phase)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=TRAIN_COLOR.get(tr.name, "#16a085"), width=3),
            marker=dict(size=6),
            name=f"{tr.name} ({tr.composition}/{tr.profile_name})",
            hovertemplate="%{x|%H:%M:%S}<br>PK %{y:.1f}<extra></extra>",
        ))

    # "Now" cursor.
    fig.add_vline(x=now, line=dict(color="#7f8c8d", dash="dash", width=1))

    # Meeting marker.
    if meeting_pt is not None and meeting_t is not None:
        fig.add_trace(go.Scatter(
            x=[meeting_t], y=[meeting_pt], mode="markers",
            marker=dict(symbol="x", size=16, color="#c0392b", line=dict(width=2)),
            name="Crossing",
            hovertemplate=f"Crossing @ PK {meeting_pt:.2f}<br>%{{x|%H:%M:%S}}<extra></extra>",
        ))

    fig.update_yaxes(title="PK (km)", range=[-2, LGV_LENGTH_KM + 2])
    fig.update_xaxes(title="Time")
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=40),
                      legend=dict(orientation="h", y=-0.15),
                      plot_bgcolor="#fafafa")
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ONCF LGV Conflict Predictor",
                   layout="wide", page_icon="🚆")

st.title("🚆 ONCF — LGV Traffic & Power Conflict Predictor")
st.caption("Al Boraq · Tanger Ville ↔ Kenitra · "
           "based on EPGV eco-conduite (v01 — 01/09/2022)")

with st.sidebar:
    st.header("⚙️ Inputs")

    today = datetime.now().replace(second=0, microsecond=0)

    st.subheader("🟢 Train A — Pair (T→K)")
    dep_a = st.time_input("Departure A", value=today.time(), key="dep_a")
    prof_a = st.selectbox("Profile A", ["Eco", "Normale"], key="prof_a",
                          help="Eco = piecewise 290/320/coast (report §VII.2). "
                               "Normale = constant 320 km/h baseline.")
    comp_a = st.selectbox("Composition A", ["US", "UM"], key="comp_a")
    delay_a = st.slider("Reported delay A (min)", -5.0, 15.0, 0.0, 0.5)

    st.markdown("---")

    st.subheader("🔵 Train B — Impair (K→T)")
    dep_b = st.time_input("Departure B", value=today.time(), key="dep_b")
    prof_b = st.selectbox("Profile B", ["Eco", "Normale"], key="prof_b")
    comp_b = st.selectbox("Composition B", ["US", "UM"], key="comp_b")
    delay_b = st.slider("Reported delay B (min)", -5.0, 15.0, 0.0, 0.5)

    st.markdown("---")
    st.subheader("▶️ Live simulation")
    sim_speed = st.slider("Sim speed (× real time)", 1, 240, 60)
    run_sim = st.toggle("Run simulation", value=False)
    if st.button("Reset clock"):
        st.session_state.pop("sim_clock", None)
        st.session_state.pop("sim_anchor", None)


dep_a_dt = datetime.combine(today.date(), dep_a)
dep_b_dt = datetime.combine(today.date(), dep_b)

train_a = Train("Train A (Pair)", direction=+1, departure=dep_a_dt,
                profile_name=prof_a, composition=comp_a, delay_min=delay_a)
train_b = Train("Train B (Impair)", direction=-1, departure=dep_b_dt,
                profile_name=prof_b, composition=comp_b, delay_min=delay_b)

# Sim clock
earliest = min(dep_a_dt, dep_b_dt)
if "sim_clock" not in st.session_state:
    st.session_state.sim_clock = earliest
if "sim_anchor" not in st.session_state:
    st.session_state.sim_anchor = (time.time(), earliest)

if run_sim:
    real_now = time.time()
    real_anchor, sim_anchor = st.session_state.sim_anchor
    st.session_state.sim_clock = sim_anchor + timedelta(
        seconds=(real_now - real_anchor) * sim_speed)
else:
    st.session_state.sim_anchor = (time.time(), st.session_state.sim_clock)

now = st.session_state.sim_clock

# Intersection scenarios
opt = find_meeting(train_a, train_b, "optimistic")
pes = find_meeting(train_a, train_b, "pessimistic")

intersection_range = None
meeting_main = meeting_t_main = None
if opt:
    meeting_main, meeting_t_main, *_ = opt
    if pes:
        pks = sorted([opt[0], pes[0]])
        intersection_range = (pks[0], pks[1])
    else:
        intersection_range = (meeting_main - 3, meeting_main + 3)

# --- KPI strip -----------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sim clock", now.strftime("%H:%M:%S"))
if meeting_main is not None:
    sst = substation_for(meeting_main)
    c2.metric("Crossing PK", f"{meeting_main:.2f}")
    c3.metric("Sector", sst["name"] if sst else "—")
    if opt:
        c4.metric("Crossing ETA", meeting_t_main.strftime("%H:%M:%S"))
else:
    c2.metric("Crossing PK", "—")
    c3.metric("Sector", "—")
    c4.metric("Crossing ETA", "—")

# --- Track map -----------------------------------------------------------
st.subheader("🗺️ Track map (live)")
st.plotly_chart(
    build_track_map([train_a, train_b], now, intersection_range, meeting_main),
    use_container_width=True,
)

# --- Substation gauges --------------------------------------------------
st.subheader("⚡ Substation load — right now")
loads = substation_load([train_a, train_b], now)
gcols = st.columns(len(SUBSTATIONS))
risk_overall = False
for col, sst in zip(gcols, SUBSTATIONS):
    info = loads[sst["name"]]
    kva = kw_to_kva(info["kw"])
    limit = sst["kva_souscrite"]
    pct = (kva / limit * 100) if limit else 0
    delta = kva - limit
    col.metric(
        sst["name"],
        f"{kva} KVA",
        delta=f"{delta:+d} vs PS ({limit})",
        delta_color="inverse",
    )
    col.progress(min(pct / 100, 1.0))
    if info["trains"]:
        col.caption(" · ".join(info["trains"]))
    if delta > 0:
        risk_overall = True

# --- Meeting / conflict banner ------------------------------------------
if meeting_main is not None:
    sst = substation_for(meeting_main)
    # Estimate worst-case combined draw if both at peak in this sector.
    worst_kw = max(train_a.power_kw_at(meeting_t_main), 0) + \
               max(train_b.power_kw_at(meeting_t_main), 0)
    worst_kva = kw_to_kva(worst_kw)
    if sst and worst_kva > sst["kva_souscrite"]:
        st.error(
            f"⚠ **Power conflict risk** — crossing near PK {meeting_main:.1f} "
            f"inside **{sst['name']}**. Combined draw at crossing instant ≈ "
            f"{worst_kva} KVA, which exceeds the {sst['kva_souscrite']} KVA "
            f"souscrite. Suggest *marche sur l'erre* on the Pair train, or "
            f"shift its accel-to-320 PK so its traction phase doesn't overlap "
            f"with the Impair train (cf. report §VII.2)."
        )
    elif sst:
        st.success(
            f"✅ Crossing near PK {meeting_main:.1f} in **{sst['name']}** — "
            f"combined draw {worst_kva} KVA within {sst['kva_souscrite']} KVA "
            f"souscrite."
        )
else:
    st.info("Trains do not cross on the LGV with the current inputs.")

# --- Marche graphique ---------------------------------------------------
st.subheader("📉 Marche graphique (time-distance)")
st.plotly_chart(
    build_marche_graphique([train_a, train_b], now, meeting_main, meeting_t_main),
    use_container_width=True,
)

# --- Scenario table -----------------------------------------------------
st.subheader("📊 Crossing scenarios")
rows = []
for label, res in [("Optimistic (declared delays)", opt),
                   ("Pessimistic (±2 min spread)", pes)]:
    if res is None:
        rows.append({"Scenario": label, "Crossing PK": "—",
                     "ETA": "—", "Sector": "—",
                     "Phase A": "—", "Phase B": "—"})
        continue
    pk, when, pa_phase, pb_phase = res
    sst = substation_for(pk)
    rows.append({
        "Scenario": label,
        "Crossing PK": f"{pk:.2f}",
        "ETA": when.strftime("%H:%M:%S"),
        "Sector": sst["name"] if sst else "Outside",
        "Phase A": pa_phase,
        "Phase B": pb_phase,
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("ℹ️ Speed/power model used"):
    st.markdown("""
**Eco-conduite profiles** (from EPGV report v01, §VII.2 — applied 15/07/2022):

- **Pair (Tanger → Kenitra):** accel to 290 → cruise 290 to PK 60 → cruise 320
  to PK 98+404 (sectioning) → cruise 290 to PK 164 (PCV Benmansour) → marche
  sur l'erre to Kenitra.
- **Impair (Kenitra → Tanger):** accel to 290 → cruise 290 to PK 111 (PCV
  Chouaafa) → cruise 320 to PK 70 → cruise 290 to PK 25 → marche sur l'erre
  to Tanger Ville.

**Per-rame power** (η = 0.86, RGV M, 2 motrice + 8 remorques):
cruise 320 ≈ 6.4 MW · cruise 290 ≈ 4.8 MW · accel ≈ 8.5–9 MW · coast ≈ 0.

**Substations:** AOUAMA SST1 (PS 12 000 KVA) and OULAD SLAMA SST2
(PS 14 500 KVA), 40 MVA installé chacune. Sector boundary = sectionnement
PK 98+404. Reported peak overload event: 17 490 KVA on OULAD SLAMA on
08–09/07/2022 during UM × US crossings.
    """)

if run_sim:
    time.sleep(1.0)
    st.rerun()
