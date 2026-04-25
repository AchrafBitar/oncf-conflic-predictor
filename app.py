"""
ONCF — Prédicteur de Conflits Trafic & Puissance
LGV Al Boraq · Tanger Ville ↔ Kenitra (≈193 km)

S'appuie sur le profil d'éco-conduite réel du rapport EPGV (v01, 01/09/2022)
et sur les deux sous-stations LGV : AOUAMA SST1 (secteur Nord) et
OULAD SLAMA SST2 (secteur Sud).
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
# Données de référence LGV
# ---------------------------------------------------------------------------

LGV_LENGTH_KM = 193.0
US_LENGTH_KM = 0.200
UM_LENGTH_KM = 0.400

# Affichage : taille visuelle d'un train (sinon invisible à l'échelle 193 km)
DISPLAY_TRAIN_KM = {"US": 5.0, "UM": 9.0}

WAYPOINTS = [
    {"name": "Tanger Ville",          "pk": 0.0,   "type": "gare"},
    {"name": "Zone de séparation",    "pk": 11.0,  "type": "zone"},
    {"name": "PCV Aquass Briech",     "pk": 29.5,  "type": "pcv"},
    {"name": "PCV Sidi El Yamani",    "pk": 55.2,  "type": "pcv"},
    {"name": "PCVE Laaouamra",        "pk": 85.7,  "type": "pcv"},
    {"name": "Zone de sectionnement", "pk": 98.4,  "type": "zone"},
    {"name": "PCV Chouaafa",          "pk": 111.4, "type": "pcv"},
    {"name": "PCV Bahara Ouled Ayad", "pk": 136.9, "type": "pcv"},
    {"name": "PCV Benmansour",        "pk": 163.5, "type": "pcv"},
    {"name": "Kenitra",               "pk": 193.0, "type": "gare"},
]

SUBSTATIONS = [
    {
        "name": "AOUAMA SST1",
        "secteur": "Secteur Nord",
        "pk_start": 0.0, "pk_end": 98.4,
        "kva_souscrite": 12000,
        "color": "#3498db",
        "fill": "rgba(52,152,219,0.12)",
    },
    {
        "name": "OULAD SLAMA SST2",
        "secteur": "Secteur Sud",
        "pk_start": 98.4, "pk_end": LGV_LENGTH_KM,
        "kva_souscrite": 14500,
        "color": "#e67e22",
        "fill": "rgba(230,126,34,0.12)",
    },
]

# ---------------------------------------------------------------------------
# Profils de vitesse / puissance (rapport EPGV §VII.2, en vigueur 15/07/2022)
# ---------------------------------------------------------------------------

ECO_PAIR = [
    (0.0,  11.0,  "accel290"),
    (11.0, 60.0,  "cruise290"),
    (60.0, 98.4,  "cruise320"),
    (98.4, 164.0, "cruise290"),
    (164.0, 193.0, "coast"),
]

ECO_IMPAIR = [
    (193.0, 183.5, "accel290"),
    (183.5, 111.0, "cruise290"),
    (111.0, 70.0,  "cruise320"),
    (70.0,  25.0,  "cruise290"),
    (25.0,  0.0,   "coast"),
]

NORMAL_PAIR = [(0.0, 193.0, "cruise320")]
NORMAL_IMPAIR = [(193.0, 0.0, "cruise320")]

PHASE_LABELS_FR = {
    "accel290":  "Accélération → 290",
    "cruise290": "Croisière 290 km/h",
    "accel320":  "Accélération → 320",
    "cruise320": "Croisière 320 km/h",
    "coast":     "Marche sur l'erre",
}

PHASE_COLORS = {
    "accel290":  "#f39c12",
    "cruise290": "#3498db",
    "accel320":  "#e67e22",
    "cruise320": "#2c3e50",
    "coast":     "#27ae60",
}

PHASE_SPEED_KMH = {
    "accel290":  220.0,
    "cruise290": 290.0,
    "accel320":  305.0,
    "cruise320": 320.0,
    "coast":     230.0,
}

# Puissance électrique (kW) absorbée par rame, par phase (η = 0.86, RGV M).
PHASE_POWER_KW_PER_RAME = {
    "accel290":  8500,
    "cruise290": 4843,
    "accel320":  9000,
    "cruise320": 6377,
    "coast":     150,
}


# ---------------------------------------------------------------------------
# Modèle Train
# ---------------------------------------------------------------------------

@dataclass
class Train:
    name: str
    direction: int            # +1 = Pair (T→K), -1 = Impair (K→T)
    departure: datetime
    profile_name: str         # "Éco" | "Normale"
    composition: str          # "US" | "UM"
    delay_min: float = 0.0

    @property
    def length_km(self) -> float:
        return UM_LENGTH_KM if self.composition == "UM" else US_LENGTH_KM

    @property
    def display_length_km(self) -> float:
        return DISPLAY_TRAIN_KM[self.composition]

    @property
    def n_rames(self) -> int:
        return 2 if self.composition == "UM" else 1

    @property
    def origin_pk(self) -> float:
        return 0.0 if self.direction == +1 else LGV_LENGTH_KM

    @property
    def sens_label(self) -> str:
        return "Tanger → Kenitra" if self.direction == +1 else "Kenitra → Tanger"

    def profile(self):
        if self.profile_name == "Normale":
            return NORMAL_PAIR if self.direction == +1 else NORMAL_IMPAIR
        return ECO_PAIR if self.direction == +1 else ECO_IMPAIR

    def effective_departure(self, extra_min: float = 0.0) -> datetime:
        return self.departure + timedelta(minutes=self.delay_min + extra_min)

    def position_at(self, t: datetime, extra_min: float = 0.0):
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
        return None

    def total_trip_minutes(self) -> float:
        h = sum(abs(b - a) / PHASE_SPEED_KMH[p] for a, b, p in self.profile())
        return h * 60.0

    def power_kw_at(self, t: datetime) -> int:
        pos = self.position_at(t)
        if pos is None:
            return 0
        _, phase = pos
        return PHASE_POWER_KW_PER_RAME[phase] * self.n_rames


# ---------------------------------------------------------------------------
# Détection de croisement
# ---------------------------------------------------------------------------

def find_meeting(a: Train, b: Train, scenario="optimiste", step_s=5.0):
    if a.direction == b.direction:
        return None
    if scenario == "optimiste":
        delays = [(0.0, 0.0)]
    else:
        delays = [(+2.0, -2.0), (-2.0, +2.0)]
    best = None
    for da, db in delays:
        r = _scan_meeting(a, b, da, db, step_s)
        if r is None:
            continue
        if best is None or abs(r[0] - best[0]) > 0:
            best = r
    return best


def _scan_meeting(a, b, da, db, step_s):
    earliest = min(a.effective_departure(da), b.effective_departure(db))
    n = int(3 * 3600 / step_s)
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
            frac = prev_diff / (prev_diff - diff) if (prev_diff - diff) else 0.5
            pk = (pa[0] + pb[0]) / 2
            t_cross = prev_t + (t - prev_t) * frac
            return pk, t_cross, pa[1], pb[1]
        prev_diff = diff
        prev_t = t
    return None


def substation_for(pk):
    for sst in SUBSTATIONS:
        if sst["pk_start"] <= pk <= sst["pk_end"]:
            return sst
    return None


def substation_load(trains, t):
    out = {sst["name"]: {"kw": 0, "details": []} for sst in SUBSTATIONS}
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
        out[sst["name"]]["details"].append(
            f"{tr.name} · {PHASE_LABELS_FR[phase]} · {kw} kW"
        )
    return out


def kw_to_kva(kw, pf=0.95):
    return int(round(kw / pf)) if pf > 0 else kw


# ---------------------------------------------------------------------------
# Visualisation : carte de la voie
# ---------------------------------------------------------------------------

TRAIN_BRAND = {
    "Train A (Pair)":   {"primary": "#16a085", "dark": "#0e6655"},
    "Train B (Impair)": {"primary": "#8e44ad", "dark": "#5b2c6f"},
}


def _draw_train_shape(fig, tr, pk, phase, y_center):
    """Dessine un train comme un rectangle coloré avec une pointe directionnelle."""
    L = tr.display_length_km
    color = TRAIN_BRAND[tr.name]["primary"]
    dark = TRAIN_BRAND[tr.name]["dark"]
    phase_color = PHASE_COLORS[phase]

    if tr.direction == +1:
        body_x0, body_x1 = pk - L, pk - L * 0.25
        nose_x0, nose_x1 = pk - L * 0.25, pk
    else:
        body_x0, body_x1 = pk + L * 0.25, pk + L
        nose_x0, nose_x1 = pk, pk + L * 0.25

    h = 0.18

    # Bande de phase (sous le train) — couleur indiquant le mode
    fig.add_shape(type="rect",
                  x0=min(body_x0, nose_x0) - 0.5, x1=max(body_x1, nose_x1) + 0.5,
                  y0=y_center - h - 0.06, y1=y_center - h - 0.02,
                  fillcolor=phase_color, line_width=0)

    # Corps du train
    fig.add_shape(type="rect",
                  x0=body_x0, x1=body_x1,
                  y0=y_center - h, y1=y_center + h,
                  fillcolor=color, line=dict(color=dark, width=2))

    # Pointe / nez (triangle via path)
    if tr.direction == +1:
        path = (f"M {nose_x0},{y_center - h} "
                f"L {nose_x1},{y_center} "
                f"L {nose_x0},{y_center + h} Z")
    else:
        path = (f"M {nose_x1},{y_center - h} "
                f"L {nose_x0},{y_center} "
                f"L {nose_x1},{y_center + h} Z")
    fig.add_shape(type="path", path=path,
                  fillcolor=color, line=dict(color=dark, width=2))

    # Fenêtres (petits rectangles blancs)
    n_windows = 4
    for i in range(n_windows):
        wx0 = body_x0 + (body_x1 - body_x0) * (0.1 + i * 0.22)
        wx1 = wx0 + (body_x1 - body_x0) * 0.12
        fig.add_shape(type="rect",
                      x0=wx0, x1=wx1,
                      y0=y_center + h * 0.15, y1=y_center + h * 0.65,
                      fillcolor="rgba(255,255,255,0.85)", line_width=0)

    # Étiquette nom + vitesse
    speed_text = (f"{int(PHASE_SPEED_KMH[phase])} km/h"
                  if phase != "coast" else "↘ erre")
    fig.add_annotation(
        x=(body_x0 + nose_x1) / 2,
        y=y_center + h + 0.18,
        text=f"<b>{tr.name}</b> · {tr.composition}<br>"
             f"<span style='color:{phase_color}'>{PHASE_LABELS_FR[phase]} ({speed_text})</span>",
        showarrow=False, font=dict(size=11),
        align="center",
    )


def build_track_map(trains, now, intersection_range, meeting):
    fig = go.Figure()

    # Bandes de secteurs
    for sst in SUBSTATIONS:
        fig.add_shape(type="rect",
                      x0=sst["pk_start"], x1=sst["pk_end"],
                      y0=-1.1, y1=1.1,
                      fillcolor=sst["fill"], line_width=0, layer="below")
        fig.add_annotation(
            x=(sst["pk_start"] + sst["pk_end"]) / 2, y=1.0,
            text=f"<b style='color:{sst['color']}'>{sst['secteur']}</b><br>"
                 f"{sst['name']} · PS {sst['kva_souscrite']:,} KVA".replace(",", " "),
            showarrow=False, font=dict(size=12),
        )

    # Frontière de secteur (sectionnement)
    fig.add_shape(type="line", x0=98.4, x1=98.4, y0=-0.9, y1=0.9,
                  line=dict(color="#7f8c8d", width=2, dash="dot"))

    # Voie (deux rails représentés par deux lignes)
    for off in (-0.04, 0.04):
        fig.add_trace(go.Scatter(
            x=[0, LGV_LENGTH_KM], y=[off, off],
            mode="lines", line=dict(color="#34495e", width=3),
            showlegend=False, hoverinfo="skip",
        ))

    # Traverses (petits traits gris) — espacement régulier
    for pk in range(0, int(LGV_LENGTH_KM) + 1, 5):
        fig.add_shape(type="line", x0=pk, x1=pk, y0=-0.07, y1=0.07,
                      line=dict(color="#95a5a6", width=1), layer="below")

    # Points de repère (gares + PCV + zones)
    for w in WAYPOINTS:
        if w["type"] == "gare":
            sym, sz, col = "square", 16, "#c0392b"
        elif w["type"] == "zone":
            sym, sz, col = "diamond", 11, "#7f8c8d"
        else:
            sym, sz, col = "circle", 8, "#34495e"
        fig.add_trace(go.Scatter(
            x=[w["pk"]], y=[0], mode="markers",
            marker=dict(symbol=sym, size=sz, color=col,
                        line=dict(color="white", width=2)),
            hovertemplate=f"<b>{w['name']}</b><br>PK {w['pk']:.1f}<extra></extra>",
            showlegend=False,
        ))
        # Étiquette en bas pour les gares, plus discrète pour les PCV
        if w["type"] == "gare":
            fig.add_annotation(x=w["pk"], y=-0.55,
                               text=f"<b>{w['name']}</b><br>PK {w['pk']:.0f}",
                               showarrow=False, font=dict(size=12, color=col))
        else:
            fig.add_annotation(x=w["pk"], y=-0.42,
                               text=f"{w['name']}<br><i>PK {w['pk']:.1f}</i>",
                               showarrow=False, font=dict(size=9, color="#7f8c8d"))

    # Zone de danger
    if intersection_range:
        lo, hi = intersection_range
        fig.add_shape(type="rect",
                      x0=lo, x1=hi, y0=-0.9, y1=0.9,
                      fillcolor="rgba(231,76,60,0.18)",
                      line=dict(color="#c0392b", width=2, dash="dash"),
                      layer="below")
        fig.add_annotation(
            x=(lo + hi) / 2, y=-0.85,
            text=f"⚠ <b>Zone de danger</b> · PK {lo:.1f} – {hi:.1f}",
            showarrow=False,
            font=dict(size=12, color="#c0392b"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#c0392b", borderwidth=1, borderpad=4,
        )

    # Point de croisement
    if meeting is not None:
        fig.add_trace(go.Scatter(
            x=[meeting], y=[0], mode="markers",
            marker=dict(symbol="x-thin", size=22,
                        color="#c0392b", line=dict(width=4, color="#c0392b")),
            hovertemplate=f"<b>Croisement</b> @ PK {meeting:.2f}<extra></extra>",
            showlegend=False,
        ))

    # Trains
    for tr in trains:
        pos = tr.position_at(now)
        if pos is None:
            continue
        pk, phase = pos
        y = 0.40 if tr.direction == +1 else -0.40
        _draw_train_shape(fig, tr, pk, phase, y)

    fig.update_xaxes(
        title=dict(text="<b>PK (km)</b>", font=dict(size=13)),
        range=[-5, LGV_LENGTH_KM + 5],
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
        tickmode="linear", dtick=20,
    )
    fig.update_yaxes(visible=False, range=[-1.2, 1.3])
    fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=20, b=40),
        plot_bgcolor="#fdfdfd", paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Marche graphique (espace-temps)
# ---------------------------------------------------------------------------

def build_marche_graphique(trains, now, meeting_pt, meeting_t):
    fig = go.Figure()

    earliest = min(t.effective_departure() for t in trains)
    latest = max(t.effective_departure() + timedelta(minutes=t.total_trip_minutes() + 5)
                 for t in trains)

    for sst in SUBSTATIONS:
        fig.add_shape(type="rect",
                      x0=earliest, x1=latest,
                      y0=sst["pk_start"], y1=sst["pk_end"],
                      fillcolor=sst["fill"], line_width=0, layer="below")
        fig.add_annotation(
            x=earliest, y=(sst["pk_start"] + sst["pk_end"]) / 2,
            text=f"<b style='color:{sst['color']}'>{sst['secteur']}</b>",
            showarrow=False, xanchor="left", font=dict(size=11),
            xshift=8,
        )

    for tr in trains:
        xs, ys, hovers = [], [], []
        dep = tr.effective_departure()
        elapsed = 0.0
        xs.append(dep)
        ys.append(tr.origin_pk)
        hovers.append("Départ")
        for pk_a, pk_b, phase in tr.profile():
            seg_h = abs(pk_b - pk_a) / PHASE_SPEED_KMH[phase]
            elapsed += seg_h
            xs.append(dep + timedelta(hours=elapsed))
            ys.append(pk_b)
            hovers.append(PHASE_LABELS_FR[phase])
        color = TRAIN_BRAND[tr.name]["primary"]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color, line=dict(color="white", width=2)),
            name=f"{tr.name} ({tr.composition} · {tr.profile_name})",
            customdata=hovers,
            hovertemplate="<b>%{customdata}</b><br>%{x|%H:%M:%S}<br>PK %{y:.1f}<extra></extra>",
        ))

    fig.add_vline(x=now, line=dict(color="#7f8c8d", dash="dash", width=2),
                  annotation_text="Maintenant", annotation_position="top")

    if meeting_pt is not None and meeting_t is not None:
        fig.add_trace(go.Scatter(
            x=[meeting_t], y=[meeting_pt], mode="markers",
            marker=dict(symbol="x-thin", size=20,
                        color="#c0392b", line=dict(width=4, color="#c0392b")),
            name="Croisement",
            hovertemplate=f"<b>Croisement</b><br>PK {meeting_pt:.2f}<br>%{{x|%H:%M:%S}}<extra></extra>",
        ))

    fig.update_yaxes(title="<b>PK (km)</b>", range=[-2, LGV_LENGTH_KM + 2],
                     gridcolor="rgba(0,0,0,0.05)")
    fig.update_xaxes(title="<b>Heure</b>", gridcolor="rgba(0,0,0,0.05)")
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.18, xanchor="center", x=0.5),
        plot_bgcolor="#fdfdfd", paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ONCF · Prédicteur de Conflits LGV",
    layout="wide", page_icon="🚆",
    initial_sidebar_state="expanded",
)

# CSS personnalisé
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
h1 { color: #1a5276; font-weight: 700; }
.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1rem; border-radius: 10px;
    border-left: 4px solid #1a5276;
    margin-bottom: 0.5rem;
}
.train-legend {
    display: inline-block; width: 14px; height: 14px;
    border-radius: 3px; margin-right: 6px; vertical-align: middle;
}
.phase-pill {
    display: inline-block; padding: 2px 10px;
    border-radius: 12px; font-size: 12px; color: white;
    font-weight: 600; margin: 2px;
}
[data-testid="stMetricValue"] { font-size: 1.7rem; }
[data-testid="stMetricLabel"] { font-weight: 600; }
.stProgress > div > div > div { background-color: #16a085; }
</style>
""", unsafe_allow_html=True)

# En-tête
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("# 🚆")
with col_title:
    st.markdown("# Prédicteur de Conflits LGV — ONCF")
    st.markdown("**Al Boraq · Tanger Ville ↔ Kenitra** &nbsp;·&nbsp; "
                "Basé sur l'éco-conduite EPGV (v01 — 01/09/2022)")

st.divider()

# ---------- Barre latérale ----------
with st.sidebar:
    st.markdown("## ⚙️ Paramètres de simulation")

    today = datetime.now().replace(second=0, microsecond=0)

    st.markdown("### 🟢 Train A — Pair")
    st.caption("Sens : Tanger → Kenitra")
    dep_a = st.time_input("Heure de départ", value=today.time(), key="dep_a")
    prof_a = st.selectbox("Profil de conduite", ["Éco", "Normale"], key="prof_a",
                          help="Éco = profil en paliers 290/320/marche sur l'erre. "
                               "Normale = 320 km/h constants (référence).")
    comp_a = st.selectbox("Composition", ["US", "UM"], key="comp_a",
                          help="US = 1 rame (200 m). UM = 2 rames couplées (400 m).")
    delay_a = st.slider("Retard signalé (min)", -5.0, 15.0, 0.0, 0.5, key="del_a")

    st.markdown("---")
    st.markdown("### 🟣 Train B — Impair")
    st.caption("Sens : Kenitra → Tanger")
    dep_b = st.time_input("Heure de départ", value=today.time(), key="dep_b")
    prof_b = st.selectbox("Profil de conduite", ["Éco", "Normale"], key="prof_b")
    comp_b = st.selectbox("Composition", ["US", "UM"], key="comp_b")
    delay_b = st.slider("Retard signalé (min)", -5.0, 15.0, 0.0, 0.5, key="del_b")

    st.markdown("---")
    st.markdown("### ▶️ Lecture")
    sim_speed = st.slider("Vitesse simulation (× réel)", 1, 240, 60)
    run_sim = st.toggle("Lancer la simulation", value=False)
    if st.button("🔄 Réinitialiser l'horloge", use_container_width=True):
        st.session_state.pop("sim_clock", None)
        st.session_state.pop("sim_anchor", None)


# ---------- Construction des trains ----------
dep_a_dt = datetime.combine(today.date(), dep_a)
dep_b_dt = datetime.combine(today.date(), dep_b)

train_a = Train("Train A (Pair)",   direction=+1, departure=dep_a_dt,
                profile_name=prof_a, composition=comp_a, delay_min=delay_a)
train_b = Train("Train B (Impair)", direction=-1, departure=dep_b_dt,
                profile_name=prof_b, composition=comp_b, delay_min=delay_b)

# Horloge
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

# ---------- Calculs ----------
opt = find_meeting(train_a, train_b, "optimiste")
pes = find_meeting(train_a, train_b, "pessimiste")

intersection_range = None
meeting_main = meeting_t_main = None
if opt:
    meeting_main, meeting_t_main, *_ = opt
    if pes:
        pks = sorted([opt[0], pes[0]])
        intersection_range = (pks[0] - 1, pks[1] + 1)
    else:
        intersection_range = (meeting_main - 3, meeting_main + 3)


# ---------- Bandeau KPI ----------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("⏱️ Horloge simulation", now.strftime("%H:%M:%S"))
with k2:
    if meeting_main is not None:
        st.metric("📍 PK de croisement", f"{meeting_main:.1f}",
                  help="Point kilométrique où les deux trains se croisent (scénario optimiste).")
    else:
        st.metric("📍 PK de croisement", "—")
with k3:
    if meeting_main is not None:
        sst = substation_for(meeting_main)
        st.metric("⚡ Secteur électrique",
                  sst["secteur"] if sst else "—",
                  help=sst["name"] if sst else None)
    else:
        st.metric("⚡ Secteur électrique", "—")
with k4:
    if meeting_t_main is not None:
        st.metric("🕐 Heure de croisement", meeting_t_main.strftime("%H:%M:%S"))
    else:
        st.metric("🕐 Heure de croisement", "—")

st.markdown(" ")

# ---------- Onglets principaux ----------
tab1, tab2, tab3 = st.tabs([
    "🗺️ Carte de la voie",
    "📉 Marche graphique",
    "📊 Détails & scénarios",
])

with tab1:
    st.markdown(
        f"<div style='text-align:center;color:#7f8c8d;margin-bottom:0.5rem'>"
        f"<span class='train-legend' style='background:{TRAIN_BRAND['Train A (Pair)']['primary']}'></span>"
        f"Train A (Pair · {comp_a} · {prof_a}) &nbsp;&nbsp;"
        f"<span class='train-legend' style='background:{TRAIN_BRAND['Train B (Impair)']['primary']}'></span>"
        f"Train B (Impair · {comp_b} · {prof_b})"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_track_map([train_a, train_b], now, intersection_range, meeting_main),
        use_container_width=True,
    )

    # Légende des phases
    st.markdown("**Légende des phases de conduite :**")
    cols = st.columns(len(PHASE_COLORS))
    for c, (phase, color) in zip(cols, PHASE_COLORS.items()):
        with c:
            st.markdown(
                f"<span class='phase-pill' style='background:{color}'>"
                f"{PHASE_LABELS_FR[phase]}</span>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### ⚡ Charge des sous-stations — instant courant")
    loads = substation_load([train_a, train_b], now)
    cols = st.columns(len(SUBSTATIONS))
    for col, sst in zip(cols, SUBSTATIONS):
        info = loads[sst["name"]]
        kva = kw_to_kva(info["kw"])
        limit = sst["kva_souscrite"]
        pct = (kva / limit * 100) if limit else 0
        delta = kva - limit
        with col:
            st.markdown(
                f"<div class='metric-card' style='border-left-color:{sst['color']}'>"
                f"<div style='font-size:14px;color:#7f8c8d'>{sst['secteur']}</div>"
                f"<div style='font-size:16px;font-weight:600;color:{sst['color']}'>"
                f"{sst['name']}</div>"
                f"<div style='font-size:28px;font-weight:700;margin-top:6px'>"
                f"{kva:,} <span style='font-size:14px;color:#7f8c8d'>KVA</span></div>"
                f"<div style='color:{'#c0392b' if delta > 0 else '#27ae60'};font-weight:600'>"
                f"{'+' if delta>=0 else ''}{delta:,} vs PS {limit:,} KVA</div>"
                f"</div>".replace(",", " "),
                unsafe_allow_html=True,
            )
            st.progress(min(pct / 100, 1.0))
            if info["details"]:
                for d in info["details"]:
                    st.caption("• " + d)
            else:
                st.caption("Aucun train dans ce secteur.")

    # Bandeau de risque
    st.markdown(" ")
    if meeting_main is not None:
        sst = substation_for(meeting_main)
        if meeting_t_main is not None:
            worst_kw = train_a.power_kw_at(meeting_t_main) + train_b.power_kw_at(meeting_t_main)
        else:
            worst_kw = 0
        worst_kva = kw_to_kva(worst_kw)
        if sst and worst_kva > sst["kva_souscrite"]:
            st.error(
                f"### ⚠️ Risque de conflit de puissance\n"
                f"Croisement prévu vers **PK {meeting_main:.1f}** dans le "
                f"**{sst['secteur']}** ({sst['name']}).\n\n"
                f"Appel combiné estimé à l'instant du croisement : "
                f"**≈ {worst_kva:,} KVA**, soit **{worst_kva - sst['kva_souscrite']:,} KVA "
                f"au-dessus** de la puissance souscrite ({sst['kva_souscrite']:,} KVA).\n\n"
                f"💡 *Action recommandée :* passer le train Pair en **marche sur l'erre** "
                f"plus tôt, ou décaler la phase d'accélération vers 320 km/h pour éviter "
                f"que les deux manipulateurs de traction soient en position MAX simultanément "
                f"(cf. rapport EPGV §VII.2).".replace(",", " ")
            )
        elif sst:
            st.success(
                f"### ✅ Croisement sûr\n"
                f"Croisement prévu vers **PK {meeting_main:.1f}** dans le **{sst['secteur']}**.\n\n"
                f"Appel combiné ≈ **{worst_kva:,} KVA**, dans la limite de la puissance "
                f"souscrite ({sst['kva_souscrite']:,} KVA).".replace(",", " ")
            )
    else:
        st.info("Les deux trains ne se croisent pas sur la LGV avec les paramètres actuels.")

with tab2:
    st.markdown(
        "Le **graphique espace-temps** montre la trajectoire de chaque train. "
        "Le ✕ rouge marque le croisement prévu."
    )
    st.plotly_chart(
        build_marche_graphique([train_a, train_b], now, meeting_main, meeting_t_main),
        use_container_width=True,
    )

with tab3:
    st.markdown("### 📊 Scénarios de croisement")
    rows = []
    for label, res in [("Optimiste (retards déclarés)", opt),
                       ("Pessimiste (±2 min d'écart)", pes)]:
        if res is None:
            rows.append({"Scénario": label, "PK croisement": "—",
                         "Heure": "—", "Secteur": "—",
                         "Phase Train A": "—", "Phase Train B": "—"})
            continue
        pk, when, pa_phase, pb_phase = res
        sst = substation_for(pk)
        rows.append({
            "Scénario": label,
            "PK croisement": f"{pk:.2f}",
            "Heure": when.strftime("%H:%M:%S"),
            "Secteur": sst["secteur"] if sst else "Hors secteur",
            "Phase Train A": PHASE_LABELS_FR[pa_phase],
            "Phase Train B": PHASE_LABELS_FR[pb_phase],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 🚆 Caractéristiques des trains")
    cara_rows = []
    for tr in [train_a, train_b]:
        cara_rows.append({
            "Train": tr.name,
            "Sens": tr.sens_label,
            "Composition": f"{tr.composition} ({tr.n_rames} rame{'s' if tr.n_rames>1 else ''})",
            "Profil": tr.profile_name,
            "Départ effectif": tr.effective_departure().strftime("%H:%M:%S"),
            "Durée trajet (min)": f"{tr.total_trip_minutes():.1f}",
        })
    st.dataframe(pd.DataFrame(cara_rows), use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Modèle utilisé (rapport EPGV)"):
        st.markdown("""
**Profils d'éco-conduite** (rapport EPGV v01, §VII.2 — appliqué le 15/07/2022) :

- **Pair (Tanger → Kenitra) :** accélération vers 290 → croisière 290 jusqu'au
  PK 60 → croisière 320 jusqu'au PK 98+404 (sectionnement) → croisière 290
  jusqu'au PK 164 (PCV Benmansour) → marche sur l'erre jusqu'à Kenitra.
- **Impair (Kenitra → Tanger) :** accélération vers 290 → croisière 290 jusqu'au
  PK 111 (PCV Chouaafa) → croisière 320 jusqu'au PK 70 → croisière 290 jusqu'au
  PK 25 → marche sur l'erre jusqu'à Tanger Ville.

**Puissance par rame** (η = 0,86 ; RGV M, 2 motrices + 8 remorques) :
- croisière 320 km/h ≈ 6,4 MW
- croisière 290 km/h ≈ 4,8 MW
- accélération ≈ 8,5–9 MW
- marche sur l'erre ≈ 0 (auxiliaires uniquement)

**Sous-stations LGV** :
- AOUAMA SST1 — Secteur Nord, PS **12 000 KVA**
- OULAD SLAMA SST2 — Secteur Sud, PS **14 500 KVA**
- 40 MVA installés chacune. Frontière de secteur = sectionnement PK 98+404.

**Incident référencé** : pic de **17 490 KVA** sur OULAD SLAMA les 08–09/07/2022
lors de croisements UM × US, à l'origine du nouveau plan d'éco-conduite.
""")

if run_sim:
    time.sleep(1.0)
    st.rerun()
