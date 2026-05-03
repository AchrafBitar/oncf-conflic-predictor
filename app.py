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
from dataclasses import dataclass, replace
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
# Énergie & analyse économique
# ---------------------------------------------------------------------------

def train_energy_kwh(train: Train) -> float:
    """Énergie absorbée par le train sur un trajet complet (kWh)."""
    total = 0.0
    for pk_a, pk_b, phase in train.profile():
        seg_h = abs(pk_b - pk_a) / PHASE_SPEED_KMH[phase]
        total += PHASE_POWER_KW_PER_RAME[phase] * train.n_rames * seg_h
    return total


def phase_at_pk(train: Train, pk: float):
    """Phase du train quand sa tête se trouve au PK indiqué."""
    for pk_a, pk_b, phase in train.profile():
        lo, hi = min(pk_a, pk_b), max(pk_a, pk_b)
        if lo - 1e-6 <= pk <= hi + 1e-6:
            return phase
    return None


def crossing_power_profile(a: Train, b: Train, step_km=1.0):
    """
    Pour chaque PK candidat, donne l'appel combiné en KVA si les deux
    rames se rencontraient à ce point. Permet de visualiser les zones
    « chères » de la ligne.
    """
    pks, kvas = [], []
    pk = 0.0
    while pk <= LGV_LENGTH_KM + 1e-6:
        pa = phase_at_pk(a, pk)
        pb = phase_at_pk(b, pk)
        if pa is not None and pb is not None:
            kw = (PHASE_POWER_KW_PER_RAME[pa] * a.n_rames
                  + PHASE_POWER_KW_PER_RAME[pb] * b.n_rames)
            pks.append(pk)
            kvas.append(kw_to_kva(kw))
        pk += step_km
    return pks, kvas


def scan_optimal_delay(a: Train, b: Train,
                       extra_range=(-5.0, 10.0), step=0.5):
    """
    Balaye un retard supplémentaire appliqué au train Pair pour trouver
    celui qui minimise l'appel combiné au croisement.
    """
    candidates = []
    d = extra_range[0]
    while d <= extra_range[1] + 1e-9:
        a_try = replace(a, delay_min=a.delay_min + d)
        res = find_meeting(a_try, b, "optimiste")
        if res is not None:
            pk, when, pa_phase, pb_phase = res
            kw = (PHASE_POWER_KW_PER_RAME[pa_phase] * a.n_rames
                  + PHASE_POWER_KW_PER_RAME[pb_phase] * b.n_rames)
            candidates.append({
                "delay": round(d, 2), "pk": pk, "when": when,
                "phase_a": pa_phase, "phase_b": pb_phase,
                "kw": kw, "kva": kw_to_kva(kw),
            })
        d += step
    return candidates


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

    h = 0.16

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

    # Étiquette nom + phase, dans une boîte blanche pour la lisibilité.
    label_y = y_center + h + 0.22 if tr.direction == +1 else y_center - h - 0.22
    fig.add_annotation(
        x=(body_x0 + nose_x1) / 2,
        y=label_y,
        text=f"<b style='color:{dark}'>{tr.name}</b> "
             f"<span style='color:#7f8c8d'>· {tr.composition}</span><br>"
             f"<span style='color:{phase_color};font-weight:600'>"
             f"● {PHASE_LABELS_FR[phase]}</span>",
        showarrow=False, font=dict(size=11, color="#2c3e50"),
        align="center",
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=color, borderwidth=1, borderpad=4,
    )


def build_track_map(trains, now, intersection_range, meeting):
    fig = go.Figure()

    # Bandes de secteurs
    for sst in SUBSTATIONS:
        fig.add_shape(type="rect",
                      x0=sst["pk_start"], x1=sst["pk_end"],
                      y0=-1.3, y1=1.3,
                      fillcolor=sst["fill"], line_width=0, layer="below")
        ps_txt = f"{sst['kva_souscrite']:,}".replace(",", " ")
        fig.add_annotation(
            x=(sst["pk_start"] + sst["pk_end"]) / 2, y=1.15,
            text=f"<b>{sst['secteur']}</b> &nbsp;·&nbsp; "
                 f"{sst['name']} &nbsp;·&nbsp; PS {ps_txt} KVA",
            showarrow=False,
            font=dict(size=12, color=sst["color"], family="sans-serif"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=sst["color"], borderwidth=1, borderpad=6,
        )

    # Frontière de secteur (sectionnement) — ligne verticale traversant tout
    fig.add_shape(type="line", x0=98.4, x1=98.4, y0=-1.4, y1=1.1,
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

    # Points de repère (gares + PCV + zones).
    # Les étiquettes des PCV/zones alternent entre deux niveaux verticaux
    # pour éviter le chevauchement aux abords du sectionnement.
    intermediates = [w for w in WAYPOINTS if w["type"] != "gare"]
    levels = {id(w): (-1.05 if i % 2 == 0 else -1.35)
              for i, w in enumerate(intermediates)}

    for w in WAYPOINTS:
        if w["type"] == "gare":
            sym, sz, col = "square", 16, "#c0392b"
        elif w["type"] == "zone":
            sym, sz, col = "diamond", 11, "#5d6d7e"
        else:
            sym, sz, col = "circle", 8, "#34495e"
        fig.add_trace(go.Scatter(
            x=[w["pk"]], y=[0], mode="markers",
            marker=dict(symbol=sym, size=sz, color=col,
                        line=dict(color="white", width=2)),
            hovertemplate=f"<b>{w['name']}</b><br>PK {w['pk']:.1f}<extra></extra>",
            showlegend=False,
        ))
        if w["type"] == "gare":
            fig.add_annotation(x=w["pk"], y=-1.20,
                               text=f"<b>{w['name']}</b><br>PK {w['pk']:.0f}",
                               showarrow=False,
                               font=dict(size=13, color=col),
                               bgcolor="rgba(255,255,255,0.95)",
                               bordercolor=col, borderwidth=1, borderpad=4)
        else:
            y_lvl = levels[id(w)]
            # Tirette grise reliant le marqueur à l'étiquette
            fig.add_shape(type="line", x0=w["pk"], x1=w["pk"],
                          y0=-0.06, y1=y_lvl + 0.10,
                          line=dict(color="#bdc3c7", width=1, dash="dot"),
                          layer="below")
            fig.add_annotation(
                x=w["pk"], y=y_lvl,
                text=f"<b>{w['name']}</b><br><span style='color:#7f8c8d'>PK {w['pk']:.1f}</span>",
                showarrow=False,
                font=dict(size=10, color="#34495e"),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#ecf0f1", borderwidth=1, borderpad=3,
            )

    # Zone de danger : bandeau vertical rouge couvrant tout le trajet.
    if intersection_range:
        lo, hi = intersection_range
        fig.add_shape(type="rect",
                      x0=lo, x1=hi, y0=-1.6, y1=1.05,
                      fillcolor="rgba(231,76,60,0.10)",
                      line=dict(color="#c0392b", width=2, dash="dash"),
                      layer="below")
        fig.add_annotation(
            x=(lo + hi) / 2, y=1.45,
            text=f"⚠ <b>Zone de danger</b> · PK {lo:.1f} – {hi:.1f}",
            showarrow=False,
            font=dict(size=12, color="#c0392b"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#c0392b", borderwidth=1, borderpad=5,
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

    # Trains : Pair au-dessus de la voie, Impair en-dessous (les têtes
    # triangulaires donnent en plus le sens visuel).
    for tr in trains:
        pos = tr.position_at(now)
        if pos is None:
            continue
        pk, phase = pos
        y = 0.55 if tr.direction == +1 else -0.55
        _draw_train_shape(fig, tr, pk, phase, y)

    fig.update_xaxes(
        title=dict(text="<b>PK (km)</b>", font=dict(size=13, color="#2c3e50")),
        range=[-6, LGV_LENGTH_KM + 6],
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        tickmode="linear", dtick=20,
        tickfont=dict(color="#2c3e50"),
    )
    fig.update_yaxes(visible=False, range=[-1.65, 1.65])
    fig.update_layout(
        height=580,
        margin=dict(l=20, r=20, t=30, b=50),
        plot_bgcolor="#ffffff", paper_bgcolor="white",
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

    # add_vline + annotation_text crashes on datetime axes ; on dessine
    # explicitement la ligne et l'étiquette « Maintenant ».
    fig.add_shape(type="line", x0=now, x1=now, yref="paper", y0=0, y1=1,
                  line=dict(color="#7f8c8d", dash="dash", width=2))
    fig.add_annotation(x=now, y=1.02, yref="paper",
                       text="<b>Maintenant</b>", showarrow=False,
                       font=dict(size=11, color="#7f8c8d"),
                       bgcolor="rgba(255,255,255,0.9)",
                       bordercolor="#7f8c8d", borderwidth=1, borderpad=3)

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
    st.markdown("### ⚡ Puissances souscrites (PS)")
    st.caption("Ajustables si la souscription est révisée auprès de l'ONEE.")
    ps_aouama = st.number_input(
        "AOUAMA SST1 — Secteur Nord (KVA)",
        min_value=1000, max_value=40000,
        value=12000, step=500, key="ps_aouama",
    )
    ps_oulad = st.number_input(
        "OULAD SLAMA SST2 — Secteur Sud (KVA)",
        min_value=1000, max_value=40000,
        value=14500, step=500, key="ps_oulad",
    )

    st.markdown("---")
    st.markdown("### 💰 Paramètres économiques")
    tarif_kwh = st.number_input(
        "Tarif énergie (MAD/kWh)",
        min_value=0.10, max_value=5.00,
        value=1.05, step=0.05, format="%.2f",
        help="Tarif HT moyen ONEE (≈ 1,05 MAD/kWh).",
    )
    trains_per_day = st.number_input(
        "Circulations / jour / sens",
        min_value=1, max_value=50, value=8, step=1,
        help="Nombre d'Al Boraq Pair (T→K) ou Impair (K→T) par jour.",
    )
    penalite_kva = st.number_input(
        "Pénalité dépassement PS (MAD/KVA/mois)",
        min_value=0, max_value=500, value=50, step=5,
        help="Coût mensuel facturé par l'ONEE pour chaque KVA "
             "excédant la puissance souscrite.",
    )

    st.markdown("---")
    st.markdown("### ▶️ Lecture")
    sim_speed = st.slider("Vitesse simulation (× réel)", 1, 240, 60)
    run_sim = st.toggle("Lancer la simulation", value=False)
    if st.button("🔄 Réinitialiser l'horloge", use_container_width=True):
        st.session_state.pop("sim_clock", None)
        st.session_state.pop("sim_anchor", None)


# ---------- Application des PS saisies ----------
SUBSTATIONS[0]["kva_souscrite"] = int(ps_aouama)
SUBSTATIONS[1]["kva_souscrite"] = int(ps_oulad)

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
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Carte de la voie",
    "📉 Marche graphique",
    "📊 Détails & scénarios",
    "💰 Économie & gains",
])

with tab1:
    st.markdown(
        f"""
<div style='text-align:center;margin:0.4rem 0 0.8rem 0;'>
  <span style='display:inline-block;padding:6px 14px;background:#fff;
               border:1px solid #e1e5ea;border-radius:20px;margin:0 6px;
               font-size:13px;color:#2c3e50;'>
    <span class='train-legend' style='background:{TRAIN_BRAND['Train A (Pair)']['primary']}'></span>
    <b>Train A — Pair</b> · {comp_a} · profil {prof_a} · Tanger → Kenitra
  </span>
  <span style='display:inline-block;padding:6px 14px;background:#fff;
               border:1px solid #e1e5ea;border-radius:20px;margin:0 6px;
               font-size:13px;color:#2c3e50;'>
    <span class='train-legend' style='background:{TRAIN_BRAND['Train B (Impair)']['primary']}'></span>
    <b>Train B — Impair</b> · {comp_b} · profil {prof_b} · Kenitra → Tanger
  </span>
</div>
""",
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
            kva_txt = f"{kva:,}".replace(",", " ")
            limit_txt = f"{limit:,}".replace(",", " ")
            delta_txt = f"{'+' if delta >= 0 else ''}{delta:,}".replace(",", " ")
            delta_color = "#c0392b" if delta > 0 else "#27ae60"
            st.markdown(
                f"""
<div style="
    background:#ffffff;
    border:1px solid #e1e5ea;
    border-left:6px solid {sst['color']};
    border-radius:10px;
    padding:14px 18px;
    box-shadow:0 1px 3px rgba(0,0,0,0.04);
">
  <div style="font-size:13px;color:#7f8c8d;letter-spacing:0.5px;text-transform:uppercase;">
    {sst['secteur']}
  </div>
  <div style="font-size:17px;font-weight:700;color:{sst['color']};margin-bottom:8px;">
    {sst['name']}
  </div>
  <div style="font-size:34px;font-weight:800;color:#2c3e50;line-height:1.1;">
    {kva_txt}
    <span style="font-size:14px;color:#7f8c8d;font-weight:600;">KVA</span>
  </div>
  <div style="color:{delta_color};font-weight:700;font-size:14px;margin-top:4px;">
    {delta_txt} vs PS {limit_txt} KVA
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            bar_color = "#c0392b" if pct > 100 else (
                "#f39c12" if pct > 80 else "#27ae60")
            bar_pct = min(pct, 100)
            st.markdown(
                f"""
<div style="
    background:#ecf0f1;border-radius:6px;height:10px;margin:6px 0 10px 0;
    overflow:hidden;border:1px solid #d5dbdb;">
  <div style="background:{bar_color};width:{bar_pct:.1f}%;height:100%;"></div>
</div>
<div style="font-size:12px;color:#7f8c8d;margin-bottom:6px;">
  {pct:.0f}% de la puissance souscrite
</div>
""",
                unsafe_allow_html=True,
            )
            if info["details"]:
                for d in info["details"]:
                    st.markdown(
                        f"<div style='font-size:12px;color:#34495e;"
                        f"padding:2px 0;'>• {d}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<div style='font-size:12px;color:#95a5a6;font-style:italic;'>"
                    "Aucun train dans ce secteur</div>",
                    unsafe_allow_html=True,
                )

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

with tab4:
    st.markdown("## 💰 Analyse économique & optimisation des croisements")
    st.markdown(
        "Cet onglet quantifie le **gain en MAD** apporté par l'outil. "
        "Le levier est simple : décaler légèrement le départ d'un train "
        "pour déplacer le point de croisement vers une zone où les deux "
        "rames consomment peu, ce qui évite les pénalités de dépassement "
        "de la puissance souscrite (PS)."
    )

    # ----- Section 1 : énergie & coût par train --------------------------
    st.markdown("### 🔋 Consommation par train sur un trajet complet")
    rows = []
    for tr in [train_a, train_b]:
        e_kwh = train_energy_kwh(tr)
        c_mad = e_kwh * tarif_kwh
        rows.append({
            "Train": tr.name,
            "Sens": tr.sens_label,
            "Composition": f"{tr.composition} ({tr.n_rames} rame{'s' if tr.n_rames>1 else ''})",
            "Profil": tr.profile_name,
            "Énergie (kWh)": f"{e_kwh:,.0f}".replace(",", " "),
            "Coût (MAD)":   f"{c_mad:,.0f}".replace(",", " "),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ----- Section 2 : agrégation journalière / mensuelle / annuelle -----
    st.markdown("### 📆 Coût d'exploitation extrapolé")
    daily_kwh = sum(train_energy_kwh(tr) for tr in [train_a, train_b]) * trains_per_day
    daily_cost = daily_kwh * tarif_kwh
    monthly_cost = daily_cost * 30
    yearly_cost = daily_cost * 365

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("⚡ Énergie / jour",
               f"{daily_kwh:,.0f} kWh".replace(",", " "),
               help=f"({trains_per_day} circulations × 2 sens).")
    cc2.metric("💵 Coût énergie / jour",
               f"{daily_cost:,.0f} MAD".replace(",", " "))
    cc3.metric("📅 Coût / mois",
               f"{monthly_cost:,.0f} MAD".replace(",", " "))
    cc4.metric("🗓️ Coût / an",
               f"{yearly_cost:,.0f} MAD".replace(",", " "))

    # ----- Section 3 : profil de risque le long de la ligne --------------
    st.markdown("### 📈 Profil de pic combiné — où coûte un croisement ?")
    st.caption(
        "Pour chaque PK candidat, le graphique montre l'appel combiné "
        "des deux trains s'ils s'y rencontraient. Les barres rouges "
        "dépassent la PS du secteur — c'est là que vous payez des pénalités."
    )

    pks_p, kvas_p = crossing_power_profile(train_a, train_b, step_km=1.0)
    if pks_p:
        bar_colors = []
        for pk_v, kva_v in zip(pks_p, kvas_p):
            sst = substation_for(pk_v)
            limit = sst["kva_souscrite"] if sst else 14500
            if kva_v > limit:
                bar_colors.append("#c0392b")
            elif kva_v > 0.85 * limit:
                bar_colors.append("#f39c12")
            else:
                bar_colors.append("#27ae60")

        fig_prof = go.Figure()
        y_max = max(kvas_p) * 1.15

        for sst in SUBSTATIONS:
            fig_prof.add_shape(
                type="rect",
                x0=sst["pk_start"], x1=sst["pk_end"],
                y0=0, y1=y_max,
                fillcolor=sst["fill"], line_width=0, layer="below",
            )
            # Ligne PS du secteur
            fig_prof.add_shape(
                type="line",
                x0=sst["pk_start"], x1=sst["pk_end"],
                y0=sst["kva_souscrite"], y1=sst["kva_souscrite"],
                line=dict(color=sst["color"], dash="dash", width=2),
            )
            ps_lbl = f"{sst['kva_souscrite']:,}".replace(",", " ")
            fig_prof.add_annotation(
                x=(sst["pk_start"] + sst["pk_end"]) / 2,
                y=sst["kva_souscrite"],
                text=f"<b>PS {sst['name']}</b> · {ps_lbl} KVA",
                showarrow=False, yshift=12,
                font=dict(color=sst["color"], size=11),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor=sst["color"], borderwidth=1, borderpad=3,
            )

        fig_prof.add_trace(go.Bar(
            x=pks_p, y=kvas_p, marker=dict(color=bar_colors),
            hovertemplate="PK %{x:.0f}<br><b>%{y:,.0f} KVA</b><extra></extra>",
            showlegend=False,
        ))

        # Marqueur du croisement actuel
        if meeting_main is not None:
            fig_prof.add_shape(
                type="line",
                x0=meeting_main, x1=meeting_main,
                y0=0, y1=y_max,
                line=dict(color="#2c3e50", dash="dot", width=2),
            )
            fig_prof.add_annotation(
                x=meeting_main, y=y_max * 0.95,
                text=f"<b>Croisement actuel</b><br>PK {meeting_main:.1f}",
                showarrow=False,
                font=dict(size=11, color="#2c3e50"),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#2c3e50", borderwidth=1, borderpad=3,
            )

        fig_prof.update_xaxes(
            title=dict(text="<b>PK candidat (km)</b>", font=dict(size=13)),
            range=[0, LGV_LENGTH_KM], dtick=20,
        )
        fig_prof.update_yaxes(
            title=dict(text="<b>Pic combiné (KVA)</b>", font=dict(size=13)),
            range=[0, y_max],
        )
        fig_prof.update_layout(
            height=380, margin=dict(l=20, r=20, t=20, b=40),
            plot_bgcolor="#ffffff", paper_bgcolor="white",
        )
        st.plotly_chart(fig_prof, use_container_width=True)

    # ----- Section 4 : optimisation par retard ---------------------------
    st.markdown("### 🎯 Optimiseur — retard à appliquer pour économiser")

    if meeting_main is None:
        st.info("Pas de croisement détecté avec les paramètres actuels.")
    else:
        cands = scan_optimal_delay(train_a, train_b)
        if not cands:
            st.info("Aucune solution dans la plage de scan (-5 à +10 min).")
        else:
            base = min(cands, key=lambda c: abs(c["delay"]))
            best = min(cands, key=lambda c: c["kva"])

            sst_b = substation_for(base["pk"])
            sst_o = substation_for(best["pk"])
            ps_b = sst_b["kva_souscrite"] if sst_b else 14500
            ps_o = sst_o["kva_souscrite"] if sst_o else 14500
            dep_b = max(0, base["kva"] - ps_b)
            dep_o = max(0, best["kva"] - ps_o)
            kva_evite = dep_b - dep_o

            cb, co = st.columns(2)
            with cb:
                st.markdown(
                    f"""
<div style="border:1px solid #f5b7b1;background:#fdedec;border-radius:10px;padding:14px;">
  <div style="color:#c0392b;font-weight:700;margin-bottom:6px;">🔴 Sans optimisation</div>
  <div style="color:#2c3e50;font-size:14px;line-height:1.7;">
    PK croisement : <b>{base['pk']:.1f}</b><br>
    Phases : {PHASE_LABELS_FR[base['phase_a']]} × {PHASE_LABELS_FR[base['phase_b']]}<br>
    Secteur : <b>{sst_b['name'] if sst_b else '—'}</b> (PS {ps_b:,} KVA)<br>
    Pic combiné : <b style="color:#c0392b;font-size:18px;">{base['kva']:,} KVA</b><br>
    Dépassement : <b>{dep_b:,} KVA</b>
  </div>
</div>
""".replace(",", " "),
                    unsafe_allow_html=True,
                )
            with co:
                st.markdown(
                    f"""
<div style="border:1px solid #abebc6;background:#eafaf1;border-radius:10px;padding:14px;">
  <div style="color:#27ae60;font-weight:700;margin-bottom:6px;">🟢 Avec retard optimal</div>
  <div style="color:#2c3e50;font-size:14px;line-height:1.7;">
    Retard appliqué Pair : <b>{best['delay']:+.1f} min</b><br>
    PK croisement : <b>{best['pk']:.1f}</b><br>
    Phases : {PHASE_LABELS_FR[best['phase_a']]} × {PHASE_LABELS_FR[best['phase_b']]}<br>
    Secteur : <b>{sst_o['name'] if sst_o else '—'}</b> (PS {ps_o:,} KVA)<br>
    Pic combiné : <b style="color:#27ae60;font-size:18px;">{best['kva']:,} KVA</b><br>
    Dépassement : <b>{dep_o:,} KVA</b>
  </div>
</div>
""".replace(",", " "),
                    unsafe_allow_html=True,
                )

            # Estimation du gain en MAD
            gain_par_evt = kva_evite * penalite_kva  # par mois si appliqué chaque jour
            # Plus juste : pénalité = excédent × tarif × nb croisements concernés / mois
            # On suppose 1 croisement / jour entre A et B (cas étudié), 30 j/mois.
            gain_mois = kva_evite * penalite_kva  # MAD/mois pour cette paire
            gain_an = gain_mois * 12

            if kva_evite > 0:
                st.markdown(
                    f"""
<div style="margin-top:14px;padding:18px;border-radius:10px;
            background:linear-gradient(135deg,#27ae60 0%,#16a085 100%);
            color:white;">
  <div style="font-size:14px;opacity:0.9;">💰 Gain estimé pour cette paire</div>
  <div style="font-size:32px;font-weight:800;line-height:1.2;margin:6px 0;">
    ≈ {gain_mois:,.0f} MAD / mois
  </div>
  <div style="font-size:18px;opacity:0.95;">
    soit <b>≈ {gain_an:,.0f} MAD / an</b>
  </div>
  <div style="font-size:13px;opacity:0.85;margin-top:8px;">
    En retardant le Train Pair de {best['delay']:+.1f} min, on évite
    <b>{kva_evite:,} KVA</b> de dépassement (pénalité {penalite_kva} MAD/KVA/mois).
  </div>
</div>
""".replace(",", " "),
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "✅ Le croisement actuel reste sous la PS du secteur — "
                    "pas de pénalité à éviter sur cette paire."
                )

            # Courbe de scan
            with st.expander("📊 Courbe de scan : pic combiné selon le retard"):
                fig_scan = go.Figure()
                fig_scan.add_trace(go.Scatter(
                    x=[c["delay"] for c in cands],
                    y=[c["kva"] for c in cands],
                    mode="lines+markers",
                    line=dict(color="#1a5276", width=3),
                    marker=dict(size=7, color="#1a5276"),
                    hovertemplate="Retard %{x:+.1f} min<br>"
                                  "<b>%{y:,.0f} KVA</b><extra></extra>",
                    name="Pic combiné",
                ))
                fig_scan.add_shape(
                    type="line", x0=best["delay"], x1=best["delay"],
                    y0=0, y1=max(c["kva"] for c in cands) * 1.05,
                    line=dict(color="#27ae60", dash="dash", width=2),
                )
                fig_scan.add_annotation(
                    x=best["delay"], y=best["kva"],
                    text=f"<b>Optimum</b><br>{best['delay']:+.1f} min",
                    showarrow=True, arrowhead=2, ax=30, ay=-40,
                    font=dict(color="#27ae60"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#27ae60", borderwidth=1, borderpad=3,
                )
                # Ligne PS de référence (secteur du croisement actuel)
                ref_ps = ps_b
                fig_scan.add_hline(
                    y=ref_ps,
                    line=dict(color="#c0392b", dash="dot", width=2),
                    annotation_text=f"PS {ref_ps:,} KVA".replace(",", " "),
                    annotation_position="top right",
                    annotation_font_color="#c0392b",
                )
                fig_scan.update_xaxes(
                    title="<b>Retard supplémentaire Train Pair (min)</b>",
                )
                fig_scan.update_yaxes(
                    title="<b>Pic combiné au croisement (KVA)</b>",
                )
                fig_scan.update_layout(
                    height=320,
                    margin=dict(l=20, r=20, t=20, b=40),
                    plot_bgcolor="#ffffff", paper_bgcolor="white",
                )
                st.plotly_chart(fig_scan, use_container_width=True)

    with st.expander("ℹ️ Méthode de calcul du gain"):
        st.markdown("""
**Énergie par train** : intégration de la puissance par phase
× durée de phase ; durée = longueur du segment / vitesse de phase.

**Profil de pic combiné** : pour chaque PK candidat de croisement, on
suppose que les deux rames y sont simultanément et on additionne leur
puissance instantanée selon leurs phases respectives à ce PK. Le résultat
est converti en KVA (`kw / cosφ`, cosφ ≈ 0,95).

**Optimisation** : balayage de retards supplémentaires sur le Train Pair
de −5 à +10 min (pas 30 s). Pour chaque retard, on recalcule le PK et
l'instant du croisement, puis le pic combiné. On retient le retard qui
minimise ce pic.

**Gain MAD** : la différence de dépassement de PS entre la situation
nominale et la situation optimisée, multipliée par le tarif de pénalité
(MAD/KVA/mois) — extrapolée sur 12 mois.

> ⚠ *Hypothèse simplificatrice :* la pénalité ONEE réelle dépend du
> contrat (PMD, prime fixe, surfacturation des dépassements). Le chiffre
> affiché reste un ordre de grandeur — à recaler avec le contrat exact
> avant chiffrage final.
""")


if run_sim:
    time.sleep(1.0)
    st.rerun()
