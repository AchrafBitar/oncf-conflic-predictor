# ONCF — LGV Traffic & Power Conflict Predictor

A Streamlit MVP that predicts where two opposing **Al Boraq** trains will cross
on the **LGV Tanger Ville ↔ Kenitra** (≈193 km) and whether their combined
power draw risks exceeding the *puissance souscrite* of the feeding
substations.

Built from two ONCF documents:

- *Carte Réseau à jour* (network map + substation table)
- *Efficacité énergétique dans la conduite des trains AL BORAQ* — EPGV,
  v01 du 01/09/2022 (eco-conduite speed profile, energy figures, the
  17 490 KVA OULAD SLAMA overload incident).

## What it models

- **Real eco-conduite profile** from §VII.2 of the report — piecewise
  290 / 320 km/h cruises and *marche sur l'erre* coast segments at the
  documented PK switch points (PK 11 / 60 / 98+404 / 164 in the Pair sense,
  PK 183+500 / 111 / 70 / 25 in the Impair sense).
- **Two LGV substations**: AOUAMA SST1 (sector Nord, PS 12 000 KVA) and
  OULAD SLAMA SST2 (sector Sud, PS 14 500 KVA), with the boundary placed
  at the PK 98+404 sectioning zone.
- **Per-rame electrical draw** from the report's cruise-phase calculation
  table (η = 0.86): ≈6.4 MW @ 320 km/h, ≈4.8 MW @ 290 km/h, ≈0 on the *erre*.
  Doubled for **UM** (two coupled rames).
- **Optimistic / pessimistic crossing range** (±2 min) — drawn as a
  *Danger Zone* on the track map.

## Visuals

- Track map with substation bands, PCV waypoints (Aquass Briech, Sidi El
  Yamani, Laaouamra, Chouaafa, Bahara Ouled Ayad, Benmansour) and live
  train markers.
- *Marche graphique* (time-distance diagram) showing both trajectories and
  their crossing point.
- Live substation gauges (KVA vs PS) updated as the sim clock advances.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

This is an MVP for predictive simulation. Speed-profile timings,
substation boundaries and per-phase power figures are reasonable
approximations derived from the EPGV report; they are not the
authoritative dispatcher tool.
