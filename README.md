# Real-Time Process Control for Anodization

**Closed-loop voltage control for anodization using EIS-derived oxide thickness feedback and JAX-accelerated inference.**

---

## Overview

Anodization produces a self-limiting oxide layer whose growth rate drops as the film thickens — the oxide itself becomes the dominant barrier to ion transport. This system monitors oxide growth in real time via EIS and adjusts applied voltage to maintain a target growth rate throughout the run.

### The Problem

```
Set voltage → Start anodization → Current drops after 30s → Rate slows → Thin, non-uniform oxide
```

### With Real-Time Control

```
Set target growth rate → Monitor C_oxide every 10s → Adjust voltage → Consistent oxide thickness
```

---

## How It Works

### EIS → Oxide Thickness

The growing oxide layer is well-modeled as a simple RC circuit:

```
R_Ω --- [ R_oxide || C_oxide ]
```

As oxide thickness `d` increases, **C_oxide drops directly**:

```
C_oxide = ε₀ · εᵣ · A / d
```

This makes C_oxide a real-time, physically direct measurement of oxide thickness — stronger than a proxy signal. R_oxide provides secondary information about film quality and resistivity.

**Frequency range:** 0.1 Hz – 100 Hz (lower than typical deposition EIS, to resolve the oxide capacitance through dilute electrolyte).

### Control Loop (10s cycle)

```
Measure EIS (100ms)
    ↓
Extract C_oxide → compute thickness (10ms)
    ↓
Predict growth rate — JAX model (<5ms)
    ↓
PID: compute ΔV (5ms)
    ↓
Send new voltage to Gamry (50ms)
    ↓
Repeat
```

**Total loop time: <200ms**

---

## Key Differences From Deposition Control

| | Copper Deposition | Anodization |
|---|---|---|
| Key EIS parameter | R_ct (proxy for uniformity) | C_oxide (direct thickness) |
| Circuit model | Randles (R_Ω + R_ct\|\|C_dl) | Simple RC (R_Ω + R_oxide\|\|C_oxide) |
| Frequency range | 1 Hz – 10 kHz | 0.1 Hz – 100 Hz |
| Control direction | Voltage up/down for kinetics | Voltage up to compensate thickening oxide |
| Signal reliability | Moderate (geometry-sensitive) | High (C_oxide is physically direct) |

---

## Requirements

### Hardware
- **Gamry Reference 3000** (or equivalent with EIS capability)
- **Electrochemical cell** — fixed geometry, consistent electrode spacing critical
- **Reference electrode** appropriate for your electrolyte

### Software
- Python 3.10+, Windows (Gamry COM interface)
- See `requirements.txt`

---

## Quick Start

```bash
# 1. Verify hardware
python scripts/check_hardware.py

# 2. Run open-loop baseline (collect training data)
python run_baseline.py --voltage 15 --duration 300

# 3. Run controlled anodization
python run_controlled.py --target-thickness 2.0 --duration 300

# 4. Compare results
python analyze_results.py --baseline baseline_001.json --controlled controlled_001.json
```

---

## Project Structure

```
realtime-anodization-control/
├── src/
│   ├── gamry_interface.py      # Gamry potentiostat control + EIS
│   ├── feature_extraction.py   # EIS → C_oxide, R_oxide extraction
│   ├── predictive_model.py     # JAX growth rate predictor
│   ├── controller.py           # PID controller
│   └── control_loop.py         # Main control loop
├── scripts/
│   ├── run_baseline.py
│   ├── run_controlled.py
│   └── analyze_results.py
├── models/
│   └── growth_rate_predictor.pkl
└── data/
    ├── raw/
    └── experiments.csv
```

---

## Safety Limits

```python
MIN_VOLTAGE = 5.0     # V — below this, oxide won't form
MAX_VOLTAGE = 60.0    # V — above this, risk of breakdown/sparking
MAX_DELTA_V = 2.0     # V per control interval — prevent sudden jumps
MAX_CURRENT = 500.0   # mA — overcurrent protection
```

**Do not disable without good reason.** Oxide breakdown (sparking/burning) is rapid and irreversible.

---

## Validation Protocol

**Phase 1 — Baseline:** Run 10 open-loop depositions at fixed voltage. Log EIS every 10s. Confirm C_oxide decreases monotonically and correlates with measured oxide thickness post-run.

**Phase 2 — Model Training:** Train JAX predictor on Phase 1 data. Target R² > 0.85 on validation set.

**Phase 3 — Controlled Runs:** Run 10 controlled depositions. Compare thickness uniformity and growth rate consistency to baseline.

**Success criteria:** 80% of controlled runs hit target thickness ±10%, statistically significant improvement over baseline (p < 0.05).

---

## Notes on Dilute Electrolyte

High R_Ω in dilute electrolyte can dominate the EIS spectrum and make C_oxide harder to resolve. If fit quality is poor:
- Increase AC excitation amplitude slightly
- Push frequency range lower (down to 0.01 Hz if needed)
- Consider a 4-terminal measurement to eliminate R_Ω from the sense path

---

## Citation

```bibtex
@software{anodization_control_2026,
  author = {Guarino, Christopher},
  title = {Real-Time Process Control for Anodization},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/cguarino/realtime-anodization-control}
}
```

---

**Christopher Guarino** — IBM Watson Research Center — chris.francis.guarino@gmail.com
