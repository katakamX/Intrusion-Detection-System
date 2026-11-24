# ML-NIDS — Machine Learning-Based Network Intrusion Detection System

A final-year engineering project that detects malicious network traffic using machine learning instead of fixed signature rules, so it can generalize to attack variants it wasn't explicitly trained on.

**Student:** Aditya Katakam (23BTRCL027) · **Guide:** Dr. J Somasekar · **Dept:** CSE-AIML, JAIN (Deemed-to-be University)

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Project Status](#project-status)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Results So Far](#results-so-far)
- [Roadmap](#roadmap)

---

## Problem Statement
Traditional signature/rule-based Intrusion Detection Systems (IDS) match traffic against a database of known attack patterns. They cannot detect new or modified attacks and require constant manual rule updates. This project builds an IDS that learns the *behavioral pattern* of malicious traffic from data, so it generalizes to attacks it has never explicitly seen.

## Objectives
1. Study existing IDS approaches — signature-based, anomaly-based, ML-based (literature survey).
2. Preprocess a standard, benchmark network-traffic dataset.
3. Train and evaluate ML classifier(s) that separate normal traffic from attacks.
4. Extend from binary detection to multi-class attack-type identification.
5. Package the trained model into an interactive dashboard for live/batch traffic classification.

## Project Status
**Current stage: Phase 3 — Interactive dashboard (complete).**

| Phase | Description | Status |
|---|---|---|
| 1 | Data pipeline + baseline binary classifier (Random Forest) on NSL-KDD, with evaluation metrics | ✅ Done |
| 2 | Multi-class attack-category classification (DoS/Probe/R2L/U2R/normal); compare Random Forest, XGBoost, Neural Net; persist best model | ✅ Done |
| 3 | Interactive Streamlit dashboard — upload/stream traffic, live predictions, visual alerts | ✅ Done |
| 4 | Stretch: live packet capture integration, real-time monitoring | 💭 Stretch goal |

- `poc.py` — Phase 1 binary (normal vs. attack) baseline.
- `phase2.py` — Phase 2: classifies traffic into 5 categories (normal, DoS, Probe, R2L, U2R), trains and compares Random Forest / XGBoost / a Neural Network (MLP), and saves the best model + encoders + scaler to `model/` via `joblib` for reuse without retraining.
- `app.py` — Phase 3: a Streamlit dashboard that loads the saved Phase 2 model and classifies traffic sampled from the NSL-KDD test set or uploaded as a CSV, showing live predictions, an alert count, a per-category chart, and a highlighted results table.

## Architecture
```
NSL-KDD dataset  →  Preprocessing  →  ML Classifiers  →  Evaluation  →  Streamlit dashboard
 (labeled network     (encode categorical   (Random Forest,     (accuracy, precision,    (sample/upload
  connection records)   features, binarize    XGBoost, Neural      recall, confusion        traffic, live
                        or multi-class label)  Net — best saved)    matrix)                  predictions)
```
- **Data layer**: NSL-KDD — each record is a network connection summarized into 41 features (duration, protocol type, service, flag, byte counts, error rates, etc.) plus a label.
- **Preprocessing**: categorical columns label-encoded; label collapsed to `normal` vs `attack` for the Phase 1 binary baseline, or grouped into 5 attack categories for Phase 2+.
- **Models**: Random Forest (Phase 1 baseline), compared against XGBoost and a Neural Network in Phase 2 — all fast to train, no GPU needed; best model persisted via `joblib`.
- **Evaluation**: accuracy, precision, recall, confusion matrix — recall matters most here, since a missed attack (false negative) is worse than a false alarm.
- **Serving layer**: the Streamlit dashboard (`app.py`) loads the persisted model and classifies traffic interactively, so nothing needs retraining at demo time.

## Repository Structure
```
.
├── README.md             — you are here
├── .gitignore            — excludes venv/, caches, and saved model binaries
├── poc.py                — Phase 1: binary (normal/attack) proof-of-concept
├── phase2.py             — Phase 2: multi-class classification + model comparison
├── app.py                — Phase 3: Streamlit dashboard
├── requirements.txt      — Python dependencies
├── data/
│   ├── KDDTrain+.txt     — NSL-KDD training set
│   └── KDDTest+.txt      — NSL-KDD test set
└── model/                — trained model + encoders + scaler, produced by phase2.py (gitignored)
```

## Tech Stack
- **Language**: Python 3
- **ML**: scikit-learn (Random Forest, Neural Network/MLP), XGBoost
- **Data**: pandas
- **Dashboard**: Streamlit
- **Persistence**: joblib

## Dataset
[**NSL-KDD**](https://www.unb.ca/cic/datasets/nsl.html) — the de-duplicated, improved version of the classic KDD'99 Cup dataset, and the most widely cited benchmark in IDS research. Using a recognized benchmark keeps results comparable to published literature. Notably, its official test set deliberately includes attack types absent from the training set, to test generalization rather than memorization — this project's results are discussed with that in mind (see [Results](#results-so-far)).

## Getting Started
```bash
pip install -r requirements.txt
python poc.py        # Phase 1: binary baseline
python phase2.py      # Phase 2: multi-class model comparison, saves model/ artifacts
streamlit run app.py  # Phase 3: interactive dashboard (requires model/ from phase2.py)
```
`poc.py` prints dataset size, training progress, and the evaluation report (accuracy, precision, recall, confusion matrix, top feature importances). `phase2.py` must be run at least once before `app.py`, since the dashboard loads its saved model from `model/`.

## Results So Far
Phase 1 baseline (Random Forest, binary normal-vs-attack) on the official NSL-KDD test split:

| Metric | Value |
|---|---|
| Accuracy | 0.77 |
| Precision (attack) | 0.66 |
| Recall (attack) | 0.97 |

**Why accuracy isn't ~97-99% like some papers report:** those figures usually come from a random train/test split of the *same* distribution. NSL-KDD's official `KDDTest+` set intentionally contains attack types never seen during training, to measure true generalization. A recall of 0.97 means the model still catches almost all attacks, including unfamiliar ones — it over-flags some normal traffic (lower precision), which is the safer trade-off for a security system: a false alarm is far cheaper than a missed intrusion.

Top predictive features: `src_bytes`, `dst_bytes`, `same_srv_rate`, `dst_host_same_srv_rate`, `flag`.

### Phase 2 — multi-class model comparison
Attack categories follow the standard NSL-KDD grouping: **DoS**, **Probe**, **R2L** (remote-to-local), **U2R** (user-to-root), and **normal**.

| Model | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---|---|---|---|
| Neural Network (MLP) | 0.777 | 0.880 | 0.527 | 0.545 |
| XGBoost | 0.757 | 0.796 | 0.522 | 0.534 |
| Random Forest | 0.751 | 0.773 | 0.497 | 0.506 |

The Neural Network edges out the others on macro-F1 and was saved as the production model (`model/best_model.joblib`).

**Why R2L/U2R recall is low across all three models:** both categories are extremely rare in training data (995 R2L and just 52 U2R records out of ~126,000) and the official test set again injects attack subtypes absent from training — the same generalization challenge noted in Phase 1, now visible per-category instead of being averaged away by the binary view. This is a known, published limitation of NSL-KDD and a legitimate discussion point: class imbalance + unseen-attack generalization is exactly the kind of real-world problem production IDS have to deal with, and it motivates Phase 3's focus on an interactive dashboard that can surface these per-class confidence differences to a human analyst rather than a single opaque verdict.

## Dashboard (Phase 3)
`app.py` is a Streamlit app that loads the Phase 2 model and lets you:
- **Sample bundled NSL-KDD test traffic** — pick a sample size, see live predictions against known ground truth (shows a running accuracy metric for that batch).
- **Upload your own CSV** — same raw NSL-KDD format (41 features, with or without the trailing label/difficulty columns); unseen categorical values fall back gracefully instead of crashing.
- View a **bar chart** of predicted categories, an **alert count**, and a **results table** with attack rows highlighted in red.

Run it with `streamlit run app.py` (after `phase2.py` has produced `model/best_model.joblib` at least once).

## Roadmap
- [x] Multi-class classification (identify specific attack type, not just normal/attack)
- [x] Compare Random Forest against XGBoost and a neural network baseline
- [x] Persist trained models with `joblib` for reuse without retraining
- [x] Streamlit dashboard: upload/stream traffic → live predictions → visual alerts and charts
- [ ] Stretch: live packet capture (e.g. via `scapy`) for real-time monitoring instead of static CSVs
- [ ] Stretch: address R2L/U2R class imbalance (e.g. SMOTE oversampling, class weighting)
