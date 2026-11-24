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
**Current stage: Phase 1 — Proof of Concept (complete).**

| Phase | Description | Status |
|---|---|---|
| 1 | Data pipeline + baseline binary classifier (Random Forest) on NSL-KDD, with evaluation metrics | ✅ Done |
| 2 | Multi-class attack-type classification; compare multiple algorithms (e.g. XGBoost, neural net) | ⏳ Planned |
| 3 | Interactive Streamlit dashboard — upload/stream traffic, live predictions, visual alerts | ⏳ Planned |
| 4 | Stretch: live packet capture integration, real-time monitoring | 💭 Stretch goal |

Phase 1's runnable script (`code/poc.py`) loads NSL-KDD, trains a Random Forest, and prints accuracy/precision/recall/confusion-matrix/feature-importance — this is the evidence presented at Review 1 that the ML approach works on real benchmark data.

## Architecture
```
NSL-KDD dataset  →  Preprocessing  →  ML Classifier  →  Evaluation  →  (Phase 3) Live dashboard
 (labeled network     (encode categorical   (Random Forest,     (accuracy, precision,
  connection records)   features, binarize    multi-class in      recall, confusion
                        label)                Phase 2)             matrix)
```
- **Data layer**: NSL-KDD — each record is a network connection summarized into 41 features (duration, protocol type, service, flag, byte counts, error rates, etc.) plus a label.
- **Preprocessing**: categorical columns label-encoded; label collapsed to `normal` vs `attack` for the binary baseline.
- **Model**: Random Forest Classifier — fast to train, handles mixed feature types, gives interpretable feature importances.
- **Evaluation**: accuracy, precision, recall, confusion matrix — recall matters most here, since a missed attack (false negative) is worse than a false alarm.

## Repository Structure
```
.
├── README.md             — you are here
├── poc.py                — Phase 1 proof-of-concept script
├── requirements.txt      — Python dependencies
├── data/
│   ├── KDDTrain+.txt     — NSL-KDD training set
│   └── KDDTest+.txt      — NSL-KDD test set
└── model/                — reserved for saved trained models (Phase 2+)
```

## Tech Stack
- **Language**: Python 3
- **ML**: scikit-learn (Random Forest; more algorithms in Phase 2)
- **Data**: pandas
- **Planned (Phase 3)**: Streamlit for the dashboard, joblib for model persistence

## Dataset
[**NSL-KDD**](https://www.unb.ca/cic/datasets/nsl.html) — the de-duplicated, improved version of the classic KDD'99 Cup dataset, and the most widely cited benchmark in IDS research. Using a recognized benchmark keeps results comparable to published literature. Notably, its official test set deliberately includes attack types absent from the training set, to test generalization rather than memorization — this project's results are discussed with that in mind (see [Results](#results-so-far)).

## Getting Started
```bash
pip install -r requirements.txt
python poc.py
```
This prints dataset size, training progress, and the evaluation report (accuracy, precision, recall, confusion matrix, top feature importances).

## Results So Far
Phase 1 baseline (Random Forest, binary normal-vs-attack) on the official NSL-KDD test split:

| Metric | Value |
|---|---|
| Accuracy | 0.77 |
| Precision (attack) | 0.66 |
| Recall (attack) | 0.97 |

**Why accuracy isn't ~97-99% like some papers report:** those figures usually come from a random train/test split of the *same* distribution. NSL-KDD's official `KDDTest+` set intentionally contains attack types never seen during training, to measure true generalization. A recall of 0.97 means the model still catches almost all attacks, including unfamiliar ones — it over-flags some normal traffic (lower precision), which is the safer trade-off for a security system: a false alarm is far cheaper than a missed intrusion.

Top predictive features: `src_bytes`, `dst_bytes`, `same_srv_rate`, `dst_host_same_srv_rate`, `flag`.

## Roadmap
- [ ] Multi-class classification (identify specific attack type, not just normal/attack)
- [ ] Compare Random Forest against XGBoost and a neural network baseline
- [ ] Persist trained models with `joblib` for reuse without retraining
- [ ] Streamlit dashboard: upload/stream traffic → live predictions → visual alerts and charts
- [ ] Stretch: live packet capture (e.g. via `scapy`) for real-time monitoring instead of static CSVs
