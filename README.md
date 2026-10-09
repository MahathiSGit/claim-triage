# Air Passenger Compensation Claim Triage

A small ML service that takes a flight-delay compensation claim and routes it to one of three outcomes:

1. **Auto-approve**: the claim is clearly eligible.
2. **Auto-reject**: the claim is clearly ineligible (for example, the delay was caused by weather).
3. **Human review**: the model is not sure, so an agent decides.

## Problem

When a flight is delayed, passengers can claim compensation under rules like EU261. Airlines receive thousands of claims, and reviewing each one by hand is slow, inconsistent and costly, because obvious cases take as much effort as hard ones.

**Success metrics**
- Share of claims resolved automatically
- Precision on auto-approve (wrongly paying a claim costs money)
- Recall on eligible claims (wrongly rejecting a valid claim creates legal risk)

## Solution

The system has two layers:

- **Rules layer** for legal facts that are never learned: a delay of at least 3 hours, and the compensation tier (EUR 250 up to 1,500 km, EUR 400 up to 3,500 km, EUR 600 above that).
- **ML layer**: one model that predicts the probability that a claim is approved, which covers the fuzzy part, namely whether the delay was the airline's fault or an extraordinary circumstance such as weather.

A final **triage step** compares the probability against two thresholds:

| Probability of approval | Decision |
|---|---|
| P >= 0.90 | Auto-approve |
| P <= 0.10 | Auto-reject, with a reason |
| Otherwise | Human review |

The thresholds are placeholders for now. In Phase 3 they are chosen from the cost of each type of mistake, not from accuracy.

## Architecture

The project has two workflows: one that is done once to build the model, and one that runs for every claim.

```
BUILD ONCE
  Flight data (Kaggle CSV) --> Features + labels --> Train model (LightGBM) --> Docker image (model inside)

FOR EVERY CLAIM
  Claim --> FastAPI service (Kubernetes) --> Rules check --> ML model --> Triage --> Approve / Review / Reject
                                                                                          |
                                                                                    Decision log
```

## Data

There is no public dataset of real compensation claims, because airlines keep that data private. This project uses the real **2015 US flight delays dataset** (Kaggle: `usdot/flight-delays`) and derives claims and labels from it.

- A flight counts as a claim if it arrived 180 minutes or more late and was not cancelled.
- The label `approved` is 1 if the delay was mainly the airline's fault, and 0 if it was mainly weather or security.
- About 5% of labels are flipped at random to imitate inconsistent human agents.
- US flights fall under US DOT rules, not EU261, so EU261-style rules are applied as a **simulated policy**.

The cause-of-delay columns are used only to build the label. They are never used as features, because in a real claim the cause is not yet known at submission time. Using them would be data leakage.

## ML system design concepts

| Concept | Where it appears |
|---|---|
| Hybrid design | Rules for legal logic, ML for uncertainty |
| Label construction | Labels derived from cause-of-delay data |
| Leakage prevention | Only claim-time features are used |
| Time-based split | Train on January-September, test on October-December |
| Class imbalance | About 86% of claims are approved, so accuracy alone is misleading |
| Evaluation | Precision, recall, PR-AUC and a cost matrix |
| Calibration | Check that a predicted 0.9 really means about 90% approved |
| Threshold tuning | Two thresholds chosen from the cost of mistakes |
| Serving | Real-time `POST /predict` endpoint |
| Containerization | Docker image with the API and model inside |
| Orchestration | Kubernetes with replicas, health checks, rolling update and rollback |

## Phases

| Phase | Goal | Output | Status |
|---|---|---|---|
| 1 | Set up the project and build the dataset | `data/claims.csv` | Done, see [PHASE1.md](PHASE1.md) |
| 2 | Train a baseline model and evaluate it | `model/model_v1.pkl`, metrics | In progress |
| 3 | Write the rules and tune the triage thresholds | `rules.py`, thresholds | Planned |
| 4 | Build the API | FastAPI `/predict` and `/health` | Planned |
| 5 | Containerize | `Dockerfile`, running image | Planned |
| 6 | Deploy to Kubernetes | `k8s/` manifests | Planned |
| 7 | Write up | Final README with results and future work | Planned |

Each phase is developed on its own Git branch and merged into `main` through a pull request.

## Tech stack

Python, pandas, scikit-learn, LightGBM, FastAPI, Docker, Kubernetes (minikube or kind), Git and GitHub.

## Planned project structure

```
claim-triage/
├── data/                 # not committed, see .gitignore
├── model/                # saved model files
├── app/                  # FastAPI app, rules and shared feature code (Phases 3-4)
├── k8s/                  # Kubernetes manifests (Phase 6)
├── prepare_data.py       # Phase 1
├── train.py              # Phase 2
├── Dockerfile            # Phase 5
├── README.md
└── PHASE1.md
```

## How to reproduce Phase 1

1. Create and activate a virtual environment, then install the packages:
   ```
   pip install pandas numpy scikit-learn lightgbm fastapi uvicorn joblib
   ```
2. Download `flights.csv` from the Kaggle dataset `usdot/flight-delays` and put it in `data/`.
3. Run:
   ```
   python prepare_data.py
   ```

The raw `flights.csv` file is about 565 MB, so it is excluded from the repository.

## Future work

Features left out to keep the project simple: a feature store, document OCR, fraud and duplicate detection, a text model for free-text claim reasons, a review dashboard, and automated retraining with drift monitoring.