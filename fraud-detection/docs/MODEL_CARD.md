# Model Card: PaySim Fraud Classifier

> Status: draft. Sections marked TODO are filled in as the work progresses.

## Intended use
Demonstration of an end-to-end fraud-scoring service for a portfolio. It is **not** suitable
for scoring real transactions.

## Data
PaySim: ~6.3M synthetic mobile-money transactions over 30 days (744 hourly steps),
generated from aggregated real logs. Source: https://www.kaggle.com/datasets/ealaxi/paysim1

## Known dataset limitations
- **Synthetic.** Real fraud behaviour is more varied and adapts to defences.
- **Balance-field leakage.** Simulator artefacts (e.g. emptied origin accounts) make fraud
  easier to detect than in reality. Results are reported with and without balance-derived features.
- **Fraud only in TRANSFER and CASH_OUT.** A real system cannot assume this.
- **Random account IDs.** Per-customer history features carry no genuine signal, so IDs are dropped.
- **`isFlaggedFraud` excluded.** It is derived from the simulator's own rule and is label-adjacent.
- **Extreme imbalance (~0.13% fraud).** Accuracy is meaningless; see metrics below.

## Evaluation
- Split: chronological on `step` (train <= 600, validation 601-680, test > 680 by default).
- Metrics: PR-AUC, precision and recall at the chosen threshold, confusion matrix.
- TODO: results table (baseline, logistic regression, XGBoost; with and without balance features).

## Decision threshold
- TODO: chosen threshold and the cost assumption behind it (cost of a missed fraud vs a false alert).

## Known failure modes
- TODO: fill in from error analysis.

## Privacy and logging
The service never logs raw payloads or account identifiers. It logs request ID, latency,
model version, score and decision only.

## Not for
Real financial decisions, automated account blocking without human review, or any use affecting
real customers.
