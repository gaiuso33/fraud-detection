# Fraud Detection Service

Tabular ML on the PaySim dataset, served through a validated, tested, containerised API with
CI, Kubernetes deployment and Prometheus/Grafana monitoring.

> Status: week 1, data exploration and modelling.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
make install
# download PaySim into data/ (see below), then:
make data-check
make test
```

## Dataset
Download `ealaxi/paysim1` from Kaggle and place the CSV in `data/`:
```bash
kaggle datasets download -d ealaxi/paysim1 -p data --unzip
```
Raw data is gitignored. Dataset limitations are documented in [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Layout
```
src/fraud/     data, features, model, schemas, api
tests/         unit, API and prediction tests
notebooks/     exploration only
docs/          model card, design notes
```

## Decisions log
| Decision | Choice | Why |
|---|---|---|
| Split | Chronological on `step` | Avoids future leakage |
| Primary metric | PR-AUC | Accuracy is meaningless at 0.13% fraud |
| Threshold | TBD | Cost-based, documented in the model card |
