# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Webscrapes the Brazilian presidential public agenda (`gov.br/planalto`), stores it in a local SQLite database, processes the data, and exposes it through a Plotly Dash dashboard. Source code is written in Portuguese (variable/function names, comments, UI strings) — keep new code consistent with that.

## Architecture

The system is a four-stage pipeline. Each stage is a standalone script in `agenda-presidencial/code/`; there is no orchestrator — scripts are run manually in order.

1. **`main.py`** — Scrapes one URL per day from `gov.br/planalto/.../agenda-do-presidente-da-republica/<YYYY-MM-DD>`, parses meeting items with BeautifulSoup (CSS classes `item-compromisso`, `compromisso-inicio/-fim/-titulo/-local`), and appends rows to the SQLite table `AGENDA_PRESIDENCIAL`. On a normal run it auto-resumes: `get_max_id_and_max_date()` queries the DB for the latest meeting and only scrapes from the next day forward. The commented-out block at the bottom is the one-time initial backfill (`transform_compromises_in_dataframe(2019, 1, 1, 0)`).

2. **`calendario.py`** — One-time loader: reads a manually prepared `calendario.csv` and writes it to the SQLite table `CALENDARIO`. This table supplies the business-calendar reference data (`DIA_UTIL`, `FERIADO`, `DIA_DA_SEMANA`, `MES_REFERENCIA`, `SEMANA_DO_ANO`, etc.) used to compare presidential activity against a standard CLT 8h workday.

3. **`gerar_dados.py`** — The processing stage. Reads both DB tables, joins agenda against the calendar, computes all aggregations (duration per month/weekday/start-hour/location, cumulative totals, first/last activity of day) and a TF-IDF + PCA 2D projection of meeting titles for the similarity scatter. Writes ~11 pre-computed CSVs into `code/data/`.

4. **`dashboard.py`** — The Dash app. Loads only the pre-computed CSVs (does **not** touch the database), builds Plotly figures, and serves the dashboard. Run after `gerar_dados.py` to pick up fresh data.

Data flow: `gov.br` → `main.py` → SQLite → `gerar_dados.py` → CSVs in `code/data/` → `dashboard.py` → browser.

## Hardcoded paths — important

Paths are hardcoded to the original author's machine and to the production host. Anything touching the filesystem will fail elsewhere and must be adjusted before running:

- DB: `sqlite:////home/barbaruiva/Documents/Database/AGENDA_PRESIDENCIAL.db` (in `main.py`, `calendario.py`, `gerar_dados.py`)
- `calendario.py` reads `/home/barbaruiva/Downloads/calendario.csv`
- `gerar_dados.py` writes CSVs to `.../agenda-presidencial/code/data/`
- `dashboard.py` reads CSVs from `/home/agendapresidencial/mysite/data/` (the production deployment path, different from where `gerar_dados.py` writes)

## Running

No requirements file exists. Dependencies must be installed manually:

```
pip install requests pandas sqlalchemy beautifulsoup4 dash dash-html-components dash-core-components dash-bootstrap-components plotly scikit-learn nltk
```

`gerar_dados.py` downloads NLTK Portuguese stopwords at runtime (`nltk.download('stopwords')`).

Run the pipeline stages directly with Python, e.g. `python agenda-presidencial/code/main.py`. The dashboard runs via `python agenda-presidencial/code/dashboard.py` and serves on the Dash default port (`debug=True`).

There is no build, no linter, and no test suite configured.

## Conventions

- `setup.py` declares the package `agenda-presidencial` but the actual scripts are not packaged as a module — they are run as standalone files.
- DB column names are UPPER_SNAKE_CASE English; everything else (logic, UI) is Portuguese.
- `dashboard.py` registers a Dash callback exception suppressor but currently defines no callbacks — all figures are static, computed at import time.
