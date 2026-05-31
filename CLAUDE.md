# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Webscrapes the Brazilian presidential public agenda (`gov.br/planalto`), stores it in a local SQLite database, processes the data, and exposes it through a Plotly Dash dashboard. Source code is written in Portuguese (variable/function names, comments, UI strings) — keep new code consistent with that. DB column names are the exception and stay in UPPER_SNAKE_CASE English.

## Architecture

Two runtime pipelines plus a dashboard. Each entry point is a standalone script in `agenda-presidencial/code/`; there is no orchestrator — scripts are run manually.

**Entry points**

1. **`main.py`** — Scraper. Resumes from `repositorio.RepositorioAgenda.checkpoint()` (latest `MEETING_ID` + next date in `AGENDA_PRESIDENCIAL`), fetches one URL per day, parses meetings with BeautifulSoup, appends rows. Split into three seams: `fetch_dia(data)` (only HTTP), `parse_dia(html, data)` (pure, fixture-testable), and `coletar_intervalo(...)` (orchestrator with `fetch` injectable for tests).

2. **`gerar_dados.py`** — Processing. Reads `AGENDA_PRESIDENCIAL` via the repository, builds the calendar in-memory via `calendario.calendario_para()`, calls the pure aggregations in `agregacoes`, writes 11 CSVs into `code/data/`. `main(hoje=None)` is deterministic given an as-of date.

3. **`dashboard.py`** — The Dash app. Loads the typed `DadosDoDashboard` snapshot via `dados_do_dashboard.carregar()`. KPIs are pure functions on the dataclass.

**Library modules**

- **`repositorio.py`** — `RepositorioAgenda`: the only place that knows the SQLite schema. URL comes from `AGENDA_DB_URL` env var, fallback `data/agenda.db`. Pass `engine=create_engine('sqlite:///:memory:')` for tests.
- **`calendario.py`** — Pure `calendario_para(inicio, fim) → DataFrame`. No CSV, no DB, no caching.
- **`agregacoes.py`** — Pure aggregations (DataFrame in → DataFrame out): monthly/weekday/hourly/location, cumulative, TF-IDF + PCA projection, first/last activity.
- **`dados_do_dashboard.py`** — Frozen `DadosDoDashboard` dataclass (11 named DataFrames) plus four pure KPIs (`media_horas_por_dia`, `media_horas_ultimos_30_dias`, `dias_uteis_sem_atividades`, `comparativo_clt_percentual`).

Data flow: `gov.br` → `main.py` → SQLite (`AGENDA_PRESIDENCIAL`) → `gerar_dados.py` (+ `calendario.calendario_para` in-memory) → CSVs in `code/data/` → `dados_do_dashboard.carregar()` → `dashboard.py` → browser.

## Configuration

All paths resolve relative to the source file. No hardcoded user-home paths.

- `AGENDA_DB_URL` — SQLAlchemy URL for the database. Defaults to `sqlite:///<repo>/agenda-presidencial/code/data/agenda.db`. Set to `postgresql://...` to run on Postgres (docker-compose does this).
- `AGENDA_SCRAPE_INTERVAL_HOURS` — interval between scrape cycles in the scheduler container. Default `24`.
- `AGENDA_SNAPSHOT_RETENTION_DAYS` — how long `data/backups/agenda-*.sql.gz` snapshots are kept. Default `30`, set `0` to keep all.
- `GIT_REMOTE`, `GIT_TOKEN`, `GIT_BRANCH`, `GIT_USER_EMAIL`, `GIT_USER_NAME` — for the optional off-host backup via `git push` of the CSV dump. Leave `GIT_TOKEN` blank to disable.
- `DASHBOARD_PORT`, `GUNICORN_WORKERS` — dashboard tuning.

## Running

### Docker (primary path)

```bash
cp .env.example .env                                       # fill POSTGRES_PASSWORD
docker compose up -d
open http://localhost:8050
```

Three services come up: `db` (Postgres 16 with named volume `agenda_postgres_data`), `scheduler` (runs `bootstrap → scrape → process → snapshot → git push` in a loop), and `dashboard` (gunicorn on :8050).

First-run bootstrap: if the Postgres table is empty, `scheduler.py` calls `bootstrap.py` which loads `data/df.csv` (already in git) into `AGENDA_PRESIDENCIAL`. So `git clone && docker compose up` reconstructs the entire database with no manual steps.

### Bare-metal (for development)

Two dependency sets:

```bash
pip install -r requirements-pipeline.txt   # for main.py and gerar_dados.py
pip install -r requirements.txt            # for dashboard.py
python agenda-presidencial/code/main.py
python agenda-presidencial/code/gerar_dados.py
python agenda-presidencial/code/dashboard.py
```

`gerar_dados.py` downloads NLTK Portuguese stopwords on first run (`nltk.download('stopwords', quiet=True)`). Dashboard serves on the Dash default port (`debug=True`).

There is no build, no linter, and no test suite configured. Pure functions in `agregacoes`, `calendario`, `dados_do_dashboard`, and `main.parse_dia` are fixture-testable; the repository supports in-memory engines.

## DB durability

Five layers of defense; database survives any single (and most combined) failure modes:

1. **Postgres named volume** (`agenda_postgres_data`) — survives `docker rm`, `docker compose down`, image rebuilds.
2. **Rotating snapshots** in `data/backups/agenda-*.sql.gz` — `pg_dump` after every scrape, pruned at 30 days. Bind-mounted to host; survives loss of the Postgres volume.
3. **Git-tracked CSV dump** (`data/df.csv`) — auto-committed after every successful scrape if `GIT_TOKEN` is set. Survives total host loss.
4. **Bootstrap-from-CSV** (`bootstrap.py`) — empty Postgres + `data/df.csv` present → reload automatically on container start.
5. **Re-scrape from gov.br/planalto** — ultimate last resort; full backfill from 2019-01-01 takes ~5h.

Recovery cookbook:

| Scenario | Recovery |
|---|---|
| `docker compose down -v` (volume deleted) | `docker compose up` → bootstrap restores from `data/df.csv` |
| `rm -rf agenda-presidencial/code/data/` | `git pull && docker compose up` |
| Host disk gone | `git clone` elsewhere → `cp .env.example .env` → `docker compose up` |
| GitHub repo gone | Full re-scrape (~5h) |

## Conventions

- `setup.py` declares the package `agenda-presidencial` but the scripts are not packaged as a module — they are run as standalone files.
- DB column names are UPPER_SNAKE_CASE English; everything else (module names, function names, comments, UI strings) is Portuguese.
- `dashboard.py` registers a Dash callback exception suppressor but currently defines no callbacks — all figures are static, computed at import time.
