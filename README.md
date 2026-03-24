# evolve_stats

Statistical analysis and visualization module for the **WHO Global Health Estimates** project. This package provides Python-based exploratory data analysis (EDA), charting, and reporting on top of the SQL data warehouse defined in the sibling `sql/` directory.

## Overview

The project analyzes World Health Organization (WHO) data to answer:

> How has life expectancy and the burden of disease mortality evolved globally? What are the most lethal diseases — both communicable and non-communicable — per country, and how many people die from them?

### Key Focus Areas

- **Life expectancy evolution** — Global trends from 2000-2024, regional and gender variations
- **Non-communicable disease (NCD) mortality** — Cancers, diabetes, cardiovascular disease, respiratory diseases
- **Communicable disease deaths** — TB, malaria, HIV/AIDS, Hepatitis B and C
- **Epidemiological transition** — Shift from communicable to non-communicable disease burden
- **Geographic & demographic patterns** — 228 countries across 6 WHO regions

## Tech Stack

- **Python 3.10+**
- pandas, matplotlib, seaborn — data manipulation and visualization
- scikit-learn, scipy, numpy — statistical analysis
- PyMySQL — MySQL database connection
- **MySQL 8.0+** (via Docker) — data warehouse backend

## Project Structure

```
evolve_stats/
├── README.md
├── .gitignore
└── (Python analysis modules — in progress)
```

The companion SQL project lives at `../sql/sql_final_work/` and contains:

```
sql_final_work/
├── sql/                        # Layered SQL pipeline
│   ├── 01_schema.sql           # Database schema (staging + core)
│   ├── 02_load_staging.sql     # CSV imports into staging
│   ├── 03_transform_core.sql   # Core fact/dimension tables
│   ├── 04_semantic_views.sql   # Semantic layer views
│   ├── 05_analysis_queries.sql # 16 analytical queries
│   ├── 06_quality_checks.sql   # Data validation
│   └── 07_advanced_sql.sql     # Window functions, CTEs, procedures
├── data/                       # Raw CSVs + download script
├── generate_charts.py          # 20 visualization PNGs + HTML report
├── charts/                     # Output PNGs
└── docker-compose.yml          # MySQL container setup
```

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r ../requirements.txt

# 3. Start MySQL (from sql/sql_final_work/)
docker-compose up -d

# 4. Run the SQL pipeline (01 through 04) to populate the database
```

## Data Model

The SQL warehouse follows a **Staging -> Core -> Semantic** architecture:

- **Staging**: raw CSV imports (`stg_health_estimates_raw`, `stg_ncd_deaths_raw`, `stg_communicable_deaths_raw`, `stg_countries_raw`)
- **Core**: typed fact and dimension tables (`fct_health_estimate`, `fct_ncd_deaths`, `fct_communicable_deaths`, `dim_country`, `dim_indicator`, `dim_sex`, `dim_cause`, `dim_disease`)
- **Semantic**: enriched views for analysis (`vw_health_enriched`, `vw_yearly_kpi`, `vw_deaths_enriched`, `vw_communicable_enriched`)

## Data Source

All data is sourced from the [WHO Global Health Observatory (GHO) OData API](https://www.who.int/data/gho).

## Related Projects

Part of the `evolve_master` workspace alongside:

- **epa_project** — Spanish labour market analysis (EPA data)
- **ree_project** — Spanish electrical grid analysis (REE data)
