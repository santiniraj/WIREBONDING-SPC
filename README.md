# WirebondingSPC — Unified Anonymous Master Project

This is the **single working project folder** for VS Code, thesis analysis, Streamlit and a private GitHub repository.

The project follows the master-project research design:

**Historical SPC data → data preparation → population selection → X̄–R SPC + Nelson Rules → feature engineering → parallel ARIMA + MLP Fork → common-test evaluation → engineering DSS.**

ARIMA and MLP are independent parallel branches. MLP does **not** use the ARIMA forecast as an input. The two branches are joined only at the decision level.

## Data used by this project

The default source is:

```text
data/SPC_Events_Anonymous.csv
```

It contains the full anonymized event-level population needed to reproduce the analysis. Commercial identifiers have been replaced with generic codes and calendar dates are rebased to a neutral timeline. There is no reverse mapping in this project.

The raw company workbook is **not required** to run this version.

## Folder structure

```text
WirebondingSPC/
├── config.py
├── run_all.py
├── requirements.txt
├── README.md
├── ANONYMIZATION_GUIDE.md
├── GITHUB_UPLOAD_GUIDE.md
├── data/
│   ├── SPC_Events_Anonymous.csv
│   ├── FAILURE_MODE_MAPPING.csv
│   └── README_DATA.md
├── src/
│   ├── anonymization.py
│   ├── data_cleaning.py
│   ├── spc.py
│   ├── features.py
│   ├── models.py
│   ├── report_summary.py
│   └── thesis_outputs.py
├── stages/
│   ├── 01_data_cleaning.py
│   ├── 02_population_selection.py
│   ├── 03_spc_nelson_capability.py
│   ├── 04_feature_engineering_split.py
│   ├── 05_arima_forecasting.py
│   ├── 06_mlp_classification.py
│   ├── 07_hybrid_evaluation.py
│   └── 08_dashboard_and_outputs.py
├── dashboard/
│   └── app.py
├── docs/
│   ├── RESEARCH_ALIGNMENT.md
│   ├── CHAPTER_4_5_OUTPUT_MAP.md
│   ├── VS_CODE_WORKFLOW.md
│   └── VIVA_FLOW.md
└── outputs/
    ├── data/
    ├── eda/
    ├── spc/
    ├── models/
    ├── evaluation/
    ├── deployment/
    └── thesis/
```

## Run in VS Code

Open the folder in VS Code and run:

```bat
cd /d "C:\Users\User\src\WirebondingSPC"
python -m pip install -r requirements.txt
python run_all.py
```

The pipeline has eight stages:

```text
[1/8] Data preparation and cleaning
[2/8] EDA and study population selection
[3/8] SPC, capability, Nelson Rules and break-mode analysis
[4/8] Feature engineering and training / validation / test preparation
[5/8] Parallel Fork modelling: MLP and ARIMA
[6/8] Common-test evaluation and Hybrid integration
[7/8] Engineering DSS output
[8/8] Thesis-ready tables and figures
```

Wait until the terminal shows:

```text
PIPELINE COMPLETE
```

## Open the dashboard

```bat
python -m streamlit run dashboard/app.py
```

The dashboard contains only two views:

1. **Production / Engineering View** — current SPC, X̄/R, Nelson signals, capability, break mode, ARIMA +5, MLP +5 and engineering action.
2. **Research / Admin View** — data preparation, SPC, ARIMA, MLP and common-test model evaluation for thesis/VIVA evidence.

## Chapter 4 and Chapter 5 outputs

Use:

```text
outputs/thesis/
```

The figures and CSV tables in this folder are generated directly by the code. `OUTPUT_INDEX.csv` lists the available thesis evidence.

Use `outputs/RUN_SUMMARY.md` for the latest numerical results after every full run.

## VS Code interactive stage files

The `stages/` scripts contain `# %% [markdown]` and `# %%` cells. They are designed so that each research stage can be explained and run separately during review or VIVA.

## Interpretation rules

- Nelson Rules indicate statistical non-random behaviour, not confirmed physical root cause.
- ARIMA forecasts standardized SPC trend to the fifth future event.
- MLP predicts the fifth-next process state from SPC-derived features.
- Hybrid uses the more severe ARIMA/MLP future state as conservative decision support.
- Capability and break-mode information are supporting engineering evidence.
- Dashboard actions are recommendations only; they do not automatically stop equipment, change bonding parameters or disposition product.
