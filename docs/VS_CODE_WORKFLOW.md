# VS Code Workflow

## Recommended approach

1. Open `WirebondingSPC` as a folder in VS Code.
2. Select your Python/Conda interpreter.
3. Run the full pipeline once with `python run_all.py`.
4. Open the scripts in `stages/` to review each research stage. They contain VS Code interactive cells (`# %%`).
5. Use `outputs/thesis/OUTPUT_INDEX.csv` to locate report-ready evidence.
6. Start the dashboard only after the pipeline completes.

## Useful commands

```bat
python run_all.py
python -m streamlit run dashboard/app.py
pytest -q
```

## Why the project uses modules plus stage scripts

The `src/` modules are the source of truth for reproducible processing. The `stages/` scripts are readable learning/defence files so the code can be explained in a VIVA without scrolling through one very large program.
