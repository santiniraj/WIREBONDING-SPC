# %% [markdown]
# # Stage 7–8 — DSS and Thesis Outputs
#
# The dashboard has only two pages:
# 1. Production / Engineering View
# 2. Research / Admin View
#
# Run the complete pipeline first, then generate report-safe figures and launch Streamlit.

# %%
from src.thesis_outputs import main as generate_thesis_outputs

generate_thesis_outputs()
print("\nOpen outputs/thesis/OUTPUT_INDEX.csv to see all report-ready files.")

# %% [markdown]
# ## Launch dashboard from the VS Code terminal
#
# ```bat
# python -m streamlit run dashboard/app.py
# ```
