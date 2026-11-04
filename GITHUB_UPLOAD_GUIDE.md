# Private GitHub Upload Guide

You can use this same folder as your local project and private GitHub repository. No second project copy is required.

## Recommended files to keep in the repository

- `src/`
- `stages/`
- `dashboard/`
- `docs/`
- `data/SPC_Events_Anonymous.csv`
- `data/FAILURE_MODE_MAPPING.csv`
- `config.py`
- `run_all.py`
- `requirements.txt`
- `README.md`
- `ANONYMIZATION_GUIDE.md`
- `.gitignore`

## Optional

You may also upload `outputs/thesis/` if you want the final figures/tables stored with the project.

Large regenerated folders such as `outputs/models/` or all intermediate outputs are optional because `python run_all.py` recreates them.

## Do not add later

Do not place the original company workbook, reverse ID mappings, customer/product names or other confidential raw files inside this repository folder.
