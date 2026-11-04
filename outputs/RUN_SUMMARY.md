# WirebondingSPC — Latest Pipeline Run Summary

This file is generated automatically after `python run_all.py`.

## 1. Selected study population

- Machines: **17**
- WP events: **603**
- BS events: **1387**
- SP events: **1356**
- Total events: **3346**

## 2. Model dataset

| SPLIT      |   ROWS |   PERCENT_OF_LABELLED |
|:-----------|-------:|----------------------:|
| TRAIN      |   1810 |              64.4128  |
| VALIDATION |    407 |              14.484   |
| TEST       |    341 |              12.1352  |
| PURGED     |    252 |               8.96797 |

## 3. Nelson Rule results

| RULE   |   TRIGGER_COUNT |   EPISODE_COUNT |
|:-------|----------------:|----------------:|
| N1     |             650 |             472 |
| N2     |              78 |              20 |
| N3     |              11 |              10 |
| N4     |               2 |               1 |
| N5     |             757 |             282 |
| N6     |             462 |             157 |
| N7     |               0 |               0 |
| N8     |             150 |              44 |

## 4. ARIMA

- Evaluated streams: **21**
- ARIMA RMSE lower than naïve: **18**
- Adequate-evidence streams: **13**

## 5. MLP

- Selected hidden layers: **(16,)**
- Alpha: **0.001**
- Accuracy: **0.3959**
- Balanced accuracy: **0.3594**
- Macro F1: **0.3526**
- Abnormal recall: **0.3293**

## 6. Common-test comparison

| MODEL           |   TEST_N |   ACCURACY |   BALANCED_ACCURACY |   MACRO_F1 |   RECALL_ABNORMAL |
|:----------------|---------:|-----------:|--------------------:|-----------:|------------------:|
| SPC Persistence |      305 |   0.47541  |            0.3977   |   0.397353 |          0.266667 |
| ARIMA           |      305 |   0.557377 |            0.356552 |   0.291855 |          0.04     |
| MLP             |      305 |   0.403279 |            0.36603  |   0.361586 |          0.333333 |
| Hybrid          |      305 |   0.406557 |            0.372076 |   0.366238 |          0.373333 |

## 7. Report-ready evidence

See `outputs/thesis/OUTPUT_INDEX.csv` and `docs/CHAPTER_4_5_OUTPUT_MAP.md`.

## Interpretation rule

ARIMA and MLP are parallel branches. MLP does not use ARIMA prediction as input. Hybrid integration occurs only at decision level.
