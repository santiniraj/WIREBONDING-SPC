# Research Design ↔ Coding Alignment

## Research Question 1 / Objective 1

**Question:** How can historical SPC data be pre-processed and analysed to identify rule violations and abnormal wire-bond behaviour?

**Coding flow:**

1. `src/data_cleaning.py`
   - validates WP/BS/SP sheets;
   - pairs `ITEM_VAL` and `FAILURE_CODE` rows into one event;
   - keeps VALUE1–VALUE5 paired with BREAK_MODE1–BREAK_MODE5;
   - parses dates, numerical measurements and specification limits;
   - removes unusable events and normalizes categories.
2. Study population
   - selects the Package + Wire Type + Wire Diameter combination with the largest usable record count;
   - retains all associated machines;
   - anonymizes machine IDs before public output.
3. `src/spc.py`
   - builds X̄–R reference limits from training-period history only;
   - applies Nelson Rules N1–N8;
   - derives capability indices;
   - labels each event Normal / Warning / Abnormal.

## Research Question 2 / Objective 2

**Question:** To what extent can ARIMA and MLP improve future process-deviation prediction compared with traditional SPC monitoring?

**Fork design:**

```text
                       ┌── ARIMA: forecast standardized SPC deviation to event +5
Cleaned SPC + Rules ───┤
                       └── MLP: classify event +5 as Normal/Warning/Abnormal
```

The branches are independent. MLP uses engineered SPC features, not ARIMA output.

### ARIMA

- per Machine–CTQ stream;
- candidate orders `(1,0,0)`, `(1,1,0)`, `(0,1,1)`, `(1,1,1)`;
- validation chooses lowest event+5 RMSE;
- test uses expanding-origin fifth-event forecasts;
- metrics: MAE, MSE, RMSE, naïve RMSE and 95% interval coverage.

### MLP

- three-class event+5 classifier;
- current SPC statistics + capability + Nelson + lag/rolling history + anonymized Machine/CTQ;
- chronological train/validation/test data;
- random oversampling only on development data;
- validation Macro F1 selects architecture;
- final test: Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC and confusion matrix.

## Research Question 3 / Objective 3

**Question:** How can the integrated prediction be converted into decision support?

- `evaluate_fork()` compares SPC Persistence, ARIMA, MLP and Hybrid on the same common test observations.
- Hybrid is decision-level integration, not a sequential model.
- `dashboard/app.py` gives a two-page Production and Research/Admin prototype.
- DSS action is guidance for engineering review, not autonomous equipment control.
