# Simple VIVA Coding Explanation

Use this sequence when explaining the implementation:

1. **Data** — WP, BS and SP historical SPC records are cleaned and paired into event-level records.
2. **Population** — one consistent package/wire/diameter population is selected; all associated machines remain.
3. **SPC** — X̄–R and Nelson Rules identify present statistical behaviour.
4. **State** — each eligible event is labelled Normal, Warning or Abnormal.
5. **Fork** — ARIMA forecasts the next five-event trend while MLP independently predicts the fifth-next process state.
6. **Evaluate** — SPC Persistence, ARIMA, MLP and Hybrid are compared on common test observations.
7. **DSS** — current SPC + predictive evidence is shown in a two-page engineering dashboard.

Key sentence:

> ARIMA and MLP are parallel models. MLP is trained from cleaned SPC-derived features and does not use the ARIMA forecast. Their outputs are combined only at the decision stage.
