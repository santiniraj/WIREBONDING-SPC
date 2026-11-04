# Anonymization Guide

This project uses one anonymized working dataset for coding, thesis figures, dashboard and private GitHub use.

## What is anonymized in the source dataset

- machine identifiers → generic source codes, then report-facing `WB01`, `WB02`, ... labels;
- package → generic package groups;
- wire type → generic wire-type groups;
- wire diameter → generic diameter groups;
- device, lot and record identifiers → generic codes;
- calendar dates → rebased neutral timeline while preserving sequence and elapsed intervals;
- plant/customer/commercial identifiers → not included in the distributed source dataset.

No reverse mapping is included in the project.

## Thesis and dashboard outputs

Report-facing outputs further remove or mask sensitive fields. The X̄/R visualizations use standardized or normalized SPC scales where appropriate.

## Important

The anonymization is designed to preserve the analytical relationships required for:

- population selection;
- X̄–R SPC;
- Nelson Rules;
- capability analysis;
- ARIMA forecasting;
- MLP classification;
- common-test Hybrid evaluation.

The project can therefore be run without the original company workbook.
