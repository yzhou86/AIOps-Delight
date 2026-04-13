# Example Datasets

This folder contains sample CSV and Excel files for the unified AI analysis app.

## Files

### `ops_capacity_forecast.csv`

Best for:

- Dataset Profile
- Correlation Explorer
- Anomaly Detector
- Forecast Baseline

Suggested settings:

- Time column: `timestamp`
- Value column: `cpu_load` or `latency_ms`

### `incident_log_topics.csv`

Best for:

- Dataset Profile
- Text Clusterer
- KMeans Segmentation

Suggested settings:

- Text columns: `incident_summary`, `resolution_hint`

### `fraud_risk_classification.csv`

Best for:

- Dataset Profile
- Correlation Explorer
- Classification Explorer
- KMeans Segmentation

Suggested settings:

- Target column: `fraud_flag`

### `service_health_anomalies.xlsx`

This workbook contains two sheets:

- `service_health`: anomaly-friendly operations metrics
- `support_tickets`: mixed support/ticket text data

Best for:

- Dataset Profile
- Anomaly Detector
- Forecast Baseline
- Text Clusterer
- Classification Explorer

Suggested settings for `service_health`:

- Time column: `timestamp`
- Value column: `queue_depth` or `cpu_load`
- Target column: `incident_level`
- Text columns: `operator_note`

Suggested settings for `support_tickets`:

- Target column: `label`
- Text columns: `ticket_text`
