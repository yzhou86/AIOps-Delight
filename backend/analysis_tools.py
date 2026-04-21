import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from langchain_community.chat_models.tongyi import ChatTongyi
except Exception:
    ChatTongyi = None


ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT_DIR / "tools"

TOOL_CATALOG = [
    {
        "id": "data_profile",
        "name": "Dataset Profile",
        "category": "Foundation",
        "description": "Summarize schema, missing values, duplicates, and numeric distributions.",
        "source": "new backend workflow",
        "requires": [],
    },
    {
        "id": "correlation_explorer",
        "name": "Correlation Explorer",
        "category": "Signals",
        "description": "Surface the strongest positive and negative numeric relationships.",
        "source": "tools/python/xgb_param.py",
        "requires": ["numeric columns"],
    },
    {
        "id": "anomaly_detector",
        "name": "Anomaly Detector",
        "category": "Operations",
        "description": "Use isolation-forest style scoring to flag unusual rows.",
        "source": "tools/python/anomaly_detect.py",
        "requires": ["numeric columns"],
    },
    {
        "id": "kmeans_segmentation",
        "name": "KMeans Segmentation",
        "category": "Grouping",
        "description": "Group similar records into numeric segments and compare centroids.",
        "source": "tools/python/cluster_kmeans.py",
        "requires": ["numeric columns"],
    },
    {
        "id": "text_clusterer",
        "name": "Text Clusterer",
        "category": "NLP",
        "description": "Cluster free-text fields with TF-IDF keywords for topic discovery.",
        "source": "tools/python/cluster_model.py",
        "requires": ["text columns"],
    },
    {
        "id": "forecast_baseline",
        "name": "Forecast Baseline",
        "category": "Time Series",
        "description": "Build a lightweight trend forecast from a time column plus a numeric value.",
        "source": "tools/python/job_resource_predict.py",
        "requires": ["time column", "value column"],
    },
    {
        "id": "classification_explorer",
        "name": "Classification Explorer",
        "category": "Supervised",
        "description": "Train a baseline classifier against a selected target column and rank feature importance.",
        "source": "tools/python/xgb_param.py",
        "requires": ["target column"],
    },
]


def serialize_tool_catalog():
    return TOOL_CATALOG


def load_dataframe(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        encodings = ["utf-8", "utf-8-sig", "gbk", "latin1"]
        last_error = None
        for encoding in encodings:
            try:
                dataframe = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        else:
            raise ValueError(f"Could not decode CSV file: {last_error}")
    elif suffix in {".xls", ".xlsx"}:
        dataframe = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Upload CSV, XLS, or XLSX.")

    if dataframe.empty:
        raise ValueError("The uploaded file is empty.")

    dataframe.columns = _dedupe_columns([str(column).strip() or f"column_{idx + 1}" for idx, column in enumerate(dataframe.columns)])
    return dataframe


def inspect_dataframe(dataframe, filename=None, dataset_id=None):
    column_profiles = []
    numeric_columns = []
    categorical_columns = []
    text_columns = []
    datetime_columns = []

    for column in dataframe.columns:
        series = dataframe[column]
        kind = _infer_column_kind(series)
        profile = {
            "name": column,
            "kind": kind,
            "dtype": str(series.dtype),
            "nonNullCount": int(series.notna().sum()),
            "missingCount": int(series.isna().sum()),
            "uniqueCount": int(series.nunique(dropna=True)),
        }
        column_profiles.append(profile)
        if kind == "numeric":
            numeric_columns.append(column)
        elif kind == "datetime":
            datetime_columns.append(column)
        elif kind == "text":
            text_columns.append(column)
        else:
            categorical_columns.append(column)

    return {
        "datasetId": dataset_id,
        "fileName": filename,
        "rowCount": int(len(dataframe)),
        "columnCount": int(len(dataframe.columns)),
        "columns": column_profiles,
        "numericColumns": numeric_columns,
        "categoricalColumns": categorical_columns,
        "textColumns": text_columns,
        "datetimeColumns": datetime_columns,
        "preview": _frame_to_table(dataframe.head(8)),
    }


def run_tool(tool_id, dataframe, context):
    runners = {
        "data_profile": run_data_profile,
        "correlation_explorer": run_correlation_explorer,
        "anomaly_detector": run_anomaly_detector,
        "kmeans_segmentation": run_kmeans_segmentation,
        "text_clusterer": run_text_clusterer,
        "forecast_baseline": run_forecast_baseline,
        "classification_explorer": run_classification_explorer,
    }
    if tool_id not in runners:
        raise ValueError(f"Unknown tool: {tool_id}")
    return runners[tool_id](dataframe, context)


def run_data_profile(dataframe, context):
    info = context["dataset_info"]
    duplicate_rows = int(dataframe.duplicated().sum())
    missing_rows = []
    for column in info["columns"]:
        if column["missingCount"] > 0:
            missing_rows.append(
                {
                    "column": column["name"],
                    "missing_count": column["missingCount"],
                    "missing_pct": round(column["missingCount"] / max(info["rowCount"], 1), 4),
                }
            )
    missing_rows = sorted(missing_rows, key=lambda item: item["missing_count"], reverse=True)[:10]

    numeric_frame = _numeric_frame(dataframe)
    tables = []
    if not numeric_frame.empty:
        summary = numeric_frame.describe().transpose().round(3).reset_index().rename(columns={"index": "column"})
        tables.append(_table_payload("Numeric Summary", summary.head(10)))
    if missing_rows:
        tables.append(_rows_table("Missing Values", missing_rows))

    insights = [
        f"The dataset contains {info['rowCount']} rows and {info['columnCount']} columns.",
        f"Detected {len(info['numericColumns'])} numeric, {len(info['categoricalColumns'])} categorical, {len(info['textColumns'])} text, and {len(info['datetimeColumns'])} datetime-like columns.",
        f"Found {duplicate_rows} duplicate rows.",
    ]
    if missing_rows:
        worst = missing_rows[0]
        insights.append(
            f"The most incomplete field is {worst['column']} with {worst['missing_count']} missing values."
        )

    return _result(
        "data_profile",
        "ok",
        f"Profiled {info['rowCount']} rows across {info['columnCount']} columns.",
        insights,
        tables=tables,
    )


def run_correlation_explorer(dataframe, context):
    numeric_frame = _numeric_frame(dataframe)
    if numeric_frame.shape[1] < 2:
        return _skipped("correlation_explorer", "Need at least two numeric columns for correlation analysis.")

    corr = numeric_frame.corr(numeric_only=True)
    pairs = []
    columns = corr.columns.tolist()
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "correlation": round(float(value), 4),
                        "absolute_correlation": round(abs(float(value)), 4),
                    }
                )

    if not pairs:
        return _skipped("correlation_explorer", "No usable numeric relationships were found.")

    strongest = sorted(pairs, key=lambda item: item["absolute_correlation"], reverse=True)[:10]
    top = strongest[0]
    direction = "positive" if top["correlation"] >= 0 else "negative"
    focus_columns = list(dict.fromkeys([pair["left"] for pair in strongest] + [pair["right"] for pair in strongest]))[:6]
    matrix = corr.loc[focus_columns, focus_columns].round(3).reset_index().rename(columns={"index": "column"})

    insights = [
        f"The strongest {direction} relationship is {top['left']} vs {top['right']} with correlation {top['correlation']}.",
        "Use these pairs to guide feature engineering, alert thresholds, or dashboard drill-downs.",
    ]

    return _result(
        "correlation_explorer",
        "ok",
        f"Computed numeric correlations across {numeric_frame.shape[1]} columns.",
        insights,
        tables=[
            _rows_table("Strongest Correlations", strongest),
            _table_payload("Correlation Matrix", matrix),
        ],
    )


def run_anomaly_detector(dataframe, context):
    numeric_frame = _numeric_frame(dataframe)
    if numeric_frame.shape[1] < 1 or len(numeric_frame) < 20:
        return _skipped("anomaly_detector", "Need at least 20 rows and one numeric column for anomaly detection.")

    imputer = SimpleImputer(strategy="median")
    filled = imputer.fit_transform(numeric_frame)
    scaled = StandardScaler().fit_transform(filled)
    contamination = max(0.02, min(0.08, 12 / len(numeric_frame)))
    model = IsolationForest(random_state=42, contamination=contamination)
    labels = model.fit_predict(scaled)
    scores = -model.score_samples(scaled)

    anomaly_mask = labels == -1
    anomaly_count = int(anomaly_mask.sum())
    if anomaly_count == 0:
        return _skipped("anomaly_detector", "No strong anomalies were flagged with the current baseline settings.")

    scored = dataframe.copy()
    scored["anomaly_score"] = scores
    anomalies = scored.loc[anomaly_mask].sort_values("anomaly_score", ascending=False)
    preview_columns = list(numeric_frame.columns[:5]) + ["anomaly_score"]
    preview = anomalies[preview_columns].head(10).round(4)

    insights = [
        f"Flagged {anomaly_count} rows as anomalous, about {anomaly_count / max(len(dataframe), 1):.1%} of the dataset.",
        "Review the highest-scoring rows first; they are the farthest from the dataset baseline.",
    ]

    return _result(
        "anomaly_detector",
        "ok",
        f"Flagged {anomaly_count} anomalous rows.",
        insights,
        tables=[_table_payload("Top Anomalies", preview)],
    )


def run_kmeans_segmentation(dataframe, context):
    numeric_frame = _numeric_frame(dataframe)
    if numeric_frame.shape[1] < 1 or len(numeric_frame) < 8:
        return _skipped("kmeans_segmentation", "Need at least 8 rows and one numeric column for KMeans segmentation.")

    imputer = SimpleImputer(strategy="median")
    filled = imputer.fit_transform(numeric_frame)
    scaled = StandardScaler().fit_transform(filled)
    cluster_count = max(2, min(4, int(round(math.sqrt(len(numeric_frame) / 2)))))

    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)

    segmented = dataframe.copy()
    segmented["segment"] = labels
    segment_sizes = (
        segmented["segment"]
        .value_counts()
        .sort_index()
        .rename_axis("segment")
        .reset_index(name="rows")
    )
    segment_sizes["share"] = (segment_sizes["rows"] / len(segmented)).round(4)

    centroids = segmented.groupby("segment")[numeric_frame.columns].mean().round(3).reset_index()
    insights = [
        f"Built {cluster_count} numeric segments from {numeric_frame.shape[1]} numeric columns.",
        f"Segment {segment_sizes.iloc[0]['segment']} is the largest group with {segment_sizes.iloc[0]['rows']} rows.",
    ]

    return _result(
        "kmeans_segmentation",
        "ok",
        f"Built {cluster_count} KMeans segments.",
        insights,
        tables=[
            _table_payload("Segment Sizes", segment_sizes),
            _table_payload("Segment Centroids", centroids),
        ],
    )


def run_text_clusterer(dataframe, context):
    text_columns = context.get("text_columns") or context["dataset_info"]["textColumns"]
    usable_columns = [column for column in text_columns if column in dataframe.columns]
    if not usable_columns:
        return _skipped("text_clusterer", "Select at least one text column to run text clustering.")

    corpus = dataframe[usable_columns].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    corpus = corpus[corpus.str.len() > 0]
    if len(corpus) < 8:
        return _skipped("text_clusterer", "Need at least 8 non-empty text rows for clustering.")

    vectorizer = TfidfVectorizer(max_features=400)
    matrix = vectorizer.fit_transform(corpus)
    if matrix.shape[1] < 2:
        return _skipped("text_clusterer", "The text columns do not contain enough vocabulary diversity.")

    cluster_count = max(2, min(4, int(round(math.sqrt(len(corpus) / 2)))))
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)

    feature_names = vectorizer.get_feature_names_out()
    order = model.cluster_centers_.argsort()[:, ::-1]
    cluster_rows = []
    labeled = pd.DataFrame({"text": corpus.values, "cluster": labels})
    for cluster_id in range(cluster_count):
        terms = [feature_names[idx] for idx in order[cluster_id][:5]]
        cluster_rows.append(
            {
                "cluster": cluster_id,
                "rows": int((labels == cluster_id).sum()),
                "top_terms": ", ".join(term for term in terms if term),
            }
        )

    preview = labeled.sort_values(["cluster", "text"]).head(10).copy()
    preview["text"] = preview["text"].str.slice(0, 120)

    insights = [
        f"Clustered {len(corpus)} text rows across {cluster_count} topic groups.",
        "Use the top terms to label recurring themes, error modes, or user intents.",
    ]

    return _result(
        "text_clusterer",
        "ok",
        f"Grouped text into {cluster_count} topical clusters.",
        insights,
        tables=[
            _rows_table("Topic Clusters", cluster_rows),
            _table_payload("Sample Text by Cluster", preview),
        ],
    )


def run_forecast_baseline(dataframe, context):
    time_column = context.get("time_column") or _pick_datetime_column(context["dataset_info"])
    value_column = context.get("value_column") or _pick_numeric_column(context["dataset_info"])

    if not time_column or not value_column:
        return _skipped("forecast_baseline", "Choose a time column and a numeric value column for forecasting.")
    if time_column not in dataframe.columns or value_column not in dataframe.columns:
        return _skipped("forecast_baseline", "The selected forecast columns were not found in the dataset.")

    forecast_frame = dataframe[[time_column, value_column]].copy()
    forecast_frame[time_column] = pd.to_datetime(forecast_frame[time_column], errors="coerce")
    forecast_frame[value_column] = pd.to_numeric(forecast_frame[value_column], errors="coerce")
    forecast_frame = forecast_frame.dropna().sort_values(time_column)
    forecast_frame = forecast_frame.groupby(time_column, as_index=False)[value_column].mean()

    if len(forecast_frame) < 8:
        return _skipped("forecast_baseline", "Need at least 8 time points for a baseline forecast.")

    x = np.arange(len(forecast_frame)).reshape(-1, 1)
    y = forecast_frame[value_column].to_numpy()
    model = LinearRegression()
    model.fit(x, y)
    training_score = float(model.score(x, y))

    horizon = max(3, min(6, int(round(len(forecast_frame) * 0.15))))
    future_x = np.arange(len(forecast_frame), len(forecast_frame) + horizon).reshape(-1, 1)
    future_values = model.predict(future_x)

    deltas = forecast_frame[time_column].diff().dropna()
    median_delta = deltas.median() if not deltas.empty else pd.Timedelta(days=1)
    if pd.isna(median_delta) or median_delta == pd.Timedelta(0):
        median_delta = pd.Timedelta(days=1)

    last_time = forecast_frame[time_column].iloc[-1]
    future_times = [last_time + median_delta * step for step in range(1, horizon + 1)]
    forecast_rows = pd.DataFrame(
        {
            "forecast_time": future_times,
            "predicted_value": np.round(future_values, 4),
        }
    )

    slope_direction = "upward" if model.coef_[0] >= 0 else "downward"
    insights = [
        f"The baseline trend is {slope_direction} for {value_column}.",
        f"The simple linear fit explains about {training_score:.1%} of the observed variance, so treat it as a directional forecast.",
    ]

    return _result(
        "forecast_baseline",
        "ok",
        f"Forecasted {horizon} future points for {value_column}.",
        insights,
        tables=[_table_payload("Forecast", forecast_rows)],
    )


def run_classification_explorer(dataframe, context):
    target_column = context.get("target_column")
    if not target_column:
        return _skipped("classification_explorer", "Choose a target column to run supervised classification.")
    if target_column not in dataframe.columns:
        return _skipped("classification_explorer", "The selected target column does not exist in the dataset.")

    modeling = dataframe.copy()
    modeling = modeling.dropna(subset=[target_column])
    if len(modeling) < 30:
        return _skipped("classification_explorer", "Need at least 30 labeled rows to train a baseline classifier.")

    target = modeling[target_column]
    unique_values = target.nunique(dropna=True)
    if unique_values < 2 or unique_values > 12:
        return _skipped("classification_explorer", "The target column should contain between 2 and 12 classes.")

    features = modeling.drop(columns=[target_column]).copy()
    if features.empty:
        return _skipped("classification_explorer", "No feature columns remain after removing the target.")

    for column in features.columns:
        if _infer_column_kind(features[column]) == "datetime":
            parsed = pd.to_datetime(features[column], errors="coerce")
            features[column] = parsed.view("int64").replace(-9223372036854775808, np.nan)

    features = pd.get_dummies(features, dummy_na=True)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.loc[:, features.notna().any()]
    if features.empty:
        return _skipped("classification_explorer", "Features could not be converted into a usable training matrix.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target if unique_values <= max(2, len(target) // 5) else None,
    )

    imputer = SimpleImputer(strategy="median")
    x_train_ready = imputer.fit_transform(x_train)
    x_test_ready = imputer.transform(x_test)

    model = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(x_train_ready, y_train)
    predictions = model.predict(x_test_ready)

    accuracy = float(accuracy_score(y_test, predictions))
    weighted_f1 = float(f1_score(y_test, predictions, average="weighted"))
    feature_importance = (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
    )

    class_balance = (
        target.value_counts()
        .rename_axis("class")
        .reset_index(name="rows")
    )

    insights = [
        f"Baseline classification accuracy is {accuracy:.1%} with weighted F1 of {weighted_f1:.1%}.",
        "Treat this as a quick benchmark to identify signal strength before a fuller modeling pass.",
    ]

    return _result(
        "classification_explorer",
        "ok",
        f"Trained a baseline classifier on target '{target_column}'.",
        insights,
        tables=[
            _table_payload("Class Balance", class_balance),
            _table_payload("Top Feature Importance", feature_importance.round(4)),
        ],
    )


def build_agent_summary(dataset_info, prompt, results):
    successful = [result for result in results if result["status"] == "ok"]
    skipped = [result for result in results if result["status"] == "skipped"]

    fallback_lines = [
        f"Analysis finished for {dataset_info['fileName']} with {dataset_info['rowCount']} rows and {dataset_info['columnCount']} columns.",
    ]
    if prompt:
        fallback_lines.append(f"Prompt focus: {prompt}")
    if successful:
        fallback_lines.append("Completed tools:")
        for result in successful:
            fallback_lines.append(f"- {result['toolName']}: {result['headline']}")
    if skipped:
        fallback_lines.append("Skipped tools:")
        for result in skipped:
            fallback_lines.append(f"- {result['toolName']}: {result['headline']}")

    llm_summary = _maybe_build_llm_summary(dataset_info, prompt, results)
    return llm_summary or "\n".join(fallback_lines)


def build_agent_answer(dataset_info, prompt, results, chat_history=None):
    fallback_answer = _build_fallback_answer(dataset_info, prompt, results)
    llm_answer = _maybe_build_llm_answer(dataset_info, prompt, results, chat_history or [])
    return llm_answer or fallback_answer


def _build_fallback_answer(dataset_info, prompt, results):
    successful = [result for result in results if result["status"] == "ok"]
    skipped = [result for result in results if result["status"] == "skipped"]

    lines = []
    if prompt:
        lines.append(f"For your question, my short answer is based on {dataset_info['fileName']} and the tools you selected.")
    else:
        lines.append(f"I analyzed {dataset_info['fileName']} and here is the most useful answer I can give.")

    if successful:
        top = successful[0]
        lines.append(f"The strongest signal from the current run is: {top['headline']}")
        if top["insights"]:
            lines.append(top["insights"][0])

    if len(successful) > 1:
        lines.append("Other completed tools also added supporting evidence:")
        for result in successful[1:3]:
            lines.append(f"- {result['toolName']}: {result['headline']}")

    if skipped:
        lines.append("A few tools could not add signal with the current dataset or settings:")
        for result in skipped[:2]:
            lines.append(f"- {result['toolName']}: {result['headline']}")

    lines.append("You can review the detailed tables below for the evidence behind this answer.")
    return "\n".join(lines)


def _maybe_build_llm_summary(dataset_info, prompt, results):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key or ChatTongyi is None:
        return None

    compact_results = []
    for result in results:
        compact_results.append(
            {
                "tool": result["toolName"],
                "status": result["status"],
                "headline": result["headline"],
                "insights": result["insights"][:3],
                "warnings": result["warnings"][:2],
            }
        )

    model = ChatTongyi(api_key=api_key, model=os.getenv("DASHSCOPE_MODEL", "qwen-turbo"))
    summary_prompt = (
        "You are an AI data science copilot. Summarize the uploaded dataset analysis in concise business language. "
        "Call out the strongest signals, caveats, and next steps.\n\n"
        f"User prompt: {prompt or 'No additional prompt provided.'}\n"
        f"Dataset info: {json.dumps(dataset_info, ensure_ascii=False)}\n"
        f"Tool results: {json.dumps(compact_results, ensure_ascii=False)}"
    )
    try:
        response = model.invoke(summary_prompt)
        return getattr(response, "content", None) or str(response)
    except Exception:
        return None


def _maybe_build_llm_answer(dataset_info, prompt, results, chat_history):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key or ChatTongyi is None or not prompt:
        return None

    compact_results = []
    for result in results:
        compact_results.append(
            {
                "tool": result["toolName"],
                "status": result["status"],
                "headline": result["headline"],
                "insights": result["insights"][:3],
                "warnings": result["warnings"][:2],
            }
        )

    history_window = chat_history[-6:]
    model = ChatTongyi(api_key=api_key, model=os.getenv("DASHSCOPE_MODEL", "qwen-turbo"))
    answer_prompt = (
        "You are an AI data science copilot inside an AIOps analytics workspace. "
        "Answer the user's question directly and clearly using the dataset context and tool results. "
        "Be conversational but concise. Mention caveats when the tools are weak or skipped. "
        "Do not invent data beyond the supplied context.\n\n"
        f"Dataset info: {json.dumps(dataset_info, ensure_ascii=False)}\n"
        f"Recent chat history: {json.dumps(history_window, ensure_ascii=False)}\n"
        f"Tool results: {json.dumps(compact_results, ensure_ascii=False)}\n"
        f"User question: {prompt}"
    )
    try:
        response = model.invoke(answer_prompt)
        return getattr(response, "content", None) or str(response)
    except Exception:
        return None


def _infer_column_kind(series):
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return "categorical"

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    parse_ratio = parsed.notna().mean()
    if parse_ratio >= 0.8:
        return "datetime"

    average_length = sample.str.len().mean()
    unique_ratio = sample.nunique() / max(len(sample), 1)
    if average_length >= 18 or unique_ratio >= 0.7:
        return "text"
    return "categorical"


def _pick_datetime_column(dataset_info):
    columns = dataset_info.get("datetimeColumns") or []
    return columns[0] if columns else None


def _pick_numeric_column(dataset_info):
    columns = dataset_info.get("numericColumns") or []
    return columns[0] if columns else None


def _numeric_frame(dataframe):
    numeric = dataframe.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.loc[:, numeric.notna().sum() > 0]
    return numeric


def _result(tool_id, status, headline, insights, warnings=None, tables=None):
    tool = _tool_by_id(tool_id)
    return {
        "toolId": tool_id,
        "toolName": tool["name"],
        "status": status,
        "headline": headline,
        "insights": insights,
        "warnings": warnings or [],
        "tables": tables or [],
    }


def _skipped(tool_id, reason):
    return _result(tool_id, "skipped", reason, [], warnings=[reason], tables=[])


def _table_payload(title, dataframe):
    if isinstance(dataframe, pd.Series):
        dataframe = dataframe.to_frame().reset_index()
    return {
        "title": title,
        "columns": [str(column) for column in dataframe.columns],
        "rows": [_sanitize_record(record) for record in dataframe.to_dict(orient="records")],
    }


def _rows_table(title, rows):
    if not rows:
        return {"title": title, "columns": [], "rows": []}
    return {
        "title": title,
        "columns": list(rows[0].keys()),
        "rows": [_sanitize_record(row) for row in rows],
    }


def _frame_to_table(dataframe):
    return {
        "columns": [str(column) for column in dataframe.columns],
        "rows": [_sanitize_record(record) for record in dataframe.to_dict(orient="records")],
    }


def _sanitize_record(record):
    return {str(key): _sanitize_value(value) for key, value in record.items()}


def _sanitize_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value) if isinstance(value, Path) else value


def _dedupe_columns(columns):
    seen = {}
    deduped = []
    for column in columns:
        count = seen.get(column, 0)
        if count == 0:
            deduped.append(column)
        else:
            deduped.append(f"{column}_{count + 1}")
        seen[column] = count + 1
    return deduped


def _tool_by_id(tool_id):
    for tool in TOOL_CATALOG:
        if tool["id"] == tool_id:
            return tool
    raise ValueError(f"Unknown tool: {tool_id}")
