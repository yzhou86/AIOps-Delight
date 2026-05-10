import json
import os
import urllib.request
import uuid
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, g, jsonify, request, send_from_directory, session
from werkzeug.utils import secure_filename

try:
    from analysis_tools import (
        auto_select_tools,
        build_agent_answer,
        build_agent_answer_bundle,
        build_agent_summary,
        build_agent_summary_bundle,
        get_llm_runtime_info,
        inspect_dataframe,
        load_dataframe,
        run_tool,
        serialize_tool_catalog,
    )
    from dao import SqliteDao
    from pdf_export import build_chat_pdf
except ModuleNotFoundError:
    from .analysis_tools import (
        auto_select_tools,
        build_agent_answer,
        build_agent_answer_bundle,
        build_agent_summary,
        build_agent_summary_bundle,
        get_llm_runtime_info,
        inspect_dataframe,
        load_dataframe,
        run_tool,
        serialize_tool_catalog,
    )
    from .dao import SqliteDao
    from .pdf_export import build_chat_pdf


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "datapilot.db"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
GUEST_EXAMPLE_ID = "service_health_anomalies"
GUEST_TOOL_IDS = ["data_profile", "anomaly_detector"]

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
DATASET_STORE = {}

EXAMPLE_CATALOG = [
    {
        "id": "ops_capacity_forecast",
        "fileName": "ops_capacity_forecast.csv",
        "label": "Ops Capacity Forecast",
        "description": "Time-series operations metrics with load, latency, and queue depth for forecasting and anomaly checks.",
        "recommendedTools": ["data_profile", "correlation_explorer", "anomaly_detector", "forecast_baseline"],
    },
    {
        "id": "incident_log_topics",
        "fileName": "incident_log_topics.csv",
        "label": "Incident Log Topics",
        "description": "Free-text incident summaries and resolution hints for text clustering and segmentation.",
        "recommendedTools": ["data_profile", "text_clusterer", "kmeans_segmentation"],
    },
    {
        "id": "fraud_risk_classification",
        "fileName": "fraud_risk_classification.csv",
        "label": "Fraud Risk Classification",
        "description": "Tabular fraud-likelihood records for classification, segmentation, and signal discovery.",
        "recommendedTools": ["data_profile", "correlation_explorer", "classification_explorer"],
    },
    {
        "id": "service_health_anomalies",
        "fileName": "service_health_anomalies.xlsx",
        "label": "Service Health Anomalies",
        "description": "Workbook with service-health metrics and ticket text, useful for anomalies and mixed-signal analysis.",
        "recommendedTools": ["data_profile", "anomaly_detector", "forecast_baseline", "text_clusterer"],
    },
    {
        "id": "customer_churn_signals",
        "fileName": "customer_churn_signals.csv",
        "label": "Customer Churn Signals",
        "description": "Subscription-health and support-behavior data for churn classification and feature ranking.",
        "recommendedTools": ["data_profile", "correlation_explorer", "classification_explorer", "kmeans_segmentation"],
    },
    {
        "id": "cloud_cost_guardrails",
        "fileName": "cloud_cost_guardrails.csv",
        "label": "Cloud Cost Guardrails",
        "description": "Daily cloud spend, traffic, and efficiency metrics for anomaly detection and forecasting.",
        "recommendedTools": ["data_profile", "correlation_explorer", "anomaly_detector", "forecast_baseline"],
    },
]

dao = SqliteDao(DB_PATH)
dao.init_db()

app = Flask(__name__, static_folder=str(FRONTEND_DIST_DIR), static_url_path="")
app.secret_key = os.getenv("APP_SECRET_KEY", "datapilot-dev-secret-change-me")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def error_response(message, status=400):
    return jsonify({"error": message}), status


def frontend_ready():
    return (FRONTEND_DIST_DIR / "index.html").exists()


def frontend_not_ready_response():
    return (
        "Frontend build not found. Run `npm install && npm run build` in the frontend directory first.",
        503,
    )


def qianfan_bearer_token():
    return os.getenv("QIANFAN_BEARER_TOKEN", "").strip()


def serialize_examples():
    return EXAMPLE_CATALOG


def find_example(example_id):
    for example in EXAMPLE_CATALOG:
        if example["id"] == example_id:
            return example
    return None


def serialize_user(user):
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "createdAt": user.get("created_at"),
        "updatedAt": user.get("updated_at"),
    }


def is_guest_user(user=None):
    user = user or get_current_user()
    return bool(user and user.get("username") == "guest")


def get_current_user():
    if hasattr(g, "current_user"):
        return g.current_user
    user_id = session.get("user_id")
    user = dao.get_user_by_id(user_id) if user_id else None
    g.current_user = user
    return user


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = get_current_user()
        if not user:
            return error_response("Please log in first.", 401)
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = get_current_user()
        if not user:
            return error_response("Please log in first.", 401)
        if user["role"] != "admin":
            return error_response("Admin access is required.", 403)
        return view(*args, **kwargs)

    return wrapped


def register_dataset(file_path, filename, owner_user_id, source="upload", example_id=None):
    dataset_id = str(uuid.uuid4())
    dataframe = load_dataframe(file_path)
    dataset_info = inspect_dataframe(dataframe, filename=filename, dataset_id=dataset_id)
    dataset_info["source"] = source
    if example_id:
        dataset_info["exampleId"] = example_id

    DATASET_STORE[dataset_id] = {
        "dataset_id": dataset_id,
        "file_name": filename,
        "path": str(file_path),
        "info": dataset_info,
        "chat_history": [],
        "source": source,
        "example_id": example_id,
        "owner_user_id": owner_user_id,
    }
    return dataset_info


def guest_prompt(language):
    if str(language).lower().startswith("zh"):
        return "请展示这个固定演示数据集的异常检测结果。"
    return "Show the anomaly-detection result for this fixed demo dataset."


def build_guest_demo_response(language, owner_user_id):
    example = find_example(GUEST_EXAMPLE_ID)
    if not example:
        raise ValueError("Guest demo example is not configured.")

    file_path = EXAMPLES_DIR / example["fileName"]
    if not file_path.exists():
        raise ValueError("Guest demo example file is missing.")

    dataset_info = register_dataset(
        file_path,
        example["fileName"],
        owner_user_id=owner_user_id,
        source="guest-demo",
        example_id=GUEST_EXAMPLE_ID,
    )
    dataframe = load_dataframe(file_path)
    prompt = guest_prompt(language)
    context = {
        "prompt": prompt,
        "language": language,
        "target_column": None,
        "time_column": None,
        "value_column": None,
        "text_columns": [],
        "dataset_info": dataset_info,
    }
    results = [run_tool(tool_id, dataframe.copy(), context) for tool_id in GUEST_TOOL_IDS]
    anomaly_result = next((result for result in results if result["toolId"] == "anomaly_detector"), results[-1])
    profile_result = next((result for result in results if result["toolId"] == "data_profile"), results[0])

    if str(language).lower().startswith("zh"):
        answer = (
            f"这是访客预览模式。当前固定演示文件是 {example['fileName']}。"
            f"本次演示重点是异常检测：{anomaly_result['headline']}"
            "你可以查看下方图表和表格了解异常记录，但访客账号不能上传文件，也不能发起新的智能问数对话。"
        )
        summary = (
            f"固定演示已加载：{profile_result['headline']} "
            f"{anomaly_result['headline']}"
        )
    else:
        answer = (
            f"This is the guest preview mode. The fixed demo file is {example['fileName']}. "
            f"The preview focuses on anomaly detection: {anomaly_result['headline']} "
            "You can inspect the charts and tables below, but guest accounts cannot upload files or start new agent conversations."
        )
        summary = f"Fixed demo loaded: {profile_result['headline']} {anomaly_result['headline']}"

    return {
        "dataset": dataset_info,
        "prompt": prompt,
        "answer": answer,
        "summary": summary,
        "results": results,
        "selectedTools": GUEST_TOOL_IDS,
        "toolMode": "manual",
        "resolvedContext": {
            "targetColumn": None,
            "timeColumn": None,
            "valueColumn": None,
            "textColumns": [],
        },
        "guestMode": True,
        "exampleId": GUEST_EXAMPLE_ID,
    }


def normalize_llm_payload(payload):
    provider = str(payload.get("provider") or "auto").strip().lower()
    if provider not in {"auto", "qwen", "openai_compatible"}:
        provider = "auto"
    openai_base_url = str(payload.get("openaiBaseUrl") or payload.get("openai_base_url") or "https://api.openai.com/v1").strip()
    if "api.fasttoken.ai" in openai_base_url:
        openai_base_url = openai_base_url.replace("api.fasttoken.ai", "api.fastoken.ai")
    return {
        "provider": provider,
        "qwen_api_key": str(payload.get("qwenApiKey") or payload.get("qwen_api_key") or "").strip(),
        "qwen_model": str(payload.get("qwenModel") or payload.get("qwen_model") or "qwen-turbo").strip() or "qwen-turbo",
        "openai_api_key": str(payload.get("openaiApiKey") or payload.get("openai_api_key") or "").strip(),
        "openai_base_url": openai_base_url or "https://api.openai.com/v1",
        "openai_model": str(payload.get("openaiModel") or payload.get("openai_model") or "gpt-4o-mini").strip()
        or "gpt-4o-mini",
    }


def build_runtime_llm_config():
    stored = dao.get_llm_config()
    openai_base_url = (
        stored.get("openai_base_url")
        or os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
        or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    if "api.fasttoken.ai" in openai_base_url:
        openai_base_url = openai_base_url.replace("api.fasttoken.ai", "api.fastoken.ai")
    return {
        "provider": stored.get("provider") or os.getenv("LLM_PROVIDER", "auto"),
        "qwen_api_key": stored.get("qwen_api_key") or os.getenv("DASHSCOPE_API_KEY", ""),
        "qwen_model": stored.get("qwen_model") or os.getenv("DASHSCOPE_MODEL", "qwen-turbo"),
        "openai_api_key": stored.get("openai_api_key")
        or os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", ""),
        "openai_base_url": openai_base_url,
        "openai_model": stored.get("openai_model")
        or os.getenv("OPENAI_COMPATIBLE_MODEL", "")
        or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/auth/me")
def get_me():
    user = get_current_user()
    return jsonify({"user": serialize_user(user) if user else None})


@app.post("/api/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    if not username or not password:
        return error_response("Username and password are required.")

    user = dao.verify_user(username, password)
    if not user:
        return error_response("Invalid username or password.", 401)

    session.clear()
    session["user_id"] = user["id"]
    return jsonify({"user": serialize_user(user)})


@app.post("/api/auth/logout")
@login_required
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.post("/api/auth/change-password")
@login_required
def change_own_password():
    payload = request.get_json(silent=True) or {}
    current_password = str(payload.get("currentPassword") or "")
    new_password = str(payload.get("newPassword") or "")
    if not current_password or not new_password:
        return error_response("Current password and new password are required.")
    if len(new_password) < 4:
        return error_response("New password must be at least 4 characters long.")

    user = get_current_user()
    if is_guest_user(user):
        return error_response("Guest preview password cannot be changed.", 403)
    if not dao.verify_user(user["username"], current_password):
        return error_response("Current password is incorrect.", 403)

    updated = dao.update_user_password(user["id"], new_password)
    g.current_user = updated
    return jsonify({"user": serialize_user(updated)})


@app.get("/api/admin/users")
@admin_required
def get_users():
    return jsonify({"users": dao.list_users()})


@app.post("/api/admin/users")
@admin_required
def create_user():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    if not username or not password:
        return error_response("Username and password are required.")
    if len(password) < 4:
        return error_response("Password must be at least 4 characters long.")
    if dao.get_user_by_username(username):
        return error_response("That username already exists.")

    user = dao.create_user(username, password, role="user")
    return jsonify({"user": serialize_user(user)}), 201


@app.put("/api/admin/users/<int:user_id>/password")
@admin_required
def update_user_password(user_id):
    payload = request.get_json(silent=True) or {}
    password = str(payload.get("password") or "")
    if len(password) < 4:
        return error_response("Password must be at least 4 characters long.")

    user = dao.get_user_by_id(user_id)
    if not user:
        return error_response("User not found.", 404)

    updated = dao.update_user_password(user_id, password)
    return jsonify({"user": serialize_user(updated)})


@app.get("/api/admin/llm-config")
@admin_required
def get_llm_config():
    return jsonify({"config": normalize_llm_payload(dao.get_llm_config())})


@app.put("/api/admin/llm-config")
@admin_required
def update_llm_config():
    payload = request.get_json(silent=True) or {}
    config = dao.update_llm_config(normalize_llm_payload(payload))
    return jsonify({"config": config})


@app.get("/api/tools")
@login_required
def get_tools():
    return jsonify({"tools": serialize_tool_catalog()})


@app.get("/api/examples")
@login_required
def get_examples():
    if is_guest_user():
        example = find_example(GUEST_EXAMPLE_ID)
        return jsonify({"examples": [example] if example else []})
    return jsonify({"examples": serialize_examples()})


@app.get("/api/guest-demo")
@login_required
def get_guest_demo():
    user = get_current_user()
    if not is_guest_user(user):
        return error_response("Guest demo is only available for the guest account.", 403)
    language = (request.args.get("language") or "en").strip()
    try:
        demo = build_guest_demo_response(language, owner_user_id=user["id"])
    except Exception as exc:
        return error_response(f"Failed to load guest demo: {exc}", 500)
    return jsonify(demo)


@app.post("/api/news-search")
@login_required
def proxy_news_search():
    payload = request.get_json(silent=True) or {}
    keyword = (payload.get("keyword") or "").strip()
    if not keyword:
        return error_response("Keyword is required.")

    bearer_token = qianfan_bearer_token()
    if not bearer_token:
        return error_response(
            "QIANFAN_BEARER_TOKEN is not configured in the OS environment.",
            500,
        )

    search_url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
    request_body = {
        "messages": [{"role": "user", "content": keyword}],
        "edition": "standard",
        "search_source": "baidu_search_v2",
        "search_recency_filter": "week",
    }
    data = json.dumps(request_body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer_token}",
    }

    try:
        req = urllib.request.Request(search_url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            response_body = response.read().decode("utf-8")
            return app.response_class(
                response=response_body,
                status=response.status,
                mimetype="application/json",
            )
    except Exception as exc:
        return error_response(f"News search proxy failed: {exc}", 502)


@app.post("/api/datasets/inspect")
@login_required
def inspect_dataset():
    if is_guest_user():
        return error_response("Guest preview is read-only. File upload is disabled.", 403)
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return error_response("Please upload a CSV or Excel file.")

    filename = secure_filename(uploaded_file.filename)
    if not allowed_file(filename):
        return error_response("Unsupported file type. Upload CSV, XLS, or XLSX.")

    storage_id = str(uuid.uuid4())
    stored_name = f"{storage_id}_{filename}"
    file_path = UPLOAD_DIR / stored_name
    uploaded_file.save(file_path)

    try:
        dataset_info = register_dataset(file_path, filename, owner_user_id=get_current_user()["id"], source="upload")
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        return error_response(str(exc))
    return jsonify(dataset_info)


@app.post("/api/examples/load")
@login_required
def load_example_dataset():
    if is_guest_user():
        return error_response("Guest preview is locked to the built-in anomaly demo.", 403)
    payload = request.get_json(silent=True) or {}
    example_id = (payload.get("exampleId") or "").strip()
    example = find_example(example_id)
    if not example:
        return error_response("Example dataset not found.", 404)

    file_path = EXAMPLES_DIR / example["fileName"]
    if not file_path.exists():
        return error_response("Example file is missing from the examples directory.", 500)

    try:
        dataset_info = register_dataset(
            file_path,
            example["fileName"],
            owner_user_id=get_current_user()["id"],
            source="example",
            example_id=example_id,
        )
    except Exception as exc:
        return error_response(str(exc), 500)
    return jsonify(dataset_info)


@app.post("/api/analyze")
@login_required
def analyze_dataset():
    if is_guest_user():
        return error_response("Guest preview cannot start new agent conversations.", 403)
    payload = request.get_json(silent=True) or {}
    dataset_id = payload.get("datasetId")
    selected_tools = payload.get("selectedTools") or []
    tool_mode = (payload.get("toolMode") or "manual").strip().lower()
    prompt = (payload.get("prompt") or "").strip()
    language = (payload.get("language") or "en").strip()

    if not dataset_id or dataset_id not in DATASET_STORE:
        return error_response("Dataset not found. Upload the file again.", 404)
    if tool_mode != "auto" and not selected_tools:
        return error_response("Select at least one analysis tool.")

    dataset_record = DATASET_STORE[dataset_id]
    if dataset_record.get("owner_user_id") != get_current_user()["id"]:
        return error_response("You do not have access to this dataset.", 403)

    try:
        dataframe = load_dataframe(dataset_record["path"])
        dataset_info = inspect_dataframe(
            dataframe,
            filename=dataset_record["file_name"],
            dataset_id=dataset_id,
        )
    except Exception as exc:
        return error_response(f"Failed to load dataset: {exc}", 500)

    context = {
        "prompt": prompt,
        "language": language,
        "target_column": payload.get("targetColumn"),
        "time_column": payload.get("timeColumn"),
        "value_column": payload.get("valueColumn"),
        "text_columns": payload.get("textColumns") or [],
        "dataset_info": dataset_info,
    }

    if tool_mode == "auto":
        auto_plan = auto_select_tools(dataset_info, prompt, context)
        selected_tools = auto_plan["selected_tools"]
        context["target_column"] = context["target_column"] or auto_plan["target_column"]
        context["time_column"] = context["time_column"] or auto_plan["time_column"]
        context["value_column"] = context["value_column"] or auto_plan["value_column"]
        context["text_columns"] = context["text_columns"] or auto_plan["text_columns"]

    if not selected_tools:
        return error_response("No suitable analysis tools were found for the current dataset and question.")

    results = []
    for tool_id in selected_tools:
        try:
            results.append(run_tool(tool_id, dataframe.copy(), context))
        except Exception as exc:
            results.append(
                {
                    "toolId": tool_id,
                    "toolName": tool_id.replace("_", " ").title(),
                    "status": "error",
                    "headline": "The tool could not complete.",
                    "insights": [],
                    "warnings": [str(exc)],
                    "tables": [],
                }
            )

    chat_history = dataset_record.setdefault("chat_history", [])
    llm_settings = build_runtime_llm_config()
    answer_bundle = build_agent_answer_bundle(
        dataset_info,
        prompt,
        results,
        chat_history,
        language=language,
        llm_settings=llm_settings,
    )
    summary_bundle = build_agent_summary_bundle(
        dataset_info,
        prompt,
        results,
        language=language,
        llm_settings=llm_settings,
    )
    answer = answer_bundle["text"]
    summary = summary_bundle["text"]
    llm_runtime = get_llm_runtime_info(llm_settings)

    if prompt:
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": answer})
        dataset_record["chat_history"] = chat_history[-12:]

    return jsonify(
        {
            "dataset": dataset_info,
            "prompt": prompt,
            "answer": answer,
            "summary": summary,
            "results": results,
            "selectedTools": selected_tools,
            "toolMode": tool_mode,
            "llm": {
                "runtime": llm_runtime,
                "answer": answer_bundle["llm"],
                "summary": summary_bundle["llm"],
                "answerSource": answer_bundle["source"],
                "summarySource": summary_bundle["source"],
            },
            "resolvedContext": {
                "targetColumn": context.get("target_column"),
                "timeColumn": context.get("time_column"),
                "valueColumn": context.get("value_column"),
                "textColumns": context.get("text_columns") or [],
            },
        }
    )


@app.post("/api/export-chat-pdf")
@login_required
def export_chat_pdf():
    payload = request.get_json(silent=True) or {}
    try:
        pdf_bytes = build_chat_pdf(
            {
                "messages": payload.get("messages") or [],
                "language": payload.get("language") or "en",
                "datasetName": payload.get("datasetName") or "",
                "username": get_current_user()["username"],
                "title": payload.get("title") or "SciPilot Chat Export",
            }
        )
    except RuntimeError as exc:
        return error_response(str(exc), 500)
    except Exception as exc:
        return error_response(f"Failed to export PDF: {exc}", 500)

    filename = secure_filename(payload.get("fileName") or "datapilot-chat-export.pdf")
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return app.response_class(
        response=pdf_bytes,
        status=200,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


@app.get("/")
def serve_index():
    if not frontend_ready():
        return frontend_not_ready_response()
    return send_from_directory(FRONTEND_DIST_DIR, "index.html")


@app.get("/<path:path>")
def serve_frontend(path):
    if path.startswith("api/"):
        return error_response("API route not found.", 404)
    if not frontend_ready():
        return frontend_not_ready_response()

    target = FRONTEND_DIST_DIR / path
    if target.exists() and target.is_file():
        return send_from_directory(FRONTEND_DIST_DIR, path)
    return send_from_directory(FRONTEND_DIST_DIR, "index.html")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5005"))
    debug = str(os.getenv("APP_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
        use_reloader=False,
        threaded=True,
    )
