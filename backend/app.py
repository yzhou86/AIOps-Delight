import os
import uuid
from pathlib import Path
import json
import urllib.request

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from analysis_tools import (
    build_agent_answer,
    build_agent_summary,
    inspect_dataframe,
    load_dataframe,
    run_tool,
    serialize_tool_catalog,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
DATASET_STORE = {}

app = Flask(__name__, static_folder=str(FRONTEND_DIST_DIR), static_url_path="")


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


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/tools")
def get_tools():
    return jsonify({"tools": serialize_tool_catalog()})


@app.post("/api/news-search")
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
def inspect_dataset():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return error_response("Please upload a CSV or Excel file.")

    filename = secure_filename(uploaded_file.filename)
    if not allowed_file(filename):
        return error_response("Unsupported file type. Upload CSV, XLS, or XLSX.")

    dataset_id = str(uuid.uuid4())
    stored_name = f"{dataset_id}_{filename}"
    file_path = UPLOAD_DIR / stored_name
    uploaded_file.save(file_path)

    try:
        dataframe = load_dataframe(file_path)
        dataset_info = inspect_dataframe(dataframe, filename=filename, dataset_id=dataset_id)
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        return error_response(str(exc))

    DATASET_STORE[dataset_id] = {
        "dataset_id": dataset_id,
        "file_name": filename,
        "path": str(file_path),
        "info": dataset_info,
        "chat_history": [],
    }
    return jsonify(dataset_info)


@app.post("/api/analyze")
def analyze_dataset():
    payload = request.get_json(silent=True) or {}
    dataset_id = payload.get("datasetId")
    selected_tools = payload.get("selectedTools") or []
    prompt = (payload.get("prompt") or "").strip()

    if not dataset_id or dataset_id not in DATASET_STORE:
        return error_response("Dataset not found. Upload the file again.", 404)
    if not selected_tools:
        return error_response("Select at least one analysis tool.")

    dataset_record = DATASET_STORE[dataset_id]

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
        "target_column": payload.get("targetColumn"),
        "time_column": payload.get("timeColumn"),
        "value_column": payload.get("valueColumn"),
        "text_columns": payload.get("textColumns") or [],
        "dataset_info": dataset_info,
    }

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
    answer = build_agent_answer(dataset_info, prompt, results, chat_history)
    summary = build_agent_summary(dataset_info, prompt, results)

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
        }
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
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
