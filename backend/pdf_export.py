from datetime import datetime
from io import BytesIO


def build_chat_pdf(payload):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ModuleNotFoundError as exc:
        raise RuntimeError("PDF export dependency is missing. Install reportlab first.") from exc

    locale = (payload.get("language") or "en").strip().lower()
    messages = payload.get("messages") or []
    dataset_name = payload.get("datasetName") or "No dataset"
    username = payload.get("username") or "unknown"
    title = payload.get("title") or ("DataPilot Chat Export" if not locale.startswith("zh") else "DataPilot 对话导出")
    exported_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        title=title,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChatTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#14352b"),
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "ChatMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#51635d"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "ChatSection",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1d5f49"),
        spaceBefore=6,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "ChatBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12,
        textColor=colors.HexColor("#22332f"),
        spaceAfter=4,
    )
    subtle_style = ParagraphStyle(
        "ChatSubtle",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#6a7a74"),
        spaceAfter=4,
    )

    story = [
        Paragraph(_escape(title), title_style),
        Paragraph(
            _escape(
                (
                    f"Dataset: {dataset_name} | User: {username} | Exported: {exported_at}"
                    if not locale.startswith("zh")
                    else f"数据集：{dataset_name} | 用户：{username} | 导出时间：{exported_at}"
                )
            ),
            meta_style,
        ),
    ]

    for index, message in enumerate(messages, start=1):
        role = _message_role_label(message.get("role"), locale)
        headline = message.get("text") or ""
        story.append(Paragraph(_escape(f"{index}. {role}"), section_style))

        if message.get("kind") == "analysis":
            answer = message.get("answer") or headline
            summary = message.get("summary") or ""
            story.append(Paragraph(_escape(answer).replace("\n", "<br/>"), body_style))
            if summary and summary != answer:
                summary_label = "Analysis Summary" if not locale.startswith("zh") else "分析摘要"
                story.append(Paragraph(_escape(f"{summary_label}:").replace("\n", "<br/>"), subtle_style))
                story.append(Paragraph(_escape(summary).replace("\n", "<br/>"), body_style))

            analysis = message.get("analysis") or {}
            selected_tools = analysis.get("selectedTools") or []
            if selected_tools:
                label = "Tools" if not locale.startswith("zh") else "工具"
                story.append(Paragraph(_escape(f"{label}: {', '.join(selected_tools)}"), subtle_style))

            for result in analysis.get("results") or []:
                tool_title = f"{result.get('toolName') or result.get('toolId')}: {result.get('headline') or ''}"
                story.append(Paragraph(_escape(tool_title), body_style))
                for insight in result.get("insights") or []:
                    story.append(Paragraph(_escape(f"• {insight}"), body_style))
                for warning in result.get("warnings") or []:
                    warning_prefix = "Warning" if not locale.startswith("zh") else "提示"
                    story.append(Paragraph(_escape(f"{warning_prefix}: {warning}"), subtle_style))
                for table in (result.get("tables") or [])[:2]:
                    table_story = _build_table(table, locale, Table, TableStyle, colors)
                    if table_story:
                        story.extend(table_story)
        elif message.get("kind") == "dataset" and message.get("dataset"):
            dataset = message["dataset"]
            story.append(Paragraph(_escape(headline).replace("\n", "<br/>"), body_style))
            profile_line = (
                f"Rows: {dataset.get('rowCount')} | Columns: {dataset.get('columnCount')} | Numeric: {len(dataset.get('numericColumns') or [])} | Text: {len(dataset.get('textColumns') or [])}"
                if not locale.startswith("zh")
                else f"行数：{dataset.get('rowCount')} | 列数：{dataset.get('columnCount')} | 数值列：{len(dataset.get('numericColumns') or [])} | 文本列：{len(dataset.get('textColumns') or [])}"
            )
            story.append(Paragraph(_escape(profile_line), subtle_style))
        else:
            story.append(Paragraph(_escape(headline).replace("\n", "<br/>"), body_style))

        story.append(Spacer(1, 4))

    document.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


def _build_table(table, locale, Table, TableStyle, colors):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Spacer

    columns = table.get("columns") or []
    rows = table.get("rows") or []
    if not columns or not rows:
        return []

    label = table.get("title") or ("Table" if not locale.startswith("zh") else "表格")
    data = [columns]
    for row in rows[:10]:
        data.append([_cell_value(row.get(column)) for column in columns])

    width = 170
    column_widths = [max(22, width / max(1, len(columns))) for _ in columns]
    table_widget = Table(data, colWidths=column_widths)
    table_widget.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0ec")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#14352b")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c6d4ce")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#faf8f3")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    return [Paragraph(_escape(label), ParagraphStyle("TableTitle", fontName="Helvetica-Bold", fontSize=8.5, leading=10, textColor=colors.HexColor("#38574f"), spaceAfter=3)), table_widget, Spacer(1, 4)]


def _message_role_label(role, locale):
    if locale.startswith("zh"):
        return {"user": "用户", "assistant": "智能体", "system": "系统"}.get(role, "消息")
    return {"user": "User", "assistant": "Agent", "system": "System"}.get(role, "Message")


def _cell_value(value):
    if value is None:
        return ""
    text = str(value)
    if len(text) > 60:
        return text[:57] + "..."
    return text


def _escape(text):
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
