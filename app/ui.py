"""Enterprise Knowledge Assistant — Professional Gradio UI.

Run with: python app/ui.py
"""

from __future__ import annotations

import json
from urllib.parse import quote

import gradio as gr
import requests

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

API_BASE = "http://127.0.0.1:8000/api/v1"

CSS = """
/* ----- reset & base ---------------------------------------------------- */
* { box-sizing: border-box; }
footer { visibility: hidden !important; }

.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ----- app shell ------------------------------------------------------- */
.app-shell {
    display: flex;
    height: 100vh;
    overflow: hidden;
}

/* ----- sidebar --------------------------------------------------------- */
.sidebar {
    width: 260px;
    min-width: 260px;
    background: #0f172a;
    color: #e2e8f0;
    display: flex;
    flex-direction: column;
    padding: 24px 0;
    border-right: 1px solid #1e293b;
    z-index: 10;
}

.sidebar-brand {
    padding: 0 20px 24px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 16px;
}

.sidebar-brand h2 {
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    letter-spacing: -0.3px;
}

.sidebar-brand span {
    font-size: 12px;
    color: #64748b;
    margin-top: 2px;
    display: block;
}

.sidebar-nav {
    flex: 1;
    padding: 8px 12px;
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.sidebar-nav button {
    width: 100%;
    text-align: left;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    color: #94a3b8;
    background: transparent;
    border: none;
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    gap: 10px;
}

.sidebar-nav button:hover {
    background: #1e293b;
    color: #e2e8f0;
}

.sidebar-nav button.active {
    background: #1d4ed8;
    color: white;
}

.sidebar-nav button .nav-icon {
    font-size: 18px;
    width: 22px;
    text-align: center;
}

.sidebar-status {
    padding: 16px 20px;
    border-top: 1px solid #1e293b;
    font-size: 12px;
    color: #64748b;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}

.status-dot.online { background: #22c55e; }
.status-dot.offline { background: #ef4444; }

/* ----- main content ---------------------------------------------------- */
.main-content {
    flex: 1;
    overflow-y: auto;
    background: #f8fafc;
    padding: 32px 40px;
}

.page-header {
    margin-bottom: 28px;
}

.page-header h1 {
    font-size: 26px;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 4px;
}

.page-header p {
    font-size: 14px;
    color: #64748b;
    margin: 0;
}

/* ----- cards ----------------------------------------------------------- */
.card {
    background: white;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.card-header {
    font-size: 15px;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 16px;
}

/* ----- stat row -------------------------------------------------------- */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.stat-card {
    background: white;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.stat-card .stat-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
}

.stat-card .stat-label {
    font-size: 13px;
    color: #64748b;
    margin-top: 4px;
}

/* ----- chatbot area ---------------------------------------------------- */
.chat-container {
    background: white;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    overflow: hidden;
}

/* ----- upload zone ----------------------------------------------------- */
.upload-zone {
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    background: white;
    transition: border-color 0.2s;
}

.upload-zone:hover {
    border-color: #3b82f6;
}

/* ----- file table ------------------------------------------------------ */
.file-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid #f1f5f9;
    font-size: 14px;
}

.file-row:last-child { border-bottom: none; }

.file-name { font-weight: 500; color: #0f172a; }
.file-meta { font-size: 12px; color: #94a3b8; }

/* ----- Gradio overrides ------------------------------------------------ */
.gr-button-primary {
    background: #1d4ed8 !important;
    border: none !important;
}

.gr-button-primary:hover {
    background: #1e40af !important;
}

.gr-textbox input, .gr-textbox textarea {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 14px !important;
}

.gr-textbox input:focus, .gr-textbox textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}

.gr-file {
    border-radius: 12px !important;
}
"""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api(method: str, path: str, **kwargs) -> dict | None:
    url = f"{API_BASE}{path}"
    try:
        r = requests.request(method, url, timeout=120, **kwargs)
        if r.status_code >= 400:
            return {"error": f"{r.status_code}: {r.text[:300]}"}
        return r.json() if r.text else {"ok": True}
    except requests.ConnectionError:
        return {"error": "Cannot reach backend. Start the server first."}
    except Exception as exc:
        return {"error": str(exc)}


def check_backend_health() -> str:
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if r.status_code == 200:
            return "🟢 Online"
        return "🔴 Error"
    except Exception:
        return "🔴 Offline"


# ---------------------------------------------------------------------------
# page: Chat
# ---------------------------------------------------------------------------


def ask_question(question: str, history: list[dict]) -> tuple[str, list[dict]]:
    if not question.strip():
        return "", history

    result = _api("POST", "/query", json={"question": question.strip()})

    if result is None or "error" in result:
        answer = f"❌ {result.get('error', 'No response')}" if result else "❌ No response"
    else:
        answer = result.get("answer", "No answer returned.")
        sources = result.get("sources", [])
        latency = result.get("latency_ms", 0)
        if sources:
            answer += f"\n\n**Sources:** {', '.join(sources)}"
        answer += f"\n\n*{latency:.0f}ms*"

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return "", history


# ---------------------------------------------------------------------------
# page: Upload
# ---------------------------------------------------------------------------


def upload_file(file: str | None) -> tuple[str, list[list[str]]]:
    if file is None:
        return "Please select a file first.", _list_files_data()

    try:
        fname = file.split("\\")[-1].split("/")[-1]
        with open(file, "rb") as fh:
            files = {"file": (fname, fh)}
            r = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
        if r.status_code >= 400:
            detail = r.json().get("detail", r.text)
            return f"❌ Upload failed: {detail}", _list_files_data()
        data = r.json()
        return (
            f"✅ **{data['filename']}** uploaded and indexed.\n"
            f"Job: `{data['job_id'][:8]}...`"
        ), _list_files_data()
    except requests.ConnectionError:
        return "❌ Cannot reach backend.", _list_files_data()
    except Exception as exc:
        return f"❌ {exc}", _list_files_data()


# ---------------------------------------------------------------------------
# page: Documents (files)
# ---------------------------------------------------------------------------


def _list_files_data() -> list[list[str]]:
    result = _api("GET", "/uploads")
    if result is None or "error" in result or not result:
        return []
    return [[f["filename"], _fmt_size(f["size"]), f["uploaded_at"][:19]] for f in result]


def delete_file_handler(filename: str) -> tuple[str, list[list[str]]]:
    if not filename.strip():
        return "Enter a filename.", _list_files_data()
    result = _api("DELETE", f"/uploads/{quote(filename.strip())}")
    if result is None or "error" in result:
        return f"❌ {result.get('error', 'Error')}", _list_files_data()
    return f"🗑️ Deleted: {filename}", _list_files_data()


# ---------------------------------------------------------------------------
# page: Settings / Index
# ---------------------------------------------------------------------------


def index_dir(directory: str) -> str:
    if not directory.strip():
        return "Enter a directory path."
    result = _api("POST", "/index", json={"directory": directory.strip()})
    if result is None or "error" in result:
        return f"❌ {result.get('error', 'Error')}"
    return (
        f"✅ Indexing started.\n"
        f"Job: `{result['job_id'][:8]}...`  |  Status: {result['status']}"
    )


def check_job_status(job_id: str) -> str:
    if not job_id.strip():
        return "Enter a job ID."
    result = _api("GET", f"/jobs/{job_id.strip()}")
    if result is None or "error" in result:
        return f"❌ {result.get('error', 'Error')}"
    return f"**{result.get('status')}**  |  Files: {result.get('files_processed',0)}  |  Chunks: {result.get('chunks_created',0)}  |  Embeddings: {result.get('embeddings_generated',0)}"


def get_system_info() -> str:
    result = _api("GET", "/uploads")
    file_count = len(result) if isinstance(result, list) else 0
    health = check_backend_health()
    return f"Backend: {health}  |  Files: {file_count}  |  Provider: DeepSeek v4"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fmt_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------

with gr.Blocks(
    css=CSS,
    title="Enterprise Knowledge Assistant",
    theme=gr.themes.Soft(primary_hue="blue", font=gr.themes.GoogleFont("Inter")),
    head="""<meta name="viewport" content="width=device-width, initial-scale=1">""",
) as demo:

    # Hidden state for sidebar active page
    current_page = gr.State("chat")

    # ---- Full-screen row: sidebar | main content ------------------------
    with gr.Row(equal_height=True, elem_classes="app-shell"):

        # =================================================================
        # SIDEBAR
        # =================================================================
        with gr.Column(scale=1, min_width=260, elem_classes="sidebar"):
            # Brand
            gr.HTML("""
            <div class="sidebar-brand">
                <h2>Knowledge<br>Assistant</h2>
                <span>Enterprise RAG Platform</span>
            </div>
            """)

            # Nav buttons
            chat_btn = gr.Button("💬  Chat", elem_classes="sidebar-nav")
            upload_btn = gr.Button("📤  Upload", elem_classes="sidebar-nav")
            docs_btn = gr.Button("📁  Documents", elem_classes="sidebar-nav")
            index_btn = gr.Button("⚙️  Console", elem_classes="sidebar-nav")

            # Status
            sys_info = gr.Markdown("🟢 Backend online  |  2 files", elem_classes="sidebar-status")

        # =================================================================
        # MAIN CONTENT
        # =================================================================
        with gr.Column(scale=4, elem_classes="main-content"):

            # --- PAGE: Chat -------------------------------------------------
            with gr.Column(visible=True) as chat_page:
                gr.HTML("""
                <div class="page-header">
                    <h1>Ask a question</h1>
                    <p>Search your document corpus with AI-powered RAG</p>
                </div>
                """)

                chatbot = gr.Chatbot(
                    label="",
                    height=480,
                    type="messages",
                    show_copy_button=True,
                    elem_classes="chat-container",
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your question and press Enter...",
                        label="",
                        scale=9,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear chat", variant="secondary", size="sm")

                msg.submit(ask_question, [msg, chatbot], [msg, chatbot])
                send_btn.click(ask_question, [msg, chatbot], [msg, chatbot])
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

            # --- PAGE: Upload ------------------------------------------------
            with gr.Column(visible=False) as upload_page:
                gr.HTML("""
                <div class="page-header">
                    <h1>Upload documents</h1>
                    <p>Add PDF, DOCX, TXT, or MD files to your knowledge base</p>
                </div>
                """)

                with gr.Column(elem_classes="upload-zone"):
                    gr.Markdown("### 📎 Drop or select a file")
                    gr.Markdown("*Supported: PDF · DOCX · TXT · Markdown  —  Max 50 MB*")
                    upload_input = gr.File(
                        label="",
                        file_types=[".pdf", ".docx", ".txt", ".md"],
                    )
                    upload_action_btn = gr.Button("Upload & Index", variant="primary", size="lg")
                    upload_status_text = gr.Markdown("")

                upload_action_btn.click(
                    upload_file, [upload_input], [upload_status_text]
                )

            # --- PAGE: Documents ---------------------------------------------
            with gr.Column(visible=False) as docs_page:
                gr.HTML("""
                <div class="page-header">
                    <h1>Documents</h1>
                    <p>Manage your uploaded files</p>
                </div>
                """)

                files_table = gr.Dataframe(
                    headers=["Filename", "Size", "Uploaded"],
                    label="Uploaded files",
                    interactive=False,
                    value=_list_files_data,
                )

                with gr.Row():
                    delete_input = gr.Textbox(
                        placeholder="Filename to delete...",
                        label="Delete a file",
                        scale=3,
                    )
                    delete_btn = gr.Button("Delete", variant="stop", scale=1)
                    refresh_btn = gr.Button("Refresh", variant="secondary", scale=1)
                    delete_result_text = gr.Markdown("")

                delete_btn.click(
                    delete_file_handler, [delete_input],
                    [delete_result_text, files_table]
                )
                refresh_btn.click(
                    lambda: _list_files_data(), outputs=[files_table]
                )

            # --- PAGE: Console -----------------------------------------------
            with gr.Column(visible=False) as console_page:
                gr.HTML("""
                <div class="page-header">
                    <h1>Console</h1>
                    <p>Index management and job tracking</p>
                </div>
                """)

                with gr.Column(elem_classes="card"):
                    gr.Markdown("### Index a directory")
                    dir_input = gr.Textbox(
                        placeholder="data/uploads",
                        label="Directory path",
                        value="data/uploads",
                    )
                    index_action_btn = gr.Button("Start indexing", variant="primary")
                    index_result_text = gr.Markdown("")

                index_action_btn.click(index_dir, [dir_input], [index_result_text])

                with gr.Column(elem_classes="card"):
                    gr.Markdown("### Check job status")
                    with gr.Row():
                        job_input = gr.Textbox(
                            placeholder="Paste job ID...",
                            label="Job ID",
                            scale=3,
                        )
                        job_check_btn = gr.Button("Check", variant="secondary", scale=1)
                    job_status_text = gr.Markdown("")

                job_check_btn.click(check_job_status, [job_input], [job_status_text])

                with gr.Column(elem_classes="card"):
                    gr.Markdown("### System")
                    sys_detail = gr.Markdown(get_system_info)
                    refresh_sys_btn = gr.Button("Refresh", variant="secondary", size="sm")
                    refresh_sys_btn.click(get_system_info, outputs=[sys_detail])

    # =====================================================================
    # Sidebar navigation logic
    # =====================================================================

    def show_page(page: str) -> list[dict]:
        return [
            gr.update(visible=(page == "chat")),
            gr.update(visible=(page == "upload")),
            gr.update(visible=(page == "docs")),
            gr.update(visible=(page == "console")),
        ]

    chat_btn.click(
        show_page, gr.State("chat"),
        [chat_page, upload_page, docs_page, console_page],
    ).then(get_system_info, outputs=[sys_info])

    upload_btn.click(
        show_page, gr.State("upload"),
        [chat_page, upload_page, docs_page, console_page],
    ).then(get_system_info, outputs=[sys_info])

    docs_btn.click(
        show_page, gr.State("docs"),
        [chat_page, upload_page, docs_page, console_page],
    ).then(lambda: _list_files_data(), outputs=[files_table]).then(get_system_info, outputs=[sys_info])

    index_btn.click(
        show_page, gr.State("console"),
        [chat_page, upload_page, docs_page, console_page],
    ).then(get_system_info, outputs=[sys_info])


if __name__ == "__main__":
    print("\n    Enterprise Knowledge Assistant UI")
    print("    http://127.0.0.1:7860\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
