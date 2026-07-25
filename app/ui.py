"""Gradio UI for the Enterprise Knowledge Assistant.

Run with: python app/ui.py
"""

from __future__ import annotations

from urllib.parse import quote

import gradio as gr
import requests

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

API_BASE = "http://127.0.0.1:8000/api/v1"

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api(method: str, path: str, **kwargs) -> dict | None:
    """Thin wrapper around the backend API."""
    url = f"{API_BASE}{path}"
    try:
        r = requests.request(method, url, timeout=120, **kwargs)
        if r.status_code >= 400:
            return {"error": f"{r.status_code}: {r.text[:300]}"}
        return r.json() if r.text else {"ok": True}
    except requests.ConnectionError:
        return {"error": "Cannot reach backend. Is the server running? (uvicorn app.main:app --port 8000)"}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tab 1: Chat / Query
# ---------------------------------------------------------------------------


def ask_question(question: str, history: list[list[str | None]]) -> str:
    """Send a query to the RAG pipeline and return the answer."""
    if not question.strip():
        return "Please enter a question."

    result = _api("POST", "/query", json={"question": question.strip()})

    if result is None:
        return "No response from server."
    if "error" in result:
        return f"❌ {result['error']}"

    answer = result.get("answer", "No answer returned.")
    sources = result.get("sources", [])
    latency = result.get("latency_ms", 0)

    parts = [answer]
    if sources:
        parts.append(f"\n\n📚 **Sources:** {', '.join(sources)}")
    parts.append(f"\n⏱️ {latency:.0f}ms")
    return "".join(parts)


def stream_question(question: str) -> str:
    """Stream tokens from the backend."""
    if not question.strip():
        return "Please enter a question."

    try:
        tokens = []
        with requests.post(
            f"{API_BASE}/query/stream",
            json={"question": question.strip()},
            stream=True,
            timeout=120,
        ) as r:
            if r.status_code >= 400:
                return f"❌ Error {r.status_code}"
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    tokens.append(line)
        return "".join(tokens) if tokens else "No response."
    except requests.ConnectionError:
        return "❌ Cannot reach backend. Start the server first."
    except Exception as exc:
        return f"❌ {exc}"


# ---------------------------------------------------------------------------
# Tab 2: Upload
# ---------------------------------------------------------------------------


def upload_file(file: str | None) -> str:
    """Upload a single document."""
    if file is None:
        return "No file selected."

    try:
        with open(file, "rb") as fh:
            files = {"file": (file.split("\\")[-1].split("/")[-1], fh)}
            r = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
        if r.status_code >= 400:
            detail = r.json().get("detail", r.text)
            return f"❌ Upload failed: ({r.status_code}) {detail}"
        data = r.json()
        return (
            f"✅ **{data['filename']}** uploaded.\n"
            f"Job ID: `{data['job_id']}`\n"
            f"Status: {data['status']}"
        )
    except requests.ConnectionError:
        return "❌ Cannot reach backend."
    except Exception as exc:
        return f"❌ {exc}"


# ---------------------------------------------------------------------------
# Tab 3: Manage Files
# ---------------------------------------------------------------------------


def list_files() -> list[list[str]]:
    """Return table of uploaded files."""
    result = _api("GET", "/uploads")
    if result is None or "error" in result:
        return [["Error", "", ""]]
    if not result:
        return [["No files uploaded yet", "", ""]]
    return [[f["filename"], _fmt_size(f["size"]), f["uploaded_at"]] for f in result]


def delete_file(filename: str) -> str:
    """Delete an uploaded file."""
    if not filename.strip():
        return "Enter a filename to delete."
    result = _api("DELETE", f"/uploads/{quote(filename.strip())}")
    if result is None:
        return "No response."
    if "error" in result:
        return f"❌ {result['error']}"
    return f"🗑️ {result.get('detail', 'Deleted')}"


def refresh_files():
    """Return updated table + count."""
    rows = list_files()
    count = len(rows) if rows[0][0] != "No files uploaded yet" else 0
    return rows, f"{count} file(s)"


# ---------------------------------------------------------------------------
# Tab 4: Index
# ---------------------------------------------------------------------------


def index_directory(directory: str) -> str:
    """Kick off a background indexing job."""
    if not directory.strip():
        return "Enter a directory path."
    result = _api("POST", "/index", json={"directory": directory.strip()})
    if result is None:
        return "No response."
    if "error" in result:
        return f"❌ {result['error']}"
    return (
        f"✅ Indexing started.\n"
        f"Job ID: `{result['job_id']}`\n"
        f"Status: {result['status']}"
    )


def check_job(job_id: str) -> str:
    """Poll job status."""
    if not job_id.strip():
        return "Enter a job ID."
    result = _api("GET", f"/jobs/{job_id.strip()}")
    if result is None:
        return "No response."
    if "error" in result:
        return f"❌ {result['error']}"

    lines = [f"**Job:** `{result.get('job_id', '?')}`", f"**Status:** {result.get('status', '?')}"]
    if result.get("files_processed"):
        lines.append(f"Files: {result['files_processed']}  |  "
                      f"Chunks: {result.get('chunks_created', 0)}  |  "
                      f"Embeddings: {result.get('embeddings_generated', 0)}")
    if result.get("error_message"):
        lines.append(f"❌ Error: {result['error_message']}")
    if result.get("created_at"):
        lines.append(f"Created: {result['created_at']}")
    if result.get("completed_at"):
        lines.append(f"Completed: {result['completed_at']}")
    return "\n".join(lines)


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
# App
# ---------------------------------------------------------------------------


HEADER = """
# 📚 Enterprise Knowledge Assistant

**RAG-powered question answering over your documents.**
Upload PDFs, DOCX, TXT, or MD files — then ask questions.
"""

with gr.Blocks(theme=THEME, title="Enterprise Knowledge Assistant", css="""
    footer {visibility: hidden}
""") as demo:

    gr.Markdown(HEADER)

    with gr.Tabs():
        # ---- Chat ---------------------------------------------------------
        with gr.TabItem("💬 Chat", id="chat"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=450,
                        type="messages",
                        show_copy_button=True,
                    )
                    msg = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        label="Your question",
                        scale=4,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("🔍 Ask", variant="primary")
                        stream_btn = gr.Button("⚡ Stream", variant="secondary")
                        clear_btn = gr.Button("🗑️ Clear", variant="stop", size="sm")

            def respond(message: str, history: list[dict]) -> tuple[str, list[dict]]:
                answer = ask_question(message, history)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": answer})
                return "", history

            def respond_stream(message: str, history: list[dict]) -> tuple[str, list[dict]]:
                answer = stream_question(message)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": answer})
                return "", history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
            stream_btn.click(respond_stream, [msg, chatbot], [msg, chatbot])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

        # ---- Upload -------------------------------------------------------
        with gr.TabItem("📤 Upload", id="upload"):
            gr.Markdown("### Upload documents to the knowledge base")
            with gr.Row():
                upload_input = gr.File(
                    label="Select a document",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                )
            upload_btn = gr.Button("🚀 Upload & Index", variant="primary")
            upload_status = gr.Markdown("")

            upload_btn.click(
                upload_file, [upload_input], [upload_status]
            ).then(lambda: None, outputs=[upload_input])

        # ---- Files --------------------------------------------------------
        with gr.TabItem("📁 Files", id="files"):
            gr.Markdown("### Manage uploaded files")
            files_table = gr.Dataframe(
                headers=["Filename", "Size", "Uploaded"],
                label="Uploaded Files",
                interactive=False,
                every=5,  # auto-refresh
                value=list_files,
            )
            with gr.Row():
                delete_input = gr.Textbox(placeholder="Filename to delete...", label="Delete file")
                delete_btn = gr.Button("🗑️ Delete", variant="stop")
                delete_status = gr.Markdown("")

            delete_btn.click(
                delete_file, [delete_input], [delete_status]
            ).then(list_files, outputs=[files_table])

        # ---- Index --------------------------------------------------------
        with gr.TabItem("⚙️ Index", id="index"):
            gr.Markdown("### Manual indexing & job tracking")
            with gr.Group():
                gr.Markdown("**Index a directory of documents**")
                dir_input = gr.Textbox(
                    placeholder="data/uploads",
                    label="Directory path",
                    value="data/uploads",
                )
                index_btn = gr.Button("🔄 Start Indexing", variant="primary")
                index_status = gr.Markdown("")

            index_btn.click(index_directory, [dir_input], [index_status])

            with gr.Group():
                gr.Markdown("**Check job status**")
                job_input = gr.Textbox(placeholder="Paste job ID here...", label="Job ID")
                job_btn = gr.Button("🔍 Check", variant="secondary")
                job_status = gr.Markdown("")

            job_btn.click(check_job, [job_input], [job_status])


if __name__ == "__main__":
    print(f"\n*** Starting UI at http://127.0.0.1:7860 ***")
    print(f"    Backend: {API_BASE}\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
