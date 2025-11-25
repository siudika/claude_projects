import streamlit as st
from anthropic import Anthropic, APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import re
import uuid
import requests
import hashlib
import streamlit_nested_layout  # type: ignore # imported for side effects

# --- Helpers ---

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def prettify_model_name(model_id: str) -> str:
    """
    Convert 'claude-sonnet-4-5-20250929' -> 'Claude Sonnet 4.5 (20250929)'
    """
    parts = model_id.split("-")
    if len(parts) < 3:
        return model_id.title()

    # e.g. ['claude', 'sonnet', '4', '5', '20250929']
    model_family = parts[1].capitalize()  # Sonnet, Opus, Haiku

    version_parts = []
    date_part = None
    for p in parts[2:]:
        if p.isdigit() and len(p) == 8:  # date
            date_part = p
        else:
            version_parts.append(p)

    version_str = ".".join(version_parts) if version_parts else ""

    if date_part:
        return f"Claude {model_family} {version_str} ({date_part})"
    else:
        return f"Claude {model_family} {version_str}".strip()

# --- Basic setup ---

load_dotenv()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PROJECT_INDEX_PATH = DATA_DIR / "project_index.json"
FILES_INDEX_PATH = DATA_DIR / "files_index.json"
USAGE_LOG_PATH = DATA_DIR / "usage_log.json"

st.set_page_config(
    page_title="Claude Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

system_prompt = "You are a helpful AI assistant. Be concise, accurate and clear."

# Inject Tabler Icons + custom styles
st.markdown(
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
    <style>
    .ti { font-size: 1.0rem; }
    /* Sticky bottom container - completely fixed to bottom */
    .sticky-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #0e1117;
        padding: 1rem;
        z-index: 999;
        border-top: 1px solid #333;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        max-height: 200px;
    }
    /* Main content padding to avoid overlap */
    .main > div {
        padding-bottom: 220px !important;
    }
    /* Thread list item hover effect */
    .thread-item-container:hover .thread-actions {
        opacity: 1;
    }
    .thread-actions {
        opacity: 0;
        transition: opacity 0.2s;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Anthropic client ---

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    st.error("ANTHROPIC_API_KEY not found")
    st.stop()

try:
    client = Anthropic(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Anthropic client: {e}")
    st.stop()

FILES_BETA_HEADER = "files-api-2025-04-14"
ANTHROPIC_VERSION = "2023-06-01"
API_BASE_URL = "https://api.anthropic.com/v1"

MODEL_PRICES = {
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
}
DEFAULT_PRICES = {"input": 3.0, "output": 15.0}

@st.cache_data(ttl=3600)
def fetch_models():
    try:
        return [m.id for m in client.models.list().data]
    except Exception:
        return ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]

AVAILABLE_MODELS = fetch_models()

# --- Helpers: local indexes ---

def load_project_index() -> dict:
    if PROJECT_INDEX_PATH.exists():
        try:
            with open(PROJECT_INDEX_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_project_index(index: dict):
    with open(PROJECT_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

def load_files_index() -> dict:
    if FILES_INDEX_PATH.exists():
        try:
            with open(FILES_INDEX_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_files_index(index: dict):
    with open(FILES_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

def load_usage_log() -> list:
    if USAGE_LOG_PATH.exists():
        try:
            with open(USAGE_LOG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_usage_log(entries: list):
    with open(USAGE_LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2)

def record_usage(model: str, input_tokens: int, output_tokens: int):
    prices = MODEL_PRICES.get(model, DEFAULT_PRICES)
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost

    entry = {
        "ts": datetime.utcnow().isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

    log = load_usage_log()
    log.append(entry)
    save_usage_log(log)
    return entry

def summarize_usage_last_24h():
    log = load_usage_log()
    if not log:
        return None

    now = datetime.utcnow()
    cutoff = now - timedelta(hours=24)

    total_input = total_output = 0
    total_cost = 0.0

    for entry in log:
        try:
            ts = datetime.fromisoformat(entry["ts"])
        except Exception:
            continue
        if ts < cutoff:
            continue
        total_input += entry.get("input_tokens", 0)
        total_output += entry.get("output_tokens", 0)
        total_cost += entry.get("total_cost", 0.0)

    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_cost": total_cost,
    }

# --- Helpers: code snippets ---

def normalize_code_snippets(content: str) -> str:
    """
    Convert various XML code formats to markdown:
    - <code_snippet language="...">...</code_snippet>
    - <code_change> blocks with <before>/<after>
    - Preserve <file>, <operation>, <description> as metadata
    """
    
    # Pattern 1: Handle <code_change> blocks with before/after
    code_change_pattern = r'<code_change>\s*<file>(.*?)</file>\s*<operation>(.*?)</operation>\s*<description>(.*?)</description>\s*<before>(.*?)</before>\s*<after>(.*?)</after>\s*</code_change>'
    
    def repl_code_change(match: re.Match) -> str:
        file = match.group(1).strip()
        operation = match.group(2).strip()
        description = match.group(3).strip()
        before_code = match.group(4).strip()
        after_code = match.group(5).strip()
        
        # Detect language from file extension
        lang = ""
        if "." in file:
            ext = file.split(".")[-1].lower()
            lang_map = {
                "py": "python", "js": "javascript", "ts": "typescript",
                "java": "java", "cpp": "cpp", "c": "c", "cs": "csharp",
                "rb": "ruby", "go": "go", "rs": "rust", "php": "php",
                "swift": "swift", "kt": "kotlin", "scala": "scala",
                "html": "html", "css": "css", "json": "json", "xml": "xml",
                "yaml": "yaml", "yml": "yaml", "md": "markdown", "sql": "sql",
                "sh": "bash", "bash": "bash", "ps1": "powershell",
            }
            lang = lang_map.get(ext, "")
        
        result = f"""
📝 {operation.title()}: {file}

**Description:** {description}

**Before:**
{before_code}
**After:**
{after_code}
"""
        return result
    
    content = re.sub(code_change_pattern, repl_code_change, content, flags=re.DOTALL)
    
    # Pattern 2: Handle standalone <before>/<after> blocks
    before_after_pattern = r'<before>(.*?)</before>\s*<after>(.*?)</after>'
    
    def repl_before_after(match: re.Match) -> str:
        before_code = match.group(1).strip()
        after_code = match.group(2).strip()
        
        result = f"""
**Before:**
{before_code}
**After:**
{after_code}
"""
        return result
    
    content = re.sub(before_after_pattern, repl_before_after, content, flags=re.DOTALL)
    
    # Pattern 3: Handle standard <code_snippet> tags
    code_snippet_pattern = r'<code_snippet(?:\s+language="([^"]*)")?\s*>([\s\S]*?)</code_snippet>'
    
    def repl_code_snippet(match: re.Match) -> str:
        lang = match.group(1) or ""
        code = match.group(2).strip("\n")
        return f"``````"
    
    content = re.sub(code_snippet_pattern, repl_code_snippet, content)
    
    # Pattern 4: Clean up remaining standalone XML tags
    metadata_tags = ['file', 'operation', 'description']
    for tag in metadata_tags:
        content = re.sub(f'<{tag}>(.*?)</{tag}>', r'**\1**', content, flags=re.DOTALL)
    
    return content
def render_message(content: str):
    """Render a message with normalized code blocks"""
    normalized = normalize_code_snippets(content)
    st.markdown(normalized, unsafe_allow_html=False)

# --- Helpers: threads
def load_threads():
    files = list(DATA_DIR.glob("*.json"))
    files = [f for f in files if f.name not in {"project_index.json", "files_index.json", "usage_log.json"}]
    return sorted(
        [f.stem for f in files],
        key=lambda x: (DATA_DIR / f"{x}.json").stat().st_mtime,
        reverse=True,
    )

def save_thread(name, conv):
    if conv:
        with open(DATA_DIR / f"{name}.json", "w") as f:
            json.dump(conv, f, indent=2)

def load_thread(name):
    try:
        with open(DATA_DIR / f"{name}.json", "r") as f:
            return json.load(f)
    except Exception:
        return []

def delete_thread(name):
    try:
        (DATA_DIR / f"{name}.json").unlink()
        return True
    except Exception:
        return False

def rename_thread(old_name: str, new_name: str) -> bool:
    try:
        old_path = DATA_DIR / f"{old_name}.json"
        new_path = DATA_DIR / f"{new_name}.json"
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)
            return True
        return False
    except Exception:
        return False

def generate_thread_name(message: str | None) -> str:
    base = "Chat"
    if message:
        words = message.strip().split()[:3]
        if words:
            base = " ".join(words).title()
    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base} [{suffix}]"

# --- Helpers: Files API ---

def upload_file_to_anthropic(file_name: str, file_bytes: bytes, mime_type: str = "text/plain") -> str:
    """
    Upload file to Anthropic Files API.
    Note: Only PDF and text/plain are supported. Convert all text-based files to text/plain.
    """
    # Map various text MIME types to text/plain (only supported text format)
    if mime_type and mime_type.startswith("text/"):
        mime_type = "text/plain"
    elif mime_type == "application/json":
        mime_type = "text/plain"
    elif mime_type == "application/xml":
        mime_type = "text/plain"
    elif mime_type not in ["application/pdf", "text/plain"]:
        # Default to text/plain for any unsupported type
        mime_type = "text/plain"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": FILES_BETA_HEADER,
    }
    files = {"file": (file_name, file_bytes, mime_type)}
    data = {"purpose": "document"}
    resp = requests.post(f"{API_BASE_URL}/files", headers=headers, files=files, data=data)
    resp.raise_for_status()
    payload = resp.json()
    return payload["id"]

def delete_file_from_anthropic(file_id: str):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": FILES_BETA_HEADER,
    }
    resp = requests.delete(f"{API_BASE_URL}/files/{file_id}", headers=headers)
    return resp.status_code

# --- Session state ---

if "model" not in st.session_state:
    st.session_state.model = AVAILABLE_MODELS[0]

if "thread" not in st.session_state:
    threads = load_threads()
    st.session_state.thread = threads[0] if threads else None

if "conversation" not in st.session_state:
    st.session_state.conversation = (
        load_thread(st.session_state.thread) if st.session_state.thread else []
    )

if "files" not in st.session_state:
    st.session_state.files = []

if "project_index" not in st.session_state:
    st.session_state.project_index = load_project_index()

if "files_index" not in st.session_state:
    st.session_state.files_index = load_files_index()

if "files_api_processed_batch" not in st.session_state:
    st.session_state.files_api_processed_batch = False

if "files_action" not in st.session_state:
    st.session_state.files_action = "analyze"

if "last_usage" not in st.session_state:
    st.session_state.last_usage = None

if "usage_summary_24h" not in st.session_state:
    st.session_state.usage_summary_24h = summarize_usage_last_24h()

# Thread editing state
if "editing_thread" not in st.session_state:
    st.session_state.editing_thread = None

if "editing_thread_text" not in st.session_state:
    st.session_state.editing_thread_text = ""

# --- SIDEBAR ---

with st.sidebar:
    # Top metric: current chat name
    if st.session_state.thread:
        st.metric(label="Current Chat", value=st.session_state.thread)
    else:
        st.metric(label="Current Chat", value="New Chat")

    # Model selector with prettified names
    model_options = AVAILABLE_MODELS
    model_labels = {m: prettify_model_name(m) for m in model_options}
    selected_label = st.selectbox(
        "Model",
        options=[model_labels[m] for m in model_options],
        index=[model_labels[m] for m in model_options].index(model_labels[st.session_state.model]),
        key="model_select_sidebar",
    )
    # reverse lookup
    for m_id, lbl in model_labels.items():
        if lbl == selected_label:
            st.session_state.model = m_id
            break

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.thread = None
        st.session_state.conversation = []
        st.session_state.files = []
        st.session_state.editing_thread = None
        st.rerun()

    st.markdown("---")
    st.markdown("### Chats")

    threads = load_threads()
    if not threads:
        st.caption("No chats yet")
    else:
        for thread in threads:
            # Check if this thread is being edited
            is_editing = st.session_state.editing_thread == thread
            active = st.session_state.thread == thread

            if is_editing:
                # Edit mode: show input and confirm/cancel buttons
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    new_name = st.text_input(
                        "Rename",
                        value=st.session_state.editing_thread_text,
                        key=f"edit_input_{thread}",
                        label_visibility="collapsed",
                    )
                with col2:
                    if st.button("✓", key=f"confirm_{thread}", help="Confirm", use_container_width=True):
                        if new_name and new_name != thread:
                            if rename_thread(thread, new_name):
                                if st.session_state.thread == thread:
                                    st.session_state.thread = new_name
                        st.session_state.editing_thread = None
                        st.rerun()
                with col3:
                    if st.button("✕", key=f"cancel_{thread}", help="Cancel", use_container_width=True):
                        st.session_state.editing_thread = None
                        st.rerun()
            else:
                # Normal mode: left-aligned title with hover buttons
                col1, col2, col3 = st.columns([3, 0.5, 0.5])

                with col1:
                    # Left-aligned button
                    icon = "✓ " if active else ""
                    if st.button(
                        f"{icon}{thread}",
                        key=f"thread_{thread}",
                        use_container_width=True,
                        type="primary" if active else "tertiary",
                    ):
                        st.session_state.thread = thread
                        st.session_state.conversation = load_thread(thread)
                        st.session_state.files = []
                        st.session_state.editing_thread = None
                        st.rerun()

                # Show edit/delete buttons in columns 2 and 3
                # Note: CSS hover effects are handled by the injected styles
                with col2:
                    if st.button("✎", key=f"edit_{thread}", help="Edit", use_container_width=True):
                        st.session_state.editing_thread = thread
                        st.session_state.editing_thread_text = thread
                        st.rerun()

                with col3:
                    if st.button("🗑", key=f"delete_{thread}", help="Delete", use_container_width=True):
                        if delete_thread(thread):
                            if st.session_state.thread == thread:
                                st.session_state.thread = None
                                st.session_state.conversation = []
                            st.rerun()

    st.markdown("---")

    # --- USAGE & COST (Collapsed by default) ---
    with st.expander("📊 Usage & Cost (24h)", expanded=False):
        usage = summarize_usage_last_24h()
        if usage:
            total_tokens = usage['input_tokens'] + usage['output_tokens']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input Tokens", f"{usage['input_tokens']:,}")
            with col2:
                st.metric("Output Tokens", f"{usage['output_tokens']:,}")
            with col3:
                st.metric(
                    "Total Cost",
                    f"${usage['total_cost']:.4f}",
                    delta=f"{total_tokens:,} tokens",
                    delta_color="off"
                )
        else:
            st.caption("No usage data in last 24 hours")

        # Show recent requests
        log = load_usage_log()
        if log:
            st.markdown("**Recent requests:**")
            recent = log[-5:][::-1]  # Last 5, reversed
            for entry in recent:
                st.caption(
                    f"• {prettify_model_name(entry['model'])}: "
                    f"{entry['input_tokens']:,} in / {entry['output_tokens']:,} out "
                    f"(${entry['total_cost']:.4f})"
                )

    st.markdown("---")

    # --- FILES API SECTION (Collapsed by default) ---
    with st.expander("📁 Files API", expanded=False):
        st.info("ℹ️ Files API supports PDF and text files only. Code files will be uploaded as plain text.")

        # File uploader with multiple file support
        uploaded = st.file_uploader(
            "Upload files to Files API",
            type=["pdf", "txt", "md", "json", "py", "js", "html", "css", "csv", "xml", "yaml", "yml"],
            accept_multiple_files=True,
            key="files_api_uploader",
            help="Upload files to Anthropic's Files API for document analysis (uploaded as PDF or text/plain)"
        )

        # Process uploaded files
        if uploaded:
            for file in uploaded:
                file_bytes = file.read()
                file_hash = sha256_bytes(file_bytes)
                idx = st.session_state.files_index

                # Check if file already uploaded (by name)
                if file.name not in idx:
                    try:
                        with st.spinner(f"Uploading {file.name}..."):
                            file_id = upload_file_to_anthropic(file.name, file_bytes, file.type or "text/plain")
                            idx[file.name] = {
                                "file_id": file_id,
                                "size": len(file_bytes),
                                "size_kb": len(file_bytes) / 1024,
                                "mime_type": file.type or "text/plain",
                                "sha256": file_hash,
                                "uploaded_at": datetime.utcnow().isoformat(),
                            }
                            save_files_index(idx)
                            st.session_state.files_index = idx
                            st.success(f"✓ Uploaded {file.name}")
                    except Exception as e:
                        st.error(f"Upload failed for {file.name}: {e}")
                else:
                    st.info(f"✓ {file.name} already uploaded")

        st.markdown("---")
        st.markdown("**Uploaded Files:**")

        idx = st.session_state.files_index
        if not idx:
            st.caption("No Files API uploads yet.")
        else:
            for file_name, meta in list(idx.items()):
                file_id = meta.get("file_id", "")
                size_kb = meta.get("size_kb", meta.get("size", 0) / 1024)

                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.caption(f"📄 {file_name}")
                    st.caption(f"Size: {size_kb:.1f} KB")
                with col2:
                    if st.button("📌", key=f"attach_{file_name}", help="Attach to chat", use_container_width=True):
                        if file_name not in st.session_state.files:
                            st.session_state.files.append(file_name)
                        st.rerun()
                with col3:
                    if st.button("🗑", key=f"del_file_{file_name}", help="Delete", use_container_width=True):
                        try:
                            delete_file_from_anthropic(file_id)
                            del idx[file_name]
                            save_files_index(idx)
                            st.session_state.files_index = idx
                            if file_name in st.session_state.files:
                                st.session_state.files.remove(file_name)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

# --- MAIN CONTENT AREA ---

# Title & current thread display
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💬 Claude Chat")
with col2:
    if st.session_state.thread:
        st.caption(f"Thread: **{st.session_state.thread}**")

st.markdown("---")

# Conversation history
if st.session_state.conversation:
    for msg in st.session_state.conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            render_message(content)

# Attached Files API documents display
if st.session_state.files:
    st.markdown("---")
    st.markdown("**📎 Attached Files:**")
    idx = st.session_state.files_index
    for file_name in st.session_state.files:
        # file_name is now the key in files_index
        if file_name in idx:
            meta = idx[file_name]
            file_size = meta.get("size", 0)
            st.caption(f"📄 {file_name} ({file_size:,} bytes)")

# Add spacer to ensure content doesn't overlap with sticky bottom
st.markdown("<div style='height: 220px;'></div>", unsafe_allow_html=True)

# --- STICKY BOTTOM: Chat Input + Files API Checkbox ---
st.markdown('<div class="sticky-bottom">', unsafe_allow_html=True)

# Files API checkbox and controls (at top of sticky section)
use_files = st.checkbox(
    "📎 Use Files API documents",
    value=False,
    key="use_files_main",
    help="Include attached Files API documents as context.",
)

selected_files = []
selected_action = st.session_state.files_action

if use_files:
    idx = st.session_state.files_index
    if idx:
        file_names = list(idx.keys())
        selected_files = st.multiselect(
            "Select files:",
            options=file_names,
            default=[],
            key="files_api_selected",
        )

        if selected_files:
            action_pill = st.pills(
                "Action:",
                options=["🔍 Analyze", "✏️ Edit", "📝 Summarize"],
                default="🔍 Analyze",
                key="files_action_pills",
            )
            action_map = {
                "🔍 Analyze": "analyze",
                "✏️ Edit": "edit",
                "📝 Summarize": "summarize",
            }
            st.session_state.files_action = action_map.get(action_pill, "analyze")
            selected_action = st.session_state.files_action
    else:
        st.caption("ℹ No Files API uploads yet. Upload in sidebar.")

# Chat input (always at bottom)
user_input = st.chat_input("Message Claude...", key="chat_input_main")

st.markdown("</div>", unsafe_allow_html=True)

# --- MESSAGE PROCESSING ---
if user_input:
    # Build message content
    full_msg = user_input

    # Add project context if available
    index = st.session_state.project_index
    if index:
        summaries_text = "\n\n".join(
            f"### {file_name}\n{meta.get('summary', meta) if isinstance(meta, dict) else meta}"
            for file_name, meta in list(index.items())[:5]
        )
        full_msg = f"""You have previously analyzed these project files:

{summaries_text}

Now answer the following question or request, using those summaries as context:

{user_input}
"""

    # Add Files API context if selected
    if use_files and selected_files:
        action_text = {
            "analyze": "Analyze the selected files and answer the question.",
            "edit": "Propose code changes to the selected files as requested.",
            "summarize": "Summarize the selected files at a high level.",
        }[selected_action]
        full_msg = f"""{action_text}

Question: {user_input}
"""

    # Auto-generate thread name if needed
    if not st.session_state.thread:
        st.session_state.thread = generate_thread_name(user_input)

    # Add user message to conversation
    st.session_state.conversation.append({
        "role": "user",
        "content": full_msg,
    })

    # Build content blocks
    content_blocks = [{"type": "text", "text": full_msg}]

    # Add Files API documents if selected
    if use_files and selected_files:
        idx = st.session_state.files_index
        for file_name in selected_files:
            # The filename IS the key in files_index
            if file_name in idx:
                meta = idx[file_name]
                file_id = meta.get("file_id", "")

                if file_id:
                    content_blocks.append({
                        "type": "document",
                        "source": {
                            "type": "file",
                            "file_id": file_id,
                        },
                        "title": f"{file_name} ({selected_action})",
                        "context": f"User selected action: {selected_action} on this file.",
                    })
                    
    # Build API messages
    api_messages = []
    for msg in st.session_state.conversation[:-1]:
        api_messages.append({
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}],
        })
    api_messages.append({
        "role": "user",
        "content": content_blocks,
    })

    # API request
    try:
        with st.spinner("Claude is thinking..."):
            extra_headers = {}
            if use_files and selected_files:
                extra_headers["anthropic-beta"] = FILES_BETA_HEADER

            # Trim to last 30 messages
            trimmed_messages = api_messages[-30:]

            response = client.messages.create(
                model=st.session_state.model,
                max_tokens=4096,
                system=system_prompt,
                messages=trimmed_messages,
                extra_headers=extra_headers if extra_headers else None,
            )

            # Extract assistant response
            assistant_message = response.content[0].text if response.content else ""

            # Record usage
            usage_entry = record_usage(
                model=st.session_state.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            st.session_state.last_usage = usage_entry
            st.session_state.usage_summary_24h = summarize_usage_last_24h()

            # Add assistant response to conversation
            st.session_state.conversation.append({
                "role": "assistant",
                "content": assistant_message,
            })

            # Save thread
            save_thread(st.session_state.thread, st.session_state.conversation)

            st.rerun()

    except (APIConnectionError, APIStatusError, AuthenticationError, RateLimitError) as e:
        st.error(f"API Error: {e}")
        # Remove last user message on error
        if st.session_state.conversation and st.session_state.conversation[-1]["role"] == "user":
            st.session_state.conversation.pop()
    except Exception as e:
        st.error(f"Error: {e}")
        if st.session_state.conversation and st.session_state.conversation[-1]["role"] == "user":
            st.session_state.conversation.pop()