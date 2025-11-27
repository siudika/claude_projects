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
import streamlit_nested_layout  # imported for side effects
from cryptography.fernet import Fernet
import pyotp  # for TOTP-based login

# --- ENCRYPTION SETUP ---
_encryption_key = os.environ.get("CLAUDE_CHAT_KEY")
if not _encryption_key:
    st.error(
        "CLAUDE_CHAT_KEY not found. Set it in your environment or .env file.\n"
        "Example (bash): export CLAUDE_CHAT_KEY=$(python -c \"from cryptography.fernet import Fernet; "
        "print(Fernet.generate_key().decode())\")"
    )
    st.stop()

try:
    cipher_suite = Fernet(_encryption_key)
except Exception as e:
    st.error(f"Invalid CLAUDE_CHAT_KEY: {e}")
    st.stop()

# --- Helpers ---
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def prettify_model_name(model_id: str) -> str:
    """Convert 'claude-sonnet-4-5-20250929' -> 'Claude Sonnet 4.5 (20250929)'"""
    parts = model_id.split("-")
    if len(parts) < 3:
        return model_id.title()
    model_family = parts[1].capitalize()
    version_parts = []
    date_part = None
    for p in parts[2:]:
        if p.isdigit() and len(p) == 8:
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

# Mobile-first: centered layout, collapsed sidebar
st.set_page_config(
    page_title="Claude Chat",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Secure Claude Chat with TOTP & Encryption"
    }
)

system_prompt = "You are a helpful AI assistant. Be concise, accurate and clear."

# --- TOTP LOGIN GATE ---
def require_totp_login():
    """Gate the entire app behind a TOTP-based login."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return
    
    secret = os.environ.get("CLAUDE_TOTP_SECRET")
    if not secret:
        st.error(
            "CLAUDE_TOTP_SECRET not set. Configure it in your environment or .env file "
            "to enable TOTP login."
        )
        st.stop()
    
    totp = pyotp.TOTP(secret)
    
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("🔐 Claude Chat")
    st.write("Enter your authenticator code to continue")
    
    code = st.text_input("6-digit code", type="password", max_chars=8, key="totp_code")
    
    if st.button("🔓 Unlock", type="primary", use_container_width=True, key="unlock_btn"):
        if totp.verify(code, valid_window=1):
            st.session_state.authenticated = True
            st.success("✓ Authenticated")
            st.rerun()
        else:
            st.error("❌ Invalid code")
    
    st.caption("💡 Codes refresh every 30 seconds")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

require_totp_login()

# --- MOBILE-RESPONSIVE STYLES ---
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
<style>
    @media (max-width: 768px) {
        .stApp {
            padding-bottom: 180px !important;
        }
        
        [data-testid="collapsedControl"] {
            top: 0.5rem;
            left: 0.5rem;
        }
        
        h1 {
            font-size: 1.5rem !important;
            margin: 0.5rem 0 !important;
        }
        
        [data-testid="stChatMessageContainer"] {
            max-width: 100% !important;
            padding: 0.5rem !important;
        }
        
        button {
            min-height: 44px !important;
            font-size: 1rem !important;
        }
        
        input, textarea, select {
            font-size: 16px !important;
            min-height: 44px !important;
        }
        
        .sidebar-content {
            padding: 0.5rem !important;
        }
    }
    
    .sticky-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: var(--background-color);
        padding: 0.75rem;
        z-index: 999;
        border-top: 1px solid var(--border-color);
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    @media (min-width: 769px) {
        .sticky-bottom {
            max-width: 730px;
            margin: 0 auto;
        }
        .stApp {
            padding-bottom: 200px !important;
        }
    }
    
    [data-testid="stChatInput"] {
        border-radius: 24px !important;
    }
    
    .ti {
        font-size: 1.2rem;
    }
    
    .thread-item:hover {
        background: var(--hover-color);
        cursor: pointer;
    }
    
    .code-block {
        margin: 12px 0;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .code-header {
        background: var(--secondary-background-color);
        padding: 8px 12px;
        font-size: 0.75rem;
    }
    
    pre {
        margin: 0 !important;
        padding: 12px !important;
        overflow-x: auto !important;
        background: var(--code-background) !important;
    }
    
    @media (max-width: 768px) {
        .hide-on-mobile {
            display: none !important;
        }
        
        [data-testid="stMetric"] {
            padding: 0.5rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

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
    """Convert various XML code formats to markdown"""
    code_change_pattern = r'de_change>\s*<file>(.*?)</file>\s*<operation>(.*?)</operation>\s*<description>(.*?)</description>\s*<before>(.*?)</before>\s*<after>(.*?)</after>\s*</code_change>'
    
    def repl_code_change(match: re.Match) -> str:
        file = match.group(1).strip()
        operation = match.group(2).strip()
        description = match.group(3).strip()
        before_code = match.group(4).strip()
        after_code = match.group(5).strip()
        
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
    
    code_snippet_pattern = r'de_snippet(?:\s+language="([^"]*)")?>([\\s\\S]*?)</code_snippet>'
    
    def repl_code_snippet(match: re.Match) -> str:
        lang = match.group(1) or ""
        code = match.group(2).strip("\n")
        return f"``````"
    
    content = re.sub(code_snippet_pattern, repl_code_snippet, content)
    
    metadata_tags = ['file', 'operation', 'description']
    for tag in metadata_tags:
        content = re.sub(f'<{tag}>(.*?)</{tag}>', r'**\1**', content, flags=re.DOTALL)
    
    return content

def render_message(content: str):
    """Render a message with normalized code blocks"""
    normalized = normalize_code_snippets(content)
    st.markdown(normalized, unsafe_allow_html=False)

# --- Helpers: threads (encrypted on disk) ---
def _thread_plain_path(name: str) -> Path:
    return DATA_DIR / f"{name}.json"

def _thread_encrypted_path(name: str) -> Path:
    return DATA_DIR / f"{name}.json.enc"

def load_threads():
    enc_files = list(DATA_DIR.glob("*.json.enc"))
    threads: list[str] = [f.stem.replace(".json", "") for f in enc_files]
    
    plain_files = [
        f for f in DATA_DIR.glob("*.json")
        if f.name not in {"project_index.json", "files_index.json", "usage_log.json"}
    ]
    for f in plain_files:
        name = f.stem
        if name not in threads:
            threads.append(name)
    
    def _mtime(thread_name: str) -> float:
        enc = _thread_encrypted_path(thread_name)
        plain = _thread_plain_path(thread_name)
        if enc.exists():
            return enc.stat().st_mtime
        if plain.exists():
            return plain.stat().st_mtime
        return 0.0
    
    return sorted(threads, key=_mtime, reverse=True)

def save_thread(name, conv):
    if not conv:
        return
    try:
        raw = json.dumps(conv, indent=2).encode("utf-8")
        token = cipher_suite.encrypt(raw)
        enc_path = _thread_encrypted_path(name)
        with open(enc_path, "wb") as f:
            f.write(token)
    except Exception as e:
        st.error(f"Failed to save encrypted thread '{name}': {e}")
        return
    
    plain_path = _thread_plain_path(name)
    if plain_path.exists():
        try:
            plain_path.unlink()
        except Exception:
            pass

def load_thread(name):
    enc_path = _thread_encrypted_path(name)
    if enc_path.exists():
        try:
            with open(enc_path, "rb") as f:
                token = f.read()
            raw = cipher_suite.decrypt(token)
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            st.error(f"Failed to decrypt thread '{name}': {e}")
            return []
    
    plain_path = _thread_plain_path(name)
    if plain_path.exists():
        try:
            with open(plain_path, "r", encoding="utf-8") as f:
                conv = json.load(f)
            save_thread(name, conv)
            return conv
        except Exception as e:
            st.error(f"Failed to load legacy thread '{name}': {e}")
            return []
    
    return []

def delete_thread(name):
    ok = False
    enc_path = _thread_encrypted_path(name)
    plain_path = _thread_plain_path(name)
    
    if enc_path.exists():
        try:
            enc_path.unlink()
            ok = True
        except Exception:
            pass
    
    if plain_path.exists():
        try:
            plain_path.unlink()
            ok = True
        except Exception:
            pass
    
    return ok

def rename_thread(old_name: str, new_name: str) -> bool:
    try:
        old_enc = _thread_encrypted_path(old_name)
        new_enc = _thread_encrypted_path(new_name)
        old_plain = _thread_plain_path(old_name)
        new_plain = _thread_plain_path(new_name)
        
        if old_enc.exists():
            if new_enc.exists():
                return False
            old_enc.rename(new_enc)
            if new_plain.exists():
                try:
                    new_plain.unlink()
                except Exception:
                    pass
            return True
        
        if old_plain.exists():
            if new_plain.exists():
                return False
            old_plain.rename(new_plain)
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
    if mime_type and mime_type.startswith("text/"):
        mime_type = "text/plain"
    elif mime_type == "application/json":
        mime_type = "text/plain"
    elif mime_type == "application/xml":
        mime_type = "text/plain"
    elif mime_type not in ["application/pdf", "text/plain"]:
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

def refresh_files_index_from_anthropic() -> dict:
    idx = load_files_index()
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": FILES_BETA_HEADER,
        }
        resp = requests.get(f"{API_BASE_URL}/files", headers=headers)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data", []) or payload.get("files", [])
        
        for fobj in items:
            file_id = fobj.get("id")
            fname = fobj.get("filename") or file_id
            size = fobj.get("size", 0)
            key = fname
            
            existing = idx.get(key, {})
            existing.update({
                "file_id": file_id,
                "size": size,
                "size_kb": size / 1024 if size else existing.get("size_kb", 0),
                "mime_type": fobj.get("mime_type", existing.get("mime_type", "text/plain")),
                "uploaded_at": fobj.get("created_at", existing.get("uploaded_at")),
            })
            idx[key] = existing
        
        save_files_index(idx)
        return idx
    except Exception as e:
        st.error(f"Failed to refresh Files index from Anthropic: {e}")
        return idx

# --- Session state initialization ---
if "model" not in st.session_state:
    st.session_state.model = AVAILABLE_MODELS[0]

if "thread" not in st.session_state:
    threads = load_threads()
    st.session_state.thread = threads[0] if threads else None

if "conversation" not in st.session_state:
    st.session_state.conversation = (
        load_thread(st.session_state.thread) if st.session_state.thread else []
    )

if "project_index" not in st.session_state:
    st.session_state.project_index = load_project_index()

if "files_index" not in st.session_state:
    st.session_state.files_index = load_files_index()

if "files_action" not in st.session_state:
    st.session_state.files_action = "analyze"

if "last_usage" not in st.session_state:
    st.session_state.last_usage = None

if "usage_summary_24h" not in st.session_state:
    st.session_state.usage_summary_24h = summarize_usage_last_24h()

if "editing_thread" not in st.session_state:
    st.session_state.editing_thread = None

if "editing_thread_text" not in st.session_state:
    st.session_state.editing_thread_text = ""

if "pending_file_upload" not in st.session_state:
    st.session_state.pending_file_upload = False

# --- CALLBACKS (No reruns unless necessary) ---
def switch_thread(thread_name):
    """Switch to a different thread without triggering rerun"""
    st.session_state.thread = thread_name
    st.session_state.conversation = load_thread(thread_name)
    st.session_state.editing_thread = None

def start_new_chat():
    """Start a new chat without rerun"""
    st.session_state.thread = None
    st.session_state.conversation = []
    st.session_state.editing_thread = None

def start_edit_thread(thread_name):
    """Enter edit mode for a thread"""
    st.session_state.editing_thread = thread_name
    st.session_state.editing_thread_text = thread_name

def confirm_rename():
    """Confirm thread rename"""
    old = st.session_state.editing_thread
    new = st.session_state.editing_thread_text
    if old and new and new != old:
        if rename_thread(old, new):
            if st.session_state.thread == old:
                st.session_state.thread = new
    st.session_state.editing_thread = None

def cancel_edit():
    """Cancel edit mode"""
    st.session_state.editing_thread = None

def delete_thread_action(thread_name):
    """Delete a thread"""
    if delete_thread(thread_name):
        if st.session_state.thread == thread_name:
            st.session_state.thread = None
            st.session_state.conversation = []

# --- MOBILE-OPTIMIZED SIDEBAR ---
with st.sidebar:
    # Top metric: current chat name
    if st.session_state.thread:
        st.metric(label="Current Chat", value=st.session_state.thread)
    else:
        st.caption("📌 New Chat")
    
    # Model selector
    model_options = AVAILABLE_MODELS
    model_labels = {m: prettify_model_name(m) for m in model_options}
    selected_label = st.selectbox(
        "Model",
        options=[model_labels[m] for m in model_options],
        index=[model_labels[m] for m in model_options].index(model_labels[st.session_state.model]),
        key="model_select_sidebar",
    )
    
    for m_id, lbl in model_labels.items():
        if lbl == selected_label:
            st.session_state.model = m_id
            break
    
    if st.button("➕ New Chat", use_container_width=True, type="primary", on_click=start_new_chat):
        pass  # Callback handles it
    
    st.markdown("---")
    st.markdown("### 💬 Chats")
    
    threads = load_threads()
    if not threads:
        st.caption("No chats yet")
    else:
        # Show recent threads
        for thread in threads[:10]:
            is_editing = st.session_state.editing_thread == thread
            active = st.session_state.thread == thread
            
            if is_editing:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text_input(
                        "Rename",
                        value=st.session_state.editing_thread_text,
                        key=f"edit_input_{thread}",
                        label_visibility="collapsed",
                        on_change=lambda: setattr(st.session_state, 'editing_thread_text', st.session_state[f"edit_input_{thread}"])
                    )
                with col2:
                    if st.button("✓", key=f"confirm_{thread}", help="Confirm", use_container_width=True, on_click=confirm_rename):
                        pass
                with col3:
                    if st.button("✕", key=f"cancel_{thread}", help="Cancel", use_container_width=True, on_click=cancel_edit):
                        pass
            else:
                col1, col2, col3 = st.columns([3, 0.5, 0.5])
                with col1:
                    icon = "✓ " if active else ""
                    display_name = thread if len(thread) < 25 else thread[:22] + "..."
                    if st.button(
                        f"{icon}{display_name}",
                        key=f"thread_{thread}",
                        use_container_width=True,
                        type="primary" if active else "tertiary",
                        on_click=switch_thread,
                        args=(thread,)
                    ):
                        pass  # Callback handles it
                
                with col2:
                    if st.button("✎", key=f"edit_{thread}", help="Edit", use_container_width=True, on_click=start_edit_thread, args=(thread,)):
                        pass
                
                with col3:
                    if st.button("🗑", key=f"delete_{thread}", help="Delete", use_container_width=True, on_click=delete_thread_action, args=(thread,)):
                        pass
    
    st.markdown("---")
    
    # Usage & Cost
    with st.expander("📊 Usage (24h)"):
        usage = st.session_state.usage_summary_24h
        if usage:
            st.metric("Cost", f"${usage['total_cost']:.4f}")
            st.caption(f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out")
        else:
            st.caption("No usage data")
    
    # Files API
    with st.expander("📁 Files API", expanded=False):
        st.caption("PDF and text files only")
        
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_files"):
            st.session_state.files_index = refresh_files_index_from_anthropic()
        
        # File uploader (key stays constant to avoid reset)
        uploaded = st.file_uploader(
            "Upload",
            type=["pdf", "txt", "md", "json", "py", "js", "html", "css", "csv", "xml", "yaml", "yml"],
            accept_multiple_files=True,
            key="files_api_uploader",
            label_visibility="collapsed"
        )
        
        # Process uploads WITHOUT rerun
        if uploaded:
            idx = st.session_state.files_index
            for file in uploaded:
                file_bytes = file.read()
                file_hash = sha256_bytes(file_bytes)
                
                if file.name not in idx:
                    try:
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
                        st.success(f"✓ {file.name}", icon="✅")
                    except Exception as e:
                        st.error(f"❌ {file.name}: {e}")
        
        # Show uploaded files
        idx = st.session_state.files_index
        if idx:
            st.caption(f"**{len(idx)} files:**")
            for file_name, meta in list(idx.items())[:5]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"📄 {file_name[:25]}...")
                with col2:
                    if st.button("🗑", key=f"del_{file_name}", help="Delete", use_container_width=True):
                        try:
                            delete_file_from_anthropic(meta.get("file_id", ""))
                            del idx[file_name]
                            save_files_index(idx)
                            st.session_state.files_index = idx
                            st.success("Deleted")
                        except Exception:
                            st.error("Failed")

# --- MAIN CONTENT ---

# Header
if st.session_state.thread:
    st.title("🗨️ "+st.session_state.thread)
else:
    st.title("📌 New Chat")

# Conversation history
if st.session_state.conversation:
    for msg in st.session_state.conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            render_message(content)

# Spacer for sticky bottom
st.markdown('<div style="height: 180px;"></div>', unsafe_allow_html=True)

# --- STICKY BOTTOM ---
st.markdown('<div class="sticky-bottom">', unsafe_allow_html=True)

# Files attachment UI
use_files = st.checkbox(
    "📎 Attach files",
    value=False,
    key="use_files_main",
    help="Add files as context",
)

selected_files = []
selected_action = st.session_state.files_action

if use_files:
    idx = st.session_state.files_index
    if idx:
        file_names = list(idx.keys())
        selected_files = st.multiselect(
            "Select files",
            options=file_names,
            default=[],
            key="files_api_selected",
        )
        
        if selected_files:
            action_pill = st.pills(
                "Action",
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
        st.caption("No files uploaded yet")

# Chat input
user_input = st.chat_input("Type your message...", key="chat_input_main")

st.markdown('</div>', unsafe_allow_html=True)

# --- MESSAGE PROCESSING ---
if user_input:
    full_msg = user_input
    
    # Add context for file operations
    if use_files and selected_files:
        action_text = {
            "analyze": "Analyze the selected files and answer the question.",
            "edit": "Propose code changes to the selected files as requested.",
            "summarize": "Summarize the selected files at a high level.",
        }[selected_action]
        full_msg = f"{action_text}\n\nQuestion: {user_input}"
    
    # Create thread if needed
    if not st.session_state.thread:
        st.session_state.thread = generate_thread_name(user_input)
    
    # Add user message
    st.session_state.conversation.append({
        "role": "user",
        "content": full_msg,
    })
    
    # Build content blocks
    content_blocks = [{"type": "text", "text": full_msg}]
    
    if use_files and selected_files:
        idx = st.session_state.files_index
        for file_name in selected_files:
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
    
    # Call Claude API
    try:
        with st.spinner("💭 Thinking..."):
            extra_headers = {}
            if use_files and selected_files:
                extra_headers["anthropic-beta"] = FILES_BETA_HEADER
            
            trimmed_messages = api_messages[-30:]
            response = client.messages.create(
                model=st.session_state.model,
                max_tokens=4096,
                system=system_prompt,
                messages=trimmed_messages,
                extra_headers=extra_headers if extra_headers else None,
            )
            
            assistant_message = response.content[0].text if response.content else ""
            
            # Record usage
            usage_entry = record_usage(
                model=st.session_state.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            
            st.session_state.last_usage = usage_entry
            st.session_state.usage_summary_24h = summarize_usage_last_24h()
            
            # Add assistant response
            st.session_state.conversation.append({
                "role": "assistant",
                "content": assistant_message,
            })
            
            # Save thread
            save_thread(st.session_state.thread, st.session_state.conversation)
            
            # NOW rerun to show the response
            st.rerun()
            
    except (APIConnectionError, APIStatusError, AuthenticationError, RateLimitError) as e:
        st.error(f"API Error: {e}")
        if st.session_state.conversation and st.session_state.conversation[-1]["role"] == "user":
            st.session_state.conversation.pop()
    except Exception as e:
        st.error(f"Error: {e}")
        if st.session_state.conversation and st.session_state.conversation[-1]["role"] == "user":
            st.session_state.conversation.pop()
