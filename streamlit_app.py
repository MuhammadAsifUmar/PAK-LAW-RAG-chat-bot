import uuid
import streamlit as st
from datetime import datetime

# Assuming these are from your custom modules
from rag_chain import stream_question, inject_metadata_into_answer
import persistence_database as db

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    
    page_title="PakLaw AI · Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
# ADVANCED GEMINI-STYLE DARK MODE CSS (ALL ISSUES FIXED)
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* 1. GLOBAL TYPOGRAPHY & DARK THEME (Safe application to avoid breaking icons) */
/* Poore app ka font aur base theme set karna */
html, body, .stApp, p, h1, h2, h3, h4, h5, h6, span, div, button, input, textarea {
    font-family: 'Inter', sans-serif;
}

/* Restore font for Streamlit's built-in Material icons */
.material-symbols-rounded, .material-icons {
    font-family: 'Material Symbols Rounded' !important;
}

html, body, .stApp {
    background-color: #0A0A0A !important;
    color: #FFFFFF !important;
}

/* FIX HEADER (Upar wala white bar hatane ke liye) */
header[data-testid="stHeader"], [data-testid="stHeader"] {
    background-color: #0A0A0A !important;
    background: #0A0A0A !important;
}

/* 2. FIX CHAT OVERLAP & SCROLLING ISSUES */
/* Neeche extra padding taake aakhri message input box ke peechay na chupe */
.main .block-container {
    padding: 2rem 2rem 150px !important; 
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* 3. INPUT FOOTER */
/* Chat input ke area ka background black karna */
[data-testid="stBottom"], 
[data-testid="stBottom"] > div, 
.stBottomBlockContainer {
    background-color: #0A0A0A !important; 
    padding-bottom: 20px !important;
    padding-top: 20px !important;
    z-index: 99999 !important; 
}

/* Chat input bar styling */
[data-testid="stChatInput"] {
    background-color: #0A0A0A !important;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] div[data-baseweb="base-input"] {
    background-color: #18181A !important;
    border-radius: 24px !important;
}

[data-testid="stChatInput"] > div {
    border: 1px solid #27272A !important;
    padding: 4px 8px !important;
    transition: border-color 0.3s ease;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px #3B82F6 !important;
}

[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important; 
    border: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #A1A1AA !important;
    -webkit-text-fill-color: #A1A1AA !important;
}

/* Send button */
[data-testid="stChatInputSubmitButton"] {
    background-color: #FFFFFF !important;
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
    margin-top: 6px !important;
}
[data-testid="stChatInputSubmitButton"] svg {
    fill: #000000 !important;
}

/* 4. SIDEBAR STYLING (DARK GREY/BLACK WITH WHITE TEXT) */
section[data-testid="stSidebar"] {
    background-color: #121212 !important;
    border-right: 1px solid #27272A !important;
}
section[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background-color: #18181B !important;
    color: #FFFFFF !important;
    border: 1px solid #27272A !important;
    border-radius: 12px !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #27272A !important;
    border-color: #3B82F6 !important;
}

/* 5. GEMINI-STYLE CHAT BUBBLES */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 20px 0 !important;
}

/* User Message (Right-aligned dark grey bubble) */
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #27272A !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 16px 24px !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    max-width: 80% !important;
    border: 1px solid #3F3F46 !important;
}

/* Assistant Message (Left-aligned, transparent) */
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background-color: transparent !important;
    padding: 10px 0 !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    max-width: 95% !important;
}

/* FORCE ALL CHAT TEXT TO WHITE */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] * {
    color: #FFFFFF !important; 
    -webkit-text-fill-color: #FFFFFF !important; 
    line-height: 1.6 !important;
}

/* 6. SUGGESTION BUTTONS */
.stButton > button {
    background-color: #18181A !important;
    color: #A1A1AA !important;
    border: 1px solid #27272A !important;
    border-radius: 16px !important;
    padding: 14px 20px !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #27272A !important;
    color: #FFFFFF !important;
    border-color: #3B82F6 !important;
}

/* ════════════════════════════════════════════════════════════
   7. EXPANDER FIX (Settings aur View References ka White Background Problem) 
   ════════════════════════════════════════════════════════════ */

/* Expander (Dropdown) ke header ka base color set karna */
details summary, 
[data-testid="stExpander"] details summary {
    background-color: #18181B !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: 1px solid #27272A !important;
}

/* Click karne ya hover karne par white color ko rokna */
details summary:hover, 
details summary:focus, 
details summary:active,
[data-testid="stExpander"] details summary:hover,
[data-testid="stExpander"] details summary:focus,
[data-testid="stExpander"] details summary:active {
    background-color: #27272A !important;
    color: #FFFFFF !important;
    border-color: #3B82F6 !important;
    outline: none !important;
}

/* Arrow icon ko white rakhna */
details summary svg,
[data-testid="stExpander"] details summary svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* Expander khulne ke baad uske andar ka box design */
div[data-testid="stExpanderDetails"] {
    background-color: #121212 !important;
    border: 1px solid #27272A !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 15px !important;
}

/* ════════════════════════════════════════════════════════════
   8. SELECTBOX FIX (Model Routing White Background Problem) 
   ════════════════════════════════════════════════════════════ */
/* Select box closed state */
div[data-baseweb="select"] > div {
    background-color: #18181B !important;
    color: #FFFFFF !important;
    border: 1px solid #27272A !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] * {
    color: #FFFFFF !important;
}
/* Select box dropdown list (popover) */
div[data-baseweb="popover"] > div, 
div[data-baseweb="popover"] ul {
    background-color: #18181B !important;
    border: 1px solid #27272A !important;
    border-radius: 8px !important;
}
div[data-baseweb="popover"] li {
    color: #FFFFFF !important;
    background-color: transparent !important;
}
div[data-baseweb="popover"] li:hover {
    background-color: #27272A !important;
    color: #3B82F6 !important;
}


/* Custom Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0A0A; }
::-webkit-scrollbar-thumb { background: #3F3F46; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #3B82F6; }

/* Hide default streamlit marks */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SESSION STATE & HELPER
# ════════════════════════════════════════════════════════════
def new_thread() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "messages": [],
    }

if "db_loaded" not in st.session_state:
    db.init_db()
    saved_threads = db.get_all_threads()
    
    if saved_threads:
        st.session_state.threads = saved_threads
    else:
        st.session_state.threads = [new_thread()]
        
    st.session_state.active_idx = 0
    st.session_state.pending_q = None
    st.session_state.db_loaded = True
    
    # Active thread ke messages database se load karna
    active_id = st.session_state.threads[0]["id"]
    st.session_state.threads[0]["messages"] = db.get_chat_history(active_id)

def current() -> dict:
    return st.session_state.threads[st.session_state.active_idx]

# ════════════════════════════════════════════════════════════
# SIDEBAR (WHITE TEXT ON GREY/BLACK)
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚖️ PakLaw AI")
    st.caption("Pakistan Legal Reference Assistant")
    
    st.write("") # Spacing
    if st.button("➕  New Chat", key="new_chat_btn", use_container_width=True):
        st.session_state.threads.insert(0, new_thread())
        st.session_state.active_idx = 0
        st.session_state.pending_q = None
        st.rerun()

    st.markdown("---")
    st.markdown("<p style='color:#A1A1AA; font-size:0.8rem; font-weight:600;'>RECENT CONVERSATIONS</p>", unsafe_allow_html=True)

    for idx, thread in enumerate(st.session_state.threads):
        is_active = (idx == st.session_state.active_idx)
        title_short = thread["title"][:28] + ("…" if len(thread["title"]) > 28 else "")

        if is_active:
            st.markdown(
                f'<div style="background-color:#27272A; border-left: 3px solid #3B82F6; color:#FFFFFF;'
                f'border-radius:0 8px 8px 0; padding:10px 14px; margin-bottom:8px;'
                f'font-size:0.9rem; font-weight:500;">'
                f'{title_short}</div>',
                unsafe_allow_html=True,
            )
        else:
            if st.button(title_short, key=f"thread_{thread['id']}", use_container_width=True):
                st.session_state.active_idx = idx
                st.session_state.pending_q = None
                # DB se is specific chat ke messages fetch karna
                st.session_state.threads[idx]["messages"] = db.get_chat_history(thread["id"])
                st.rerun()
    st.markdown("---")
    
    # Advanced Options (Ab iska design white nahi hoga)
    with st.expander("⚙️ Advanced Settings"):
        st.markdown("**Response Language**")
        language = st.radio(
            "lang",
            ["English", "اردو (Urdu)"],
            index=0,
            key="language",
            label_visibility="collapsed",
        )
        st.markdown("**Model Routing**")
        st.selectbox("Engine", ["Groq Mixtral-8x7b", "Llama-3-70b", "Gemma-7b"], label_visibility="collapsed")
        
    st.markdown("---")
    
    # Coverage Info
    st.markdown("<p style='color:#A1A1AA; font-size:0.8rem; font-weight:600;'>LEGAL COVERAGE</p>", unsafe_allow_html=True)
    st.markdown("- Pakistan Penal Code (PPC) 1860\n- Code of Criminal Procedure (CrPC) 1898")

    st.write("")
    if st.button("🧹 Clear Chat History", key="clear_btn", use_container_width=True):
        db.delete_thread(current()["id"])
        
        st.session_state.threads.pop(st.session_state.active_idx)
        if not st.session_state.threads:
            st.session_state.threads = [new_thread()]
            
        st.session_state.active_idx = 0
        st.session_state.threads[0]["messages"] = db.get_chat_history(st.session_state.threads[0]["id"])
        st.session_state.pending_q = None
        st.rerun()

# ════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════
if not current()["messages"]:
    # Beautiful empty state
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #3B82F6; font-size: 2.5rem;'>Hello, I'm PakLaw AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #E2E8F0; font-weight: 400; margin-bottom: 3rem;'>How can I assist you with Pakistan's Legal Codes today?</h3>", unsafe_allow_html=True)
    
    SUGGESTIONS = [
        "What is Section 302 of PPC?",
        "Punishment for theft under PPC?",
        "Define Qatl-i-Amd",
        "What are Diyat provisions?",
        "Section 497 CrPC explained",
        "Bail rules under CrPC"
    ]
    
    cols = st.columns(3)
    for i, s in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_q = s
                st.rerun()

# ── Chat history ──────────────────────────────────────────
for msg in current()["messages"]:
    avatar_icon = "👤" if msg["role"] == "user" else "⚖️"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        if msg["role"] == "assistant" and "|||REFS|||" in msg["content"]:
            answer_part, refs_part = msg["content"].split("|||REFS|||", 1)
            st.markdown(answer_part.strip())
            
            ref_label = "📎 حوالہ جات دیکھیں" if st.session_state.get("language") == "اردو (Urdu)" else "📎 View References"
            with st.expander(ref_label):
                st.markdown(refs_part.strip())
        else:
            st.markdown(msg["content"])

# ── Streaming response ────────────────────────────────────
def render_streaming_response(question: str) -> str:
    lang_hint = "\n\nPlease respond in Urdu." if st.session_state.get("language") == "اردو (Urdu)" else ""
    
    placeholder = st.empty()
    full_response = ""
    
    # Modern thinking indicator
    placeholder.markdown("<span style='color:#3B82F6'>✦</span> <span style='color:#FFFFFF'>*Scanning Legal Database...*</span>", unsafe_allow_html=True)

    try:
        for token in stream_question((question + lang_hint).strip()):
            if token is None: continue
            full_response += token
            placeholder.markdown(full_response + " █")

        full_response = (
            full_response
            .replace("(Not specified)", "")
            .replace("Not available", "")
            .replace("Not specified", "")
        )

        if "\n\n**References:**" in full_response:
            answer_text, refs = full_response.split("\n\n**References:**", 1)
            answer_text = answer_text.strip()
            refs = refs.strip()
            placeholder.markdown(answer_text)
            
            ref_label = "📎 حوالہ جات دیکھیں" if st.session_state.get("language") == "اردو (Urdu)" else "📎 View References"
            with st.expander(ref_label):
                st.markdown(refs)
            return answer_text + "|||REFS|||" + refs
        else:
            placeholder.markdown(full_response.strip())
            return full_response.strip()

    except Exception as e:
        err = f"⚠️ Could not retrieve answer: {str(e)}"
        placeholder.error(err)
        return err

# ── Handle question ───────────────────────────────────────
def handle_question(question: str):
    curr = current()
    
    if curr["title"] == "New Chat":
        title = question[:28].strip()
        if len(question) > 28: title += "…"
        curr["title"] = title
        db.save_thread(curr["id"], title)

    curr["messages"].append({"role": "user", "content": question})
    db.save_message(curr["id"], "user", question)
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="⚖️"):
        answer = render_streaming_response(question)

    curr["messages"].append({"role": "assistant", "content": answer})
    db.save_message(curr["id"], "assistant", answer)

# ── Pending suggestion ────────────────────────────────────
if st.session_state.pending_q:
    question = st.session_state.pending_q
    st.session_state.pending_q = None
    handle_question(question)
    st.rerun()

# ── Chat input (Bottom Sticky) ────────────────────────────
if question := st.chat_input("Ask about any section or legal issue..."):
    handle_question(question)