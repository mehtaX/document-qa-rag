import os
from pathlib import Path
import streamlit as st

# Set page configuration first
st.set_page_config(
    page_title="Document Q&A RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium styling matching the brand
st.markdown("""
    <style>
    /* Styling for app title & header */
    .title-gradient {
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Styling for sidebar and containers */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Grounding sources code display formatting */
    .source-block {
        background-color: #020617;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
    }
    
    .source-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #60a5fa;
        margin-bottom: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# Custom header
st.markdown('<div class="title-gradient">Document Q&A RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent Document Q&A with hybrid retrieval, Reciprocal Rank Fusion, and strict grounding.</div>', unsafe_allow_html=True)

# We check if there are environment variables or secrets for Groq
groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = groq_key

# Import RAG pipeline lazily so we can update the API key if needed before import/instantiation
from rag_pipeline import get_rag_instance

# Sidebar title and API configuration
st.sidebar.title("Configuration")

llm_provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
st.sidebar.markdown(f"**LLM Provider:** `{llm_provider.capitalize()}`")

# If using Groq and key is missing, prompt user
if llm_provider == "groq":
    api_key_from_env = os.environ.get("GROQ_API_KEY", "")
    is_placeholder = not api_key_from_env or api_key_from_env == "PLACEHOLDER"
    
    if is_placeholder:
        user_key = st.sidebar.text_input("Enter Groq API Key", type="password", key="user_groq_key_input")
        if user_key:
            os.environ["GROQ_API_KEY"] = user_key
            # Initialize or reinitialize LLM
            rag = get_rag_instance()
            rag.set_api_key(user_key)
            st.sidebar.success("Groq API Key set!")
        else:
            st.sidebar.warning("⚠️ GROQ_API_KEY is required to process queries with Groq.")
    else:
        st.sidebar.success("Groq API Key loaded successfully.")

# Get RAG Instance and ensure uploads are processed
rag = get_rag_instance()

if "rag_loaded" not in st.session_state:
    with st.spinner("Loading uploaded documents & building index..."):
        try:
            rag.load_existing_uploads()
            st.session_state.rag_loaded = True
        except Exception as e:
            st.error(f"Error loading existing documents: {e}")

# Sidebar Document Upload
st.sidebar.subheader("Upload PDF Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
        
    file_name = uploaded_file.name
    if file_name not in st.session_state.processed_files:
        with st.sidebar.status(f"Ingesting {file_name}...") as status:
            # Ensure upload folder exists
            dest = rag.uploads_path / file_name
            with open(dest, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Run ingestion
            try:
                result = rag.ingest_pdf(dest)
                st.session_state.processed_files.add(file_name)
                status.update(label=f"Indexed {file_name} successfully!", state="complete", expanded=False)
                st.sidebar.success(f"Added {result['chunks']} chunks. Total chunks: {result['total_chunks']}.")
                st.rerun()
            except Exception as e:
                status.update(label=f"Ingestion failed: {e}", state="error", expanded=True)

# Sidebar list of documents
st.sidebar.subheader("Indexed Documents")
uploaded_docs = rag.list_documents()
if not uploaded_docs:
    st.sidebar.info("No documents uploaded yet.")
else:
    for idx, doc in enumerate(uploaded_docs):
        st.sidebar.markdown(f"📄 **{doc['filename']}**  \n*{doc['pages']} pages, {doc['chunks']} chunks*")

# Sidebar system logs
if rag.step_logs:
    with st.sidebar.expander("System Logs", expanded=False):
        for log in rag.step_logs:
            st.caption(log)

# Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("🔍 Grounding Sources & Citations"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-block">
                        <div class="source-header">{src['citation']} &nbsp;|&nbsp; File: {src['source_file']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.code(src["content"], language="text")

# Response Generator for streaming
def response_generator(question):
    stream = rag.ask_stream(question)
    for event in stream:
        if event["type"] == "sources":
            st.session_state.last_sources = event["sources"]
            st.session_state.last_logs = event["step_logs"]
        elif event["type"] == "token":
            yield event["token"]
        elif event["type"] == "done":
            st.session_state.last_sources = event["sources"]
            st.session_state.last_logs = event["step_logs"]
            st.session_state.last_answer = event["answer"]
        elif event["type"] == "error":
            st.error(f"Error generating response: {event['message']}")
            yield f"\n\nError: {event['message']}"

# Chat input
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    # Check if index is ready
    if not rag.is_ready():
        st.warning("Please upload and index a PDF document first!")
    elif llm_provider == "groq" and (not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "PLACEHOLDER"):
        st.error("Please configure your Groq API Key in the sidebar to ask questions.")
    else:
        # Display user question
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Reset intermediate states
        st.session_state.last_sources = []
        st.session_state.last_logs = []
        st.session_state.last_answer = ""
        
        # Render assistant response with streaming
        with st.chat_message("assistant"):
            full_response = st.write_stream(response_generator(prompt))
            sources = st.session_state.get("last_sources", [])
            logs = st.session_state.get("last_logs", [])
            
            if sources:
                with st.expander("🔍 Grounding Sources & Citations"):
                    for src in sources:
                        st.markdown(f"""
                        <div class="source-block">
                            <div class="source-header">{src['citation']} &nbsp;|&nbsp; File: {src['source_file']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.code(src["content"], language="text")
            
            # Store in history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "logs": logs
            })
            st.rerun()
