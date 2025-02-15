import streamlit as st
import requests
from streamlit_option_menu import option_menu
from io import BytesIO

# Configuration
BACKEND_URL = "http://127.0.0.1:8001"

# Dark theme configuration
THEME = {
    "bg_color": "#1e1e2d",
    "text_color": "#e2e8f0",
    "primary_color": "#3b82f6",
    "secondary_color": "#2d3748",
    "accent_color": "#60a5fa",
    "card_bg": "#2d3748",
    "sidebar_bg": "#1a1a27"
}

# LLM Options
LLM_OPTIONS = {
   "llama-3.3-70b-versatile": "LLAMA 3.3 70B",
   "llama-3.1-8b-instant": "LLAMA 3.1 8B",
    "deepseek-r1-distill-qwen-32b": "DEEPSEEK R1 DISTILL QWEN 32B",
    "mixtral-8x7b-32768": "MIXTRAL 8X7B 32768",
    "gemma2-9b-it": "GEMMA2 9B"

}
LLM_OPTIONS_REVERSE = {v: k for k, v in LLM_OPTIONS.items()}

# Database Options
DB_OPTIONS = {
    "Qdrant": "qdrant",  # You can add more DBs here later
}

# Embedding Model Options
EMBEDDING_MODEL_OPTIONS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    # Add more options here
}

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Chat"

# Database Credentials
if 'llm_selected' not in st.session_state:
    st.session_state.llm_selected = ""
if 'db_selected' not in st.session_state:
    st.session_state.db_selected = ""

# Database Credentials
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = ''
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ''
if 'qdrant_cloud_url' not in st.session_state:
    st.session_state.qdrant_cloud_url = ''
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "all-MiniLM-L6-v2"  # Default

if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None

# Page configuration
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for dark theme
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {THEME["bg_color"]};
            color: {THEME["text_color"]};
        }}
        .main .block-container {{
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: {THEME["text_color"]};
            font-family: 'Segoe UI', sans-serif;
        }}
        .stCard {{
            background: {THEME["card_bg"]};
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            border: 1px solid {THEME["secondary_color"]};
        }}
        .chat-message {{
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 10px;
            max-width: 80%;
        }}
        .user-message {{
            background: {THEME["primary_color"]};
            color: white;
            margin-left: auto;
        }}
        .bot-message {{
            background: {THEME["card_bg"]};
            border: 1px solid {THEME["secondary_color"]};
            color: {THEME["text_color"]};
            margin-right: auto;
        }}
        .stTextInput > div > div {{
            background: {THEME["card_bg"]};
            border-color: {THEME["secondary_color"]};
            color: {THEME["text_color"]};
            padding: 0.5rem;
            border-radius: 8px;
        }}
        .stButton > button {{
            width: 100%;
            padding: 0.75rem 1.5rem;
            color: white;
            background: linear-gradient(45deg, {THEME["primary_color"]}, {THEME["accent_color"]});
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .uploadedFile {{
            padding: 1rem;
            background: {THEME["card_bg"]};
            border-radius: 10px;
            border: 2px dashed {THEME["secondary_color"]};
            text-align: center;
        }}
        .css-1d391kg {{
            background: {THEME["sidebar_bg"]};
        }}
        .sidebar .sidebar-content {{
            background: {THEME["sidebar_bg"]};
            padding: 1rem;
        }}
        .footer {{
            padding: 1rem;
            text-align: center;
            color: {THEME['text_color']};
            }}
    </style>
    """, unsafe_allow_html=True)

# Sidebar with navigation
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Chat", "Settings", "About"],
        icons=["chat-dots-fill", "gear-fill", "info-circle-fill"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "1rem", "background-color": THEME["card_bg"]},
            "icon": {"color": THEME["primary_color"]},
            "nav-link": {
                "color": THEME["text_color"],
                "background": THEME["secondary_color"],
                "border-radius": "0.5rem",
                "margin": "0.5rem 0"
            },
            "nav-link-selected": {
                "background": THEME["primary_color"],
                "color": "white",
            }
        }
    )

    st.session_state.current_page = selected

# Main content area
if st.session_state.current_page == "Chat":
    st.title("📄 AI PDF Chatbot")
    st.write("Upload a PDF and ask questions about its content.")

    # Check if settings are configured
    settings_configured = (
        st.session_state.llm_selected and
        st.session_state.db_selected and
        st.session_state.qdrant_api_key and
        st.session_state.groq_api_key and
        st.session_state.qdrant_cloud_url
    )

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", key="file_uploader")

    # Process PDF upload
    if uploaded_file is not None:
        if not settings_configured:
            st.warning("⚠️ Please configure LLM and Database settings before uploading a PDF.")
        else:
            st.session_state.pdf_data = uploaded_file.getvalue()

            st.success("✅ File uploaded successfully!")
            files = {"file": (uploaded_file.name, BytesIO(st.session_state.pdf_data), "application/pdf")}

            with st.spinner("🔄 Processing PDF..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/upload-pdf", files=files, timeout=10)
                    response.raise_for_status()
                    if response.status_code == 200:
                        st.success("📄 PDF processed successfully!")
                    else:
                        st.error(f"❌ Failed to process PDF. Status Code: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"⚠️ Error: {e}")

    # Query input and display
    st.subheader("💬 Ask a Question")
    query = st.text_input("Type your question here...")

    if st.button("🔍 Get Answer"):
        if not settings_configured:
            st.warning("⚠️ Please configure LLM and Database settings before asking a question.")
        elif not query:
            st.warning("⚠️ Please enter a question.")
        elif st.session_state.pdf_data is None:
            st.warning("Please upload a PDF file first.")
        else:
            with st.spinner(" Thinking..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/query", json={"query": query}, timeout=10)
                    response.raise_for_status()

                    if response.status_code == 200:
                        answer = response.json().get("answer", "No answer found.")
                        st.markdown(
                            f'<div class="chat-message user-message">You: {query}</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="chat-message bot-message">AI: {answer}</div>', unsafe_allow_html=True)

                except requests.exceptions.RequestException as e:
                    st.error(f"⚠️ Error: {e}")

elif st.session_state.current_page == "Settings":
    st.title("🔧 Settings")

    tab1, tab2 = st.tabs(["⚙️ LLM Settings", "🗄️ Database Settings"])

    with tab1:
        with st.form("llm_settings"):
            st.markdown("### Groq API Configuration")
            groq_api_key = st.text_input("Groq API Key", type="password", value=st.session_state.groq_api_key or "")
            llm_option_keys = list(LLM_OPTIONS.keys())
            selected_llm = st.selectbox(
                 "Select LLM",
                options=list(LLM_OPTIONS.values()),
                index=llm_option_keys.index(st.session_state.llm_selected) if st.session_state.llm_selected in llm_option_keys else 0,
                key="llm_selection"
            )

            submitted = st.form_submit_button("Save LLM Settings")
            if submitted:
                if not groq_api_key:
                    st.error("⚠️ Groq API Key cannot be empty.")
                else:
                    st.session_state.groq_api_key = groq_api_key
                    selected_llm_key = next((key for key, value in LLM_OPTIONS.items() if value == selected_llm), None)
                    st.session_state.llm_selected = selected_llm_key

                    st.success("LLM Settings saved successfully!")

    with tab2:
        with st.form("vector_settings"):
            st.markdown("### Vector Database Configuration")
            qdrant_cloud_url = st.text_input("Qdrant Cloud URL", value=st.session_state.qdrant_cloud_url or "")
            qdrant_api_key = st.text_input("Qdrant API Key", type="password", value=st.session_state.qdrant_api_key or "")

            st.markdown("### Embedding Settings")
            embedding_model = st.selectbox(
                "Embedding Model",
                options=list(EMBEDDING_MODEL_OPTIONS.keys()),
                index=list(EMBEDDING_MODEL_OPTIONS.keys()).index(st.session_state.embedding_model) if st.session_state.embedding_model in EMBEDDING_MODEL_OPTIONS else 0,
                 key="embedding_model_selection"
            )

            submitted = st.form_submit_button("Save Database Settings")
            if submitted:
                if not qdrant_api_key or not qdrant_cloud_url:
                    st.error("⚠️ Qdrant API Key and Cloud URL cannot be empty.")
                else:
                    st.session_state.qdrant_api_key = qdrant_api_key
                    st.session_state.qdrant_cloud_url = qdrant_cloud_url
                    st.session_state.embedding_model = embedding_model
                    st.session_state.db_selected ="Qdrant"
                    settings_data = {
                        "qdrant_api_key": st.session_state.qdrant_api_key,
                        "groq_api_key": st.session_state.groq_api_key,
                        "qdrant_cloud_url": st.session_state.qdrant_cloud_url,
                        "embedding_model": st.session_state.embedding_model
                    }
                    try:
                        response = requests.post(f"{BACKEND_URL}/set-settings", json=settings_data, timeout=10)
                        response.raise_for_status()
                    except requests.exceptions.RequestException as e:
                       st.error(f"⚠️ Error sending settings to backend: {e}")
                    st.success("Database Settings saved successfully!")

else:  # About page
    st.title("ℹ️ About")
    st.markdown("""
    ### AI PDF Chatbot

    This application allows you to have interactive conversations with your PDF documents
    using state-of-the-art AI models.

    #### Features
    - 📄 PDF document processing
    - 💬 Natural language interaction
    - 🤖 Multiple AI model support
    - ⚙️ Customizable settings

    #### Technologies Used
    - FastAPI
    - Streamlit
    - LangChain
    - Vector Databases
    """)

# Footer
st.markdown(f"""
    <div class="footer">
        <p style="margin: 0; text-align: center;">
            Made with ❤️ by Bhaven Chheda
        </p>
    </div>
""", unsafe_allow_html=True)
