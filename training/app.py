import logging
import os

import streamlit as st

from modules.auth import require_auth, logout, check_auth
from modules.i18n import get_translation
from config import APP_ICON, BASE_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Initialize session state
if "language" not in st.session_state:
    st.session_state.language = "es"
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Set page config with theme
st.set_page_config(
    page_title=get_translation(st.session_state.language, "app_title"),
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme(theme: str):
    hide_native_nav = """
        <style>
            [data-testid="stSidebarNav"] { display: none; }
        </style>
    """
    st.markdown(hide_native_nav, unsafe_allow_html=True)

    if theme == "light":
        st.markdown(
            """
            <style>
                .stApp {
                    background-color: #FFFFFF;
                }
                [data-testid="stSidebar"] {
                    background-color: #F3F4F6;
                }
                .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, div, label {
                    color: #000000 !important;
                }
                .stButton > button,
                [data-testid="stSidebar"] .stButton > button,
                [data-testid="baseButton-secondary"],
                [data-testid="baseButton-primary"] {
                    background-color: #FFFFFF !important;
                    color: #000000 !important;
                    border: 1px solid #D1D5DB !important;
                }
                .stButton > button:hover,
                [data-testid="stSidebar"] .stButton > button:hover {
                    background-color: #E5E7EB !important;
                    color: #000000 !important;
                    border-color: #9CA3AF !important;
                }
                .stButton > button p {
                    color: inherit !important;
                }
                .stTextInput input,
                .stNumberInput input,
                .stTextArea textarea,
                [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
                [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
                    background-color: #FFFFFF !important;
                    color: #000000 !important;
                    border: 1px solid #D1D5DB !important;
                }
                [data-testid="stDataFrame"] table {
                    background-color: #FFFFFF !important;
                    color: #000000 !important;
                }
                [data-testid="stDataFrame"] th {
                    background-color: #E5E7EB !important;
                    color: #000000 !important;
                }
                [data-testid="stDataFrame"] td {
                    background-color: #FFFFFF !important;
                    color: #000000 !important;
                }
                [data-testid="stTable"] table,
                [data-testid="stTable"] th,
                [data-testid="stTable"] td {
                    background-color: #FFFFFF !important;
                    color: #000000 !important;
                }
                [data-testid="stTable"] th {
                    background-color: #E5E7EB !important;
                }
                [data-testid="stHeader"] {
                    background-color: #FFFFFF !important;
                }
                [data-testid="stToolbar"] {
                    background-color: #FFFFFF !important;
                }
                [data-testid="stHeader"] svg,
                [data-testid="stToolbar"] svg {
                    fill: #000000 !important;
                }
                [data-testid="stElementToolbar"],
                [data-testid="stElementToolbar"] * {
                    color: #FFFFFF !important;
                    fill: #FFFFFF !important;
                }
                [data-testid="stElementToolbar"] button {
                    background-color: transparent !important;
                }
                [data-testid="stElementToolbar"] button:hover {
                    background-color: #3F3F5C !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
                .stApp {
                    background-color: #0F0F1A;
                }
                [data-testid="stSidebar"] {
                    background-color: #1A1A2E;
                }
                .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, div, label {
                    color: #F1F1F6 !important;
                }
                .stButton > button,
                [data-testid="stSidebar"] .stButton > button,
                [data-testid="baseButton-secondary"],
                [data-testid="baseButton-primary"] {
                    background-color: #2D2D44 !important;
                    color: #F1F1F6 !important;
                    border: 1px solid #3F3F5C !important;
                }
                .stButton > button:hover,
                [data-testid="stSidebar"] .stButton > button:hover {
                    background-color: #3F3F5C !important;
                    color: #F1F1F6 !important;
                    border-color: #55557A !important;
                }
                .stButton > button p {
                    color: inherit !important;
                }
                .stTextInput input,
                .stNumberInput input,
                .stTextArea textarea,
                [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
                [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
                    background-color: #1A1A2E !important;
                    color: #F1F1F6 !important;
                    border: 1px solid #3F3F5C !important;
                }
                [data-testid="stDataFrame"] table {
                    background-color: #1A1A2E !important;
                    color: #F1F1F6 !important;
                }
                [data-testid="stDataFrame"] th {
                    background-color: #2D2D44 !important;
                    color: #F1F1F6 !important;
                }
                [data-testid="stDataFrame"] td {
                    background-color: #1A1A2E !important;
                    color: #F1F1F6 !important;
                }
                [data-testid="stTable"] table,
                [data-testid="stTable"] th,
                [data-testid="stTable"] td {
                    background-color: #1A1A2E !important;
                    color: #F1F1F6 !important;
                }
                [data-testid="stTable"] th {
                    background-color: #2D2D44 !important;
                }
                [data-testid="stHeader"] {
                    background-color: #0F0F1A !important;
                }
                [data-testid="stToolbar"] {
                    background-color: #0F0F1A !important;
                }
                [data-testid="stHeader"] svg,
                [data-testid="stToolbar"] svg {
                    fill: #F1F1F6 !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )


apply_theme(st.session_state.theme)

require_auth()

# ── Descargar modelos desde HuggingFace si está configurado ──
if "models_downloaded" not in st.session_state:
    hf_repo = os.getenv("HF_REPO_ID", "")
    if hf_repo:
        from download_models import download_models
        logger.info(f"HF_REPO_ID={hf_repo}. Descargando modelos...")
        with st.spinner("Descargando modelos desde HuggingFace Hub..."):
            ok = download_models(MODELS_DIR)
            if ok:
                st.sidebar.success("Modelos descargados correctamente.")
            else:
                st.sidebar.warning("Algunos modelos no pudieron descargarse. Verifica HF_TOKEN si el repo es privado.")
    else:
        logger.info("HF_REPO_ID no configurado. Usando modelos locales si existen.")
    st.session_state["models_downloaded"] = True

# Sidebar header
st.sidebar.markdown(
    f"# {APP_ICON} {get_translation(st.session_state.language, 'app_title')}",
)
st.sidebar.markdown("---")

# Theme and language toggles
st.sidebar.subheader(get_translation(st.session_state.language, "theme"))
if st.session_state.theme == "dark":
    toggle_label = f"☀️ {get_translation(st.session_state.language, 'light')}"
else:
    toggle_label = f"🌙 {get_translation(st.session_state.language, 'dark')}"
if st.sidebar.button(toggle_label, use_container_width=True):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

st.sidebar.subheader(get_translation(st.session_state.language, "language"))
lang_label = "English" if st.session_state.language == "es" else "Español"
if st.sidebar.button(f"🌐 {lang_label}", use_container_width=True):
    st.session_state.language = "en" if st.session_state.language == "es" else "es"
    st.rerun()

st.sidebar.markdown("---")

PAGES = {
    get_translation(st.session_state.language, "pages.dashboard"): "views/01_Dashboard.py",
    get_translation(st.session_state.language, "pages.eda"): "views/02_EDA.py",
    get_translation(st.session_state.language, "pages.models"): "views/03_Models.py",
    get_translation(st.session_state.language, "pages.results"): "views/04_Results.py",
    get_translation(st.session_state.language, "pages.tuning_results"): "views/05_TuningResults.py",
    get_translation(st.session_state.language, "pages.statistical_tests"): "views/06_StatisticalTests.py",
    get_translation(st.session_state.language, "pages.reports"): "views/07_Reports.py",
}

if check_auth():
    username = st.session_state.get("username", "admin")
    st.sidebar.markdown(f"**{get_translation(st.session_state.language, 'user')}:** {username}")
    st.sidebar.markdown("---")

    selected = None
    for label in PAGES:
        if st.sidebar.button(label, use_container_width=True, key=f"nav_{label}"):
            selected = label

    st.sidebar.markdown("---")
    if st.sidebar.button(get_translation(st.session_state.language, "logout"), use_container_width=True):
        logout()

    page_file = st.session_state.get("current_page", None)

    if selected:
        st.session_state["current_page"] = PAGES[selected]
        st.rerun()

    if page_file and page_file in PAGES.values():
        try:
            exec(open(BASE_DIR / page_file, encoding="utf-8").read())
        except Exception as e:
            st.error(f"Error al cargar la página: {e}")
    else:
        exec(open(BASE_DIR / "views/01_Dashboard.py", encoding="utf-8").read())
