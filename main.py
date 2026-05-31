import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, pull as cls_pull, save_model as cls_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save

st.set_page_config(page_title="AutoML App", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

dark = st.session_state["dark_mode"]

if dark:
    BG      = "#0d0d1a"
    SIDEBAR = "#13131f"
    CARD    = "#1c1c2e"
    TEXT    = "#e2e2f0"
    SUBTEXT = "#9090b0"
    BORDER  = "#2a2a3f"
else:
    BG      = "#ffffff"
    SIDEBAR = "#f5f7fb"
    CARD    = "#eef1f8"
    TEXT    = "#111827"
    SUBTEXT = "#6b7280"
    BORDER  = "#e0e4ed"

ACCENT = "#4e8cff"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG} !important; }}

    section[data-testid="stSidebar"] {{
        background-color: {SIDEBAR} !important;
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {{
        color: {TEXT} !important;
    }}

    h1, h2, h3 {{ color: {TEXT} !important; font-weight: 600; }}
    p, label, .stMarkdown {{ color: {TEXT} !important; }}

    [data-testid="stMetric"] {{
        background-color: {CARD};
        border-radius: 10px;
        padding: 1.2rem 1rem;
        border-left: 4px solid {ACCENT};
    }}
    [data-testid="stMetric"] label {{
        color: {SUBTEXT} !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
        font-size: 1.8rem;
        font-weight: 700;
    }}

    .stButton > button {{
        background-color: {ACCENT};
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: background 0.2s;
    }}
    .stButton > button:hover {{
        background-color: #3a7bd5 !important;
        color: white !important;
    }}

    div[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}
    hr {{ border-color: {BORDER} !important; margin: 0.8rem 0; }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<h2 style='color:{TEXT};margin-bottom:0'>AutoML</h2>", unsafe_allow_html=True)
    st.markdown("---")

    toggle_val = st.toggle("Dark mode", value=st.session_state["dark_mode"])
    if toggle_val != st.session_state["dark_mode"]:
        st.session_state["dark_mode"] = toggle_val
        st.rerun()

    st.markdown("---")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Upload  →  Profile  →  Train  →  Download")

if os.path.exists("sourcedata.csv"):
    st.session_state["df"] = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Dataset")
    st.markdown(f"<p style='color:{SUBTEXT}'>Upload a CSV file to get started.</p>", unsafe_allow_html=True)
    file = st.file_uploader("", type=["csv"])
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.session_state["df"] = df
        st.success("Dataset uploaded successfully.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
        st.markdown("### Preview")
        st.dataframe(df.head(50), use_container_width=True)

if choice == "Profiling":
    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
    else:
        st.title("Exploratory Data Analysis")
        with st.spinner("Generating report..."):
            profile_report = ProfileReport(st.session_state["df"])
            st_profile_report(profile_report)

if choice == "Modelling":
    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state["df"]
        st.title("Model Training")
        col1, col2 = st.columns(2)
        with col1:
            chosen_target = st.selectbox("Target Column", df.columns)
        with col2:
            task = st.radio("Task Type", ["Classification", "Regression"])
        if st.button("Run Modelling"):
            with st.spinner("Training models... this may take a few minutes."):
                if task == "Classification":
                    cls_setup(df, target=chosen_target)
                    st.markdown("### Experiment Settings")
                    st.dataframe(cls_pull(), use_container_width=True)
                    best_model = cls_compare()
                    st.markdown("### Model Comparison")
                    st.dataframe(cls_pull(), use_container_width=True)
                    cls_save(best_model, "best_model")
                else:
                    reg_setup(df, target=chosen_target)
                    st.markdown("### Experiment Settings")
                    st.dataframe(reg_pull(), use_container_width=True)
                    best_model = reg_compare()
                    st.markdown("### Model Comparison")
                    st.dataframe(reg_pull(), use_container_width=True)
                    reg_save(best_model, "best_model")
            st.success("Training complete. Go to Download to get your model.")

if choice == "Download":
    st.title("Download Your Model")
    if os.path.exists("best_model.pkl"):
        st.success("Your trained model is ready.")
        with open("best_model.pkl", "rb") as f:
            st.download_button("Download Model", f, file_name="trained_model.pkl")
    else:
        st.warning("No trained model found. Please run Modelling first.")