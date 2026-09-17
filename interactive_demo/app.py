"""
Multimodal RGB-IR Object Detection: Live Deployment & Research Demonstrator.
Main Streamlit Application Entrypoint.
Run via: streamlit run app.py
"""

import sys
from pathlib import Path

# Add app directory and project root to sys.path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

for p in [APP_DIR, PROJECT_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st
from ui.styles import LIGHT_THEME_CSS
from ui.components import render_header
from ui.dashboard import render_dashboard
from ui.comparison import render_comparison_page
from ui.recommendations import render_recommendations_page
from ui.how_it_works import render_how_it_works_page

# Page Configuration
st.set_page_config(
    page_title="Multimodal RGB-IR Detection — FYP Demonstrator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Accessible High-Contrast Light CSS Theme
st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# Main Banner
render_header()

# Main Navigation Tabs
tab_live, tab_compare, tab_recom, tab_evolution = st.tabs([
    "🚗 Live Perception Demonstrator",
    "📊 Research Comparison (Dual Protocols)",
    "🎯 Deployment Recommendations",
    "🧬 Architectural Evolution (Stages 1-6)"
])

with tab_live:
    render_dashboard()

with tab_compare:
    render_comparison_page()

with tab_recom:
    render_recommendations_page()

with tab_evolution:
    render_how_it_works_page()
