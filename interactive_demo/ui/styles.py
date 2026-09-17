"""
Custom CSS styling for high-contrast, accessible white-background dashboard aesthetic.
Designed according to WCAG 2.1 AAA contrast guidelines and modern engineering UI standards.
"""

LIGHT_THEME_CSS = """
<style>
    /* =========================================================================
       GLOBAL APP & TYPOGRAPHY STYLING (WCAG AAA COMPLIANCE)
       ========================================================================= */
    .stApp {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Headings - Bold, high-contrast, crisp visual hierarchy */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    h1 { font-size: 2.0rem !important; }
    h2 { font-size: 1.6rem !important; }
    h3 { font-size: 1.3rem !important; }
    h4 { font-size: 1.1rem !important; }

    /* Paragraphs and list items */
    p, li, span, div {
        color: #1e293b;
        line-height: 1.6;
    }
    b, strong {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* =========================================================================
       SIDEBAR NAVIGATION & CONTROLS
       ========================================================================= */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] label {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* =========================================================================
       CUSTOM HEADER BANNER
       ========================================================================= */
    .fyp-header {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #1e40af;
        padding: 20px 24px;
        margin-bottom: 24px;
        border-radius: 8px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
    }
    .fyp-title {
        font-size: 24px;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        color: #0f172a !important;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .fyp-subtitle {
        font-size: 13px;
        font-weight: 700 !important;
        color: #475569 !important;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* =========================================================================
       METRIC CARDS (HIGH CONTRAST WHITE TILES)
       ========================================================================= */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800 !important;
        color: #1e40af !important;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 12px;
        font-weight: 700 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* =========================================================================
       STATUS PILLS & BADGES
       ========================================================================= */
    .sensor-pill-online {
        background-color: #f0fdf4;
        color: #15803d !important;
        border: 1px solid #86efac;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700 !important;
        display: inline-block;
    }
    .sensor-pill-offline {
        background-color: #fef2f2;
        color: #b91c1c !important;
        border: 1px solid #fca5a5;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700 !important;
        display: inline-block;
    }
    .arch-badge {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #0f172a !important;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700 !important;
        font-family: monospace;
    }

    /* =========================================================================
       PIPELINE ARCHITECTURE DIAGRAM BOX
       ========================================================================= */
    .pipeline-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #1e40af;
        padding: 16px 20px;
        border-radius: 6px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
        font-size: 12.5px;
        color: #0f172a !important;
        margin: 14px 0;
        line-height: 1.5;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
        overflow-x: auto;
    }

    /* =========================================================================
       NAVIGATION TABS
       ========================================================================= */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        color: #475569 !important;
        padding: 10px 18px;
        font-weight: 700 !important;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1e40af !important;
        border: 1px solid #e2e8f0 !important;
        border-top: 3px solid #1e40af !important;
        border-bottom: 2px solid #ffffff !important;
        font-weight: 800 !important;
    }

    /* =========================================================================
       NATIVE STREAMLIT METRICS & DATA DISPLAYS
       ========================================================================= */
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 700 !important;
    }

    /* Callout & Alert containers */
    .stAlert {
        border-radius: 8px !important;
        font-weight: 500 !important;
        border-width: 1px !important;
    }

    /* Dataframes / Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }
</style>
"""

# Backwards compatibility alias
DARK_THEME_CSS = LIGHT_THEME_CSS
