"""Light/dark theme helper for the Streamlit UI (sun/moon switch).

The app runs on Streamlit's light base theme (.streamlit/config.toml); the dark mode is a static CSS
override injected on top of it. Only our own constant CSS is ever injected, never user or Gemma text.
"""

from __future__ import annotations

import streamlit as st

DARK_BG = "#0E1117"
DARK_TEXT = "#FAFAFA"
DARK_PANEL = "#262730"
DARK_MUTED = "#A3A8B8"
DARK_BORDER = "#41424C"

DARK_CSS = f"""
.stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"], [data-testid="stBottom"] {{ background-color: {DARK_BG}; color: {DARK_TEXT}; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background-color: {DARK_PANEL}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp li, .stApp label,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"] *,
.stApp [data-testid="stHeader"] * {{ color: {DARK_TEXT}; }}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stMetricDelta"] {{ color: {DARK_MUTED}; }}
.stApp hr {{ border-color: {DARK_BORDER}; }}
.stApp code {{ background-color: {DARK_PANEL}; color: #FF9F6B; }}
.stApp [data-testid="stCode"] pre {{ background-color: {DARK_PANEL}; }}
.stApp [data-testid="stTable"] table, .stApp [data-testid="stTable"] th,
.stApp [data-testid="stTable"] td {{ background-color: {DARK_BG}; color: {DARK_TEXT}; border-color: {DARK_BORDER}; }}
.stApp [data-testid="stDataFrame"] {{ filter: invert(0.92) hue-rotate(180deg); }}
.stApp [data-testid="stExpander"] details {{ background-color: {DARK_PANEL}; border-color: {DARK_BORDER}; }}
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary * {{ color: {DARK_TEXT}; }}
.stApp [data-baseweb="tab-list"] {{ background-color: transparent; }}
.stApp [data-baseweb="tab"], .stApp [data-baseweb="tab"] * {{ color: {DARK_MUTED}; background-color: transparent; }}
.stApp [data-baseweb="tab"][aria-selected="true"], .stApp [data-baseweb="tab"][aria-selected="true"] * {{ color: {DARK_TEXT}; }}
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="textarea"],
.stApp [data-baseweb="select"] > div, .stApp input, .stApp textarea {{ background-color: {DARK_PANEL}; color: {DARK_TEXT}; }}
.stApp [data-baseweb="select"] *, .stApp [data-baseweb="input"] * {{ color: {DARK_TEXT}; }}
[data-baseweb="popover"], [data-baseweb="popover"] ul, [data-baseweb="menu"],
[data-baseweb="popover"] li {{ background-color: {DARK_PANEL}; color: {DARK_TEXT}; }}
.stApp button[kind], .stApp [data-testid^="stBaseButton"] {{ background-color: {DARK_PANEL}; color: {DARK_TEXT}; border-color: {DARK_BORDER}; }}
.stApp [data-testid="stBaseButton-primary"], .stApp button[kind="primary"] {{ background-color: #2E6FBA; color: #FFFFFF; border-color: #2E6FBA; }}
.stApp [data-testid^="stPageLink"] * {{ color: {DARK_TEXT}; }}
.stApp [data-testid="stAlert"] [data-baseweb="notification"] {{ background-color: {DARK_PANEL}; }}
.stApp [data-testid="stAlert"] * {{ color: {DARK_TEXT}; }}
"""


def theme_toggle() -> bool:
    """Compact sun/moon switch at the top of the sidebar; returns True when dark mode is on."""
    dark = bool(st.session_state.get("dark_mode", False))
    return bool(st.sidebar.toggle("☀️ Light mode" if dark else "🌙 Dark mode", key="dark_mode", value=False))


def apply_theme(dark: bool) -> None:
    """Inject the dark CSS on top of the light base theme (nothing when light)."""
    if dark:
        st.markdown(f"<style>{DARK_CSS}</style>", unsafe_allow_html=True)


def plotly_template(dark: bool) -> str:
    return "plotly_dark" if dark else "plotly_white"
