"""TalkToTables: ask questions in plain English, see the SQL, the table, a chart and an answer.

Two datasets: the benchmarked Olist demo, or files the visitor uploads.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import json
import os
import random
import re

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import config, llm
from src.assistant import SQLAssistant
from src.db import Database
from src.retriever import Retriever
from src.workspace import UploadError, build_workspace, suggest_questions

try:  # pretty-print one-line SQL (optional: pip install sqlparse)
    import sqlparse
except ImportError:
    sqlparse = None

try:  # voice input: records in the browser, transcribes with Google's free speech API
    from streamlit_mic_recorder import speech_to_text
except ImportError:
    speech_to_text = None

st.set_page_config(page_title="TalkToTables - ask your data in plain English", page_icon=":bar_chart:", layout="wide")

# (topic tag, question) - the tag shows on the example card
EXAMPLE_QUESTIONS = [
    ("Revenue", "What was total revenue by month in 2018?"),
    ("Delivery", "Which 10 product categories have the most late deliveries?"),
    ("Reviews", "What is the average review score for late vs on-time orders?"),
    ("Logistics", "Which states have the highest average delivery time?"),
    ("Retention", "What share of customers came back for a second order?"),
    ("Payments", "Which payment types are most common for orders over 500 BRL?"),
]

# Row counts verified by python -m src.load_data
TABLES = [
    ("orders", "99,441", "One row per order: status and purchase / delivery timestamps", "order_id, customer_id"),
    ("order_items", "112,650", "Products in each order, with price and freight", "order_id, product_id, seller_id"),
    ("payments", "103,886", "How each order was paid (card, boleto, voucher...)", "order_id"),
    ("reviews", "99,224", "Customer review score (1-5) and comments", "order_id"),
    ("customers", "99,441", "Customer city and state", "customer_id"),
    ("sellers", "3,095", "Seller city and state", "seller_id"),
    ("products", "32,951", "Product category, size and weight", "product_id"),
    ("category_translation", "71", "Portuguese to English category names", "product_category_name"),
    ("geolocation", "1,000,163", "Zip code prefix to latitude / longitude", "zip_code_prefix"),
]

SURPRISE_POOL = [q for _, q in EXAMPLE_QUESTIONS] + [
    "Which 5 sellers made the most revenue?",
    "What is the average freight cost by customer state?",
    "Which product category has the best average review score?",
    "What is the average order value by payment type?",
    "Which day of the week gets the most orders?",
    "How many orders were delivered late in each month of 2017?",
]

FULL_EVAL_SIZE = 40
DIFFICULTIES = ["easy", "medium", "hard"]

# ---------- theme ----------

LIGHT = dict(bg="#f6f8fb", surface="#ffffff", surface2="#eef2f7", border="#e2e8f0", text="#0f172a",
             muted="#64748b", accent="#0891b2", accent2="#6366f1", accent_soft="rgba(8,145,178,0.08)",
             shadow="0 1px 2px rgba(15,23,42,.04), 0 4px 16px rgba(15,23,42,.06)")
DARK = dict(bg="#0b1120", surface="#111a2e", surface2="#0f172a", border="#1f2a44", text="#e5e7eb",
            muted="#94a3b8", accent="#22d3ee", accent2="#818cf8", accent_soft="rgba(34,211,238,0.08)",
            shadow="0 1px 2px rgba(0,0,0,.3), 0 4px 16px rgba(0,0,0,.35)")


def theme_css(t: dict, dark: bool) -> str:
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    :root {{
      --bg:{t['bg']}; --surface:{t['surface']}; --surface2:{t['surface2']}; --border:{t['border']};
      --text:{t['text']}; --muted:{t['muted']}; --accent:{t['accent']}; --accent2:{t['accent2']};
      --accent-soft:{t['accent_soft']}; --shadow:{t['shadow']};
    }}
    .stApp *:not([data-testid="stIconMaterial"]) {{ font-family: 'Inter', sans-serif; }}
    .stApp code, .stApp pre, .stApp pre * {{ font-family: 'JetBrains Mono', monospace !important; }}
    footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 2.2rem; max-width: 1200px; }}

    /* hero: deep night band with a glowing orb (same in light and dark mode) */
    .hero {{ position: relative; overflow: hidden; display: grid; grid-template-columns: 1.5fr 1fr; align-items: center;
      gap: 10px; padding: 34px 36px; border-radius: 22px; margin-bottom: 18px; color: #e2e8f0;
      background: radial-gradient(500px 260px at 82% 50%, rgba(34,211,238,.28), transparent 70%),
                  radial-gradient(600px 300px at 100% 100%, rgba(139,92,246,.35), transparent 70%),
                  linear-gradient(135deg, #070b19 0%, #111338 55%, #1e1b4b 100%);
      box-shadow: 0 20px 50px -20px rgba(49,46,129,.55); }}
    .hero::before {{ content: ""; position: absolute; inset: 0; opacity: .35; pointer-events: none;
      background-image: radial-gradient(rgba(255,255,255,.35) 1px, transparent 1px); background-size: 26px 26px;
      mask-image: linear-gradient(90deg, transparent, #000 60%); -webkit-mask-image: linear-gradient(90deg, transparent, #000 60%); }}
    .hero .eyebrow {{ color: #67e8f9; }}
    .hero .hero-title {{ color: #ffffff; }}
    .hero .hero-sub {{ color: #a5b4cb; }}
    .eyebrow {{ font-size: .72rem; letter-spacing: .14em; font-weight: 600; color: var(--accent); text-transform: uppercase; }}
    .hero-title {{ font-size: 2.4rem; font-weight: 800; line-height: 1.12; margin: 8px 0 10px; color: var(--text); }}
    .grad {{ background: linear-gradient(90deg, var(--accent), var(--accent2)); -webkit-background-clip: text;
      background-clip: text; color: transparent; }}
    .hero .grad {{ background-image: linear-gradient(90deg, #22d3ee, #a78bfa, #f0abfc); }}
    .hero-sub {{ color: var(--muted); font-size: 1rem; max-width: 620px; margin: 0; line-height: 1.6; }}
    .pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
    .pill {{ font-size: .75rem; font-weight: 500; padding: 5px 12px; border-radius: 999px; color: #e2e8f0;
      border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.06); backdrop-filter: blur(6px); }}
    .pill b {{ color: #67e8f9; font-weight: 600; }}

    /* the orb */
    .orb-wrap {{ position: relative; height: 230px; display: flex; align-items: center; justify-content: center; }}
    .orb {{ position: relative; width: 150px; height: 150px; border-radius: 50%;
      background: radial-gradient(circle at 34% 28%, #ffffff 0%, #cffafe 10%, #22d3ee 34%, #6366f1 68%, #7c3aed 100%);
      box-shadow: 0 0 50px 8px rgba(34,211,238,.45), 0 0 120px 40px rgba(124,58,237,.35),
                  inset -14px -22px 40px rgba(59,7,100,.55);
      animation: float 6s ease-in-out infinite; }}
    .ring {{ position: absolute; border-radius: 50%; border: 1.5px solid transparent; }}
    .ring.r1 {{ inset: -22px; border-top-color: rgba(103,232,249,.9); border-right-color: rgba(103,232,249,.25);
      animation: spin 5s linear infinite; }}
    .ring.r2 {{ inset: -40px; border-bottom-color: rgba(192,132,252,.85); border-left-color: rgba(192,132,252,.2);
      animation: spin 9s linear infinite reverse; }}
    .ring.r3 {{ inset: -60px; border: 1px dashed rgba(148,163,184,.25); animation: spin 30s linear infinite; }}
    .ring.r4 {{ inset: -8px; background: conic-gradient(from 0deg, transparent, rgba(34,211,238,.55), transparent 35%,
      rgba(217,70,239,.5), transparent 70%); filter: blur(10px); animation: spin 7s linear infinite; z-index: -1; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    @keyframes float {{ 0%,100% {{ transform: translateY(0) scale(1); }} 50% {{ transform: translateY(-8px) scale(1.03); }} }}
    @keyframes shimmer {{ to {{ background-position: -200% 0; }} }}
    @media (max-width: 800px) {{ .hero {{ grid-template-columns: 1fr; }} .orb-wrap {{ height: 180px; }} }}
    @media (prefers-reduced-motion: reduce) {{ .orb, .ring, .thinking-text {{ animation: none !important; }} }}

    /* slim header shown once answers are on screen */
    .mini-hero {{ display: flex; align-items: center; gap: 22px; padding: 16px 24px; border-radius: 16px; margin-bottom: 14px;
      background: radial-gradient(300px 120px at 8% 50%, rgba(34,211,238,.25), transparent 70%),
                  linear-gradient(135deg, #070b19 0%, #111338 60%, #1e1b4b 100%);
      box-shadow: 0 12px 30px -18px rgba(49,46,129,.6); }}
    .mini-hero .orb {{ width: 42px; height: 42px; flex: none; }}
    .mini-hero .ring.r1 {{ inset: -8px; }}
    .mini-hero .ring.r2 {{ inset: -14px; }}
    .mini-hero .eyebrow {{ color: #67e8f9; font-size: .65rem; }}
    .mini-title {{ color: #fff; font-weight: 700; font-size: 1.15rem; }}
    .mini-hero .grad {{ background-image: linear-gradient(90deg, #22d3ee, #a78bfa, #f0abfc); }}

    /* "thinking" state while the query runs */
    .thinking {{ display: flex; align-items: center; gap: 22px; padding: 22px 26px; border-radius: 16px;
      background: linear-gradient(135deg, #070b19, #1e1b4b); margin: 8px 0 16px; }}
    .thinking .orb {{ width: 44px; height: 44px; animation: float 1.2s ease-in-out infinite; }}
    .thinking .ring.r1 {{ inset: -8px; animation-duration: 1s; }}
    .thinking .ring.r2 {{ inset: -14px; animation-duration: 1.6s; }}
    .thinking-text {{ font-weight: 600; font-size: 1rem; background: linear-gradient(90deg, #94a3b8 0%, #ffffff 50%, #94a3b8 100%);
      background-size: 200% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
      animation: shimmer 1.6s linear infinite; }}
    .thinking-sub {{ font-size: .8rem; color: #94a3b8; }}

    /* action row: voice + surprise */
    .st-key-actions button {{ border-radius: 999px; font-weight: 600; }}
    .st-key-surprise button {{ background: linear-gradient(90deg, #0891b2, #7c3aed) !important; color: #fff !important;
      border: 0 !important; box-shadow: 0 6px 20px -8px rgba(124,58,237,.7); }}
    .st-key-surprise button:hover {{ filter: brightness(1.1); transform: translateY(-1px); }}
    .st-key-surprise button p {{ color: #fff !important; }}

    /* stat cards */
    .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 4px 0 22px; }}
    .stat {{ padding: 14px 16px; border-radius: 14px; border: 1px solid var(--border); background: var(--surface);
      box-shadow: var(--shadow); }}
    .stat .v {{ font-size: 1.45rem; font-weight: 700; color: var(--text); }}
    .stat .l {{ font-size: .78rem; color: var(--muted); margin-top: 2px; }}
    @media (max-width: 800px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} .hero-title {{ font-size: 1.7rem; }} }}

    .section-label {{ font-size: .8rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
      color: var(--muted); margin: 6px 0 8px; }}

    /* example-question cards (real buttons) */
    [class*="st-key-examples"] button {{ min-height: 96px; justify-content: flex-start; align-items: flex-start; text-align: left;
      padding: 14px 16px; border-radius: 14px; border: 1px solid var(--border); background: var(--surface);
      color: var(--text); box-shadow: var(--shadow); transition: transform .12s, border-color .12s, box-shadow .12s; }}
    [class*="st-key-examples"] button:hover {{ transform: translateY(-2px); border-color: var(--accent); color: var(--text);
      box-shadow: 0 0 0 3px var(--accent-soft), var(--shadow); }}
    [class*="st-key-examples"] button * {{ white-space: normal !important; overflow: visible !important;
      text-overflow: clip !important; }}
    [class*="st-key-examples"] button p {{ font-size: .92rem; line-height: 1.45; text-align: left; }}
    [class*="st-key-examples"] button > div {{ width: 100%; justify-content: flex-start; text-align: left; }}
    [class*="st-key-examples"] [data-testid="stMarkdownContainer"] {{ width: 100%; text-align: left; }}
    [class*="st-key-examples"] button::after {{ content: "→"; position: absolute; right: 16px; bottom: 12px;
      color: var(--muted); transition: transform .15s, color .15s; }}
    [class*="st-key-examples"] button:hover::after {{ color: var(--accent); transform: translateX(3px); }}
    [class*="st-key-examples"] button {{ position: relative; padding-right: 36px; }}
    [class*="st-key-examples"] button strong {{ display: inline-block; font-size: .68rem; letter-spacing: .1em; text-transform: uppercase;
      color: var(--accent); margin-bottom: 4px; }}

    /* answer cards */
    .q-title {{ font-size: 1.15rem; font-weight: 700; color: var(--text); margin-bottom: 8px; }}
    .answer {{ font-size: 1.02rem; line-height: 1.65; color: var(--text); padding: 12px 16px; border-radius: 12px;
      background: var(--accent-soft); border-left: 3px solid var(--accent); margin-bottom: 14px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
    .chip {{ font-size: .72rem; padding: 3px 9px; border-radius: 999px; background: var(--surface2);
      border: 1px solid var(--border); color: var(--muted); }}
    .chip.ok {{ color: #059669; border-color: rgba(5,150,105,.35); background: rgba(5,150,105,.08); }}

    /* dataset switch + uploader */
    [data-testid="stFileUploaderDropzone"] {{ border-radius: 16px; border: 1.5px dashed var(--accent);
      background: var(--accent-soft); }}
    .ws-note {{ font-size: .82rem; color: var(--muted); margin: 4px 0 14px; }}
    .badge {{ font-size: .68rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; margin-left: 6px; }}

    /* follow-up bar on the newest answer */
    .fu-label {{ font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
      color: var(--accent); margin: 16px 0 6px; padding-top: 14px; border-top: 1px dashed var(--border); }}
    [class*="st-key-fu_"] .stButton button {{ border-radius: 999px; font-size: .85rem; min-height: 36px;
      border: 1px solid var(--border); background: var(--accent-soft); color: var(--text); }}
    [class*="st-key-fu_"] .stButton button:hover {{ border-color: var(--accent); color: var(--accent); }}
    [class*="st-key-fu_"] [data-testid="stFormSubmitButton"] button {{ border-radius: 12px; height: 40px;
      background: linear-gradient(135deg, #0891b2, #7c3aed); color: #fff; border: 0; font-weight: 600; }}
    [class*="st-key-fu_"] [data-testid="stTextInput"] input {{ border-radius: 12px; height: 40px; }}

    /* accuracy headline */
    .big {{ font-size: 3rem; font-weight: 800; line-height: 1; }}

    /* sidebar */
    [data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}
    [data-testid="stSidebarHeader"] {{ height: 2.2rem; padding-top: .6rem; }}
    .brand-row {{ display: flex; align-items: center; gap: 12px; margin: 0 0 14px; }}
    .logo {{ width: 42px; height: 42px; border-radius: 12px; flex: none; display: flex; align-items: center;
      justify-content: center; background: linear-gradient(135deg, #22d3ee 0%, #6366f1 60%, #a855f7 100%);
      box-shadow: 0 6px 18px -6px rgba(99,102,241,.7), inset 0 1px 0 rgba(255,255,255,.35); }}
    .logo svg {{ width: 22px; height: 22px; }}
    .brand {{ font-size: 1.25rem; font-weight: 800; letter-spacing: -.02em; color: var(--text); line-height: 1.1; }}
    .brand span {{ background: linear-gradient(90deg, var(--accent), var(--accent2)); -webkit-background-clip: text;
      background-clip: text; color: transparent; }}
    .brand-sub {{ font-size: .76rem; color: var(--muted); margin-top: 3px; }}

    /* tabs as a segmented pill (role selectors work across Streamlit versions) */
    .stTabs [role="tablist"] {{ gap: 4px; padding: 4px; border-radius: 14px; width: fit-content;
      background: var(--surface2); border: 1px solid var(--border); border-bottom: 1px solid var(--border) !important; }}
    .stTabs [role="tab"] {{ height: 38px; padding: 0 18px !important; border-radius: 10px; background: transparent;
      border: 0 !important; margin: 0 !important; }}
    .stTabs [role="tab"] p {{ font-size: .9rem; font-weight: 600; color: var(--muted) !important; }}
    .stTabs [role="tab"][aria-selected="true"] {{ background: var(--surface) !important;
      box-shadow: 0 1px 2px rgba(15,23,42,.08), 0 2px 8px rgba(15,23,42,.06); }}
    .stTabs [role="tab"][aria-selected="true"] p {{ color: var(--text) !important; }}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"],
    .stTabs .react-aria-SelectionIndicator, .stTabs [role="tab"]::after {{ display: none !important; }}
    .stTabs [role="tabpanel"] {{ padding-top: 18px; }}
    .stTabs {{ padding-top: 6px; overflow: visible; }}

    /* question box */
    [data-testid="stChatInput"] > div {{ border-radius: 18px !important; border: 1.5px solid var(--border) !important;
      background: var(--surface) !important; box-shadow: 0 2px 4px rgba(15,23,42,.03), 0 10px 30px -12px rgba(8,145,178,.25);
      transition: border-color .15s, box-shadow .15s; }}
    [data-testid="stChatInput"] > div:focus-within {{ border-color: var(--accent) !important;
      box-shadow: 0 0 0 4px var(--accent-soft), 0 10px 30px -12px rgba(8,145,178,.35); }}
    [data-testid="stChatInput"] textarea {{ font-size: 1rem !important; padding-top: 6px; }}
    [data-testid="stChatInputSubmitButton"] {{ border-radius: 12px !important;
      background: linear-gradient(135deg, #0891b2, #7c3aed) !important; color: #fff !important; }}
    [data-testid="stChatInputSubmitButton"] svg {{ fill: #fff !important; color: #fff !important; }}
    [data-testid="stChatInputSubmitButton"]:disabled {{ opacity: .45; }}
    .side-note {{ font-size: .78rem; color: var(--muted); }}
    .side-note a {{ color: var(--accent); text-decoration: none; }}

    /* rounded containers and tabs */
    [data-testid="stExpander"] details {{ border-radius: 12px; }}
    """
    if dark:
        css += """
        .stApp, [data-testid="stHeader"] { background: var(--bg) !important; color: var(--text); }
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: var(--surface2) !important; }
        .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        [data-testid="stWidgetLabel"] *, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] * { color: var(--text); }
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
        .stTabs [role="tab"][aria-selected="true"] { box-shadow: 0 0 0 1px var(--border), 0 4px 12px rgba(0,0,0,.35) !important; }
        .stApp .grad, .stApp .brand span { color: transparent !important; }
        [data-testid="stSelectbox"] * { color: var(--text) !important; }
        [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] details {
          border-color: var(--border) !important; background: var(--surface); }
        [data-testid="stExpander"] summary:hover p { color: var(--accent); }
        [data-testid="stSelectbox"] div, [data-baseweb="select"] div, [data-baseweb="select"] input { background-color: var(--surface) !important; color: var(--text) !important; }
        [data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }
        [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea {
          background: var(--surface) !important; border-color: var(--border) !important; color: var(--text) !important; }
        [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--bg) !important; }
        ul[role="listbox"], [data-baseweb="popover"] li { background: var(--surface) !important; color: var(--text) !important; }
        .stCode pre, [data-testid="stCode"] pre { background: var(--surface2) !important; }
        [data-testid="stAlert"] { background: var(--surface) !important; }
        /* dataframes are drawn on a canvas, so flip their colors */
        [data-testid="stDataFrame"] { filter: invert(0.92) hue-rotate(180deg); }
        """
    css += "</style>"
    # strip indentation so Markdown doesn't turn any of it into a code block
    return "\n".join(line.strip() for line in css.splitlines() if line.strip())


def chart_theme(dark: bool) -> dict:
    t = DARK if dark else LIGHT
    return {"background": "transparent",
            "axis": {"labelColor": t["muted"], "titleColor": t["muted"], "gridColor": t["border"],
                     "domainColor": t["border"], "tickColor": t["border"], "labelFont": "Inter", "titleFont": "Inter"},
            "legend": {"labelColor": t["text"], "titleColor": t["muted"], "labelFont": "Inter", "labelLimit": 0},
            "view": {"stroke": None}}


# ---------- helpers ----------

class _NoRetriever:
    """Uploaded data has no verified queries or definitions, so retrieval is always off."""


@st.cache_resource(show_spinner="Loading database and retrieval index...")
def get_assistant() -> SQLAssistant:
    return SQLAssistant(db=Database(), retriever=Retriever())


def available_models() -> dict[str, str]:
    """Only models whose API key is set (local Ollama models only with ENABLE_LOCAL_MODELS=1)."""
    show_local = os.getenv("ENABLE_LOCAL_MODELS") == "1"
    out = {}
    for k, v in config.MODELS.items():
        if v["provider"] == "ollama" and not show_local:
            continue
        key = config.PROVIDER_KEYS.get(v["provider"])
        if key and not os.getenv(key):
            continue
        out[k] = v["label"]
    return out


def fmt_sql(sql: str | None) -> str:
    if not sql:
        return "-"
    if sqlparse is not None and "\n" not in sql.strip():
        return sqlparse.format(sql, reindent=True, keyword_case="upper")
    return sql


def set_pending(q: str) -> None:
    # Runs as a button callback, before the script reruns, so the question is ready when we read it.
    st.session_state.pending = q


def pct(v) -> str:
    return "-" if v is None or pd.isna(v) else f"{v:.1f}%"


def pick_chart(df: pd.DataFrame):
    """Return ('line'|'bar', x, y) when a simple chart makes sense, else None."""
    if df is None or len(df) < 2 or df.shape[1] < 2:
        return None
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    labels = [c for c in df.columns if c not in numeric]
    if not numeric or len(labels) != 1:
        return None
    x, y = labels[0], numeric[-1]
    if pd.api.types.is_datetime64_any_dtype(df[x]):  # DuckDB DATE/TIMESTAMP arrive as datetime64
        return "line", x, y
    return ("bar", x, y) if len(df) <= 30 else None


def pretty(df: pd.DataFrame):
    """Display copy: dates without midnight times, numbers with thousands separators."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]) and (out[c].dropna().dt.normalize() == out[c].dropna()).all():
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    return out.style.format(thousands=",", precision=2)


def draw_chart(df: pd.DataFrame, kind: str, x: str, y: str, dark: bool):
    t = DARK if dark else LIGHT
    if kind == "line":
        # scale type "utc": dates are midnight UTC; showing them in local time would shift Apr 1 to Mar 31
        base = alt.Chart(df).encode(x=alt.X(f"{x}:T", title=None, scale=alt.Scale(type="utc"),
                                            axis=alt.Axis(format="%b %Y", labelAngle=0, labelOverlap=True)),
                                    y=alt.Y(f"{y}:Q", title=y))
        c = (base.mark_area(opacity=0.15, color=t["accent"])
             + base.mark_line(color=t["accent"], strokeWidth=2.5)
             + base.mark_point(color=t["accent"], filled=True, size=45).encode(tooltip=[x, y]))
    else:
        c = alt.Chart(df).mark_bar(color=t["accent"], cornerRadiusEnd=4).encode(
            y=alt.Y(f"{x}:N", sort="-x", title=None), x=alt.X(f"{y}:Q", title=y), tooltip=[x, y])
    st.altair_chart(c.properties(height=300).configure(**chart_theme(dark)), width="stretch")


# ---------- datasets ----------

def olist_context() -> dict | None:
    try:
        assistant = get_assistant()
    except FileNotFoundError:
        st.error("No Olist database yet. Download the Olist CSVs into data/raw/ and run: python -m src.load_data")
        return None
    return dict(
        key="olist", assistant=assistant, rag=True, name="Olist",
        eyebrow="TalkToTables · demo dataset: Olist",
        title="Talk to your <span class='grad'>tables</span><br>in plain English",
        sub="Type it or say it, then keep the conversation going. The assistant writes DuckDB SQL, checks "
            "that it's read-only, runs it, and explains the answer, out loud if you like. You're on the Olist "
            "demo: 100K real orders from a Brazilian marketplace. Switch to your own data in the sidebar.",
        pills=[("Benchmarked", "40 questions"), ("Models", "Claude + open-weight"), ("Voice", "speak & listen"),
               ("Follow-ups", "remembers context"), ("Guardrails", "read-only SQL")],
        stats=[("99,441", "Orders"), ("3,095", "Sellers"), ("32,951", "Products"), ("2016-18", "Period covered")],
        examples=EXAMPLE_QUESTIONS, surprise=SURPRISE_POOL,
        placeholder="Ask a question about orders, sellers, deliveries, reviews...",
        about=olist_about,
    )


def olist_about():
    st.markdown(
        "Olist is a Brazilian e-commerce marketplace. **orders** is the center: "
        "items, payments and reviews join on `order_id`; customers join on `customer_id`; "
        "products and sellers join through **order_items**."
    )
    st.dataframe(pd.DataFrame(TABLES, columns=["Table", "Rows", "What it holds", "Join keys"]),
                 hide_index=True, width="stretch")


def fallback_questions(ws) -> list[str]:
    """Used if the model can't suggest questions: simple ones built from the column types."""
    qs = []
    for t in ws.infos:
        nums = [c for c, typ in t.columns if any(k in typ.upper() for k in ("INT", "DOUBLE", "DECIMAL", "FLOAT"))]
        texts = [c for c, typ in t.columns if "VARCHAR" in typ.upper()]
        dates = [c for c, typ in t.columns if "DATE" in typ.upper() or "TIME" in typ.upper()]
        qs.append(f"How many rows are in {t.name}?")
        if nums and texts:
            qs.append(f"What is the total {nums[-1]} by {texts[0]}?")
            qs.append(f"Which 5 {texts[0]} values have the highest average {nums[-1]}?")
        if nums and dates:
            qs.append(f"How did total {nums[-1]} change by month?")
    return qs[:6]


def upload_context(model: str) -> dict | None:
    ws = st.session_state.get("ws")
    if ws is None:
        return None
    if "ws_suggestions" not in st.session_state:
        with st.spinner("Reading your tables and suggesting questions..."):
            st.session_state.ws_suggestions = suggest_questions(ws.schema_text, model=model) or fallback_questions(ws)
    sugg = st.session_state.ws_suggestions
    assistant = SQLAssistant(db=ws.db, retriever=_NoRetriever(), schema_text=ws.schema_text,
                             allowed_tables=ws.tables, dataset="the user's uploaded data")
    n_cols = sum(len(t.columns) for t in ws.infos)
    return dict(
        key=f"ws:{st.session_state.ws_sig}", assistant=assistant, rag=False, name="your data",
        eyebrow="TalkToTables · your uploaded data",
        title="Talk to <span class='grad'>your data</span><br>in plain English",
        sub="Your files were loaded into a private, read-only DuckDB database for this session. Type or say "
            "a question, follow up naturally, and the assistant writes the SQL, runs it, and explains the answer.",
        pills=[("Tables", str(len(ws.infos))), ("Private", "this session only"), ("Voice", "speak & listen"),
               ("Follow-ups", "remembers context"), ("Guardrails", "read-only SQL")],
        stats=[(f"{len(ws.infos)}", "Tables"), (f"{ws.total_rows:,}", "Rows"), (f"{n_cols}", "Columns"),
               ("Live", "Not benchmarked")],
        examples=[("Suggested", q) for q in sugg], surprise=sugg or ["How many rows are there?"],
        placeholder="Ask a question about your data...",
        about=lambda: workspace_about(ws),
    )


def workspace_about(ws):
    for t in ws.infos:
        st.markdown(f"**{t.name}** · {t.rows:,} rows · from `{t.source}`")
        if t.renamed:
            st.caption("Renamed columns: " + ", ".join(f"{a} → {b}" for a, b in list(t.renamed.items())[:12]))
        st.dataframe(pretty(ws.preview(t.name)), hide_index=True, width="stretch")


def uploader():
    st.markdown("<div class='section-label'>Upload your data</div>", unsafe_allow_html=True)
    files = st.file_uploader(
        "CSV, TSV, Excel or Parquet. Several files become several tables; each Excel sheet becomes a table.",
        type=["csv", "tsv", "txt", "xlsx", "xls", "parquet"], accept_multiple_files=True, key="uploads")
    st.markdown(
        f"<div class='ws-note'>Up to {config.MAX_UPLOAD_FILES} files, {config.MAX_UPLOAD_MB:.0f} MB each. "
        "Your data stays in this session and is deleted when you leave. Column names, a few example values and "
        "query results are sent to the selected AI model to write the SQL and the answer.</div>",
        unsafe_allow_html=True)
    if not files:
        return
    sig = "|".join(f"{f.name}:{f.size}" for f in files)
    if st.session_state.get("ws_sig") == sig:
        return
    old = st.session_state.pop("ws", None)
    if old is not None:
        old.close()
    st.session_state.pop("ws_suggestions", None)
    try:
        with st.spinner("Loading your files into DuckDB..."):
            ws = build_workspace([(f.name, f.getvalue()) for f in files])
    except UploadError as e:
        st.session_state.pop("ws_sig", None)
        st.error(str(e))
        return
    st.session_state.ws, st.session_state.ws_sig = ws, sig
    st.rerun()


# ---------- Ask tab ----------

def hero(ctx: dict):
    pills = "".join(f"<span class='pill'><b>{a}</b> {b}</span>" for a, b in ctx["pills"])
    st.markdown(
        f"""
        <div class="hero">
          <div>
            <div class="eyebrow">{ctx.get('eyebrow', 'TalkToTables')}</div>
            <div class="hero-title">{ctx['title']}</div>
            <p class="hero-sub">{ctx['sub']}</p>
            <div class="pills">{pills}</div>
          </div>
          <div class="orb-wrap"><div class="orb"><div class="ring r4"></div><div class="ring r1"></div>
            <div class="ring r2"></div><div class="ring r3"></div></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_header(ctx: dict):
    """Slim version of the hero that stays on screen once answers are showing."""
    title = re.sub(r"<br\s*/?>", " ", ctx["title"])
    st.markdown(
        "<div class='mini-hero'><div class='orb'><div class='ring r1'></div><div class='ring r2'></div></div>"
        f"<div><div class='eyebrow'>{ctx.get('eyebrow', 'TalkToTables')}</div><div class='mini-title'>{title}</div></div></div>",
        unsafe_allow_html=True)


def stats(ctx: dict):
    cards = "".join(f"<div class='stat'><div class='v'>{v}</div><div class='l'>{l}</div></div>" for v, l in ctx["stats"])
    st.markdown(f"<div class='stats'>{cards}</div>", unsafe_allow_html=True)


def thinking_html(question: str) -> str:
    return ("<div class='thinking'><div class='orb'><div class='ring r1'></div><div class='ring r2'></div></div>"
            f"<div><div class='thinking-text'>Writing SQL and running it...</div>"
            f"<div class='thinking-sub'>{html.escape(question)}</div></div></div>")


def action_row(ctx: dict) -> str | None:
    """Voice button + Surprise me. Returns the spoken question, if any."""
    spoken = None
    with st.container(key="actions"):
        c1, c2, _ = st.columns([1.3, 1.3, 4])
        with c1:
            if speech_to_text is not None:
                spoken = speech_to_text(start_prompt="🎙️ Speak a question", stop_prompt="⏹️ Stop and ask",
                                        language="en", just_once=True, use_container_width=True, key="voice")
            else:
                st.caption("Voice needs: pip install streamlit-mic-recorder")
        with c2:
            with st.container(key="surprise"):
                st.button("🎲 Surprise me", width="stretch", on_click=set_pending,
                          args=(random.choice(ctx["surprise"]),))
    return spoken or None


def data_expander(ctx: dict):
    with st.expander("What's in the data?"):
        ctx["about"]()


def example_cards(ctx: dict, prefix: str = "ex"):
    with st.container(key=f"examples_{prefix}" if prefix != "ex" else "examples"):
        cols = st.columns(3)
        for i, (tag, q) in enumerate(ctx["examples"]):
            cols[i % 3].button(f"**{tag}**  \n{q}", key=f"{prefix}_{i}", width="stretch",
                               on_click=set_pending, args=(q,))


def speak_widget(text: str, dark: bool, auto: bool):
    """Listen button + voice and speed pickers. Uses the browser's built-in voices (free, no API key),
    so the list depends on the visitor's computer and browser. The choice is remembered in this browser.
    It runs inside a small iframe, so the click that starts speech happens in the same frame (browsers require that)."""
    t = DARK if dark else LIGHT
    components.html(f"""
    <style>
      body {{ margin: 0; font-family: Inter, system-ui, sans-serif; }}
      .row {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
      button, select {{ height: 32px; padding: 0 12px; border-radius: 999px; border: 1px solid {t['border']};
        background: {t['surface']}; color: {t['text']}; font-size: 13px; font-family: inherit; }}
      button {{ display: inline-flex; align-items: center; gap: 8px; font-weight: 600; cursor: pointer; }}
      select {{ cursor: pointer; max-width: 260px; }}
      button:hover, select:hover {{ border-color: {t['accent']}; }}
      .bars {{ display: none; gap: 2px; align-items: end; height: 12px; }}
      .on .bars {{ display: inline-flex; }}
      .bars i {{ width: 3px; background: {t['accent']}; border-radius: 2px; animation: eq .9s ease-in-out infinite; }}
      .bars i:nth-child(2) {{ animation-delay: .15s }} .bars i:nth-child(3) {{ animation-delay: .3s }}
      .bars i:nth-child(4) {{ animation-delay: .45s }}
      @keyframes eq {{ 0%,100% {{ height: 3px }} 50% {{ height: 12px }} }}
    </style>
    <div class="row">
      <button id="b"><span id="l">🔊 Listen</span><span class="bars"><i></i><i></i><i></i><i></i></span></button>
      <select id="v" title="Voice"></select>
      <select id="r" title="Speed">
        <option value="0.9">0.9x</option><option value="1" selected>1x</option>
        <option value="1.15">1.15x</option><option value="1.3">1.3x</option>
      </select>
    </div>
    <script>
      const text = {json.dumps(text)};
      const b = document.getElementById('b'), l = document.getElementById('l');
      const vs = document.getElementById('v'), rs = document.getElementById('r');
      const synth = window.speechSynthesis;
      const store = {{
        get(k) {{ try {{ return localStorage.getItem(k); }} catch (e) {{ return null; }} }},
        set(k, v) {{ try {{ localStorage.setItem(k, v); }} catch (e) {{}} }},
      }};
      let voices = [];
      function fill() {{
        if (!synth) return;
        const all = synth.getVoices();
        voices = all.filter(v => /^en/i.test(v.lang));
        if (!voices.length) voices = all;
        const saved = store.get('ttt_voice');
        const nice = voices.findIndex(v => /Google US|Aria|Jenny|Samantha|Natural/i.test(v.name));
        vs.innerHTML = voices.map((v, i) => `<option value="${{i}}">${{v.name.replace('Microsoft ', '').replace(' Online (Natural)', '')}} (${{v.lang}})</option>`).join('');
        vs.style.display = voices.length ? '' : 'none';
        const idx = voices.findIndex(v => v.name === saved);
        vs.value = String(idx >= 0 ? idx : Math.max(nice, 0));
      }}
      const savedRate = store.get('ttt_rate'); if (savedRate) rs.value = savedRate;
      vs.onchange = () => {{ store.set('ttt_voice', voices[+vs.value]?.name || ''); say(); }};
      rs.onchange = () => {{ store.set('ttt_rate', rs.value); say(); }};
      function stop() {{ synth.cancel(); b.classList.remove('on'); l.textContent = '🔊 Listen'; }}
      function say() {{
        if (!synth) {{ l.textContent = 'Voice not supported here'; return; }}
        synth.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.voice = voices[+vs.value] || null;
        u.rate = parseFloat(rs.value);
        u.onstart = () => {{ b.classList.add('on'); l.textContent = '⏹ Stop'; }};
        u.onend = u.onerror = () => {{ b.classList.remove('on'); l.textContent = '🔊 Listen'; }};
        synth.speak(u);
      }}
      b.onclick = () => synth && synth.speaking ? stop() : say();
      if (synth) {{
        fill();
        synth.onvoiceschanged = () => {{ fill(); {"say();" if auto else ""} synth.onvoiceschanged = fill; }};
        {"if (synth.getVoices().length) say();" if auto else ""}
      }} else {{ vs.style.display = rs.style.display = 'none'; }}
    </script>
    """, height=40)


FOLLOWUP_PROMPT = """A business user asked: {question}

The SQL that answered it:
{sql}

Result columns: {cols}. First rows:
{rows}

Suggest 3 natural follow-up questions they might ask next, each under 9 words, that build on this
result (drill down, compare, change the time period, rank, or explain a spike). Answerable from the
same tables. One per line, no numbering, nothing else."""


def suggest_followups(res, model: str) -> list[str]:
    """One small extra model call per answer; its tokens and cost are added to the answer's totals."""
    try:
        reply = llm.complete("You suggest short, useful analytics follow-up questions.",
                             [{"role": "user", "content": FOLLOWUP_PROMPT.format(
                                 question=res.question, sql=res.sql, cols=", ".join(map(str, res.df.columns)),
                                 rows=res.df.head(5).to_csv(index=False))}],
                             model=model, max_tokens=512)
    except Exception:
        return []
    res.llm_calls.append(reply)
    out = []
    for line in reply.text.splitlines():
        q = re.sub(r"^\s*(?:[-*\d.)]+\s*)", "", line).strip().strip('"')
        if "`" in q or re.search(r"\b(select|from|group by)\b", q, re.I) or not re.search(r"[a-z]{3}", q, re.I):
            continue  # skip code or stray formatting, keep only plain-English questions
        if 5 <= len(q) <= 90:
            out.append(q if q.endswith("?") else q + "?")
    return out[:3]


def ask_followup(q: str) -> None:
    st.session_state.pending = q
    st.session_state.force_follow = True


def followup_bar(res):
    """Keep the conversation going from the newest answer: suggested chips, a text box, and the mic."""
    st.markdown("<div class='fu-label'>↪ Keep going</div>", unsafe_allow_html=True)
    with st.container(key=f"fu_{id(res)}"):
        chips = getattr(res, "followups", None) or []
        if chips:
            cols = st.columns(len(chips))
            for i, q in enumerate(chips):
                cols[i].button(q, key=f"fuc_{id(res)}_{i}", width="stretch", on_click=ask_followup, args=(q,))
        with st.form(key=f"fuf_{id(res)}", clear_on_submit=True, border=False):
            c1, c2 = st.columns([6, 1])
            text = c1.text_input("Follow-up", placeholder="Ask a follow-up... e.g. 'now split it by state'",
                                 label_visibility="collapsed")
            if c2.form_submit_button("Ask ↪", width="stretch") and text.strip():
                ask_followup(text.strip())
                st.rerun()
        if speech_to_text is not None:
            spoken = speech_to_text(start_prompt="🎙️ Say a follow-up", stop_prompt="⏹️ Stop and ask",
                                    language="en", just_once=True, key=f"fuv_{id(res)}")
            if spoken:
                ask_followup(spoken)
                st.rerun()


def render_result(res, dark: bool, name: str, speak: str = "off", newest: bool = False):
    with st.container(border=True):
        label = config.MODELS.get(res.model, {}).get("label", res.model)
        st.markdown(f"<div class='q-title'>{html.escape(res.question)}</div>", unsafe_allow_html=True)

        if res.cannot_answer:
            st.info(f"This question can't be answered from {name}: {res.answer}")
            return
        if res.error:
            st.error(res.error)
            if res.sql:
                with st.expander("Show SQL"):
                    st.code(fmt_sql(res.sql), language="sql", wrap_lines=True)
            return

        if res.answer:
            st.markdown(f"<div class='answer'>{html.escape(res.answer)}</div>", unsafe_allow_html=True)
            if speak != "off":
                speak_widget(res.answer, dark, auto=speak == "auto")

        chart = pick_chart(res.df)
        if chart:
            left, right = st.columns([2, 3], gap="medium")
            with left:
                st.dataframe(pretty(res.df), width="stretch", hide_index=True, height=min(300, 35 * (len(res.df) + 1) + 3))
            with right:
                draw_chart(res.df, *chart, dark=dark)
        else:
            st.dataframe(pretty(res.df), width="stretch", hide_index=True)
        if res.truncated:
            st.caption(f"Showing the first {config.MAX_ROWS} rows.")

        if res.sql:
            with st.expander("Show SQL"):
                st.code(fmt_sql(res.sql), language="sql", wrap_lines=True)  # st.code has a built-in copy button

        if res.examples:
            with st.expander("What retrieval found for this question"):
                if res.terms:
                    st.markdown("**Definitions:** " + ", ".join(t["term"] for t in res.terms))
                st.markdown("**Similar verified queries:**")
                for e in res.examples:
                    st.markdown(f"- {e['question']}")

        cost = f"<span class='chip'>${res.cost_usd:.4f}</span>" if res.cost_usd is not None else ""
        st.markdown(
            "<div class='meta'>"
            "<span class='chip ok'>✓ read-only check passed</span>"
            f"<span class='chip'>{html.escape(label)}</span>"
            f"<span class='chip'>{len(res.df)} rows</span>"
            f"<span class='chip'>{res.seconds:.1f}s</span>"
            f"<span class='chip'>{res.input_tokens + res.output_tokens:,} tokens</span>{cost}"
            f"<span class='chip'>retrieval {'on' if res.used_rag else 'off'}</span>"
            + ("<span class='chip'>↪ follow-up</span>" if getattr(res, "followed_up", False) else "") +
            ""
            f"<span class='chip'>{res.attempts} attempt{'s' if res.attempts != 1 else ''}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        if newest:
            followup_bar(res)


def ask_tab(dark: bool):
    st.session_state.setdefault("asked", 0)
    st.session_state.setdefault("histories", {})

    with st.sidebar:
        source = st.radio("Dataset", ["Olist demo", "My own data"], horizontal=True, key="source",
                          help="The Olist demo is benchmarked on 40 hand-checked questions. "
                               "Your own data uses the same pipeline without retrieval.")
        models = available_models()
        if not models:
            st.error("No models available. Add ANTHROPIC_API_KEY or GROQ_API_KEY to .env")
            return
        model = st.selectbox("Model", list(models), format_func=lambda k: models[k],
                             index=list(models).index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in models else 0)
        own = source == "My own data"
        use_rag = st.toggle("Use retrieval (RAG)", value=not own, disabled=own,
                            help="Adds business definitions and similar verified queries to the prompt "
                                 "(Olist demo only).")
        talk_back = st.toggle("🔊 Read answers aloud", key="talk_back",
                              help="Speaks each new answer with your browser's built-in voice. "
                                   "Every answer also has a Listen button.")
        follow_up = st.toggle("💬 Follow-up questions", value=True, key="follow_up",
                              help="Lets you ask follow-ups like 'what about 2017?' or 'now by state'. "
                                   "The previous question and its SQL are sent along with the new one.")
        counter = st.empty()

    ctx = upload_context(model) if own else olist_context()
    if own and ctx is None:
        hero(dict(eyebrow="TalkToTables · bring your own data",
                  title="Talk to <span class='grad'>your own</span><br>tables",
                  sub="Upload a spreadsheet or CSV and ask questions in plain English, by typing or talking. "
                      "The assistant writes the SQL, checks it's read-only, runs it, and explains the answer.",
                  pills=[("CSV", "Excel · Parquet"), ("Joins", "across files"), ("Private", "this session only"),
                         ("Guardrails", "read-only SQL")]))
        uploader()
        return
    if ctx is None:
        return

    history = st.session_state.histories.setdefault(ctx["key"], [])
    question = st.chat_input(ctx["placeholder"])

    empty = not history
    if empty:
        hero(ctx)
    else:
        mini_header(ctx)
    if own:
        with st.expander("Change or add files"):
            uploader()
    spoken = action_row(ctx)
    question = question or spoken or st.session_state.pop("pending", None)

    if question:
        if st.session_state.asked >= config.MAX_QUESTIONS_PER_SESSION:
            st.warning("You've reached the question limit for this session. Refresh later to ask more.")
        else:
            st.session_state.asked += 1
            thinking = st.empty()
            thinking.markdown(thinking_html(question), unsafe_allow_html=True)
            asked = question
            prev = history[0] if history else None
            forced = st.session_state.pop("force_follow", False)
            if (follow_up or forced) and prev is not None and prev.sql and not prev.error:
                asked = (f"Earlier question: {prev.question}\nSQL used for it:\n{prev.sql}\n\n"
                         f"New question (it may build on the earlier one, e.g. 'what about 2017?' or "
                         f"'now split by state'; if it doesn't, ignore the earlier one): {question}")
            res = ctx["assistant"].ask(asked, model=model, use_rag=use_rag and ctx["rag"])
            res.question = question  # show what the person actually asked
            res.followed_up = asked != question
            st.session_state.speak_next = id(res)
            if res.ok:
                res.followups = suggest_followups(res, model)
            thinking.empty()
            history.insert(0, res)
            if not res.error and not res.cannot_answer:
                st.toast(f"Answered in {res.seconds:.1f}s · {len(res.df)} rows", icon="⚡")
            st.rerun()  # redraw with the answer on top (and the hero gone)

    left_now = max(config.MAX_QUESTIONS_PER_SESSION - st.session_state.asked, 0)
    counter.progress(left_now / config.MAX_QUESTIONS_PER_SESSION,
                     text=f"{left_now} of {config.MAX_QUESTIONS_PER_SESSION} questions left this session")

    if empty:
        stats(ctx)
        st.markdown("<div class='section-label'>Try one of these</div>", unsafe_allow_html=True)
        example_cards(ctx)
        st.write("")
        data_expander(ctx)
        return

    newest = st.session_state.pop("speak_next", None)
    for i, res in enumerate(history[:5]):
        speak = "auto" if (talk_back and id(res) == newest) else "button"
        render_result(res, dark, ctx["name"], speak, newest=(i == 0))
    with st.expander("Try another example"):
        example_cards(ctx, "more")
    data_expander(ctx)


# ---------- Accuracy tab ----------

def load_records(summary: dict, results_dir) -> list[dict]:
    """Every setup's rows point at the run file they came from (older summaries have one run_file)."""
    wanted: dict[str, set] = {}
    for row in summary["rows"]:
        rf = row.get("run_file") or summary.get("run_file")
        if rf:
            wanted.setdefault(rf, set()).add((row["model"], row["rag"]))
    records = []
    for rf, keys in wanted.items():
        path = results_dir / rf
        if path.is_file():
            records += [r for r in json.loads(path.read_text(encoding="utf-8")) if (r["model"], r["rag"]) in keys]
    return records


def accuracy_tab(dark: bool):
    results_dir = config.EVAL_DIR / "results"
    path = results_dir / "summary.json"
    st.markdown(
        "<div class='eyebrow'>Benchmark · Olist demo</div>"
        "<div class='hero-title' style='font-size:1.8rem'>How accurate is it?</div>"
        f"<p class='hero-sub'>{FULL_EVAL_SIZE} questions with hand-checked answers (12 easy, 16 medium, 12 hard). "
        "A question counts as correct when the returned data matches the verified answer. "
        "Closed models (Claude) and open-weight models (anyone can download and self-host them) "
        "answer the same questions with the same scorer.</p>",
        unsafe_allow_html=True,
    )
    st.write("")
    if not path.exists():
        st.info("Benchmark results are coming soon.")
        return

    summary = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(summary["rows"])
    rows["setup"] = rows["label"] + " · retrieval " + rows["rag"].map({True: "on", False: "off"})
    rows["type"] = rows["model"].map(lambda m: "Open-weight" if config.MODELS.get(m, {}).get("open_weights")
                                     else "Closed")
    records = load_records(summary, results_dir)

    if "n" not in rows:  # summaries from older runs
        counts = {}
        for r in records:
            counts[(r["model"], r["rag"])] = counts.get((r["model"], r["rag"]), 0) + 1
        rows["n"] = [counts.get((m, g)) for m, g in zip(rows["model"], rows["rag"])]

    partial = rows["n"].notna() & (rows["n"] < FULL_EVAL_SIZE)
    if partial.all():
        st.warning(f"This is a small test run ({int(rows['n'].max())} questions), not the full benchmark.")
    elif partial.any():
        st.caption("Setups marked with fewer than 40 questions are small test runs.")

    best = rows.sort_values("accuracy", ascending=False).iloc[0]
    with st.container(border=True):
        a, b = st.columns([1, 2], gap="large")
        with a:
            st.markdown(f"<div class='section-label'>Best setup</div><div class='big grad'>{pct(best['accuracy'])}</div>"
                        f"<div class='side-note' style='margin-top:6px'>{html.escape(best['setup'])}</div>",
                        unsafe_allow_html=True)
        with b:
            m = st.columns(3)
            m[0].metric("Hard questions", pct(best["hard"]))
            m[1].metric("Seconds / question", f"{best['avg_seconds']:.1f}")
            c = best["cost_per_100_questions_usd"]
            m[2].metric("Cost / 100 questions", "free (local)" if c is None or pd.isna(c) else f"${c:.2f}")
        st.caption(f"Last run: {summary.get('generated_at', '')[:16].replace('T', ' ')} · "
                   f"{len(rows)} setup{'s' if len(rows) != 1 else ''} compared")

    table = pd.DataFrame({
        "Setup": rows["setup"],
        "Type": rows["type"],
        "Questions": rows["n"].map(lambda v: "-" if pd.isna(v) else str(int(v))),
        "Overall": rows["accuracy"].map(pct),
        "Easy": rows["easy"].map(pct),
        "Medium": rows["medium"].map(pct),
        "Hard": rows["hard"].map(pct),
        "Avg seconds": rows["avg_seconds"].map(lambda v: f"{v:.2f}"),
        "Cost / 100 q": rows["cost_per_100_questions_usd"].map(
            lambda v: "free (local)" if v is None or pd.isna(v) else f"${v:.2f}"),
    })
    st.dataframe(table, hide_index=True, width="stretch")
    if any(config.MODELS.get(m, {}).get("provider") == "groq" for m in rows["model"]):
        st.caption("Groq (open-weight) times include waiting for the free tier's rate limit, so they "
                   "overstate how long the model itself takes. Costs use Groq's paid prices.")

    long = rows[["setup"] + DIFFICULTIES].melt(id_vars="setup", var_name="difficulty", value_name="accuracy")
    long = long.dropna(subset=["accuracy"])
    if not long.empty:
        palette = (["#0891b2", "#6366f1", "#f59e0b", "#10b981", "#ec4899", "#64748b"] if not dark
                   else ["#22d3ee", "#818cf8", "#fbbf24", "#34d399", "#f472b6", "#94a3b8"])
        chart = alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("difficulty:N", sort=DIFFICULTIES, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("setup:N"),
            y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("setup:N", title=None, scale=alt.Scale(range=palette),
                            legend=alt.Legend(orient="bottom", columns=2, columnPadding=28, symbolType="square")),
            tooltip=["setup", "difficulty", alt.Tooltip("accuracy:Q", format=".1f")],
        ).properties(height=320)
        st.altair_chart(chart.configure(**chart_theme(dark)), width="stretch")

    if records:
        misses = sorted((r for r in records if not r["correct"]), key=lambda r: (r["id"], r["model"]))
        st.markdown(f"<div class='section-label' style='margin-top:18px'>Where it went wrong · "
                    f"{len(misses)} misses across all setups</div>", unsafe_allow_html=True)
        if not misses:
            st.success("No misses in this run.")
            return
        setups = sorted({(r["model"], r["rag"]) for r in misses},
                        key=lambda k: config.MODELS.get(k[0], {}).get("label", k[0]))
        if len(setups) > 1:
            names = {k: f"{config.MODELS.get(k[0], {}).get('label', k[0])} · retrieval {'on' if k[1] else 'off'}"
                     for k in setups}
            pick = st.selectbox("Show misses for", ["All setups"] + list(names.values()))
            if pick != "All setups":
                key = next(k for k, v in names.items() if v == pick)
                misses = [r for r in misses if (r["model"], r["rag"]) == key]
        for r in misses[:40]:
            label = config.MODELS.get(r["model"], {}).get("label", r["model"])
            with st.expander(f"{r['id']} · {label} · retrieval {'on' if r['rag'] else 'off'} · {r['question']}"):
                if r.get("error"):
                    st.error(r["error"])
                c1, c2 = st.columns(2)
                c1.markdown("**Model's SQL**")
                c1.code(fmt_sql(r.get("sql")), language="sql", wrap_lines=True)
                c2.markdown("**Verified SQL**")
                c2.code(fmt_sql(r["gold_sql"]), language="sql", wrap_lines=True)


# ---------- page ----------

with st.sidebar:
    st.markdown(
        "<div class='brand-row'><div class='logo'>"
        "<svg viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.8' stroke-linecap='round' "
        "stroke-linejoin='round'><path d='M4 5h16v11H9l-5 4z'/><path d='M4 9.5h16M10 5v11'/></svg></div>"
        "<div><div class='brand'>TalkTo<span>Tables</span></div>"
        "<div class='brand-sub'>Ask your data in plain English</div></div></div>",
        unsafe_allow_html=True)
    dark_mode = st.toggle("🌙 Dark mode", key="dark_mode")
    st.divider()

st.markdown(theme_css(DARK if dark_mode else LIGHT, dark_mode), unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬  Ask", "📊  How accurate is it?"])
with tab1:
    ask_tab(dark_mode)
with tab2:
    accuracy_tab(dark_mode)

with st.sidebar:
    st.divider()
    st.markdown("<div class='side-note'>Built by Himavarsha Sreenivas<br>"
                "<a href='https://github.com/hima24' target='_blank'>GitHub</a> · "
                "<a href='https://linkedin.com/in/himavarshas' target='_blank'>LinkedIn</a></div>",
                unsafe_allow_html=True)
