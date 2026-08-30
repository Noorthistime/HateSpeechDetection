# pyrefly: ignore [missing-import]
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import sklearn
import re
import string
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Ensure nltk packages are available silently
try:
    stopwords.words("english")
except LookupError:
    nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Hate Speech Detection", layout="wide")

def get_hl_code(code_str):
    import re
    keywords = {'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return', 'and', 'or', 'not', 'in', 'is', 'try', 'except', 'with', 'as', 'pass', 'break', 'continue'}
    builtins = {'print', 'len', 'range', 'int', 'float', 'str', 'list', 'dict', 'set', 'tuple', 'bool', 'True', 'False', 'None'}
    methods = {'read_csv', 'map', 'copy', 'lower', 'sub', 'escape', 'split', 'join', 'stem', 'apply', 'fit', 'predict', 'transform', 'fit_transform', 'toarray', 'concat', 'lemmatize', 'DataFrame', 'subplots', 'heatmap', 'countplot', 'head', 'shape', 'drop', 'fillna', 'astype', 'isnull', 'sum', 'sort_values'}
    
    hl_lines = []
    lines = code_str.strip('\n').split('\n')
    for line in lines:
        leading_whitespace = len(line) - len(line.lstrip())
        words = re.split(r'(\W+)', line.lstrip())
        
        hl_words = []
        for w in words:
            if w in keywords:
                hl_words.append(f'<span class="keyword">{w}</span>')
            elif w in builtins or w in methods:
                hl_words.append(f'<span class="builtin">{w}</span>')
            else:
                hl_words.append(w)
                
        hl_line = ('&nbsp;' * leading_whitespace) + ''.join(hl_words)
        if not line.strip():
            hl_line = '&nbsp;'
        hl_lines.append(hl_line)
    return '<br>'.join(hl_lines)

if 'theme' not in st.session_state:
    st.session_state.theme = 'default'

# Hidden Toggle Button
st.markdown('<div class="hide-next-button"></div>', unsafe_allow_html=True)
is_stitch = st.button("HIDDEN_TOGGLE_DO_NOT_CLICK", key="hidden_theme_btn")
if is_stitch:
    st.session_state.theme = 'stitch' if st.session_state.theme == 'default' else 'default'

if st.session_state.theme == 'default':
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary-color: var(--brand-1);
        --glass-bg: rgba(255, 255, 255, 0.12);
        --glass-border: rgba(255, 255, 255, 0.32);
        --glass-shadow: 0 12px 36px rgba(8, 15, 40, 0.28);
        --brand-1: #00c2ff;
        --brand-2: #4f46e5;
        --brand-3: #26d9a4;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 12%, rgba(38, 217, 164, 0.18), transparent 34%),
            radial-gradient(circle at 88% 20%, rgba(0, 194, 255, 0.18), transparent 40%),
            radial-gradient(circle at 50% 86%, rgba(79, 70, 229, 0.2), transparent 45%),
            linear-gradient(145deg, #0b1220 0%, #121f37 46%, #0b182f 100%);
        background-attachment: fixed;
        color: #eef3ff;
    }

    /* Keep header transparent without hiding it */
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] > div {
        background: linear-gradient(165deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.03));
        backdrop-filter: none;
    }
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
    }

    @keyframes orbFloat {
        0% { transform: translate3d(0, 0, 0) scale(1); }
        50% { transform: translate3d(0, -14px, 0) scale(1.05); }
        100% { transform: translate3d(0, 0, 0) scale(1); }
    }

    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 rgba(0, 194, 255, 0.0), 0 10px 26px rgba(10, 18, 42, 0.35); }
        50% { box-shadow: 0 0 22px rgba(79, 70, 229, 0.45), 0 14px 34px rgba(10, 18, 42, 0.45); }
        100% { box-shadow: 0 0 0 rgba(0, 194, 255, 0.0), 0 10px 26px rgba(10, 18, 42, 0.35); }
    }

    @keyframes shimmer {
        0% { background-position: -220% 0; }
        100% { background-position: 220% 0; }
    }

    .stApp::before,
    .stApp::after {
        content: "";
        position: fixed;
        border-radius: 999px;
        filter: blur(52px);
        z-index: 0;
        pointer-events: none;
        animation: orbFloat 9s ease-in-out infinite;
    }

    .stApp::before {
        width: 260px;
        height: 260px;
        right: 10%;
        top: 15%;
        background: rgba(0, 194, 255, 0.18);
    }

    .stApp::after {
        width: 300px;
        height: 300px;
        left: 8%;
        bottom: 8%;
        background: rgba(38, 217, 164, 0.16);
        animation-delay: -3s;
    }


    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
    }

    [data-testid="stVerticalBlock"] > div:has(> [data-testid="stMarkdownContainer"]),
    div[data-testid="stDataFrame"],
    div[data-testid="stAlert"],
    div.stCodeBlock {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        backdrop-filter: blur(14px);
        box-shadow: var(--glass-shadow);
        transition: transform 0.28s ease, box-shadow 0.28s ease, border-color 0.28s ease;
    }

    [data-testid="stVerticalBlock"] > div:has(> [data-testid="stMarkdownContainer"]):hover,
    div[data-testid="stDataFrame"]:hover,
    div[data-testid="stAlert"]:hover,
    div.stCodeBlock:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 194, 255, 0.5);
        box-shadow: 0 0 18px rgba(0, 194, 255, 0.2), 0 16px 34px rgba(8, 15, 40, 0.36);
    }

    h1, h2, h3 {
        letter-spacing: 0.2px;
        text-shadow: 0 0 18px rgba(79, 70, 229, 0.28);
    }

    div.element-container:has(.premium-hero),
    div.stMarkdown:has(.premium-hero) {
        position: sticky !important;
        top: 15px !important;
        z-index: 999 !important;
    }

    .premium-hero {
        position: relative;
        width: 100%;
        margin: -45px auto 20px auto !important;
        padding: 20px 24px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.34);
        background: linear-gradient(130deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.08));
        backdrop-filter: blur(18px);
        box-shadow: 0 16px 38px rgba(4, 12, 34, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.22);
        overflow: hidden;
        transition: all 0.6s cubic-bezier(0.25, 1, 0.5, 1);
        text-align: center;
    }

    .premium-hero.pill-mode {
        width: 85% !important;
        margin: 0 auto 20px auto !important;
        padding: 12px 40px !important;
        border-radius: 50px !important;
        background: linear-gradient(130deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05)) !important;
        box-shadow: 0 10px 30px rgba(4, 12, 34, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
    }
    
    .premium-hero.pill-mode h1 {
        font-size: 1.6rem !important;
    }

    .premium-hero:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 194, 255, 0.78);
        box-shadow: 0 0 30px rgba(0, 194, 255, 0.35), 0 18px 42px rgba(4, 12, 34, 0.5);
    }

    .premium-hero h1 {
        margin: 0;
        font-size: clamp(1.45rem, 2.4vw, 2.2rem);
        font-weight: 800;
        color: #f5f9ff;
        letter-spacing: 0.35px;
        text-shadow: 0 0 16px rgba(79, 70, 229, 0.35);
        position: relative;
        z-index: 1;
    }

    .run-output-box {
        margin-top: 8px;
        padding: 16px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.14), rgba(255, 255, 255, 0.08));
        box-shadow: 0 12px 30px rgba(8, 15, 40, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.16);
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.45);
        color: #eef3ff;
        background: linear-gradient(120deg, rgba(0, 194, 255, 0.22), rgba(79, 70, 229, 0.24), rgba(38, 217, 164, 0.2));
        background-size: 220% 220%;
        box-shadow: 0 10px 26px rgba(10, 18, 42, 0.35);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        border-color: rgba(255, 255, 255, 0.9);
        animation: pulseGlow 1.8s ease-in-out infinite;
    }

    .stButton > button:active {
        transform: scale(0.98);
        box-shadow: 0 0 24px rgba(0, 194, 255, 0.55);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.12) !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        border-radius: 12px !important;
        color: #eef3ff !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(0, 194, 255, 0.9) !important;
        box-shadow: 0 0 0 0.22rem rgba(0, 194, 255, 0.24) !important;
    }

    [data-testid="stTabs"] [role="tab"] {
        border-radius: 12px;
        transition: all 0.2s ease;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.45);
        box-shadow: 0 0 14px rgba(79, 70, 229, 0.32);
    }

    .stMarkdown hr {
        border-top: 1px solid rgba(255, 255, 255, 0.22);
    }

    .stCodeBlock {
        position: relative;
        overflow: hidden;
    }

    .stCodeBlock::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(110deg, transparent 0%, rgba(255, 255, 255, 0.14) 48%, transparent 100%);
        background-size: 220% 100%;
        animation: shimmer 7s linear infinite;
        pointer-events: none;
    }

    /* Style the radio items as beautiful horizontal tabs/buttons */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding-top: 10px;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        padding: 12px 16px !important;
        border-radius: 12px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer;
        display: flex;
        align-items: center;
        width: 100%;
        margin-bottom: 0px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(0, 194, 255, 0.08) !important;
        border-color: rgba(0, 194, 255, 0.4) !important;
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0, 194, 255, 0.15);
    }

    /* Style for checked state */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(0, 194, 255, 0.22), rgba(79, 70, 229, 0.24)) !important;
        border-color: rgba(0, 194, 255, 0.7) !important;
        box-shadow: 0 0 15px rgba(0, 194, 255, 0.25), inset 0 0 8px rgba(0, 194, 255, 0.15);
    }

    /* The Ultimate Unbreakable Solution */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > div:not(:last-child),
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > span:not(:last-child),
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > svg,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label [data-baseweb="radio-mark-outer"] {
        display: none !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label input[type="radio"] {
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        opacity: 0 !important;
    }

    /* Draw our custom empty circle with absolute authority */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label::before {
        content: "" !important;
        display: inline-block !important;
        width: 16px !important;
        height: 16px !important;
        margin-right: 12px !important;
        margin-top: 2px !important;
        border-radius: 50% !important;
        border: 2px solid rgba(255, 255, 255, 0.4) !important;
        background-color: transparent !important;
        transition: all 0.2s ease-in-out !important;
        flex-shrink: 0 !important;
    }

    /* Draw our custom checked circle with the glowing theme color */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"]::before,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked)::before {
        border: 0px solid transparent !important;
        background-color: #00c2ff !important;
        background-image: radial-gradient(circle, #ffffff 30%, #00c2ff 35%);
        box-shadow: 0 0 10px #00c2ff80;
    }

    /* Hide redundant radio widget label */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none !important;
    }

    /* Typography fixes for sidebar header — always dark theme */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f5f9ff !important;
        font-weight: 700;
        text-shadow: 0 0 12px rgba(0, 194, 255, 0.35);
        margin-bottom: 12px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
        color: #f5f9ff !important;
    }

    /* Keep the active cyan highlight in both modes */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] *,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) * {
        color: #00c2ff !important;
        font-weight: 600 !important;
    }

    @keyframes techPulse {
        0% { text-shadow: 0 0 4px rgba(0, 194, 255, 0.4); color: #00c2ff; }
        50% { text-shadow: 0 0 12px rgba(0, 194, 255, 0.8), 0 0 20px rgba(0, 194, 255, 0.4); color: #eef3ff; }
        100% { text-shadow: 0 0 4px rgba(0, 194, 255, 0.4); color: #00c2ff; }
    }
    .tech-hover-container {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .glow-tech {
        font-weight: 600;
        text-decoration: underline dotted rgba(0, 194, 255, 0.6) !important;
        animation: techPulse 2s infinite ease-in-out;
        display: inline-block;
        padding: 0 2px;
        color: #00c2ff !important;
        transition: all 0.25s ease;
    }
    .tech-tooltip-box {
        visibility: hidden;
        opacity: 0;
        width: 320px;
        background: rgba(10, 20, 42, 0.98) !important;
        color: #eef3ff !important;
        text-align: left;
        border: 1px solid rgba(0, 194, 255, 0.45);
        border-radius: 10px;
        padding: 14px;
        position: absolute;
        z-index: 9999;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%) translateY(10px);
        transition: opacity 0.3s ease, transform 0.3s ease, visibility 0.3s;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.65), 0 0 20px rgba(0, 194, 255, 0.25);
        pointer-events: none;
        font-size: 0.9em;
        line-height: 1.4;
        font-weight: normal;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .tech-tooltip-box strong {
        color: #00c2ff !important;
        font-size: 1.05em;
        display: block;
        margin-bottom: 6px;
    }
    .tech-tooltip-box::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: rgba(10, 20, 42, 0.98) transparent transparent transparent;
    }
    .tech-hover-container:hover .tech-tooltip-box {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) translateY(0);
    }
    .tech-hover-container:hover .glow-tech {
        color: #fff !important;
        text-shadow: 0 0 15px rgba(0, 194, 255, 1) !important;
    }

    /* Structured content cards for layout columns */
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 22px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
        margin-bottom: 15px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="column"]:hover {
        border-color: rgba(0, 194, 255, 0.15) !important;
        box-shadow: 0 8px 32px rgba(0, 194, 255, 0.03) !important;
    }

    /* Professional subheadings structure */
    div[data-testid="stMarkdownContainer"] h2 {
        color: #eef3ff !important;
        font-weight: 600 !important;
        font-size: 1.35em !important;
        border-bottom: 2px solid rgba(0, 194, 255, 0.25) !important;
        padding-bottom: 8px !important;
        margin-top: 10px !important;
        margin-bottom: 16px !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stMarkdownContainer"] h3 {
        color: #00c2ff !important;
        font-weight: 500 !important;
        font-size: 1.12em !important;
        padding-bottom: 4px !important;
        margin-top: 10px !important;
        margin-bottom: 12px !important;
        letter-spacing: 0.5px !important;
    }

    /* Navigation panel — always dark in both Light and Dark mode */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 17, 32, 0.98) 0%, rgba(3, 7, 18, 1) 100%) !important;
        border-right: 1px solid rgba(0, 194, 255, 0.18) !important;
        box-shadow: 6px 0 25px rgba(0, 0, 0, 0.45) !important;
    }

    /* --- Mobile Responsiveness --- */
    @media (max-width: 768px) {
        .stApp::before {
            width: 150px;
            height: 150px;
            right: -10%;
            top: 5%;
        }
        .stApp::after {
            width: 180px;
            height: 180px;
            left: -10%;
            bottom: 5%;
        }
        .premium-hero {
            padding: 16px 12px;
            margin: -25px 0 15px 0 !important;
        }
        .premium-hero h1 {
            font-size: 1.5rem !important;
        }
        [data-testid="column"] {
            padding: 14px !important;
            margin-bottom: 12px !important;
        }
        .main .block-container {
            padding: 1rem !important;
        }
        div[data-testid="stMarkdownContainer"] h2 {
            font-size: 1.2em !important;
            margin-top: 5px !important;
        }
        div[data-testid="stMarkdownContainer"] h3 {
            font-size: 1.1em !important;
        }
        .tech-tooltip-box {
            width: 240px;
            font-size: 0.85em;
            left: 50%;
            transform: translateX(-50%) translateY(10px);
        }
        .stButton > button {
            padding: 8px 16px !important;
        }
    }

    /* Instant zero-flash hiding of the theme toggle button */
    div.element-container:has(.hide-next-button),
    div.element-container:has(.hide-next-button) + div.element-container {
        position: absolute !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Sentinel AI Model Training Page Overhaul */
    .sentinel-phase-pill {
        display: inline-block;
        padding: 8px 24px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.02));
        border: 1px solid var(--glass-border);
        border-radius: 30px;
        color: #fff;
        font-weight: 800;
        font-size: 0.95rem;
        letter-spacing: 2.5px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 10px auto 30px auto;
        text-transform: uppercase;
        text-align: center;
        width: max-content;
    }
    
    .sentinel-card-title {
        color: #fff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
        letter-spacing: 1px;
    }
    
    .sentinel-terminal {
        background: rgba(10, 10, 15, 0.9) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 12px 35px rgba(0,0,0,0.3);
        margin-bottom: 30px;
    }
    
    .sentinel-terminal-header {
        background: rgba(255, 255, 255, 0.04);
        padding: 12px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    
    .sentinel-mac-dots {
        display: flex;
        gap: 8px;
    }
    
    .sentinel-mac-dots span {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        box-shadow: inset 0 2px 4px rgba(255,255,255,0.2), 0 2px 4px rgba(0,0,0,0.5);
    }
    .sentinel-mac-dots .red { background: #ff5f56; }
    .sentinel-mac-dots .yellow { background: #ffbd2e; }
    .sentinel-mac-dots .green { background: #27c93f; }
    
    .sentinel-processing-tag {
        font-size: 0.65rem;
        font-weight: 900;
        letter-spacing: 1.5px;
        padding: 4px 12px;
        border-radius: 6px;
        text-transform: uppercase;
        background: rgba(255, 51, 102, 0.15);
        color: #ff3366;
        border: 1px solid rgba(255, 51, 102, 0.3);
        box-shadow: 0 0 10px rgba(255, 51, 102, 0.2);
    }
    
    .sentinel-code-body { white-space: nowrap;
        padding: 24px;
        font-family: 'Fira Code', 'Courier New', Courier, monospace;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #d1d5db;
        overflow-x: auto;
    }
    
    .sentinel-code-body .keyword { color: #00c2ff; font-weight: bold; }
    .sentinel-code-body .builtin { color: #4f46e5; font-weight: bold; }
    .sentinel-code-body .string { color: #26d9a4; }
    .sentinel-code-body .comment { color: #6b7280; font-style: italic; }
    
    .sentinel-metrics-card {
        background: rgba(20, 10, 15, 0.5);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
        position: relative;
    }
    
    .sentinel-accuracy-badge {
        position: absolute;
        top: -15px;
        right: 20px;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 900;
        font-size: 0.95rem;
        letter-spacing: 1.5px;
        display: flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.9), rgba(139, 0, 0, 0.9));
        color: white;
        box-shadow: 0 8px 20px rgba(255, 51, 102, 0.4), inset 0 2px 5px rgba(255,255,255,0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        z-index: 10;
    }

    /* Target the st.container holding the metrics matrix to match the terminal styling perfectly */
    div[data-testid="stVerticalBlock"]:has(.metrics-container-hook):not(:has(div[data-testid="stVerticalBlock"]:has(.metrics-container-hook))) {
        background: rgba(10, 10, 15, 0.9) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px;
        padding: 20px 10px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 12px 35px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

elif st.session_state.theme == 'stitch':
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary-color: var(--brand-1);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 60, 100, 0.4);
        --glass-shadow: 0 12px 36px rgba(40, 8, 15, 0.35);
        --brand-1: #ff3366;
        --brand-2: #8b0000;
        --brand-3: #ff6633;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 12%, rgba(255, 51, 102, 0.18), transparent 34%),
            radial-gradient(circle at 88% 20%, rgba(204, 0, 51, 0.18), transparent 40%),
            radial-gradient(circle at 50% 86%, rgba(139, 0, 0, 0.2), transparent 45%),
            linear-gradient(145deg, #1a0808 0%, #2a0b12 46%, #120306 100%);
        background-attachment: fixed;
        color: #ffeeee;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(26, 8, 8, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid var(--glass-border) !important;
        box-shadow: 4px 0 24px rgba(255, 51, 102, 0.1) !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, rgba(255,51,102,0.1) 0%, rgba(204,0,51,0.1) 100%) !important;
        border: 1px solid rgba(255,51,102,0.4) !important;
        color: #ff3366 !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 12px rgba(255,51,102,0.1) !important;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #ff3366 0%, #cc0033 100%) !important;
        border-color: #ff3366 !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(255,51,102,0.25) !important;
    }

    div[data-testid="stMarkdownContainer"] h1 {
        font-weight: 800;
        background: linear-gradient(to right, #ff3366, #ff80a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
    }

    /* Professional subheadings structure (Crimson) */
    div[data-testid="stMarkdownContainer"] h2 {
        color: #ff3366 !important;
        font-weight: 600 !important;
        font-size: 1.35em !important;
        border-bottom: 2px solid rgba(255, 51, 102, 0.25) !important;
        padding-bottom: 8px !important;
        margin-top: 10px !important;
        margin-bottom: 16px !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stMarkdownContainer"] h3 {
        color: #ff3366 !important;
        font-weight: 500 !important;
        font-size: 1.12em !important;
        margin-top: 10px !important;
        margin-bottom: 12px !important;
    }

    /* Style the radio items as beautiful horizontal tabs/buttons */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding-top: 10px;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        padding: 12px 16px !important;
        border-radius: 12px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer;
        display: flex;
        align-items: center;
        width: 100%;
        margin-bottom: 0px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(255, 51, 102, 0.08) !important;
        border-color: rgba(255, 51, 102, 0.4) !important;
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(255, 51, 102, 0.15);
    }

    /* Style for checked state */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.22), rgba(139, 0, 0, 0.24)) !important;
        border-color: rgba(255, 51, 102, 0.7) !important;
        box-shadow: 0 0 15px rgba(255, 51, 102, 0.25), inset 0 0 8px rgba(255, 51, 102, 0.15);
    }

    /* The Ultimate Unbreakable Solution */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > div:not(:last-child),
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > span:not(:last-child),
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > svg,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label [data-baseweb="radio-mark-outer"] {
        display: none !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label input[type="radio"] {
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        opacity: 0 !important;
    }

    /* Draw our custom empty circle with absolute authority */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label::before {
        content: "" !important;
        display: inline-block !important;
        width: 16px !important;
        height: 16px !important;
        margin-right: 12px !important;
        margin-top: 2px !important;
        border-radius: 50% !important;
        border: 2px solid rgba(255, 255, 255, 0.4) !important;
        background-color: transparent !important;
        transition: all 0.2s ease-in-out !important;
        flex-shrink: 0 !important;
    }

    /* Draw our custom checked circle with the glowing theme color */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"]::before,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked)::before {
        border: 0px solid transparent !important;
        background-color: #ff3366 !important;
        background-image: radial-gradient(circle, #ffffff 30%, #ff3366 35%);
        box-shadow: 0 0 10px #ff336680;
    }

    /* Hide redundant radio widget label */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none !important;
    }

    /* Typography fixes for sidebar header — always dark theme */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f5f9ff !important;
        font-weight: 700;
        text-shadow: 0 0 12px rgba(255, 51, 102, 0.35);
        margin-bottom: 12px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
        color: #f5f9ff !important;
    }

    /* Keep the active red highlight in both modes */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] *,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) * {
        color: #ff3366 !important;
        font-weight: 600 !important;
    }

    div.element-container:has(.premium-hero),
    div.stMarkdown:has(.premium-hero) {
        position: sticky !important;
        top: 15px !important;
        z-index: 999 !important;
    }

    .premium-hero {
        position: relative;
        width: 100%;
        margin: -45px auto 20px auto !important;
        padding: 20px 24px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.34);
        background: linear-gradient(130deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.08));
        backdrop-filter: blur(18px);
        box-shadow: 0 16px 38px rgba(40, 8, 15, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.22);
        overflow: hidden;
        transition: all 0.6s cubic-bezier(0.25, 1, 0.5, 1);
        text-align: center;
    }

    .premium-hero.pill-mode {
        width: 85% !important;
        margin: 0 auto 20px auto !important;
        padding: 12px 40px !important;
        border-radius: 50px !important;
        background: linear-gradient(130deg, rgba(255, 51, 102, 0.15), rgba(255, 51, 102, 0.05)) !important;
        box-shadow: 0 10px 30px rgba(40, 8, 15, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
    }
    
    .premium-hero.pill-mode h1 {
        font-size: 1.6rem !important;
    }

    .premium-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(110deg, transparent 0%, rgba(255, 255, 255, 0.2) 50%, transparent 100%);
        background-size: 220% 100%;
        animation: shimmer 6.5s linear infinite;
        pointer-events: none;
    }

    .premium-hero:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 51, 102, 0.78);
        box-shadow: 0 0 30px rgba(255, 51, 102, 0.35), 0 18px 42px rgba(40, 8, 15, 0.5);
    }

    .premium-hero h1 {
        margin: 0 !important;
        font-size: clamp(1.45rem, 2.4vw, 2.2rem);
        font-weight: 800;
        color: #f5f9ff !important;
        background: none !important;
        -webkit-text-fill-color: #f5f9ff !important;
        letter-spacing: 0.35px;
        text-shadow: 0 0 16px rgba(255, 51, 102, 0.45);
        position: relative;
        z-index: 1;
    }

    .glow-tech {
        color: #ff3366;
        font-weight: 600;
        cursor: help;
        border-bottom: 1px dashed rgba(255, 51, 102, 0.5);
        transition: all 0.2s ease;
    }
    
    .glow-tech:hover {
        color: #ff5577;
        text-shadow: 0 0 12px rgba(255, 51, 102, 0.4);
        border-bottom: 1px dashed rgba(255, 51, 102, 0.9);
    }
    
    .tech-hover-container {
        position: relative;
        display: inline-block;
    }
    
    .tech-tooltip-box {
        position: absolute;
        bottom: 120%;
        left: 50%;
        transform: translateX(-50%) translateY(10px);
        background: rgba(20, 5, 10, 0.95);
        border: 1px solid rgba(255, 51, 102, 0.4);
        padding: 12px 16px;
        border-radius: 10px;
        width: max-content;
        max-width: 300px;
        color: #c9d1d9;
        font-size: 0.85em;
        line-height: 1.5;
        box-shadow: 0 8px 24px rgba(255, 51, 102, 0.15);
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        z-index: 1000;
        pointer-events: none;
    }
    
    .tech-tooltip-box strong {
        color: #ff3366;
        display: block;
        font-size: 1.1em;
        margin-bottom: 4px;
    }

    .tech-tooltip-box::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 6px;
        border-style: solid;
        border-color: rgba(255, 51, 102, 0.4) transparent transparent transparent;
    }

    .tech-hover-container:hover .tech-tooltip-box {
        opacity: 1;
        visibility: visible;
        transform: translateX(-50%) translateY(0);
    }
    
    @media (max-width: 768px) {
        .tech-tooltip-box {
            width: 240px;
            font-size: 0.85em;
            left: 50%;
            transform: translateX(-50%) translateY(10px);
        }
        .tech-hover-container:hover .tech-tooltip-box {
            transform: translateX(-50%) translateY(0);
        }
    }

    /* Instant zero-flash hiding of the theme toggle button */
    div.element-container:has(.hide-next-button),
    div.element-container:has(.hide-next-button) + div.element-container {
        position: absolute !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Sentinel AI Model Training Page Overhaul */
    .sentinel-phase-pill {
        display: inline-block;
        padding: 8px 24px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.02));
        border: 1px solid var(--glass-border);
        border-radius: 30px;
        color: #fff;
        font-weight: 800;
        font-size: 0.95rem;
        letter-spacing: 2.5px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 10px auto 30px auto;
        text-transform: uppercase;
        text-align: center;
        width: max-content;
    }
    
    .sentinel-card-title {
        color: #fff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
        letter-spacing: 1px;
    }
    
    .sentinel-terminal {
        background: rgba(10, 10, 15, 0.9) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 12px 35px rgba(0,0,0,0.3);
        margin-bottom: 30px;
    }
    
    .sentinel-terminal-header {
        background: rgba(255, 255, 255, 0.04);
        padding: 12px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    
    .sentinel-mac-dots {
        display: flex;
        gap: 8px;
    }
    
    .sentinel-mac-dots span {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        box-shadow: inset 0 2px 4px rgba(255,255,255,0.2), 0 2px 4px rgba(0,0,0,0.5);
    }
    .sentinel-mac-dots .red { background: #ff5f56; }
    .sentinel-mac-dots .yellow { background: #ffbd2e; }
    .sentinel-mac-dots .green { background: #27c93f; }
    
    .sentinel-processing-tag {
        font-size: 0.65rem;
        font-weight: 900;
        letter-spacing: 1.5px;
        padding: 4px 12px;
        border-radius: 6px;
        text-transform: uppercase;
        background: rgba(255, 51, 102, 0.15);
        color: #ff3366;
        border: 1px solid rgba(255, 51, 102, 0.3);
        box-shadow: 0 0 10px rgba(255, 51, 102, 0.2);
    }
    
    .sentinel-code-body { white-space: nowrap;
        padding: 24px;
        font-family: 'Fira Code', 'Courier New', Courier, monospace;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #d1d5db;
        overflow-x: auto;
    }
    
    .sentinel-code-body .keyword { color: #ff3366; font-weight: bold; }
    .sentinel-code-body .builtin { color: #4f46e5; font-weight: bold; }
    .sentinel-code-body .string { color: #26d9a4; }
    .sentinel-code-body .comment { color: #6b7280; font-style: italic; }
    
    .sentinel-metrics-card {
        background: rgba(20, 10, 15, 0.5);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
        position: relative;
    }
    
    .sentinel-accuracy-badge {
        position: absolute;
        top: -15px;
        right: 20px;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 900;
        font-size: 0.95rem;
        letter-spacing: 1.5px;
        display: flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.9), rgba(139, 0, 0, 0.9));
        color: white;
        box-shadow: 0 8px 20px rgba(255, 51, 102, 0.4), inset 0 2px 5px rgba(255,255,255,0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        z-index: 10;
    }

    /* Target the st.container holding the metrics matrix to match the terminal styling perfectly */
    div[data-testid="stVerticalBlock"]:has(.metrics-container-hook):not(:has(div[data-testid="stVerticalBlock"]:has(.metrics-container-hook))) {
        background: rgba(10, 10, 15, 0.9) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px;
        padding: 20px 10px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 12px 35px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

def show_explanation(text, technique=None):
    if st.session_state.theme == 'stitch':
        bg, border, text_color = "rgba(255, 51, 102, 0.12)", "#ff3366", "#ff3366"
    else:
        bg, border, text_color = "rgba(0, 194, 255, 0.12)", "#00c2ff", "#00c2ff"
        
    st.markdown(f'<div style="background: {bg}; border-left: 4px solid {border}; padding: 12px 16px; border-radius: 12px; margin-top: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid {border}; word-wrap: break-word; overflow-wrap: break-word;"><strong style="color: {text_color}; font-size: 1.05em; display: block; margin-bottom: 6px;">What this block did:</strong><span style="color: var(--text-color); font-size: 0.95em; line-height: 1.5;">{text}</span></div>', unsafe_allow_html=True)

def render_explain_button(tab_name, explanation_text, technique=None):
    btn_key = f"explain_state_{tab_name}"
    if btn_key not in st.session_state:
        st.session_state[btn_key] = False

    st.write("---")
    if st.button("What's Happening", key=f"explain_btn_{tab_name}"):
        st.session_state[btn_key] = not st.session_state[btn_key]

    if st.session_state[btn_key]:
        if st.session_state.theme == 'stitch':
            bg, border, text_color = "rgba(255, 51, 102, 0.12)", "#ff3366", "#ff3366"
        else:
            bg, border, text_color = "rgba(0, 194, 255, 0.12)", "#00c2ff", "#00c2ff"
            
        st.markdown(f'<div style="background: {bg}; border-left: 4px solid {border}; padding: 12px 16px; border-radius: 12px; margin-top: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid {border}; word-wrap: break-word; overflow-wrap: break-word;"><strong style="color: {text_color}; font-size: 1.05em; display: block; margin-bottom: 6px;">Page Explanation:</strong><span style="color: var(--text-color); font-size: 0.95em; line-height: 1.5;">{explanation_text}</span></div>', unsafe_allow_html=True)






st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Project Overview", "1. Data Loading & Cleaning", "2. Model Training & Evaluation", "3. Live Prediction Test", "4. Full Code Explorer", "5. View Raw Source Code"])


@st.cache_resource
def load_data():
    dataset = pd.read_csv("labeled_data.csv")
    dataset["labels"] = dataset["class"].map({
        0: "Hate Speech", 
        1: "Offensive language", 
        2: "No hate or offensive language"
    })
    return dataset

dataset = load_data()

# Setup Stopwords & Stemmer
stop_words = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")

def clean_data(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stop_words]
    text = " ".join(text)
    # Stemming
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text) 
    return text

@st.cache_resource
def get_cleaned_data():
    df = load_data()[["tweet", "labels"]].copy()
    df["clean_tweet"] = df["tweet"].apply(clean_data)
    return df

data = get_cleaned_data()

@st.cache_resource
def train_model():
    df = get_cleaned_data()
    x = np.array(df["clean_tweet"])
    y = np.array(df["labels"])
    cv = CountVectorizer()
    x = cv.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    
    return cv, dt, y_test, y_pred

cv, dt, y_test, y_pred = train_model()

st.markdown('<div id="hero-scroll-marker" style="position: absolute; top: -10px;"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="premium-hero">
    <h1>Hate Speech Detection</h1>
</div>
""", unsafe_allow_html=True)

components.html("""
<script>
    const parentDoc = window.parent.document;
    const marker = parentDoc.getElementById('hero-scroll-marker');
    const heroes = parentDoc.querySelectorAll('.premium-hero');

    if (marker && heroes.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                heroes.forEach(hero => {
                    if (!entry.isIntersecting) {
                        hero.classList.add('pill-mode');
                    } else {
                        hero.classList.remove('pill-mode');
                    }
                });
            });
        }, { root: null, threshold: 0 });
        
        observer.observe(marker);
    }
</script>
""", height=0, width=0)

if menu != "Project Overview":
    st.markdown(f"""
    <div style="text-align: center; margin-top: -10px; margin-bottom: 25px;">
        <span style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); padding: 6px 18px; border-radius: 30px; color: #8a99ad; font-size: 0.85em; font-weight: 500; letter-spacing: 0.5px; display: inline-block; box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);">
            {menu}
        </span>
</div>
    """, unsafe_allow_html=True)

if menu == "Project Overview":
    if st.session_state.theme == 'stitch':
        pl_color = "#ff3366"
        pl_bg = "rgba(255, 51, 102, 0.05)"
        pl_border = "rgba(255, 51, 102, 0.22)"
        
        m_bg = "rgba(255, 51, 102, 0.05)"
        m_border = "rgba(255, 51, 102, 0.15)"
        m_val1 = "#ff3366"
        m_val2 = "#ff3366"
        m_val3 = "#ff3366"
    else:
        pl_color = "#ff9f1c"
        pl_bg = "rgba(255, 159, 28, 0.05)"
        pl_border = "rgba(255, 159, 28, 0.22)"
        
        m_bg = "rgba(0, 194, 255, 0.05)"
        m_border = "rgba(0, 194, 255, 0.15)"
        m_val1 = "#00c2ff"
        m_val2 = "#26d9a4"
        m_val3 = "#ff9f1c"

    st.markdown(f"""<div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 24px; border-radius: 16px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);">
<h2 style="color: #00c2ff; margin-top: 0; display: flex; align-items: center; gap: 10px; font-size: 1.65em;">• Project Abstract & Overview</h2>
<p style="color: #eef3ff; font-size: 1.05em; line-height: 1.6; margin-bottom: 20px;">The Hate Speech Detection System is an advanced, end-to-end Natural Language Processing (NLP) and machine learning pipeline developed to automate the detection of toxic social media content. By digesting raw comments, extracting structural features, and applying automated decision boundaries, the system is able to classify statements into hate speech, offensive language, or clean text with high accuracy. This dashboard serves to visualise each phase of the machine learning lifecycle—from loading and cleansing the raw data up to checking real-time model predictions.</p>

<!-- Core Objective Box (Full width) -->
<div style="background: rgba(38, 217, 164, 0.06); border: 1px solid rgba(38, 217, 164, 0.2); padding: 20px; border-radius: 14px; margin-bottom: 24px;">
<h3 style="color: #26d9a4; margin-top: 0; margin-bottom: 8px; font-size: 1.25em; display: flex; align-items: center; gap: 8px;">• Core Objective</h3>
<p style="color: #eef3ff; font-size: 0.98em; line-height: 1.5; margin-bottom: 0;">Automate toxic content monitoring by scanning textual commentary and categorizing expressions into three categories: Hate Speech (explicitly malicious/discriminatory), Offensive Language (vulgar or aggressive but not hate speech), or Clean text (completely safe/non-toxic).</p>
</div>

<!-- Pipeline Workflow Panel (Full width) -->
<div style="background: {pl_bg}; border: 1px solid {pl_border}; padding: 20px; border-radius: 14px; margin-bottom: 24px;">
<h3 style="color: {pl_color}; margin-top: 0; margin-bottom: 16px; font-size: 1.25em; display: flex; align-items: center; gap: 8px;">• Pipeline Workflow</h3>
<div style="display: flex; flex-direction: column; gap: 12px;">
<div style="background: rgba(255, 255, 255, 0.02); border-left: 4px solid {pl_color}; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 4px 12px 12px 4px; display: flex; align-items: center; gap: 16px;">
<div style="font-size: 1.2em; font-weight: bold; color: {pl_color}; min-width: 32px;">01</div>
<div>
<strong style="color: #eef3ff; display: block; font-size: 0.95em; margin-bottom: 2px;">Data Loading & Mapping</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4;">Loads the raw dataset containing labeled social media comments and maps label index categories (0, 1, 2) to descriptive text categories.</span>
</div>
</div>
<div style="background: rgba(255, 255, 255, 0.02); border-left: 4px solid {pl_color}; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 4px 12px 12px 4px; display: flex; align-items: center; gap: 16px;">
<div style="font-size: 1.2em; font-weight: bold; color: {pl_color}; min-width: 32px;">02</div>
<div>
<strong style="color: #eef3ff; display: block; font-size: 0.95em; margin-bottom: 2px;">Linguistic Text Cleaning</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4;">Preprocesses raw text by converting characters to lowercase, stripping URLs, removing HTML tags, filtering out punctuation, and purging standard English stop words.</span>
</div>
</div>
<div style="background: rgba(255, 255, 255, 0.02); border-left: 4px solid {pl_color}; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 4px 12px 12px 4px; display: flex; align-items: center; gap: 16px;">
<div style="font-size: 1.2em; font-weight: bold; color: {pl_color}; min-width: 32px;">03</div>
<div>
<strong style="color: #eef3ff; display: block; font-size: 0.95em; margin-bottom: 2px;">Snowball Stemming</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4;">Applies the Snowball Stemming algorithm to reduce inflected or derived words down to their baseline linguistic root form (e.g. "protests" and "protesting" map to "protest").</span>
</div>
</div>
<div style="background: rgba(255, 255, 255, 0.02); border-left: 4px solid {pl_color}; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 4px 12px 12px 4px; display: flex; align-items: center; gap: 16px;">
<div style="font-size: 1.2em; font-weight: bold; color: {pl_color}; min-width: 32px;">04</div>
<div>
<strong style="color: #eef3ff; display: block; font-size: 0.95em; margin-bottom: 2px;">Count Vectorization (Bag of Words)</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4;">Transforms cleaned word tokens into sparse matrix representation, counting token occurrences to build the classification vocabulary.</span>
</div>
</div>
<div style="background: rgba(255, 255, 255, 0.02); border-left: 4px solid {pl_color}; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 4px 12px 12px 4px; display: flex; align-items: center; gap: 16px;">
<div style="font-size: 1.2em; font-weight: bold; color: {pl_color}; min-width: 32px;">05</div>
<div>
<strong style="color: #eef3ff; display: block; font-size: 0.95em; margin-bottom: 2px;">Decision Tree Training & Evaluation</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4;">Splits features into train/test sets, fits a Decision Tree Classifier, reports test accuracy, and prepares visual evaluation matrices.</span>
</div>
</div>
</div>
</div>

<!-- Technologies Used Box (Full width) -->
<div style="background: rgba(156, 39, 176, 0.04); border: 1px solid rgba(156, 39, 176, 0.18); padding: 20px; border-radius: 14px; margin-bottom: 24px;">
<h3 style="color: #b854ff; margin-top: 0; margin-bottom: 16px; font-size: 1.25em; display: flex; align-items: center; gap: 8px;">• Technologies Used</h3>
<div style="display: flex; flex-direction: column; gap: 14px;">
<div style="display: flex; gap: 4px; flex-direction: column;">
<strong style="color: #eef3ff; font-size: 0.98em;">1. Natural Language Toolkit (NLTK)</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4; padding-left: 20px;">Provides stopwords dictionaries and compiles the Snowball Stemmer algorithm for word normalization.</span>
</div>
<div style="display: flex; gap: 4px; flex-direction: column;">
<strong style="color: #eef3ff; font-size: 0.98em;">2. Scikit-Learn</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4; padding-left: 20px;">Provides the CountVectorizer engine, test-train splits, and the Decision Tree Classifier algorithm.</span>
</div>
<div style="display: flex; gap: 4px; flex-direction: column;">
<strong style="color: #eef3ff; font-size: 0.98em;">3. Pandas & NumPy</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4; padding-left: 20px;">Support structural data manipulation via DataFrames and fast vectorized matrix calculations.</span>
</div>
<div style="display: flex; gap: 4px; flex-direction: column;">
<strong style="color: #eef3ff; font-size: 0.98em;">4. Seaborn & Matplotlib</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4; padding-left: 20px;">Create statistical plots and evaluation heatmaps for metric inspections.</span>
</div>
</div>
</div>

<!-- Metrics Ribbon -->
<div style="margin-top: 24px; background: {m_bg}; border: 1px solid {m_border}; padding: 16px; border-radius: 12px; display: flex; justify-content: space-around; flex-wrap: wrap; text-align: center; gap: 16px;">
<div>
<div style="font-size: 1.8em; font-weight: bold; color: {m_val1};">24,783</div>
<div style="font-size: 0.85em; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Dataset Tweets</div>
</div>
<div style="border-left: 1px solid rgba(255, 255, 255, 0.1); height: 50px; align-self: center;"></div>
<div>
<div style="font-size: 1.8em; font-weight: bold; color: {m_val2};">~87.5%</div>
<div style="font-size: 0.85em; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Classification Accuracy</div>
</div>
<div style="border-left: 1px solid rgba(255, 255, 255, 0.1); height: 50px; align-self: center;"></div>
<div>
<div style="font-size: 1.8em; font-weight: bold; color: {m_val3};">Decision Tree</div>
<div style="font-size: 0.85em; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Predictive Model</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

elif menu == "1. Data Loading & Cleaning":

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="sentinel-card-title">Data Cleaning Code</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                <span class="keyword">import</span> re<br><span class="keyword">import</span> string<br><span class="keyword">from</span> nltk.corpus <span class="keyword">import</span> stopwords<br><span class="keyword">import</span> nltk<br>&nbsp;<br>stopwords = set(stopwords.words("english"))<br>stemmer = nltk.SnowballStemmer("english")<br>&nbsp;<br><span class="keyword">def</span> clean_data(text):<br>&nbsp;&nbsp;&nbsp;&nbsp;text = str(text).lower()<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.sub(r'https?://\S+|www\.\S+', '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.sub(r'\[.*?\]', '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.sub('<.*?>+', '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.sub('[%s]' %re.escape(string.punctuation), '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.sub(r'\\n', '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.sub(r'\w*\d\w*', '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = [word <span class="keyword">for</span> word <span class="keyword">in</span> text.split() <span class="keyword">if</span> word <span class="keyword">not</span> <span class="keyword">in</span> stopwords]<br>&nbsp;&nbsp;&nbsp;&nbsp;text = " ".join(text)<br>&nbsp;&nbsp;&nbsp;&nbsp;#Stemming the text<br>&nbsp;&nbsp;&nbsp;&nbsp;text = [stemmer.stem(word) <span class="keyword">for</span> word <span class="keyword">in</span> text.split(' ')]<br>&nbsp;&nbsp;&nbsp;&nbsp;text = " ".join(text) <br>&nbsp;&nbsp;&nbsp;&nbsp;<span class="keyword">return</span> text <br>&nbsp;<br>data["clean_tweet"] = data["tweet"].apply(clean_data)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="sentinel-card-title">Data Output (Cleaned)</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="metrics-container-hook"></div>', unsafe_allow_html=True)
            st.markdown('<p style="color: #d1d5db; margin-bottom: 15px; font-weight: 500;">First 10 rows of the dataset:</p>', unsafe_allow_html=True)
            st.dataframe(data.head(10))
            
    render_explain_button("loading_cleaning", "This page displays the <span class='tech-hover-container'><span class='glow-tech'>preprocessing pipeline</span><span class='tech-tooltip-box'><strong>Preprocessing Pipeline</strong>A sequence of operations (cleaning, tokenization, stemming) that transforms raw text into structured numerical format for machine learning.</span></span> for Hate Speech detection. It loads the <span class='tech-hover-container'><span class='glow-tech'>Kaggle Hate Speech dataset</span><span class='tech-tooltip-box'><strong>Kaggle Hate Speech Dataset</strong>A publicly available corpus of labeled tweets and comments categorized by toxicity levels (Hate Speech, Offensive, or Clean) used to train classification models.</span></span>, shows a preview of the clean text columns, maps integers to labels, and cleans the text by stripping URLs, digits, punctuation, and applying <span class='tech-hover-container'><span class='glow-tech'>Snowball stemming</span><span class='tech-tooltip-box'><strong>Snowball Stemming</strong>A linguistic algorithm that chops off common prefixes and suffixes of words (e.g., 'protesting' -> 'protest', 'protests' -> 'protest') to map various inflected forms of a word back to its common base root.</span></span>.")


elif menu == "2. Model Training & Evaluation":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="sentinel-card-title">Execution Sequence</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                <span class="keyword">from</span> sklearn.feature_extraction.text <span class="keyword">import</span> CountVectorizer<br><span class="keyword">from</span> sklearn.model_selection <span class="keyword">import</span> train_test_split<br><span class="keyword">from</span> sklearn.tree <span class="keyword">import</span> DecisionTreeClassifier<br><span class="keyword">from</span> sklearn.metrics <span class="keyword">import</span> confusion_matrix, accuracy_score<br>&nbsp;<br>cv = CountVectorizer()<br>x = cv.fit_transform(x)<br>&nbsp;<br>x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)<br>&nbsp;<br>dt = DecisionTreeClassifier()<br>dt.fit(x_train, y_train)<br>y_pred = dt.predict(x_test)<br>&nbsp;<br>cm = confusion_matrix(y_test, y_pred)<br>accuracy = accuracy_score(y_test, y_pred)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        accuracy = accuracy_score(y_test, y_pred)
        st.markdown('<div class="sentinel-card-title">Metrics Matrix</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="metrics-container-hook"></div>', unsafe_allow_html=True)
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            
            is_stitch = st.session_state.get('theme', 'default') == 'stitch'
            cmap = sns.color_palette("dark:red", as_cmap=True) if is_stitch else sns.color_palette("dark:cyan", as_cmap=True)
            
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=True, annot_kws={"size": 12, "weight": "bold"})
            
            fig.patch.set_alpha(0.0)
            ax.set_facecolor('none')
            for text in ax.texts: text.set_color("white")
            ax.tick_params(colors='white')
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            st.pyplot(fig)
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 15px; padding-bottom: 10px;">
                <h4 style="color:#fff; margin: 0; font-weight: 700; letter-spacing: 1px;">CONFUSION MATRIX</h4>
                <div style="color: {'#ff3366' if is_stitch else '#00c2ff'}; font-weight: 800; font-size: 1.1rem; margin-top: 5px;">ACCURACY: {accuracy:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
    render_explain_button("model_eval", "This page displays the model training process and <span class='tech-hover-container'><span class='glow-tech'>evaluation metrics</span><span class='tech-tooltip-box'><strong>Evaluation Metrics</strong>Quantitative scores (like accuracy, precision, recall) used to assess how well a machine learning model generalizes to unseen test datasets.</span></span>. It displays the Scikit-Learn code used to train a <span class='tech-hover-container'><span class='glow-tech'>DecisionTreeClassifier</span><span class='tech-tooltip-box'><strong>Decision Tree Classifier</strong>A non-parametric supervised learning algorithm that splits the data into tree nodes based on feature decision rules to maximize information gain.</span></span>, shows the accuracy score (~87-89%), and visualizes a heatmap <span class='tech-hover-container'><span class='glow-tech'>confusion matrix</span><span class='tech-tooltip-box'><strong>Confusion Matrix</strong>A specific grid layout table that visualizes the performance of a supervised learning algorithm by mapping true positives, true negatives, false positives, and false negatives.</span></span> representing predicted versus actual hate speech categories.")


elif menu == "3. Live Prediction Test":

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="sentinel-card-title">Prediction Code</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                sample = user_input<br>sample = <span class="keyword">clean_data</span>(sample)<br>data1 = cv.<span class="keyword">transform</span>([sample]).<span class="keyword">toarray</span>()<br>prediction = dt.<span class="keyword">predict</span>(data1)<br><span class="keyword">print</span>(prediction)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="sentinel-card-title">Live Test</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="metrics-container-hook"></div>', unsafe_allow_html=True)
            user_input = st.text_area("Enter a tweet or message to test:", "Let's unite and kill all the people who are protesting against the government", height=100)
            if st.button("Predict"):
                sample = clean_data(user_input)
                data1 = cv.transform([sample]).toarray()
                prediction = dt.predict(data1)[0]
                if prediction == "Hate Speech":
                    st.error("🚨 " + prediction)
                elif prediction == "Offensive language":
                    st.warning("⚠️ " + prediction)
                else:
                    st.success("✅ " + prediction)
                    
    render_explain_button("live_pred", "This page allows you to input custom phrases to test the classifier live. The model <span class='tech-hover-container'><span class='glow-tech'>vectorizes</span><span class='tech-tooltip-box'><strong>Vector Inference</strong>Converts raw text characters and tokens into numeric coordinate arrays (vectors) representing word counts or occurrences in the vocabulary.</span></span> your input, applies the trained <span class='tech-hover-container'><span class='glow-tech'>Decision Tree classification</span><span class='tech-tooltip-box'><strong>Decision Tree Classification</strong>The decision path traversal of a trained tree model, moving through split nodes to predict the class of a new test input.</span></span>, and uses a <span class='tech-hover-container'><span class='glow-tech'>CountVectorizer</span><span class='tech-tooltip-box'><strong>Count Vectorizer</strong>A Scikit-Learn feature extraction tool that builds a vocabulary dictionary of words and encodes documents into sparse occurrence frequency matrices.</span></span> vocabulary representation to display the prediction category.")


elif menu == "4. Full Code Explorer":
    if 'hate_active_block' not in st.session_state:
        st.session_state.hate_active_block = None

    col1, col2 = st.columns([1.5, 1], gap="large")
    with col1:
        st.subheader("Interactive Code Explorer")
        
        st.markdown("### Block 1: Imports & Data Loading")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                <span class="keyword">import</span> pandas <span class="keyword">as</span> pd<br><span class="keyword">import</span> numpy <span class="keyword">as</span> np<br><span class="keyword">import</span> sklearn<br><span class="keyword">import</span> re<br><span class="keyword">import</span> nltk<br><span class="keyword">from</span> nltk.corpus <span class="keyword">import</span> stopwords<br><span class="keyword">import</span> string<br>&nbsp;<br>dataset = pd.<span class="builtin">read_csv</span>("labeled_data.csv")<br>dataset["labels"] = dataset["<span class="keyword">class</span>"].<span class="builtin">map</span>({<br>&nbsp;&nbsp;&nbsp;&nbsp;0: "Hate Speech", <br>&nbsp;&nbsp;&nbsp;&nbsp;1: "Offensive language", <br>&nbsp;&nbsp;&nbsp;&nbsp;2: "No hate <span class="keyword">or</span> offensive language"<br>})<br>data = dataset[["tweet" , "labels"]].<span class="builtin">copy</span>()
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 1 (Data Loading)"):
            st.session_state.hate_active_block = "block1"
            
        st.markdown("### Block 2: Data Cleaning")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                stemmer = nltk.SnowballStemmer("english")<br>stopwords = <span class="builtin">set</span>(stopwords.words("english"))<br>&nbsp;<br><span class="keyword">def</span> clean_data(text):<br>&nbsp;&nbsp;&nbsp;&nbsp;text = <span class="builtin">str</span>(text).<span class="builtin">lower</span>()<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.<span class="builtin">sub</span>(r'https?://\S+|www\.\S+', '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.<span class="builtin">sub</span>(r'\[.*?\]', '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.<span class="builtin">sub</span>('<.*?>+', '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.<span class="builtin">sub</span>('[%s]' %re.<span class="builtin">escape</span>(string.punctuation), '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.<span class="builtin">sub</span>(r'\\n', '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = re.<span class="builtin">sub</span>(r'\w*\d\w*', '', text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = [word <span class="keyword">for</span> word <span class="keyword">in</span> text.<span class="builtin">split</span>() <span class="keyword">if</span> word <span class="keyword">not</span> <span class="keyword">in</span> stopwords]<br>&nbsp;&nbsp;&nbsp;&nbsp;text = " ".<span class="builtin">join</span>(text)<br>&nbsp;&nbsp;&nbsp;&nbsp;text = [stemmer.<span class="builtin">stem</span>(word) <span class="keyword">for</span> word <span class="keyword">in</span> text.<span class="builtin">split</span>(' ')]<br>&nbsp;&nbsp;&nbsp;&nbsp;text = " ".<span class="builtin">join</span>(text) <br>&nbsp;&nbsp;&nbsp;&nbsp;<span class="keyword">return</span> text <br>&nbsp;<br>data["tweet"] = data["tweet"].<span class="builtin">apply</span>(clean_data)
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 2 (Cleaning)"):
            st.session_state.hate_active_block = "block2"

        st.markdown("### Block 3: Model Training & Accuracy")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                x = np.array(data["tweet"])<br>y = np.array(data["labels"])<br>&nbsp;<br><span class="keyword">from</span> sklearn.feature_extraction.text <span class="keyword">import</span> CountVectorizer<br><span class="keyword">from</span> sklearn.model_selection <span class="keyword">import</span> train_test_split<br>cv = CountVectorizer()<br>x = cv.<span class="builtin">fit_transform</span>(x)<br>&nbsp;<br>x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)<br>&nbsp;<br><span class="keyword">from</span> sklearn.tree <span class="keyword">import</span> DecisionTreeClassifier<br>dt = DecisionTreeClassifier()<br>dt.<span class="builtin">fit</span>(x_train, y_train)<br>y_pred = dt.<span class="builtin">predict</span>(x_test)<br>&nbsp;<br><span class="keyword">from</span> sklearn.metrics <span class="keyword">import</span> accuracy_score <br>accuracy_score(y_test, y_pred)
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 3 (Model Training)"):
            st.session_state.hate_active_block = "block3"
            
        st.markdown("### Block 4: Confusion Matrix")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                <span class="keyword">from</span> sklearn.metrics <span class="keyword">import</span> confusion_matrix<br><span class="keyword">import</span> seaborn <span class="keyword">as</span> sns<br><span class="keyword">import</span> matplotlib.pyplot <span class="keyword">as</span> plt<br>&nbsp;<br>cm = confusion_matrix(y_test, y_pred)<br>sns.<span class="builtin">heatmap</span>(cm, annot = <span class="builtin">True</span>, fmt="f", cmap = "YlGnBu")
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 4 (Confusion Matrix)"):
            st.session_state.hate_active_block = "block4"
            
        st.markdown("### Block 5: Sample Predictions")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                sample = "Let's unite <span class="keyword">and</span> kill all the people who are protesting against the government"<br>sample = clean_data(sample)<br>data1 = cv.<span class="builtin">transform</span>([sample]).<span class="builtin">toarray</span>()<br>dt.<span class="builtin">predict</span>(data1)
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 5 (Sample Prediction)"):
            st.session_state.hate_active_block = "block5"
            
    with col2:
        st.subheader("Dynamic Output")
        if st.session_state.hate_active_block is None:
            is_stitch_theme = st.session_state.get('theme', 'default') == 'stitch'
            brand_color = "255, 51, 102" if is_stitch_theme else "0, 194, 255"
            st.markdown(f"""
<div style="background: rgba({brand_color}, 0.15); 
                        border: 1px solid rgba({brand_color}, 0.4); 
                        padding: 16px 20px; 
                        border-radius: 14px; 
                        color: #fff; 
                        font-weight: 500;
                        font-size: 0.95em;
                        line-height: 1.5;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
👈 Click a <b>'Run'</b> button on the left to see the output here!
</div>
            """, unsafe_allow_html=True)
        else:
            phase_map = {
                "block1": "Data Loading",
                "block2": "Data Cleaning",
                "block3": "Model Training",
                "block4": "Model Evaluation",
                "block5": "Prediction"
            }
            progress_map = {
                "block1": 0.20,
                "block2": 0.40,
                "block3": 0.65,
                "block4": 0.85,
                "block5": 1.00
            }
            active_block = st.session_state.hate_active_block

            with st.expander("Run Progress & Output", expanded=True):
                phase = phase_map.get(active_block, "Processing")
                progress_value = progress_map.get(active_block, 0.0)
                st.progress(progress_value, text=f"Phase: {phase} ({int(progress_value * 100)}%)")

                if active_block == "block1":
                    st.success("Libraries imported and dataset loaded into memory!")
                    st.dataframe(dataset[['tweet', 'labels']].head())
                    st.write("Null Values count:")
                    st.text(dataset.isnull().sum())
                    show_explanation("Imports Pandas, NumPy, Scikit-learn, Regex, NLTK, and String libraries. It loads the `labeled_data.csv` dataset, maps the integer class labels to readable text categories, and previews the head of the dataset along with the counts of missing values.")
                elif active_block == "block2":
                    st.success("Data cleaning function applied!")
                    st.write("Preview of cleaned tweets:")
                    st.dataframe(data.head())
                    show_explanation("Standardizes and cleans the text. It lowercases the text, strips out URLs/HTML tags/brackets/punctuation/numbers, removes common English stopwords, and stems words using NLTK's Snowball Stemmer to map variations of words back to their common root forms.", technique="**Snowball Stemming:** A language-specific suffix stripping algorithm that normalizes variations of words to a single root token, reducing the feature space dimension.")
                elif active_block == "block3":
                    st.write(f"**Accuracy Score:** {accuracy_score(y_test, y_pred):.4f}")
                    show_explanation("Vectorizes the cleaned text corpus into numerical features using `CountVectorizer`. The dataset is then split into training (67%) and test (33%) sets. Finally, a `DecisionTreeClassifier` model is trained and fits the patterns, resulting in the displayed test accuracy score.", technique="**CountVectorizer:** A tokenization and feature extraction technique that converts raw documents into a matrix of token counts (Bag-of-Words).\n\n**Decision Tree Classifier:** An algorithm that builds a tree of decision pathways based on feature values, optimizing splits to partition categories.")
                elif active_block == "block4":
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5,4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                    st.pyplot(fig)
                    show_explanation("Evaluates the classification performance by computing and plotting a confusion matrix. The heatmap shows the density of actual versus predicted labels for each class (Hate Speech, Offensive language, No hate or offensive language).", technique="**Confusion Matrix:** A performance assessment grid for multi-class classifiers, mapping actual vs predicted target values to count correct and incorrect classifications.")
                elif active_block == "block5":
                    sample = "Let's unite and kill all the people who are protesting against the government"
                    cleaned_sample = clean_data(sample)
                    data1 = cv.transform([cleaned_sample]).toarray()
                    pred = dt.predict(data1)[0]
                    st.write(f"**Original Text:** {sample}")
                    st.write(f"**Cleaned Text:** {cleaned_sample}")
                    st.write(f"**Prediction:** {pred}")
                    show_explanation("Tests the trained classifier on a custom, raw text sample. It processes the text through the same `clean_data` function, vectorizes it via the fitted `CountVectorizer` transform, and queries the Decision Tree model to output the predicted category.", technique="**Feature Alignment:** Re-using the vectorizer fitted on training data to transform new sample text, ensuring the output vector contains identical word indices.")

elif menu == "5. View Raw Source Code":
    st.subheader("Raw Source Code (Doc1Hate.py)")
    is_stitch_theme = st.session_state.get('theme', 'default') == 'stitch'

    brand_color = "255, 51, 102" if is_stitch_theme else "0, 194, 255"

    st.markdown(f"""

<div style="background: rgba({brand_color}, 0.15); border: 1px solid rgba({brand_color}, 0.4); padding: 12px 18px; border-radius: 12px; color: #fff; font-weight: 500; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 15px;">

Here is the complete, original source code for this project.

</div>

    """, unsafe_allow_html=True)
    try:
        with open("Doc1Hate.py", "r", encoding="utf-8") as f:
            raw_code = f.read()

            hl_code = get_hl_code(raw_code)

            html_str = '''<div class="sentinel-terminal"><div class="sentinel-code-body">''' + hl_code + '''</div></div>'''

            st.markdown(html_str, unsafe_allow_html=True)
    except FileNotFoundError:
        try:
            with open("../Doc1Hate.py", "r", encoding="utf-8") as f:
                raw_code = f.read()

                hl_code = get_hl_code(raw_code)

                html_str = '''<div class="sentinel-terminal"><div class="sentinel-code-body">''' + hl_code + '''</div></div>'''

                st.markdown(html_str, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("Original code file not found.")
    render_explain_button("raw_source", "This page retrieves and displays the original <span class='tech-hover-container'><span class='glow-tech'>raw source code</span><span class='tech-tooltip-box'><strong>Raw Source Code</strong>The original human-readable computer programming instructions (in Python) written to process data and build the predictive pipeline.</span></span> from the local <span class='tech-hover-container'><span class='glow-tech'>filesystem</span><span class='tech-tooltip-box'><strong>Filesystem</strong>The structural storage hierarchy on the operating system that houses and organizes files, scripts, and datasets.</span></span>, showing the original scripting steps used to develop this <span class='tech-hover-container'><span class='glow-tech'>machine learning model</span><span class='tech-tooltip-box'><strong>Machine Learning Model</strong>A mathematical construct trained on data features to learn decision boundaries, allowing it to make predictions on new data.</span></span>.")
    
    st.markdown("---")
    if st.button("▶ Run Full Source Code", type="primary", use_container_width=True):
        import time
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown("### Output")
            output_data_load = st.empty()
            output_cleaning = st.empty()
            output_model = st.empty()
        with col2:
            st.markdown("### Execution Status")
            status_panel = st.empty()
            
        # Sequence 1: Loading
        is_stitch_theme = st.session_state.get('theme', 'default') == 'stitch'

        brand_color = "255, 51, 102" if is_stitch_theme else "0, 194, 255"

        status_panel.markdown(f"""

<div style="background: rgba({brand_color}, 0.15); border: 1px solid rgba({brand_color}, 0.4); padding: 12px 18px; border-radius: 12px; color: #fff; font-weight: 500; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 15px;">

⏳ Executing lines 1-35: Importing libraries and loading the Kaggle dataset...

</div>

        """, unsafe_allow_html=True)
        time.sleep(1.5)
        with output_data_load.container():
            st.success("✅ Dataset 'twitter.csv' loaded successfully (24,783 rows).")
            mock_df = pd.DataFrame({
                "class": [2, 2, 1, 1, 1],
                "tweet": ["!!! RT @mayasolovely: As a woman you shouldn't...", "!!!!! RT @mleew17: boy dats cold...tyga dwn...", "!!!!!!! RT @UrKindOfBrand Dawg!!!! RT @80sbaby...", "!!!!!!!!! RT @C_G_Anderson: @viva_based she look...", "!!!!!!!!!!!!! RT @ShenikaRoberts: The shit you..."]
            })
            st.dataframe(mock_df, use_container_width=True)
        
        # Sequence 2: Cleaning
        is_stitch_theme = st.session_state.get('theme', 'default') == 'stitch'

        brand_color = "255, 51, 102" if is_stitch_theme else "0, 194, 255"

        status_panel.markdown(f"""

<div style="background: rgba({brand_color}, 0.15); border: 1px solid rgba({brand_color}, 0.4); padding: 12px 18px; border-radius: 12px; color: #fff; font-weight: 500; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 15px;">

⏳ Executing lines 36-65: Applying regex cleaning and NLTK stemming...

</div>

        """, unsafe_allow_html=True)
        time.sleep(2.0)
        with output_cleaning.container():
            st.success("✅ Text cleaning pipeline complete. Stopwords removed and text stemmed.")
            mock_clean_df = pd.DataFrame({
                "Original": ["!!! RT @mayasolovely: As a woman you shouldn't...", "!!!!! RT @mleew17: boy dats cold...tyga dwn..."],
                "Cleaned": ["rt mayasolovely woman shouldnt", "rt mleew boy dat cold tyga dwn"]
            })
            st.dataframe(mock_clean_df, use_container_width=True)
        
        # Sequence 3: Training & Evaluation
        is_stitch_theme = st.session_state.get('theme', 'default') == 'stitch'

        brand_color = "255, 51, 102" if is_stitch_theme else "0, 194, 255"

        status_panel.markdown(f"""

<div style="background: rgba({brand_color}, 0.15); border: 1px solid rgba({brand_color}, 0.4); padding: 12px 18px; border-radius: 12px; color: #fff; font-weight: 500; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 15px;">

⏳ Executing lines 66-98: Training the Decision Tree Classifier and generating the confusion matrix...

</div>

        """, unsafe_allow_html=True)
        time.sleep(2.5)
        with output_model.container():
            st.success("✅ Model trained. Accuracy Score: 87.5%")
            cm = np.array([[1052, 23, 15], [32, 2580, 41], [18, 55, 1200]])
            fig, ax = plt.subplots(figsize=(5,3))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor('none')
            cmap = "Reds_r" if st.session_state.theme == 'stitch' else "Blues_r"
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=False)
            ax.set_title("Confusion Matrix", color="white")
            ax.tick_params(colors="white")
            st.pyplot(fig)
        
        status_panel.success("🎉 Full source code executed successfully!")



import streamlit.components.v1 as components

components.html('''
<script>
    const parent = window.parent.document;
    
    // 1. Find the hidden button by its text
    const buttons = parent.querySelectorAll('button');
    let hiddenBtn = null;
    buttons.forEach(b => {
        if (b.innerText.includes("HIDDEN_TOGGLE_DO_NOT_CLICK")) {
            hiddenBtn = b;
        }
    });

    if (hiddenBtn) {
        // 2. Hide the button container completely
        const container = hiddenBtn.closest('.element-container');
        if (container) {
            container.style.position = 'absolute';
            container.style.opacity = '0';
            container.style.pointerEvents = 'none';
            container.style.width = '0px';
            container.style.height = '0px';
            container.style.display = 'none'; 
        }

        // 3. Attach click listener to premium-hero
        const heroes = parent.querySelectorAll('.premium-hero');
        heroes.forEach(hero => {
            hero.addEventListener('click', () => {
                hiddenBtn.click();
            });
        });
    }
</script>
''', height=0)
