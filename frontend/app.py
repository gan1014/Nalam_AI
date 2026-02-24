
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import json
import os
import sys
from datetime import datetime
import base64
import uuid
from backend import audit_exporter

# Add current project dir to sys.path (supports both relative and package imports)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, '..'))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from alerts import email_alert
import importlib
importlib.reload(email_alert)
from backend import db
importlib.reload(db)
from backend import auth

# ── SESSION STATE & CORE CONFIG ──────────────────────────────────────────────
if 'lang' not in st.session_state:
    st.session_state.lang = "en"
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'face_verified' not in st.session_state:
    st.session_state.face_verified = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if 'temp_user' not in st.session_state:
    st.session_state.temp_user = None
if 'captcha_text' not in st.session_state:
    from backend import auth
    st.session_state.captcha_text = auth.generate_captcha_text()
if 'dispatch_verification_active' not in st.session_state:
    st.session_state.dispatch_verification_active = False
if 'dispatch_verification_attempts' not in st.session_state:
    st.session_state.dispatch_verification_attempts = 0
if 'dispatch_locked_for_session' not in st.session_state:
    st.session_state.dispatch_locked_for_session = False
if 'pending_dispatch_data' not in st.session_state:
    st.session_state.pending_dispatch_data = None

# Sync local variable for translations
lang = st.session_state.lang

# ── LOGO LOADER ─────────────────────────────────────────────────────────────────
def get_logo_b64():
    logo_path = os.path.join(os.path.dirname(__file__), 'nalamai_logo.png')
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None

LOGO_B64   = get_logo_b64()
LOGO_IMG   = f'<img src="data:image/png;base64,{LOGO_B64}" style="width:64px;height:64px;border-radius:12px;object-fit:cover;"/>' if LOGO_B64 else '<span style="font-size:2.8rem;">🩺</span>'
LOGO_SMALL = f'<img src="data:image/png;base64,{LOGO_B64}" style="width:72px;height:72px;border-radius:10px;object-fit:cover;margin-bottom:6px;"/>' if LOGO_B64 else '<span style="font-size:2rem;">🩺</span>'

# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATIONS  —  English + Tamil
# ══════════════════════════════════════════════════════════════════════════════
TRANSLATIONS = {
    # ── sidebar ──────────────────────────────────────────────────────────────
    "app_name":               {"en": "Nalam AI",                    "ta": "நலம் AI"},
    "app_subtitle":           {"en": "District Surveillance",       "ta": "மாவட்ட நோய் கண்காணிப்பு"},
    "online":                 {"en": "ONLINE",                      "ta": "இணைந்துள்ளது"},
    "ai_powered":             {"en": "AI-POWERED",                  "ta": "AI-இயக்கம்"},
    "language":               {"en": "🌐 Language",                  "ta": "🌐 மொழி"},
    "disease_filter":         {"en": "DISEASE FILTER",              "ta": "நோய் வகை"},
    "district":               {"en": "DISTRICT",                    "ta": "மாவட்டம்"},
    "forecast_horizon":       {"en": "FORECAST HORIZON",            "ta": "முன்கணிப்பு கால அளவு"},
    "weeks":                  {"en": "Weeks",                       "ta": "வாரங்கள்"},
    "system_status":          {"en": "SYSTEM STATUS",               "ta": "கணினி நிலை"},
    "xgb_model":              {"en": "🤖 XGBoost Model",            "ta": "🤖 XGBoost மாதிரி"},
    "lstm_model":             {"en": "🧠 LSTM Forecaster",          "ta": "🧠 LSTM முன்கணிப்பு"},
    "database":               {"en": "🗄️ Database",                 "ta": "🗄️ தரவுத்தளம்"},
    "email_alerts":           {"en": "📧 Email Alerts",             "ta": "📧 மின்னஞ்சல் எச்சரிக்கை"},
    "loaded":                 {"en": "LOADED",                      "ta": "ஏற்றப்பட்டது"},
    "ready":                  {"en": "READY",                       "ta": "தயார்"},
    "connected":              {"en": "CONNECTED",                   "ta": "இணைக்கப்பட்டது"},
    "standby":                {"en": "STANDBY",                     "ta": "காத்திருக்கிறது"},
    "footer_version":         {"en": "Nalam AI v1.0 · Tamil Nadu Health Dept", "ta": "நலம் AI v1.0 · தமிழ்நாடு சுகாதாரத்துறை"},
    "footer_stack":           {"en": "XGBoost · BiLSTM · SHAP · FastAPI", "ta": "XGBoost · BiLSTM · SHAP · FastAPI"},
    # ── header ───────────────────────────────────────────────────────────────
    "header_title":           {"en": "Nalam AI Clinical Surveillance",      "ta": "நலம் AI மருத்துவ கண்காணிப்பு"},
    "header_subtitle":        {"en": "Tamil Nadu Intelligent Disease Outbreak Prediction & Response System",
                               "ta": "தமிழ்நாடு உள்ளிட்ட நோய் கண்காணிப்பு மற்றும் முன்கணிப்பு மையம்"},
    "header_tagline":         {"en": "நலம் AI — மாவட்ட நோய் கண்காணிப்பு மையம்",
                               "ta": "Nalam AI — District Disease Surveillance Centre"},
    "districts_active":       {"en": "38 Districts Active",         "ta": "38 மாவட்டங்கள் செயல்பாட்டில்"},
    # ── kpi cards ────────────────────────────────────────────────────────────
    "high_risk":              {"en": "HIGH RISK",                   "ta": "அதிக ஆபத்து"},
    "medium_risk":            {"en": "MEDIUM RISK",                 "ta": "நடுத்தர ஆபத்து"},
    "low_risk":               {"en": "LOW RISK",                    "ta": "குறைந்த ஆபத்து"},
    "predicted":              {"en": "PREDICTED",                   "ta": "முன்கணிப்பு"},
    "districts_sfx":          {"en": "DISTRICTS",                   "ta": "மாவட்டங்கள்"},
    # ── tabs ─────────────────────────────────────────────────────────────────
    "tab_map":                {"en": "📍 Action Dashboard",         "ta": "📍 செயல்பாட்டு மேலாண்மை"},
    "tab_chennai":            {"en": "🏛️ Chennai Pilot",            "ta": "🏛️ சென்னை முன்னோட்டம்"},
    "tab_forecast":           {"en": "📈 14-Day Outlook",            "ta": "📈 14-நாள் முன்னறிவிப்பு"},
    "tab_risk":               {"en": "📋 Intelligence Reports",     "ta": "📋 புலனாய்வு அறிக்கைகள்"},
    "tab_alerts":             {"en": "⚙️ Advanced View",            "ta": "⚙️ மேம்பட்ட பார்வை"},
    "tab_bot":                {"en": "🤖 NalamAI Assistant",         "ta": "🤖 நலம்AI உதவியாளர்"},
    # ── tab 1 map ─────────────────────────────────────────────────────────────
    "outbreak_heatmap":       {"en": "🗺️ OUTBREAK HEATMAP",         "ta": "🗺️ நோய் வெப்ப வரைபடம்"},
    "risk_district_roster":   {"en": "🔴 RISK DISTRICT ROSTER",     "ta": "🔴 ஆபத்து மாவட்ட பட்டியல்"},
    "map_info":               {"en": "🗺️ Map will appear after predictions are generated.", "ta": "🗺️ முன்கணிப்புகள் உருவாக்கப்பட்டதும் வரைபடம் தோன்றும்."},
    "no_predictions":         {"en": "No predictions yet.",          "ta": "முன்கணிப்புகள் இல்லை."},
    # ── tab 2 forecast ───────────────────────────────────────────────────────
    "forecast_hdr":           {"en": "AI FORECAST",                 "ta": "AI முன்கணிப்பு"},
    "week_1_forecast":        {"en": "WEEK 1 FORECAST",             "ta": "வாரம் 1 முன்கணிப்பு"},
    "week_n_forecast":        {"en": "WEEK {n} FORECAST",           "ta": "வாரம் {n} முன்கணிப்பு"},
    "projected_trend":        {"en": "PROJECTED TREND",             "ta": "எதிர்பார்க்கப்படும் போக்கு"},
    "alert_threshold":        {"en": "⚠ Alert Threshold",           "ta": "⚠ எச்சரிக்கை வரம்பு"},
    "cases_label":            {"en": "Cases",                       "ta": "வழக்குகள்"},
    "forecast_legend":        {"en": "Forecast",                    "ta": "முன்கணிப்பு"},
    "upper_ci":               {"en": "Upper CI",                    "ta": "மேல் நம்பகத்தன்மை"},
    # ── tab 3 risk ──────────────────────────────────────────────────────────
    "top_high_risk":          {"en": "📊 TOP HIGH-RISK DISTRICTS",  "ta": "📊 அதிக ஆபத்து மாவட்டங்கள்"},
    "risk_distribution":      {"en": "🍩 RISK DISTRIBUTION",        "ta": "🍩 ஆபத்து பகிர்வு"},
    "records":                {"en": "Records",                     "ta": "பதிவுகள்"},
    "risk_score_label":       {"en": "Risk Score",                  "ta": "ஆபத்து மதிப்பெண்"},
    "run_engine_first":       {"en": "Run the prediction engine first.", "ta": "முதலில் முன்கணிப்பு இயந்திரத்தை இயக்கவும்."},
    # ── tab 4 alerts ─────────────────────────────────────────────────────────
    "dispatch_alert":         {"en": "🚨 DISPATCH CLINICAL ALERT",  "ta": "🚨 மருத்துவ எச்சரிக்கை அனுப்பு"},
    "alert_info":             {"en": "⚠️ Alerts dispatched to District CMOs and Field Officers. Configure .env with Gmail credentials to enable email.",
                               "ta": "⚠️ மாவட்ட CMO மற்றும் களப்பணியாளர்களுக்கு எச்சரிக்கைகள் அனுப்பப்படும். மின்னஞ்சல் செயல்படுத்த .env-ல் Gmail அமைக்கவும்."},
    "target_district":        {"en": "Target District",             "ta": "இலக்கு மாவட்டம்"},
    "disease_type":           {"en": "Disease Type",                "ta": "நோய் வகை"},
    "alert_severity":         {"en": "Alert Severity",              "ta": "எச்சரிக்கை தீவிரம்"},
    "clinical_notes":         {"en": "Clinical Notes",              "ta": "மருத்துவ குறிப்புகள்"},
    "notes_placeholder":      {"en": "Describe symptoms, affected locations, observations...",
                               "ta": "அறிகுறிகள், பாதிக்கப்பட்ட இடங்கள், கவனிப்புகள்..."},
    "dispatch_btn":           {"en": "🚨 Dispatch Alert",           "ta": "🚨 எச்சரிக்கை அனுப்பு"},
    "dispatching":            {"en": "Dispatching...",              "ta": "அனுப்புகிறோம்..."},
    "alert_sent":             {"en": "✅ Alert sent to {d} CMO!",   "ta": "✅ {d} CMO-க்கு எச்சரிக்கை அனுப்பப்பட்டது!"},
    "alert_logged":           {"en": "📋 Alert logged in DB. Add Gmail credentials in `.env` for email.",
                               "ta": "📋 எச்சரிக்கை பதிவு செய்யப்பட்டது. மின்னஞ்சல் அனுப்ப .env-ல் Gmail சேர்க்கவும்."},
    "hospital_capacity":      {"en": "🏥 HOSPITAL CAPACITY STATUS", "ta": "🏥 மருத்துவமனை படுக்கை நிலை"},
    "hospital_info":          {"en": "🏥 Hospital data will appear after DB setup.", "ta": "🏥 DB அமைவுக்குப் பிறகு மருத்துவமனை தகவல் தோன்றும்."},
    "social_signals":         {"en": "📡 SOCIAL MEDIA DISEASE SIGNALS", "ta": "📡 சமூக ஊடக நோய் சமிக்ஞைகள்"},
    "col_keyword":            {"en": "Keyword",                     "ta": "முக்கிய சொல்"},
    "col_mentions":           {"en": "Mentions (24h)",              "ta": "குறிப்புகள் (24 மணி)"},
    "col_trend":              {"en": "Trend",                       "ta": "போக்கு"},
    "col_sentiment":          {"en": "Sentiment",                   "ta": "உணர்வு"},
    "col_alert":              {"en": "Alert Level",                 "ta": "எச்சரிக்கை நிலை"},
    "negative":               {"en": "Negative",                    "ta": "எதிர்மறை"},
    "neutral":                {"en": "Neutral",                     "ta": "நடுநிலை"},
    # ── footer ───────────────────────────────────────────────────────────────
    "footer_copy":            {"en": "🩺 Nalam AI · நலம் AI · Tamil Nadu Health Department",
                               "ta": "🩺 நலம் AI · Nalam AI · தமிழ்நாடு சுகாதாரத்துறை"},
    # ── no data ──────────────────────────────────────────────────────────────
    "no_data":                {"en": "⚠️ No data found. Run `python run.py` from the project root first.",
                               "ta": "⚠️ தரவு இல்லை. முதலில் `python run.py` இயக்கவும்."},
    # ── sidebar extra ─────────────────────────────────────────────────────────
    "secure_session":         {"en": "Secure Session",               "ta": "பாதுகாப்பான அமர்வு"},
    "terminate_session":      {"en": "🚪 TERMINATE SESSION",         "ta": "🚪 அமர்வை முடிக்கவும்"},
    "online_badge":           {"en": "ONLINE",                       "ta": "இணைந்துள்ளது"},
    "ai_badge":               {"en": "AI-POWERED",                  "ta": "AI-இயக்கம்"},
    "ai_badge_hdr":           {"en": "AI-POWERED",                  "ta": "AI-இயக்கம்"},
    "lang_label":             {"en": "🌐 LANGUAGE / மொழி",          "ta": "🌐 மொழி / LANGUAGE"},
    # ── login page ───────────────────────────────────────────────────────────
    "login_title":            {"en": "Government Portal Login",      "ta": "அரசு வாயில் உள்நுழைவு"},
    "login_subtitle":         {"en": "Nalam AI Clinical Surveillance System", "ta": "நலம் AI மருத்துவ கண்காணிப்பு மையம்"},
    "username_label":         {"en": "Username or Email",            "ta": "பயனர் பெயர் அல்லது மின்னஞ்சல்"},
    "password_label":         {"en": "Password",                    "ta": "கடவுச்சொல்"},
    "captcha_label":          {"en": "Security Verification Code",  "ta": "பாதுகாப்பு சரிபார்ப்பு குறியீடு"},
    "captcha_input":          {"en": "Enter the code above",         "ta": "மேலே உள்ள குறியீட்டை உள்ளிடவும்"},
    "captcha_placeholder":    {"en": "Case-insensitive verification","ta": "சரிபார்ப்பு குறியீடு"},
    "login_btn":              {"en": "AUTHORIZE & LOGIN",            "ta": "அங்கீகரிக்கவும் & உள்நுழையவும்"},
    "login_warning":          {"en": "All fields are required for government access.","ta": "அரசு அணுகுவதற்கு அனைத்து புலங்களும் தேவை."},
    "captcha_err":            {"en": "❌ Invalid CAPTCHA. Click 🔄 to try again.","ta": "❌ தவறான CAPTCHA. மீண்டும் முயற்சிக்கவும்."},
    "login_success":          {"en": "✅ Credentials Verified. Accessing Surveillance Core...","ta": "✅ சான்றுகள் சரிபார்க்கப்பட்டன. கணினியை அணுகுகிறோம்..."},
    "login_fail":             {"en": "⛔ Authentication Denied: Invalid Username/Password.","ta": "⛔ அணுகல் மறுக்கப்பட்டது: தவறான பயனர்பெயர்/கடவுச்சொல்."},
    # ── chennai tab ───────────────────────────────────────────────────────────
    "chennai_hdr":            {"en": "🏛️ CHENNAI WARD-LEVEL INTELLIGENCE (CIVIC PILOT)", "ta": "🏛️ சென்னை வார்டு-நிலை நுண்ணறிவு (நகர முன்னோட்டம்)"},
    "ward_search_hdr":        {"en": "Ward Search & Intelligence",   "ta": "வார்டு தேடல் & நுண்ணறிவு"},
    "ward_map_hdr":           {"en": "Interactive Ward Risk Map (Click Ward to Select)", "ta": "தொடர்பு வார்டு ஆபத்து வரைபடம் (வார்டை கிளிக் செய்யவும்)"},
    "ward_select_label":      {"en": "Enter Ward Number",            "ta": "வார்டு எண் உள்ளிடவும்"},
    "ward_no_data":           {"en": "Ward identification data missing from source GIS.", "ta": "GIS மூலத்தில் வார்டு அடையாள தரவு இல்லை."},
    "dispatch_alert_ward":    {"en": "🚨 DISPATCH CLINICAL ALERT",  "ta": "🚨 மருத்துவ எச்சரிக்கை அனுப்பு"},
    "dispatch_locked":        {"en": "🔒 Alert dispatch restricted to Medical Officers and Admins.", "ta": "🔒 எச்சரிக்கை அனுப்புதல் மருத்துவர்கள் மற்றும் நிர்வாகிகளுக்கு மட்டுமே."},
    "decision_support":       {"en": "Decision Support: Evidence-based metrics provided for administrative audit.", "ta": "முடிவு ஆதாரம்: நிர்வாக தணிக்கைக்கு சான்று அடிப்படையிலான அளவீடுகள் வழங்கப்படுகின்றன."},
    "risk_drivers_hdr":       {"en": "🔍 Risk Drivers (SHAP Explainability)","ta": "🔍 ஆபத்து காரணிகள் (SHAP விளக்கம்)"},
    "resource_proj_hdr":      {"en": "🛡️ Resource Projection",       "ta": "🛡️ ஆதார திட்டமிடல்"},
    "field_teams_lbl":        {"en": "Teams",                        "ta": "குழுக்கள்"},
    "beds_lbl":               {"en": "Beds",                         "ta": "படுக்கைகள்"},
    "spray_lbl":              {"en": "Spray (L)",                    "ta": "தெளிப்பு (L)"},
    "insight_hdr":            {"en": "Clinical Insight:",             "ta": "மருத்துவ நுண்ணறிவு:"},
    "audit_trail_hdr":        {"en": "📋 Government Audit Trail (Traceability Record)", "ta": "📋 அரசு தணிக்கை பதிவு (கண்காணிப்பு)"},
    "audit_no_logs":          {"en": "No audit logs recorded yet.",   "ta": "தணிக்கை பதிவுகள் இன்னும் இல்லை."},
    "run_build_btn":          {"en": "🚀 Run Chennai Pilot Build",    "ta": "🚀 சென்னை முன்னோட்ட நிர்மாணம் இயக்கவும்"},
    "build_locked":           {"en": "🔒 System builds restricted to Administrators.", "ta": "🔒 கணினி நிர்மாணம் நிர்வாகிகளுக்கு மட்டுமே."},
    "chennai_no_data":        {"en": "⚠️ Chennai Ward-Level Pilot data not found. Processing required.", "ta": "⚠️ சென்னை வார்டு தரவு கிடைக்கவில்லை. செயலாக்கம் தேவை."},
    # ── intelligence reports tab ──────────────────────────────────────────────
    "intel_hdr":              {"en": "📋 WEEKLY DISTRICT RISK SUMMARY", "ta": "📋 வாராந்திர மாவட்ட ஆபத்து சுருக்கம்"},
    "download_report":        {"en": "📥 Download CSV Report",         "ta": "📥 CSV அறிக்கை பதிவிறக்கவும்"},
    "top_risk_hdr":           {"en": "📊 TOP HIGH-RISK DISTRICTS",   "ta": "📊 அதிக ஆபத்து மாவட்டங்கள்"},
    "risk_dist_hdr":          {"en": "🍩 RISK DISTRIBUTION",         "ta": "🍩 ஆபத்து பகிர்வு"},
    "no_pred_run":            {"en": "Run the prediction engine first.", "ta": "முதலில் முன்கணிப்பு இயந்திரத்தை இயக்கவும்."},
    # ── advanced view tab ─────────────────────────────────────────────────────
    "adv_alert_hdr":          {"en": "🚨 CLINICAL ALERT DISPATCH CENTRE", "ta": "🚨 மருத்துவ எச்சரிக்கை அனுப்பு மையம்"},
    "adv_manual_hdr":         {"en": "Manual Alert Override",         "ta": "கையேட்டு எச்சரிக்கை"},
    "adv_pipeline_hdr":       {"en": "⚙️ PREDICTION PIPELINE",       "ta": "⚙️ முன்கணிப்பு பட்டன்"},
    "run_prediction":         {"en": "🔄 RUN PREDICTION PIPELINE",   "ta": "🔄 முன்கணிப்பு பட்டன் இயக்கவும்"},
    "running_pipeline":       {"en": "Running ML Pipeline...",        "ta": "ML பட்டன் இயங்குகிறது..."},
    "pipeline_done":          {"en": "✅ Pipeline complete! Refresh to see new predictions.", "ta": "✅ முடிந்தது! புதிய முன்கணிப்புகளைப் பார்க்க புதுப்பிக்கவும்."},
    "adv_hospital_hdr":       {"en": "🏥 HOSPITAL CAPACITY STATUS",  "ta": "🏥 மருத்துவமனை படுக்கை நிலை"},
    "adv_social_hdr":         {"en": "📡 SOCIAL MEDIA DISEASE SIGNALS", "ta": "📡 சமூக ஊடக நோய் சமிக்ஞைகள்"},
    "face_verify_title":      {"en": "Biometric Verification",       "ta": "உயிரியல் சரிபார்ப்பு"},
    "face_verify_desc":       {"en": "This system uses biometric verification for authorized government access.", "ta": "அங்கீகரிக்கப்பட்ட அரசு அணுகலுக்கு இந்த அமைப்பு உயிரியல் சரிபார்ப்பைப் பயன்படுத்துகிறது."},
    "face_verify_success":    {"en": "Face verified successfully",   "ta": "முக சரிபார்ப்பு வெற்றிபெற்றது"},
    "face_verify_fail":       {"en": "Face mismatch — try again",    "ta": "முகம் பொருந்தவில்லை - மீண்டும் முயற்சிக்கவும்"},
    "face_enroll_title":      {"en": "Enroll Face for User",         "ta": "பயனருக்கு முகம் பதிவு செய்யவும்"},
    "face_enroll_desc":       {"en": "Capture or upload 2 clear face images for enrollment.", "ta": "பதிவு செய்ய 2 தெளிவான முகப் படங்களை எடுக்கவும் அல்லது பதிவேற்றவும்."},
    "capture_btn":            {"en": "CAPTURE IMAGE",                "ta": "படம் எடுக்கவும்"},
    "enroll_btn":             {"en": "ENROLL BIOMETRICS",            "ta": "உயிரியலைப் பதிவுசெய்"},
}


def t(key, lang, **kwargs):
    """Get translated string for key in given language."""
    val = TRANSLATIONS.get(key, {}).get(lang, TRANSLATIONS.get(key, {}).get("en", key))
    for k, v in kwargs.items():
        val = val.replace(f"{{{k}}}", str(v))
    return val

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nalam AI | நலம் AI · Clinical Surveillance",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── COLOR PALETTE ────────────────────────────────────────────────────────────
BG     = "#0d1117"
BG2    = "#161b22"
BORDER = "#21262d"
DIM    = "#8b949e"
HI     = "#e6edf3"
CYAN   = "#00b4d8"
RED    = "#f85149"
ORANGE = "#f97316"
GREEN  = "#22c55e"
PURPLE = "#a371f7"

def make_chart(fig, height=340, lpad=10, rpad=20, tpad=10, bpad=10):
    fig.update_layout(
        height=height,
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        font=dict(family="Inter, sans-serif", color=DIM, size=11),
        margin=dict(l=lpad, r=rpad, t=tpad, b=bpad),
        legend=dict(bgcolor=BG, bordercolor=BORDER, font=dict(size=10, color=DIM)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=BORDER, showline=False, tickfont=dict(color=DIM, size=10))
    fig.update_yaxes(gridcolor=BORDER, showline=False, tickfont=dict(color=DIM, size=10))
    return fig

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

.main, .main .block-container,
section[data-testid="stSidebar"],
.stApp {
    background-color: #0d1117 !important;
    font-family: 'Inter', sans-serif !important;
    color: #c9d1d9 !important;
}
.main .block-container {
    padding: 1.4rem 2rem !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid #21262d !important;
    background: linear-gradient(180deg,#0a0d12,#0d1117) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.2rem !important; }

/* Animated top stripe */
.stApp::before {
    content:'';
    position:fixed; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg,#00b4d8,#a371f7,#22c55e,#f97316,#00b4d8);
    background-size:300% 100%;
    animation: stripe 5s linear infinite;
    z-index:9999;
}
@keyframes stripe { 0%{background-position:0%} 100%{background-position:300%} }

/* Live dot */
@keyframes livePulse {
    0%,100%{box-shadow:0 0 0 0 rgba(34,197,94,.5)}
    50%{box-shadow:0 0 0 6px rgba(34,197,94,0)}
}
.live-dot {
    display:inline-block; width:8px; height:8px;
    background:#22c55e; border-radius:50%;
    animation:livePulse 1.8s infinite;
    vertical-align:middle; margin-right:5px;
}

/* Language toggle pill */
.lang-active {
    background: linear-gradient(135deg,#00b4d8,#0096c7) !important;
    color: #0d1117 !important;
    font-weight:700 !important;
    border-radius: 20px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:#0d1117 !important;
    border-bottom:1px solid #21262d !important;
    gap:2px; padding-bottom:0;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important;
    border:1px solid transparent !important;
    border-radius:8px 8px 0 0 !important;
    color:#6e7681 !important;
    font-size:0.78rem; font-weight:500;
    padding:0.5rem 1rem !important;
    transition:all .2s;
}
.stTabs [aria-selected="true"] {
    background:#161b22 !important;
    border-color:#21262d !important;
    border-bottom-color:#161b22 !important;
    color:#00b4d8 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background:#161b22 !important;
    border:1px solid #21262d !important;
    border-top:none !important;
    border-radius:0 0 12px 12px !important;
    padding:1.4rem !important;
}

/* Section headers */
.sec-hdr {
    font-size:.68rem; color:#6e7681;
    text-transform:uppercase; letter-spacing:1.5px;
    font-weight:700; padding-bottom:9px;
    border-bottom:1px solid #21262d; margin-bottom:13px;
}

/* Buttons */
.stButton>button {
    background:linear-gradient(135deg,#00b4d8,#0096c7) !important;
    border:none !important; border-radius:8px !important;
    color:#0d1117 !important; font-weight:700 !important;
    font-size:.8rem !important; padding:.5rem 1.3rem !important;
    transition:all .25s;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#48cae4,#00b4d8) !important;
    transform:translateY(-2px);
    box-shadow:0 6px 18px rgba(0,180,216,.3) !important;
}
[data-testid="stFormSubmitButton"]>button {
    width:100%;
    background:linear-gradient(135deg,#dc2626,#b91c1c) !important;
    color:white !important;
}
[data-testid="stFormSubmitButton"]>button:hover {
    background:linear-gradient(135deg,#ef4444,#dc2626) !important;
    box-shadow:0 6px 18px rgba(220,38,38,.3) !important;
}

/* Inputs */
.stSelectbox>div>div, .stTextInput>div>div>input, .stTextArea textarea {
    background:#161b22 !important;
    border:1px solid #30363d !important;
    border-radius:8px !important;
    color:#c9d1d9 !important;
}
div[data-baseweb="select"] div { background:#161b22 !important; color:#c9d1d9 !important; }

/* Tables */
.stDataFrame { border-radius:10px; overflow:hidden; }
.stDataFrame th, [data-testid="stDataFrame"] th {
    background:#21262d !important; color:#8b949e !important;
    font-size:.7rem; text-transform:uppercase; letter-spacing:.8px;
}
.stDataFrame td, [data-testid="stDataFrame"] td {
    background:#161b22 !important; color:#c9d1d9 !important;
    border-color:#21262d !important;
}
table { width:100%; border-collapse:collapse; }
thead tr { background:#21262d; }
th { color:#8b949e !important; font-size:.7rem; text-transform:uppercase;
     letter-spacing:.8px; padding:7px 10px !important; border:1px solid #21262d !important; }
td { color:#c9d1d9 !important; padding:7px 10px !important; border:1px solid #21262d !important; }
tbody tr:hover td { background:#21262d !important; }

/* Scrollbar */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#0d1117; }
::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }

/* Hide branding */
#MainMenu,footer,header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Language selector + filters
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo + brand ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#112240,#0d1b2a);
                border:1px solid #1f4068;border-radius:12px;
                padding:1rem;margin-bottom:1.1rem;text-align:center;">
        <div style="display:flex;justify-content:center;margin-bottom:6px;">{LOGO_SMALL}</div>
        <div style="font-weight:700;color:#e6edf3;font-size:.9rem;">{t("app_name", lang)} · நலம் AI</div>
        <div style="font-size:.68rem;color:#00b4d8;font-weight:600;letter-spacing:.5px;">{t("app_subtitle", lang)}</div>
        <div style="margin-top:8px;display:flex;justify-content:center;gap:5px;flex-wrap:wrap;">
            <span style="background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.3);
                         color:#22c55e;padding:2px 8px;border-radius:20px;font-size:.66rem;font-weight:700;">
                <span class="live-dot"></span>{t("online_badge", lang)}</span>
            <span style="background:rgba(0,180,216,.12);border:1px solid rgba(0,180,216,.3);
                         color:#00b4d8;padding:2px 8px;border-radius:20px;font-size:.66rem;font-weight:700;">
                {t("ai_badge", lang)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Language Toggle ────────────────────────────────────────────────────────
# ── AUTHENTICATION SYSTEM ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Already initialized at top level

def logout():
    db.log_event(
        username=st.session_state.user_info['username'],
        role=st.session_state.current_role,
        session_id=st.session_state.session_id,
        action="LOGOUT",
        target_type="System",
        target_id="N/A",
        details="User logged out manually"
    )
    st.session_state.authenticated = False
    st.session_state.face_verified = False
    st.session_state.user_info = None
    st.session_state.current_role = None
    st.rerun()

def face_verification_page():
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        st.markdown(f'<div style="text-align:center; margin-bottom:20px;">{LOGO_SMALL}</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:20px;">
            <h2 style="color:#e6edf3; margin-bottom:5px;">{t("face_verify_title", lang)}</h2>
            <p style="color:#8b949e; font-size:0.9rem;">{t("face_verify_desc", lang)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Camera input for live capture
        img_file = st.camera_input("Capture Face for Verification")
        
        if img_file:
            import base64
            img_bytes = img_file.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode()
            
            # Call backend to verify
            from backend import face_logic
            # For simplicity in Streamlit, we might call backend directly or via requests
            # Let's use direct backend logic call for stability in this environment
            user = st.session_state.temp_user
            encrypted_embeddings = db.get_face_embeddings(user['id'])
            
            if not encrypted_embeddings:
                st.success(t("face_verify_success", lang))
                st.session_state.face_verified = True
                # st.session_state.authenticated = True  # Already set after LDAP/DB login
                st.session_state.user_info = dict(user)
                st.session_state.current_role = user['role']
                st.rerun()
            
            live_embedding = face_logic.generate_embedding(img_bytes)
            if live_embedding is None:
                st.error("❌ No face detected. Please ensure you are in a well-lit area.")
                db.log_face_verification(user['id'], "FAILED_DETECTION")
            else:
                stored_embeddings = [face_logic.decrypt_embedding(blob) for blob in encrypted_embeddings]
                similarity = face_logic.compare_embeddings(live_embedding, stored_embeddings)
                
                if similarity >= 0.75:
                    st.success(f"{t('face_verify_success', lang)} (Score: {similarity:.2f})")
                    db.log_face_verification(user['id'], "SUCCESS", similarity_score=similarity)
                    st.session_state.face_verified = True
                    st.session_state.authenticated = True
                    st.session_state.user_info = dict(user)
                    st.session_state.current_role = user['role']
                    
                    db.log_event(
                        username=user['username'],
                        role=user['role'],
                        session_id=st.session_state.session_id,
                        action="LOGIN_COMPLETE",
                        target_type="System",
                        target_id="Biometrics",
                        details="Face verification successful"
                    )
                    st.rerun()
                else:
                    st.session_state.face_attempts = st.session_state.get('face_attempts', 0) + 1
                    remaining = 3 - st.session_state.face_attempts
                    st.error(f"{t('face_verify_fail', lang)} (Remaining: {remaining})")
                    db.log_face_verification(user['id'], "MISMATCH", similarity_score=similarity)
                    
                    if st.session_state.face_attempts >= 3:
                        st.error("⛔ Maximum attempts exceeded. Session locked.")
                        db.log_event(user['username'], user['role'], st.session_state.session_id, "LOCKOUT", "Biometrics", str(user['id']), "Max face attempts reached")
                        if st.button("Return to Login"):
                            st.session_state.temp_user = None
                            st.session_state.face_attempts = 0
                            st.rerun()

def login_page():
    # Note: captcha_text already initialized at top

    # Centered login form
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        st.markdown(f'<div style="text-align:center; margin-bottom:20px;">{LOGO_SMALL}</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:30px;">
            <h2 style="color:#e6edf3; margin-bottom:5px;">{t("login_title", lang)}</h2>
            <p style="color:#8b949e; font-size:0.9rem;">{t("login_subtitle", lang)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # User credentials with explicit keys for stability
        username = st.text_input("Username or Email", placeholder="e.g. admin", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #21262d; border-radius:8px; padding:15px; text-align:center; margin-top:10px;">
            <div style="font-size:0.65rem; color:#8b949e; margin-bottom:8px; text-transform:uppercase;">Security Verification Code</div>
            <span style="font-family:'JetBrains Mono', monospace; font-size:1.6rem; letter-spacing:10px; color:#00b4d8; font-weight:700;">
                {st.session_state.captcha_text}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        captcha_input = st.text_input("Enter the code above", placeholder="Case-insensitive verification", key="login_captcha")
        
        # Action Buttons
        log_col, ref_col = st.columns([4, 1])
        with log_col:
            submit = st.button("AUTHORIZE & LOGIN", use_container_width=True, type="primary")
        with ref_col:
            if st.button("🔄", help="Get new code"):
                st.session_state.captcha_text = auth.generate_captcha_text()
                st.rerun()

        if submit:
            if not username or not password or not captcha_input:
                st.warning("All fields are required for government access.")
            elif captcha_input.strip().upper() != st.session_state.captcha_text:
                st.error("❌ Invalid CAPTCHA. Click 🔄 to try again.")
                # We don't auto-regenerate here to allow the user to see what they typed wrong
            else:
                user = auth.get_user_from_db(username)
                if user and auth.verify_password(password, user['password_hash']):
                    st.session_state.authenticated = True
                    st.session_state.user_info = dict(user)
                    st.session_state.current_role = user['role']
                    
                    db.log_event(
                        username=user['username'],
                        role=user['role'],
                        session_id=st.session_state.session_id,
                        action="LOGIN",
                        target_type="System",
                        target_id="N/A",
                        details="Successful authentication"
                    )
                    
                    st.success(t("login_success", lang))
                    st.session_state.temp_user = dict(user)
                    st.session_state.face_attempts = 0
                    st.rerun()
                else:
                    db.log_event(
                        username=username,
                        role="GUEST",
                        session_id=st.session_state.session_id,
                        action="LOGIN_FAILURE",
                        target_type="System",
                        target_id="N/A",
                        details=f"Invalid credentials attempt for {username}"
                    )
                    st.error(t("login_fail", lang))
                    # Refresh captcha on failed auth attempt for security
                    st.session_state.captcha_text = auth.generate_captcha_text()
                    st.rerun()

# Redundant session control removed (managed at top level)

if not st.session_state.authenticated:
    if st.session_state.temp_user:
        face_verification_page()
    else:
        login_page()
    st.stop()

# --- Dispatch Verification Modal Overlay ---
if st.session_state.dispatch_verification_active:
    # STRICT: Only ADMIN users can proceed
    if st.session_state.get('current_role') != 'ADMIN':
        st.error("⛔ ACCESS DENIED: Only ADMIN users can dispatch clinical alerts.")
        st.session_state.dispatch_verification_active = False
        st.session_state.pending_dispatch_data = None
        st.stop()
    
    # High-Contrast Overlay for Government Compliance
    st.markdown("""
    <style>
    .dispatch-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
        background: rgba(10, 15, 20, 0.98); z-index: 999999; 
        display: flex; flex-direction: column; align-items: center; justify-content: flex-start;
        padding-top: 5vh; overflow-y: auto;
    }
    .dispatch-card {
        background: #1c2128; border: 2px solid #00b4d8; border-radius: 16px; 
        padding: 40px; text-align: center; max-width: 600px; width: 90%;
        box-shadow: 0 0 50px rgba(0, 180, 216, 0.4); margin-bottom: 20px;
    }
    .dispatch-title { color: #ffffff !important; font-size: 1.8rem; font-weight: 800; margin-bottom: 10px; }
    .dispatch-subtitle { color: #00b4d8 !important; font-size: 1rem; font-weight: 600; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        _, v_col, _ = st.columns([1, 2, 1])
        with v_col:
            st.markdown(f"""
            <div style="text-align:center; margin-bottom:15px; margin-top:30px;">{LOGO_SMALL}</div>
            <div style="background:#1c2128; border:2px solid #00b4d8; border-radius:16px; padding:30px; text-align:center; box-shadow:0 0 40px rgba(0,180,216,0.3);">
                <h2 style="color:white; margin-bottom:5px;">DISPATCH AUTHORIZATION</h2>
                <p style="color:#00b4d8; font-weight:600; font-size:0.95rem;">🔒 MANDATORY BIOMETRIC VERIFICATION</p>
                <p style="color:#c9d1d9; font-size:0.85rem; margin-top:10px;">Authorized personnel must verify identity via live camera to transmit Clinical Alerts.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Focused Camera Input
            v_img = st.camera_input("CAPTURE FACE TO AUTHORIZE DISPATCH", key="dispatch_cam_v2")
            
            if v_img:
                from backend import face_logic, db
                import importlib
                importlib.reload(face_logic)
                import datetime
                import uuid
                
                v_bytes = v_img.getvalue()
                user_id = st.session_state.user_info['id']
                
                with st.spinner("🔍 Performing strict identity verification..."):
                    v_result = face_logic.verify_against_file(v_bytes, user_id)
                    similarity = v_result.get("similarity", 0)
                    is_verified = v_result.get("verified", False)
                    status = v_result.get("status", "ERROR")
                    distance = v_result.get("distance", 1.0)
                    
                    if is_verified:
                        st.toast(f"✅ IDENTITY CONFIRMED (Match: {similarity:.0%}, Distance: {distance:.4f})")
                        # Generate proof and send
                        proof = {
                            'face_verified': True,
                            'admin_id': user_id,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'alert_id': str(uuid.uuid4())
                        }
                        
                        # Execute Pending Dispatch
                        p_data = st.session_state.pending_dispatch_data
                        from alerts import email_alert
                        import importlib
                        importlib.reload(email_alert)
                        
                        success = email_alert.send_alert(
                            district=p_data['district'],
                            disease=p_data['disease'],
                            risk_level=p_data['risk_level'],
                            notes=p_data['notes'],
                            triggered_by=p_data['triggered_by'],
                            resource_data=p_data.get('resource_data'),
                            verification_proof=proof
                        )
                        
                        # Log event
                        db.log_event(
                            username=st.session_state.user_info['username'],
                            role=st.session_state.current_role,
                            session_id=st.session_state.session_id,
                            action="DISPATCH_COMPLETE",
                            target_type="Verification",
                            target_id=proof['alert_id'],
                            details=f"Verified against physical file (Score: {similarity:.2f}) and alert sent."
                        )
                        
                        # Reset state
                        st.session_state.dispatch_verification_active = False
                        st.session_state.dispatch_verification_attempts = 0
                        st.session_state.pending_dispatch_data = None
                        if success: st.success("🚀 DISPATCH SUCCESSFUL")
                        else: st.error("DISPATCH FAILED: Check SMTP logs.")
                        st.rerun()
                    else:
                        st.session_state.dispatch_verification_attempts += 1
                        db.log_face_verification(user_id, f"DISPATCH_{status}", similarity_score=similarity)
                        
                        if st.session_state.dispatch_verification_attempts >= 2:
                            st.session_state.dispatch_locked_for_session = True
                            st.session_state.dispatch_verification_active = False
                            db.log_event(st.session_state.user_info['username'], st.session_state.current_role, st.session_state.session_id, "DISPATCH_LOCKED", "Security", "N/A", "Max verification attempts failed")
                            st.error("⛔ SECURITY LOCKOUT: Multiple biometric failures. Session locked.")
                        else:
                            if status == "NO_ENROLLED_FILE":
                                st.error("❌ NO REFERENCE IMAGE FOUND. Place your photo in `backend/face_data/manual_reference/REF.jpeg`.")
                            elif status == "MISMATCH":
                                st.error(f"❌ FACE MISMATCH — Distance: {distance:.4f} (must be ≤ 0.30). Attempt {st.session_state.dispatch_verification_attempts}/2.")
                            else:
                                st.error(f"❌ VERIFICATION ERROR: {v_result.get('error', 'Unknown')}. Attempt {st.session_state.dispatch_verification_attempts}/2.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("❌ CANCEL DISPATCH", use_container_width=True):
                st.session_state.dispatch_verification_active = False
                st.session_state.pending_dispatch_data = None
                st.rerun()
    st.stop() # Freeze rest of UI while verifying

# Set globals from session
user_role = st.session_state.current_role
user_name = st.session_state.user_info['username']

# ── TOP BAR / LOGOUT ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="text-align:center; margin-bottom:20px;">{LOGO_SMALL}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #21262d; border-radius:10px; padding:12px; margin-bottom:15px;">
        <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">{t("secure_session", lang)}</div>
        <div style="font-weight:700; color:#e6edf3; font-size:1rem;">{user_name}</div>
        <div style="font-size:0.7rem; color:#00b4d8; font-weight:600; margin-top:2px;">🛡️ {user_role.replace('_',' ')}</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button(t("terminate_session", lang), use_container_width=True):
        logout()
    st.markdown("---")
    st.markdown(f'<p class="sec-hdr">{t("lang_label", lang)}</p>', unsafe_allow_html=True)
    
    # Callback to handle language change instantly
    def on_lang_change():
        st.session_state.lang = "en" if st.session_state.lang_radio == "English" else "ta"

    lang_choice = st.radio(
        "Language",
        options=["English", "தமிழ்"],
        horizontal=True,
        label_visibility="collapsed",
        key="lang_radio",
        index=0 if st.session_state.lang == "en" else 1,
        on_change=on_lang_change
    )
    lang = st.session_state.lang # Ensure local var is updated
    st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Filters ─────────────────────────────────────────────────────────────
    DISEASE_ICONS = {'Dengue':'🦟','Cholera':'💧','Malaria':'🩸','Leptospirosis':'🐀','Chikungunya':'🦟'}

    @st.cache_data(ttl=300)
    def load_data():
        try:
            df = pd.read_csv('data/processed/train_scaled.csv')
            try:    preds = pd.read_csv('data/processed/latest_predictions.csv')
            except: preds = pd.DataFrame()
            try:
                with open('data/maps/tn_districts.geojson') as f: geojson = json.load(f)
            except: geojson = {}
            return df, preds, geojson
        except:
            return pd.DataFrame(), pd.DataFrame(), {}

    df, preds, geojson = load_data()

    diseases  = sorted(df['disease'].unique())  if not df.empty else ['Dengue','Cholera','Malaria','Leptospirosis','Chikungunya']
    districts = sorted(df['district'].unique()) if not df.empty else ['Chennai']

    st.markdown(f'<p class="sec-hdr">{t("disease_filter", lang)}</p>', unsafe_allow_html=True)
    selected_disease  = st.selectbox(
        t("disease_filter", lang), diseases,
        format_func=lambda d: f"{DISEASE_ICONS.get(d,'🦠')} {d}",
        label_visibility="collapsed"
    )
    st.markdown(f'<p class="sec-hdr" style="margin-top:10px;">{t("district", lang)}</p>', unsafe_allow_html=True)
    selected_district = st.selectbox(t("district", lang), districts, label_visibility="collapsed")
    st.markdown(f'<p class="sec-hdr" style="margin-top:10px;">{t("forecast_horizon", lang)}</p>', unsafe_allow_html=True)
    forecast_weeks = st.slider(t("weeks", lang), 1, 12, 8, label_visibility="collapsed")

    st.markdown("---")

    # ── System status ─────────────────────────────────────────────────────────
    st.markdown(f'<p class="sec-hdr">{t("system_status", lang)}</p>', unsafe_allow_html=True)
    for label_key, status_key, color in [
        ("xgb_model",   "loaded",    "#22c55e"),
        ("lstm_model",  "ready",     "#22c55e"),
        ("database",    "connected", "#22c55e"),
        ("email_alerts","standby",   "#f97316"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:5px 0;border-bottom:1px solid #21262d;">
            <span style="font-size:.74rem;color:#8b949e;">{t(label_key, lang)}</span>
            <span style="font-size:.66rem;font-weight:700;color:{color};">● {t(status_key, lang).upper()}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="text-align:center;padding:.7rem 0 0;font-size:.66rem;color:#484f58;">
        {t("footer_version", lang)}<br>{t("footer_stack", lang)}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
now = datetime.now()
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0a1628,#112240,#0a1628);
            border:1px solid #1f4068;border-left:4px solid #00b4d8;border-radius:14px;
            padding:1.3rem 1.8rem;margin-bottom:1.3rem;display:flex;align-items:center;gap:1.4rem;
            box-shadow:0 4px 40px rgba(0,180,216,.07);">
    <div style="flex-shrink:0;">{LOGO_IMG}</div>
    <div style="flex:1;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">
            <span style="font-size:1.55rem;font-weight:800;color:#e6edf3;letter-spacing:-.5px;">
                {t("header_title", lang)}</span>
            <span style="background:rgba(0,180,216,.15);border:1px solid rgba(0,180,216,.35);
                         color:#00b4d8;padding:2px 9px;border-radius:20px;font-size:.66rem;
                         font-weight:700;letter-spacing:1px;">{t("ai_badge_hdr", lang)}</span>
        </div>
        <p style="font-size:.8rem;color:#8b949e;margin:0;">
            {t("header_subtitle", lang)} &nbsp;·&nbsp;
            <em style="color:#6e7681;">{t("header_tagline", lang)}</em>
        </p>
    </div>
    <div style="text-align:right;font-family:'JetBrains Mono',monospace;flex-shrink:0;">
        <div style="font-size:1.4rem;font-weight:700;color:#e6edf3;">
            {now.strftime('%H:%M')}<span style="font-size:.75rem;color:#8b949e;">:{now.strftime('%S')}</span></div>
        <div style="font-size:.68rem;color:#6e7681;">{now.strftime('%A, %d %b %Y')}</div>
        <div style="margin-top:3px;font-size:.68rem;">
            <span class="live-dot"></span><span style="color:#8b949e;">{t("districts_active", lang)}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.error(t("no_data", lang))
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
dp     = preds[preds['disease']==selected_disease] if not preds.empty else pd.DataFrame()
high_n = len(dp[dp['risk_level']=='HIGH'])   if not dp.empty else 0
med_n  = len(dp[dp['risk_level']=='MEDIUM']) if not dp.empty else 0
low_n  = len(dp[dp['risk_level']=='LOW'])    if not dp.empty else 0
tot_n  = int(dp['predicted_cases'].sum())    if not dp.empty else 0

kpis = [
    (t("high_risk",  lang), high_n,       RED,    "rgba(248,81,73,.15)",  "rgba(248,81,73,.35)"),
    (t("medium_risk",lang), med_n,        ORANGE, "rgba(249,115,22,.15)", "rgba(249,115,22,.35)"),
    (t("low_risk",   lang), low_n,        GREEN,  "rgba(34,197,94,.15)",  "rgba(34,197,94,.35)"),
    (t("predicted",  lang), f"{tot_n:,}", CYAN,   "rgba(0,180,216,.12)",  "rgba(0,180,216,.3)"),
]
icons = ["🔴","🟠","🟢","📊"]
cols  = st.columns(4)
for col, (label, val, color, bg, border), icon in zip(cols, kpis, icons):
    col.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:14px;
                padding:1.1rem .8rem;text-align:center;">
        <div style="font-size:1.4rem;line-height:1.4;">{icon}</div>
        <div style="font-size:1.85rem;font-weight:800;color:{color};
                    font-family:'JetBrains Mono',monospace;line-height:1.1;">{val}</div>
        <div style="font-size:.62rem;color:#6e7681;font-weight:600;
                    text-transform:uppercase;letter-spacing:1px;margin-top:4px;">{label} {t("districts_sfx",lang)}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_labels = [
    t("tab_chennai", lang),
    t("tab_forecast",lang),
    t("tab_risk",    lang),
    t("tab_alerts",  lang),
    t("tab_bot",     lang)
]
if user_role == "ADMIN":
    tab_labels.append("🔐 Audit Logs")

all_tabs = st.tabs(tab_labels)
tab_ch, tab2, tab3, tab4, tab_bot = all_tabs[:5]
if user_role == "ADMIN":
    tab_audit = all_tabs[5]

# ═══════ TAB CHENNAI — WARD LEVEL PILOT ══════════════════════════════════════════
with tab_ch:
    st.markdown('<div class="sec-hdr">🏛️ CHENNAI WARD-LEVEL INTELLIGENCE (CIVIC PILOT)</div>', unsafe_allow_html=True)
    
    # Load Chennai Data
    load_error = None
    has_chennai = False
    path = 'data/processed/standardized/chennai_pilot_intelligence.geojson'
    if not os.path.exists(path):
        load_error = f"Standardized file not found: {path}"
    else:
        try:
            try:
                c_gdf = gpd.read_file(path)
            except Exception as e1:
                # Robust JSON fallback for GEOS/Fiona issues
                with open(path, 'r') as f:
                    c_data = json.load(f)
                c_gdf = gpd.GeoDataFrame.from_features(c_data['features'])
                c_gdf.set_crs("EPSG:4326", inplace=True)
            
            has_chennai = c_gdf is not None and not c_gdf.empty
            if not has_chennai and not load_error:
                load_error = "Standardized file exists but data is empty or malformed."
        except Exception as e:
            has_chennai = False
            load_error = f"Fatal load error: {e}"
        
    if load_error:
        st.error(f"📋 Chennai Data Sync Error: {load_error}")
        
    if has_chennai:
        # Task 7: Recalculate Summary Counts
        total_wards = len(c_gdf)
        # Task 11: Analysed Wards now includes Validated Estimates for full workability
        analysed_wards = len(c_gdf[c_gdf['status'].isin(['VALIDATED', 'VALIDATED_ESTIMATE'])]) if 'status' in c_gdf.columns else len(c_gdf)
        missing_wards = total_wards - analysed_wards
        
        # KPI Bar for Chennai
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("ward_select_label", lang) if lang == "ta" else "Wards in Boundary", total_wards)
        c2.metric("ஆயவு வார்டுகள்" if lang == "ta" else "Analysed Wards", analysed_wards)
        c3.metric("தரவு நிலுவை" if lang == "ta" else "Data Pending", missing_wards)
        
        # Defensive Metric Calculations
        crit_wards = len(c_gdf[c_gdf['risk_level']=='CRITICAL']) if 'risk_level' in c_gdf.columns else 0
        avg_dist = c_gdf[c_gdf['status']=='VALIDATED']['dist_to_phc_km'].mean() if ('dist_to_phc_km' in c_gdf.columns and 'status' in c_gdf.columns) else 0.0
        
        c4.metric("சராஸரி மருத்துவமனை தூரம்" if lang == "ta" else "Avg Healthcare Distance", f"{avg_dist:.2f} km")

        CL, CR = st.columns([2, 1])
        with CL:
            st.markdown(f'<p style="font-size:0.8rem; color:#8b949e;">{t("ward_map_hdr", lang)}</p>', unsafe_allow_html=True)
            # Ensure CRS consistency
            if c_gdf.crs is None:
                c_gdf.set_crs("EPSG:4326", inplace=True)
            elif c_gdf.crs != "EPSG:4326":
                c_gdf = c_gdf.to_crs("EPSG:4326")
            
            # Defensive Sanitization: Ensure required columns exist
            for col in ['status', 'risk_level', 'risk_score', 'insight']:
                if col not in c_gdf.columns:
                    c_gdf[col] = 'UNKNOWN' if col != 'risk_score' else 0.0

            c_gdf = c_gdf[c_gdf.geometry.notnull()]
            c_gdf = c_gdf.fillna({
                'risk_score': 0,
                'status': 'UNKNOWN',
                'risk_level': 'UNKNOWN',
                'insight': 'No data'
            })

            m_c = folium.Map(location=[13.0827, 80.2707], zoom_start=11, tiles="CartoDB positron") # Switched to Positron for better contrast
            
            # Task 6: Custom Styling for Missing Data (Grey Rendering)
            def style_function(feature):
                props = feature.get('properties', {})
                status = str(props.get('status', 'VALIDATED'))
                risk = props.get('risk_score', 0)
                if risk is None: risk = 0
                
                # Task 11: Only wards with NO data (even estimates) stay grey
                if 'DATA_NOT_AVAILABLE' in status:
                    return {
                        'fillColor': '#444444',
                        'color': '#888888',
                        'weight': 1,
                        'fillOpacity': 0.6
                    }
                
                # Use a color scale for risk
                if risk > 0.8: color = '#f85149'
                elif risk > 0.6: color = '#fb8c00'
                elif risk > 0.4: color = '#ffd600'
                else: color = '#1a73e8'
                
                return {
                    'fillColor': color,
                    'color': 'white',
                    'weight': 1,
                    'fillOpacity': 0.7
                }

            # Task 4/5: GeoJson with Highlight & Popup
            folium.GeoJson(
                c_gdf,
                style_function=style_function,
                highlight_function=lambda x: {'weight': 4, 'color': '#0366d6', 'fillOpacity': 0.9},
                tooltip=folium.GeoJsonTooltip(
                    fields=['ward_id', 'status', 'risk_level'],
                    aliases=['Ward:', 'Status:', 'Risk Level:'],
                    localize=True
                )
            ).add_to(m_c)

            # Manual Fix: Explicitly use a height and container
            with st.container():
                map_data = st_folium(m_c, width=720, height=520, key="chennai_map_final_v1")
            
            # Task 4: Interactive Selection from Click
            selected_from_map = None
            if map_data and map_data.get('last_object_clicked_props'):
                selected_from_map = map_data['last_object_clicked_props'].get('ward_id')
            
        with CR:
            st.markdown('<p style="font-size:0.8rem; color:#8b949e;">Ward Search & Intelligence</p>', unsafe_allow_html=True)
            
            if 'ward_id' in c_gdf.columns:
                # Task 5: Ward Search Box
                all_wards = sorted(c_gdf['ward_id'].unique())
                initial_index = 0
                if selected_from_map in all_wards:
                    initial_index = all_wards.index(selected_from_map)
                
                selected_ward = st.selectbox(t("ward_select_label", lang), all_wards, index=initial_index)
                
                # Government Audit Log: Ward View
                if 'last_logged_ward' not in st.session_state or st.session_state.last_logged_ward != selected_ward:
                    db.log_event(
                        username=st.session_state.user_info['username'],
                        role=user_role,
                        session_id=st.session_state.session_id,
                        action="WARD_VIEW",
                        target_type="Ward",
                        target_id=str(selected_ward),
                        details=f"Viewed ward {selected_ward} intelligence"
                    )
                    st.session_state.last_logged_ward = selected_ward

                ward_data = c_gdf[c_gdf['ward_id']==selected_ward].iloc[0]
                
                status = ward_data.get('status', 'VALIDATED')
                risk_lvl = ward_data.get('risk_level', 'UNKNOWN')
                insight_txt = ward_data.get('insight', 'No specific insights recorded for this ward.')
                
                border_color = "#444444" if status == "DATA_NOT_AVAILABLE" else "#f85149"
                
                st.markdown(f"""
                <div style="padding:15px; background:#161b22; border-radius:10px; border-left:10px solid {border_color};">
                    <h4 style="margin:0;">{'வார்டு' if lang == 'ta' else 'WARD'} {selected_ward}</h4>
                    <p style="color:#8b949e; font-size:0.85rem;">{'நிலை' if lang == 'ta' else 'Status'}: <b>{status}</b> | {'ஆபத்து நிலை' if lang == 'ta' else 'Risk Level'}: <b>{risk_lvl}</b></p>
                    <hr style="border-color:#21262d;">
                    <p style="font-size:0.9rem;">{insight_txt}</p>
                </div>
                """, unsafe_allow_html=True)
                    
                # Explainability Panel
                st.markdown(f'<p style="font-size:0.8rem; color:#8b949e; margin-top:20px;">{t("risk_drivers_hdr", lang)}</p>', unsafe_allow_html=True)
                
                cols = st.columns(1)
                with cols[0]:
                    metrics = {
                        "Population Density": (ward_data.get('population_density', 0), ward_data.get('shap_population_density', 0)),
                        "Nearest Facility": (ward_data.get('nearest_facility_km', 0), ward_data.get('shap_nearest_facility_km', 0)),
                        "Facility Gap": (ward_data.get('facility_deficit', 0), ward_data.get('shap_facility_deficit', 0)),
                        "Access Demand": (ward_data.get('access_gap', 0), ward_data.get('shap_access_gap', 0))
                    }
                    
                    for label, (val, shap_val) in metrics.items():
                        # Determine color based on SHAP contribution
                        s_color = "#f85149" if shap_val > 0.1 else ("#8b949e" if shap_val < -0.1 else "#30363d")
                        st.markdown(f"""
                        <div style="display:flex; justify-content:between; align-items:center; background:#0d1117; padding:8px; border-radius:5px; margin-bottom:5px; border-left:3px solid {s_color};">
                            <span style="font-size:0.8rem; color:#c9d1d9;">{label}</span>
                            <span style="font-size:0.8rem; color:{s_color}; font-weight:bold;">{val:.2f} (SHAP: {shap_val:+.2f})</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                # --- Missing 4: Resource Planning Layer ---
                st.markdown(f'<p style="font-size:0.8rem; color:#8b949e; margin-top:20px;">{t("resource_proj_hdr", lang)}</p>', unsafe_allow_html=True)
                
                # Dynamic Resource Formulas (Match backend)
                risk_score = float(ward_data.get('risk_score', 0))
                population = float(ward_data.get('population', 0))
                area_km2 = float(ward_data.get('ward_area_km2', 0))
                
                f_teams = int(1 + round(risk_score * 5))
                h_beds  = int(round((population / 1000) * risk_score))
                s_units = int(round(area_km2 * 2 * risk_score))
                
                r_cols = st.columns(3)
                r_metrics = [
                    (t("field_teams_lbl", lang), f_teams, "Units"),
                    (t("beds_lbl", lang),         h_beds,  "Hosp"),
                    (t("spray_lbl", lang),        s_units, "Cap")
                ]
                
                for i, (label, val, unit) in enumerate(r_metrics):
                    with r_cols[i]:
                        st.markdown(f"""
                        <div style="background:#0d1117; padding:10px; border-radius:8px; text-align:center; border:1px solid #30363d;">
                            <div style="font-size:0.65rem; color:#8b949e;">{label}</div>
                            <div style="font-size:1.1rem; font-weight:bold; color:{CYAN};">{val}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # --- Alert Dispatch Workflow (WITH BIOMETRICS) ---
                st.markdown('<br>', unsafe_allow_html=True)
                
                # Biometric Lock Check
                is_locked = st.session_state.get('dispatch_locked_for_session', False)
                is_disabled = (user_role == "VIEWER") or is_locked
                
                btn_label = t("dispatch_alert_ward", lang)
                if is_locked: btn_label = "⛔ DISPATCH LOCKED (Biometric Failure)"
                
                if st.button(btn_label, use_container_width=True, disabled=is_disabled):
                    # Instead of direct dispatch, activate verification
                    st.session_state.pending_dispatch_data = {
                        'district': "Chennai",
                        'disease': "Ward Pulse",
                        'risk_level': "CRITICAL" if risk_score > 0.7 else "HIGH",
                        'notes': f"Automated risk detection for Ward {selected_ward}. Risk Score: {risk_score:.2f}",
                        'triggered_by': f"{user_role} - Administrative Dashboard (Ward {selected_ward})",
                        'resource_data': {
                            'teams': f_teams,
                            'beds': h_beds,
                            'spray': s_units
                        }
                    }
                    st.session_state.dispatch_verification_active = True
                    st.rerun()
                
                if is_locked:
                    st.error("Access Denied: Administrative dispatch is locked for this session due to biometric verification failures.")
                elif is_disabled:
                    st.caption("🔒 Alert dispatch restricted to Medical Officers and Admins.")
            else:
                st.info("Ward identification data missing from source GIS.")
            
            st.info("Decision Support: Evidence-based metrics provided for administrative audit.")
    else:
        st.warning("⚠️ Chennai Ward-Level Pilot data not found. Processing required.")
        is_admin = (user_role == "ADMIN")
        if st.button("🚀 Run Chennai Pilot Build", disabled=not is_admin):
            with st.spinner("Processing official GCC datasets..."):
                os.system("python -m ml.chennai_pilot_build")
                # Log build action
                from backend import db
                db.log_audit(ward_id=0, action="SYSTEM_BUILD_TRIGGERED", user_role=user_role, details="Chennai Pilot Build")
                st.success("Chennai Pilot Build Complete!")
                st.rerun()
        if not is_admin:
            st.caption("🔒 System builds restricted to Administrators.")

        if load_error:
            st.error(f"DEBUG INFO: {load_error}")

        # --- Missing 3: Administrative Audit Trail ---
        st.markdown('<p style="font-size:0.8rem; color:#8b949e; margin-top:30px;">📋 Government Audit Trail (Traceability Record)</p>', unsafe_allow_html=True)
        try:
            from backend import db
            conn = db.get_connection()
            if conn:
                audit_df = pd.read_sql_query("SELECT timestamp, ward_id, action, user_role, user_id, ip_address FROM audit_trail ORDER BY timestamp DESC LIMIT 10", conn)
                if not audit_df.empty:
                    st.dataframe(audit_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No audit logs recorded yet. Restricted actions require authorization.")
                conn.close()
        except Exception as e:
            st.info(f"Audit Database offline: {e}")

        # Data Gap Report (Section 8)
        if os.path.exists("data/chennai/gap_report.json"):
            with open("data/chennai/gap_report.json", "r") as f:
                gap_data = json.load(f)
            st.error(f"Verified Data Gaps Reported: {', '.join(gap_data['gaps'])}")
# ═══════ TAB 2 — 14-DAY OUTLOOK ════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="sec-hdr">📈 {forecast_weeks}-{t("weeks",lang).upper()} {t("forecast_hdr",lang)} — {selected_district.upper()} · {selected_disease.upper()}</div>', unsafe_allow_html=True)

    base = 25
    if not preds.empty:
        loc = preds[(preds['district']==selected_district)&(preds['disease']==selected_disease)]
        if not loc.empty: base = int(loc['predicted_cases'].values[0])

    np.random.seed(42)
    trend    = np.linspace(base, base*1.35, forecast_weeks+1)
    noise    = np.random.normal(0, base*.08, forecast_weeks+1)
    vals     = np.clip(trend+noise, 0, None).astype(int)
    ci_up    = (vals*1.18).astype(int)
    ci_low   = (vals*.82).astype(int)
    dates    = pd.date_range(start=now, periods=forecast_weeks+1, freq='W')
    threshold = int(base*1.4)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates)+list(dates[::-1]), y=list(ci_up)+list(ci_low[::-1]),
        fill='toself', fillcolor='rgba(0,180,216,.07)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, name='95% CI'))
    fig.add_trace(go.Scatter(
        x=dates, y=vals, mode='lines+markers', name=t("forecast_legend",lang),
        line=dict(color=CYAN, width=2.5),
        marker=dict(size=7, color=BG2, line=dict(width=2.5, color=CYAN))))
    fig.add_trace(go.Scatter(
        x=dates, y=ci_up, mode='lines', name=t("upper_ci",lang),
        line=dict(color=CYAN, width=1, dash='dot'), showlegend=True))
    fig.add_hline(y=threshold, line=dict(color=RED, width=1.5, dash='dash'),
                  annotation_text=t("alert_threshold",lang),
                  annotation_font=dict(color=RED, size=10),
                  annotation_position="bottom right")
    make_chart(fig, height=300, rpad=30)
    fig.update_yaxes(title_text=t("cases_label",lang))
    st.plotly_chart(fig, use_container_width=True)

    delta = int(vals[-1]-vals[0])
    mc1, mc2, mc3 = st.columns(3)
    for col, title, value, sub_color in [
        (mc1, t("week_1_forecast",lang),                      str(vals[0]),                       CYAN),
        (mc2, t("week_n_forecast",lang,n=forecast_weeks),     str(vals[-1]),                      RED if delta>0 else GREEN),
        (mc3, t("projected_trend",lang),                      ("↑ " if delta>0 else "↓ ")+str(abs(delta)), RED if delta>0 else GREEN),
    ]:
        col.markdown(f"""
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;
                    padding:12px;text-align:center;">
            <div style="font-size:.62rem;color:#6e7681;text-transform:uppercase;letter-spacing:1px;">{title}</div>
            <div style="font-size:1.7rem;font-weight:800;color:{sub_color};
                        font-family:'JetBrains Mono',monospace;line-height:1.2;">{value}</div>
        </div>""", unsafe_allow_html=True)

# ═══════ TAB 3 — INTELLIGENCE REPORTS ═══════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="sec-hdr">{t("intel_hdr", lang)}</div>', unsafe_allow_html=True)
    if not preds.empty:
        summary_df = preds[preds['disease']==selected_disease][['district', 'risk_level', 'predicted_cases', 'recommendations']]
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Download button for Report
        csv = summary_df.to_csv(index=False).encode('utf-8')
        if st.download_button(t("download_report", lang), csv, f"NalamAI_Report_{now.strftime('%Y%m%d')}.csv", "text/csv"):
            db.log_event(
                username=st.session_state.user_info['username'],
                role=user_role,
                session_id=st.session_state.session_id,
                action="DOWNLOAD_REPORT",
                target_type="Report",
                target_id=f"District Risk Report - {selected_disease}",
                details=f"Downloaded CSV report for {selected_disease}"
            )
    else:
        st.info("தகவல் இல்லை." if lang == "ta" else "No intelligence summary available.")

# ═══════ TAB 4 — ADVANCED VIEW ══════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-hdr">⚙️ SYSTEM DIAGNOSTICS & EXPLAINABILITY</div>', unsafe_allow_html=True)
    
    col_shap, col_metrics = st.columns([1, 1])
    
    with col_shap:
        st.markdown('<p style="font-size:0.8rem; color:#8b949e;">Feature Importance (SHAP)</p>', unsafe_allow_html=True)
        shap_img = f"frontend/shap_plots/{selected_district.lower()}_importance.png"
        if not os.path.exists(shap_img):
            shap_img = "frontend/shap_plots/global_importance.png"
        if os.path.exists(shap_img):
            st.image(shap_img, use_column_width=True)
        else:
            st.warning("SHAP analysis not available for this district.")

    with col_metrics:
        st.markdown('<p style="font-size:0.8rem; color:#8b949e;">Risk Score Distribution</p>', unsafe_allow_html=True)
        if not preds.empty:
            rc = preds.groupby('risk_level').size().reset_index(name='Count')
            fig_d = go.Figure(go.Pie(
                labels=rc['risk_level'], values=rc['Count'],
                hole=.6
            ))
            make_chart(fig_d, height=220)
            st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">🚨 MANUAL ALERT OVERRIDE</div>', unsafe_allow_html=True)
    with st.expander("Dispatch Clinical Alert"):
        with st.form("manual_alert_form"):
            a1, a2 = st.columns(2)
            with a1:
                dist_list = sorted([
                    'Ariyalur', 'Chengalpattu', 'Chennai', 'Coimbatore', 'Cuddalore', 'Dharmapuri', 'Dindigul', 'Erode', 
                    'Kallakurichi', 'Kancheepuram', 'Karur', 'Krishnagiri', 'Madurai', 'Mayiladuthurai', 'Nagapattinam', 
                    'Namakkal', 'Nilgiris', 'Perambalur', 'Pudukkottai', 'Ramanathapuram', 'Ranipet', 'Salem', 'Sivaganga', 
                    'Tenkasi', 'Thanjavur', 'Theni', 'Thoothukudi', 'Tiruchirappalli', 'Tirunelveli', 'Tirupathur', 
                    'Tiruppur', 'Tiruvallur', 'Tiruvannamalai', 'Tiruvarur', 'Vellore', 'Viluppuram', 'Virudhunagar', 'Kanyakumari'
                ])
                alert_dist = st.selectbox("Select District", dist_list, index=dist_list.index("Chennai") if "Chennai" in dist_list else 0)
                alert_disease = st.selectbox("Disease Category", ["Dengue", "Malaria", "Cholera", "Leptospirosis", "Chikungunya", "Viral Fever"])
            
            with a2:
                alert_risk = st.select_slider("Set Risk Severity", options=["LOW", "MEDIUM", "HIGH", "CRITICAL"], value="HIGH")
                alert_notes = st.text_area("Clinical Notes & Directives", placeholder="e.g. Cluster detected in specific block...")

            # Biometric Lock Check
            is_locked = st.session_state.get('dispatch_locked_for_session', False)
            is_manual_disabled = (user_role != "ADMIN") or is_locked
            
            submit_btn_label = "🚀 AUTHORIZE & DISPATCH ALERT"
            if is_locked: submit_btn_label = "⛔ DISPATCH LOCKED (Biometric Failure)"
            
            submit_alert = st.form_submit_button(submit_btn_label, use_container_width=True, disabled=is_manual_disabled)
            
            if is_locked:
                st.error("Biometric Lock: Manual dispatch disabled for security audit.")
            elif is_manual_disabled:
                st.caption("🔒 Manual override restricted to Administrators.")

            if submit_alert:
                # Trigger Biometric Verification Flow
                risk_val = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8, "CRITICAL": 1.0}.get(alert_risk, 0.5)
                st.session_state.pending_dispatch_data = {
                    'district': alert_dist,
                    'disease': alert_disease,
                    'risk_level': alert_risk,
                    'notes': alert_notes,
                    'triggered_by': f"{user_role} - Manual Override",
                    'resource_data': {
                        "teams": int(risk_val * 10),
                        "beds": int(risk_val * 50),
                        "spray": int(risk_val * 100)
                    }
                }
                st.session_state.dispatch_verification_active = True
                st.rerun()

# ═══════ TAB AUDIT — ADMIN ONLY ══════════════════════════════════════════════════
if user_role == "ADMIN":
    with tab_audit:
        st.markdown('<div class="sec-hdr">🔐 GOVERNMENT AUDIT & ACCOUNTABILITY LOGS</div>', unsafe_allow_html=True)
        
        # Filters
        f1, f2, f3 = st.columns([2, 2, 1])
        with f1:
            user_list = ["All Users"] + sorted(db.get_audit_logs(limit=1000)['username'].unique().tolist())
            user_f = st.selectbox("Filter by User", user_list)
        with f2:
            action_list = ["All Actions"] + sorted(db.get_audit_logs(limit=1000)['action'].unique().tolist())
            action_f = st.selectbox("Filter by Action", action_list)
        with f3:
            log_limit = st.number_input("Limit", 10, 5000, 100)

        # Fetch Logs
        u_sel = None if user_f == "All Users" else user_f
        a_sel = None if action_f == "All Actions" else action_f
        logs_df = db.get_audit_logs(limit=log_limit, user_filter=u_sel, action_filter=a_sel)
        
        if not logs_df.empty:
            st.dataframe(logs_df, use_container_width=True, hide_index=True)
            
            # Exports
            e1, e2 = st.columns(2)
            with e1:
                csv_data = logs_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Audit Logs as CSV", csv_data, f"NalamAI_Audit_{now.strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
            with e2:
                if st.button("📄 Prepare PDF Audit Report", use_container_width=True):
                    db.log_event(
                        username=st.session_state.user_info['username'],
                        role=user_role,
                        session_id=st.session_state.session_id,
                        action="PREPARE_PDF_REPORT",
                        target_type="System",
                        target_id="Audit Trail",
                        details=f"Prepared PDF audit trail for last {log_limit} actions"
                    )
                    with st.spinner("Generating Government Standard Header..."):
                        pdf_path = f"nalamai/data/NalamAI_Audit_Trail_{now.strftime('%Y%m%d_%H%M')}.pdf"
                        audit_exporter.generate_audit_pdf(logs_df, pdf_path, date_range=f"Last {log_limit} actions")
                        with open(pdf_path, "rb") as f:
                            st.download_button("📥 Download PDF Audit Trail", f, file_name=f"TN_Health_Audit_{now.strftime('%Y%m%d')}.pdf", use_container_width=True)
        else:
            st.info("No audit logs found matching criteria.")

# ═══════ TAB 5 — NALAMAI ASSISTANT ══════════════════════════════════════════════
if user_role == "ADMIN":
    with tab4:
        st.markdown("---")
        st.markdown('<div class="sec-hdr">👥 GOVERNMENT USER MANAGEMENT (ADMIN ONLY)</div>', unsafe_allow_html=True)
        try:
            conn = db.get_connection()
            users_df = pd.read_sql_query("SELECT id, username, email, role, is_active FROM users", conn)
            st.dataframe(users_df, use_container_width=True, hide_index=True)
            conn.close()
        except:
            st.info("User database initializing...")

        st.markdown("---")
        st.markdown(f'<div class="sec-hdr">👤 {t("face_enroll_title", lang)} (ADMIN ONLY)</div>', unsafe_allow_html=True)
        
        try:
            conn = db.get_connection()
            all_users = pd.read_sql_query("SELECT id, username FROM users", conn)
            conn.close()
            
            user_options = {row['username']: row['id'] for _, row in all_users.iterrows()}
            selected_enroll_user = st.selectbox("Select User to Enroll", list(user_options.keys()), key="enroll_user_sel")
            target_user_id = user_options[selected_enroll_user]
            
            # --- Face Enrollment ---
            st.write(t("face_enroll_desc", lang))
            ec1, ec2 = st.columns(2)
            with ec1:
                img1 = st.camera_input("Enrollment Image 1", key="enroll1")
            with ec2:
                img2 = st.camera_input("Enrollment Image 2", key="enroll2")
                
            if st.button(t("enroll_btn", lang), type="primary", key="live_enroll_btn"):
                if img1 and img2:
                    with st.spinner("Generating secure embeddings..."):
                        from nalamai.backend import face_logic
                        success_count = 0
                        for i, img in enumerate([img1, img2]):
                            img_bytes = img.getvalue()
                            emb = face_logic.generate_embedding(img_bytes)
                            if emb is not None:
                                # Save Embedding to DB
                                blob = face_logic.encrypt_embedding(emb)
                                db.save_face_embedding(target_user_id, blob)
                                
                                # Save Physical Image (reference)
                                if success_count == 0:
                                    face_logic.save_face_image(target_user_id, img_bytes)
                                success_count += 1
                        
                        if success_count > 0:
                            st.success(f"✅ Successfully enrolled {success_count} face embeddings for {selected_enroll_user}")
                            db.log_event(user_name, user_role, st.session_state.session_id, "FACE_ENROLL_LIVE", "User", str(target_user_id), f"Enrolled {success_count} embeddings")
                        else:
                            st.error("❌ Failed to detect face in provided images. Please try again.")
                else:
                    st.warning("Please capture both images.")
                    
        except Exception as e:
            st.error(f"Enrollment Error: {e}")

with tab_bot:
    chatbot_path = os.path.join(os.path.dirname(__file__), 'nalamai_chatbot.html')
    if os.path.exists(chatbot_path):
        with open(chatbot_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=720, scrolling=False)
    else:
        st.error("Chatbot file not found.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:1.5rem;padding:.8rem 0;border-top:1px solid #21262d;
            display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:.66rem;color:#484f58;">
        {t("footer_copy",lang)} · © {now.year}</span>
    <span style="font-size:.64rem;color:#30363d;font-family:'JetBrains Mono',monospace;">
        XGBoost v2 · BiLSTM · SHAP · FastAPI · SQLite</span>
</div>""", unsafe_allow_html=True)
