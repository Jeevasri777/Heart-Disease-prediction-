import streamlit as st
import sqlite3
import hashlib
import joblib
import pandas as pd
import base64
from sklearn.metrics import accuracy_score

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="CardioAI - Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# GLOBAL STYLES
# -------------------------
st.markdown(
    """
    <style>
    /* ===== FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }

    .stApp {
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }

    /* ===== HEADINGS ===== */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        color: #1a1a2e !important;
    }

    h2, h3 {
        font-weight: 700 !important;
        color: #1a1a2e !important;
    }

    /* ===== INPUT LABELS ===== */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #2d3748 !important;
        margin-bottom: 0.3rem !important;
    }

    /* ===== INPUT FIELDS ===== */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15), 0 4px 12px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }

    /* ===== SELECT BOXES ===== */
    div[data-baseweb="select"] {
        border-radius: 12px !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
        background: white !important;
    }

    div[data-baseweb="select"] > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15) !important;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px !important;
        border: none !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }

    .stButton > button:active {
        transform: translateY(-1px) !important;
    }

    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: white !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4) !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.5) !important;
    }

    /* ===== TRANSPARENT FILE UPLOADER WITH NO OUTER BOX ===== */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e0 !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.08) !important;
    }
    
    /* Target the file uploader framing box to keep it clean */
    [data-testid="stFileUploader"] > section {
        background: transparent !important;
        border: none !important;
    }

    /* ===== DATAFRAME ===== */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
    }

    /* ===== SUCCESS/ERROR MESSAGES ===== */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important;
        border-left: 5px solid #28a745 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%) !important;
        border-left: 5px solid #dc3545 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* ===== ANIMATIONS ===== */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    @keyframes heartbeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.15); }
        28% { transform: scale(1); }
        42% { transform: scale(1.15); }
        70% { transform: scale(1); }
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }

    .icon-heart {
        display: inline-block;
        animation: heartbeat 1.5s ease-in-out infinite;
    }

    .icon-float {
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }

    .icon-pulse {
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
    }

    /* ===== GLASSMORPHISM CARD & CONTAINER CENTERING ===== */
    .login-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
        width: 100%;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
        padding: 2rem !important;
        width: 100%;
        max-width: 450px;
        margin: auto;
    }

    /* ===== RESULT CARDS ===== */
    .result-card-success {
        background: linear-gradient(135deg, #d4edda 0%, #b8dfc4 100%);
        border-left: 6px solid #28a745;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.2);
        animation: pulse 2s ease-in-out infinite;
    }

    .result-card-danger {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c2c7 100%);
        border-left: 6px solid #dc3545;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        box-shadow: 0 8px 25px rgba(220, 53, 69, 0.2);
        animation: pulse 1.5s ease-in-out infinite;
    }

    .result-card-success h3,
    .result-card-danger h3 {
        margin: 0 !important;
        font-size: 1.3rem !important;
    }

    /* ===== DASHBOARD CARDS ===== */
    .dashboard-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }

    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12);
    }

    /* ===== METRIC STYLING ===== */
    [data-testid="stMetric"] {
        background: white !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
    }

    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# BACKGROUND FUNCTIONS
# -------------------------
def set_color_bg(color):
    st.markdown(f"<style>.stApp {{ background: {color}; }}</style>", unsafe_allow_html=True)

def set_gradient_bg(gradient):
    st.markdown(f"<style>.stApp {{ background: {gradient}; background-attachment: fixed; }}</style>", unsafe_allow_html=True)

def add_bg(image_file):
    try:
        with open(image_file, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.3)), url("data:image/png;base64,{data}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        set_gradient_bg("linear-gradient(135deg, #667eea 0%, #764ba2 100%)")

# -------------------------
# DATABASE
# -------------------------
conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """
)
conn.commit()

# -------------------------
# FUNCTIONS
# -------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        cursor.execute("INSERT INTO users(username, password) VALUES(?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hash_password(password)))
    return cursor.fetchone()

# -------------------------
# SESSION STATE
# -------------------------
if "login" not in st.session_state:
    st.session_state.login = False
if "page" not in st.session_state:
    st.session_state.page = "Upload Dataset"

# -------------------------
# CALLBACK FOR NAVIGATION
# -------------------------
def handle_nav_change():
    st.session_state.page = st.session_state.navigation_control

# -------------------------
# LOGIN PAGE
# -------------------------
if not st.session_state.login:
    add_bg("picc.png")
    
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
        
        with st.container():
            st.markdown(
                """
                <div class="glass-card">
                    <div style="text-align: center; margin-bottom: 1.2rem;">
                        <span class="icon-heart" style="font-size: 3.5rem;">❤️</span>
                        <h1 style="margin-top: 0.5rem; color: #1a1a2e !important; font-size: 2.2rem !important;">CardioAI</h1>
                        <p style="color: #64748b; font-size: 0.95rem; margin-top: -0.5rem;">Heart Disease Prediction System</p>
                    </div>
                """, 
                unsafe_allow_html=True
            )
            
            choice = st.selectbox("🔐 Account Action", ["Login", "Register"], key="auth_choice")
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            
            st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)

            if choice == "Register":
                if st.button("✨ Create Account", use_container_width=True):
                    if username and password:
                        if register_user(username, password):
                            st.success("🎉 Account created successfully! Please login.")
                        else:
                            st.error("⚠️ Username already exists. Try another.")
                    else:
                        st.warning("Please fill in all fields.")
            else:
                if st.button("🚀 Login", use_container_width=True):
                    if username and password:
                        user = login_user(username, password)
                        if user:
                            st.session_state.login = True
                            st.session_state.page = "Upload Dataset"
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials. Please try again.")
                    else:
                        st.warning("Please fill in all fields.")
            st.markdown('</div></div>', unsafe_allow_html=True)

# -------------------------
# DASHBOARD (AFTER LOGIN)
# -------------------------
else:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <span class="icon-heart" style="font-size: 2.5rem;">❤️</span>
                <h2 style="color: white !important; margin: 0.5rem 0;">CardioAI</h2>
                <p style="color: #94a3b8; font-size: 0.85rem;">AI Healthcare Dashboard</p>
            </div>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 1rem 0;">
            """,
            unsafe_allow_html=True
        )
        
        st.selectbox(
            "📋 Navigation",
            ["Upload Dataset", "Prediction"],
            index=0 if st.session_state.page == "Upload Dataset" else 1,
            key="navigation_control",
            on_change=handle_nav_change
        )

        st.markdown("<br>" * 4, unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.login = False
            st.session_state.page = "Upload Dataset"
            st.rerun()

    # -------------------------
    # UPLOAD DATASET PAGE
    # -------------------------
    if st.session_state.page == "Upload Dataset":
        set_gradient_bg("linear-gradient(135deg, #e3f2fd 0%, #bbdefb 50%, #e1f5fe 100%)")
        
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem 0;">
                <span class="icon-float" style="font-size: 3rem;">📂</span>
                <h1 style="margin-top: 0.5rem;">Upload Dataset</h1>
                <p style="color: #64748b; font-size: 1.1rem;">Upload your CSV file for preprocessing and evaluation</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        file = st.file_uploader(
            "📁 Select CSV File",
            type=["csv"],
            help="Upload a CSV file containing patient data"
        )

        if file:
            df = pd.read_csv(file)
            st.markdown("<br>", unsafe_allow_html=True)

            # --- MODEL ACCURACY CALCULATION ENGINE ---
            accuracy_val = "N/A"

            try:
                model = joblib.load("heart.pkl")
                scaler = joblib.load("scaler.pkl")

                st.write("CSV Columns:", df.columns.tolist())

                possible_targets = [
                    'target',
                    'output',
                    'HeartDisease',
                    'Heart Disease',
                    'condition',
                    'diagnosis'
                ]

                target_col = next(
                    (col for col in df.columns if col.strip().lower() in 
                     [x.lower() for x in possible_targets]),
                    None
                )

                st.write("Detected target:", target_col)

                if target_col is not None:
                    eval_df = df.dropna(subset=[target_col]).copy()

                    X_test = eval_df.drop(columns=[target_col])
                    y_test = eval_df[target_col]

                    st.write("X columns before:", X_test.columns.tolist())

                    if hasattr(scaler, 'feature_names_in_'):
                        X_test = X_test[scaler.feature_names_in_]

                    X_test = X_test.fillna(X_test.mean())
                    X_scaled = scaler.transform(X_test)
                    predictions = model.predict(X_scaled)

# Convert actual labels to match model integer output
                    y_test = y_test.astype(str).str.strip().str.lower()

                    y_test = y_test.replace({
    "absence": 0,
    "presence": 1
})

                    predictions = predictions.astype(int)

                    accuracy_val = f"{accuracy_score(y_test, predictions)*100:.2f}%"
                else:
                    st.write("Target not detected")
                    accuracy_val = "Target Missing"

            except Exception as e:
                st.write("ERROR:", e)
                accuracy_val = "Error"

            # 5-Column Dashboard Statistics Layout
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f'<div class="dashboard-card" style="text-align: center;"><span class="icon-pulse" style="font-size: 2rem;">📊</span><h3 style="color: #667eea; margin: 0.5rem 0;">{df.shape[0]}</h3><p style="color: #64748b; margin: 0;">Total Rows</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="dashboard-card" style="text-align: center;"><span class="icon-pulse" style="font-size: 2rem;">📋</span><h3 style="color: #667eea; margin: 0.5rem 0;">{df.shape[1]}</h3><p style="color: #64748b; margin: 0;">Total Columns</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="dashboard-card" style="text-align: center;"><span class="icon-pulse" style="font-size: 2rem;">⚠️</span><h3 style="color: #e74c3c; margin: 0.5rem 0;">{df.isnull().sum().sum()}</h3><p style="color: #64748b; margin: 0;">Missing Values</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="dashboard-card" style="text-align: center;"><span class="icon-pulse" style="font-size: 2rem;">🔄</span><h3 style="color: #f39c12; margin: 0.5rem 0;">{df.duplicated().sum()}</h3><p style="color: #64748b; margin: 0;">Duplicate Values</p></div>', unsafe_allow_html=True)
            with col5:
                st.markdown(f'<div class="dashboard-card" style="text-align: center;"><span class="icon-pulse" style="font-size: 2rem;">🎯</span><h3 style="color: #2ecc71; margin: 0.5rem 0;">{accuracy_val}</h3><p style="color: #64748b; margin: 0;">Model Accuracy</p></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="dashboard-card"><h3>📊 Original Dataset Preview</h3></div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)

            df = df.drop_duplicates()
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype == "object":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                        df[col].fillna(df[col].mean(), inplace=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="result-card-success"><h3>✅ Data Cleaning Complete</h3><p style="margin: 0.5rem 0 0 0;">Missing values filled and duplicates removed.</p></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="dashboard-card"><h3>✨ Cleaned Dataset Preview</h3></div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)

            df.to_csv("final_dataset.csv", index=False)
            st.markdown("<br>", unsafe_allow_html=True)
            st.success("💾 Dataset saved as `final_dataset.csv`")
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
            with col_b2:
                if st.button("🔮 Go To Prediction", use_container_width=True):
                    st.session_state.page = "Prediction"
                    st.rerun()

    # -------------------------
    # PREDICTION PAGE
    # -------------------------
    if st.session_state.page == "Prediction":
        set_gradient_bg("linear-gradient(135deg, #fce4ec 0%, #f8bbd9 30%, #f3e5f5 100%)")

        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0 2rem 0;">
                <span class="icon-heart" style="font-size: 3.5rem;">❤️</span>
                <h1 style="margin-top: 0.5rem;">Heart Disease Prediction</h1>
                <p style="color: #64748b; font-size: 1.1rem;">AI-powered cardiovascular risk assessment</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        model = joblib.load("heart.pkl")
        scaler = joblib.load("scaler.pkl")

        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Patient Information")
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("🎂 Age", min_value=1, max_value=120, value=50, step=1)
            sex = st.selectbox("👤 Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.number_input("💔 Chest Pain Type (0-3)", min_value=0, max_value=3, value=0, step=1)
            bp = st.number_input("🩺 Blood Pressure", min_value=0, max_value=300, value=120, step=1)
            chol = st.number_input("🧪 Cholesterol", min_value=0, max_value=600, value=200, step=1)
        with col2:
            fbs = st.selectbox("🍬 Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            ekg = st.number_input("📈 EKG Results (0-2)", min_value=0, max_value=2, value=0, step=1)
            max_hr = st.number_input("💓 Max Heart Rate", min_value=0, max_value=250, value=150, step=1)
            ex_angina = st.selectbox("🏃 Exercise Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col3:
            st_dep = st.number_input("📉 ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            slope = st.number_input("📐 Slope of ST (0-2)", min_value=0, max_value=2, value=0, step=1)
            vessels = st.number_input("🔬 Number of Vessels (0-3)", min_value=0, max_value=3, value=0, step=1)
            thal = st.number_input("💉 Thallium (0-3)", min_value=0, max_value=3, value=0, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Centered button via a lightweight HTML container block
        st.markdown(
            """
            <div style="display: flex; justify-content: center; width: 100%;">
                <div style="width: 33%;">
            """, 
            unsafe_allow_html=True
        )
        predict_clicked = st.button("🔮 Analyze Risk", use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        if predict_clicked:
            data = pd.DataFrame(
                [[
                    int(age), sex, int(cp), int(bp), int(chol),
                    fbs, int(ekg), int(max_hr), ex_angina,
                    float(st_dep), int(slope), int(vessels), int(thal)
                ]],
                columns=[
                    "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
                    "FBS over 120", "EKG results", "Max HR", "Exercise angina",
                    "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium"
                ]
            )

            result = model.predict(scaler.transform(data))
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if result[0] == 1:
                    st.markdown('<div class="result-card-danger" style="text-align: center;"><span style="font-size: 3rem;">⚠️</span><h2 style="color: #c0392b; margin: 0.5rem 0;">Heart Disease Detected</h2><p style="color: #721c24; margin: 0;">Please consult a healthcare professional immediately.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card-success" style="text-align: center;"><span style="font-size: 3rem;">✅</span><h2 style="color: #155724; margin: 0.5rem 0;">No Heart Disease</h2><p style="color: #155724; margin: 0;">Results indicate low cardiovascular risk.</p></div>', unsafe_allow_html=True)
