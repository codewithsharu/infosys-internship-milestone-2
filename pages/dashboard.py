import streamlit as st

# Configure page
st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for black and white theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        letter-spacing: -0.02em;
    }
    
    .main-title::after {
        content: '';
        display: block;
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #ffffff 0%, #666666 100%);
        margin: 1rem auto;
        border-radius: 2px;
    }
    
    /* Welcome message styling */
    .welcome-container {
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid #333;
    }
    
    .welcome-text {
        font-size: 1.5rem;
        font-weight: 600;
        color: #000000;
        text-align: center;
        margin: 0;
    }
    
    /* Dashboard grid */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f8f8 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        border-color: #000000;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #000000 0%, #666666 100%);
    }
    
    .feature-icon {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1rem;
        filter: grayscale(100%);
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #000000;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        font-size: 0.95rem;
        color: #666666;
        text-align: center;
        line-height: 1.6;
    }
    
    /* Stats section */
    .stats-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #000000 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 1px solid #333;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        border: 1px solid #333;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #cccccc;
        font-weight: 400;
        margin-top: 0.5rem;
    }
    
    /* Logout button styling */
    .logout-container {
        display: flex;
        justify-content: center;
        margin: 3rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #000000 0%, #333333 100%);
        color: white;
        border: 2px solid #ffffff;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        color: #000000;
        border-color: #000000;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Warning message styling */
    .stAlert {
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        border: 2px solid #000000;
        border-radius: 12px;
        color: #000000;
    }
    
    /* Navigation hint */
    .nav-hint {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background: rgba(0,0,0,0.8);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        backdrop-filter: blur(10px);
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Check if logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.markdown('<div class="main-title">📊 Dashboard</div>', unsafe_allow_html=True)
    st.warning("⚠️ Please log in first to access your dashboard.")
    if st.button("Go to Login"):
        st.switch_page("pages/login.py")
    st.stop()

# Main dashboard content
st.markdown('<div class="main-title">📊 Dashboard</div>', unsafe_allow_html=True)

# Welcome message
st.markdown(f"""
<div class="welcome-container">
    <div class="welcome-text">
        Welcome back, {st.session_state.username}! 🎉
    </div>
</div>
""", unsafe_allow_html=True)

# Stats section
st.markdown("""
<div class="stats-container">
    <div class="stats-grid">
        <div class="stat-item">
            <span class="stat-number">3</span>
            <div class="stat-label">Available Tools</div>
        </div>
        <div class="stat-item">
            <span class="stat-number">24/7</span>
            <div class="stat-label">Access</div>
        </div>
        <div class="stat-item">
            <span class="stat-number">∞</span>
            <div class="stat-label">Possibilities</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Feature cards
st.markdown('<div class="dashboard-grid">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📝</div>
        <div class="feature-title">Text Summarizer</div>
        <div class="feature-description">
            Transform lengthy documents into concise, meaningful summaries. 
            Perfect for research, articles, and reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔄</div>
        <div class="feature-title">Paraphrasing Tool</div>
        <div class="feature-description">
            Rewrite and enhance your content while maintaining the original meaning. 
            Improve clarity and style effortlessly.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">👤</div>
        <div class="feature-title">User Profile</div>
        <div class="feature-description">
            Manage your account settings, preferences, and view your usage statistics. 
            Customize your experience.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Action buttons
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

with col1:
    if st.button("📝 Summarizer", use_container_width=True):
        st.switch_page("pages/summarizer.py")

with col2:
    if st.button("🔄 Paraphraser", use_container_width=True):
        st.switch_page("pages/paraphraser.py")

with col3:
    if st.button("👤 Profile", use_container_width=True):
        st.switch_page("pages/profile.py")

with col4:
    if st.button("⚙️ Settings", use_container_width=True):
        st.info("Settings page coming soon!")

with col5:
    if st.button("📊 Analytics", use_container_width=True):
        st.info("Analytics page coming soon!")

# Logout section
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="logout-container">', unsafe_allow_html=True)

if st.button("🚪 Logout", key="logout_btn"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out successfully!")
    st.switch_page("app.py")

st.markdown('</div>', unsafe_allow_html=True)

# Navigation hint
st.markdown("""
<div class="nav-hint">
    💡 Tip: Click on any tool to get started
</div>
""", unsafe_allow_html=True)