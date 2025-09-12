import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.db_connection import validate_user

# Page Configuration
st.set_page_config(
    page_title="Sign In",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Professional Black & White UI Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Reset & Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.main {
    background: #000000;
    color: #ffffff;
    min-height: 100vh;
    padding: 0;
}

.stApp {
    background: #000000;
    color: #ffffff;
}

.main .block-container {
    padding: 4rem 1rem;
    max-width: 440px;
    margin: 0 auto;
}

/* Main Container */
.login-container {
    background: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 3rem 2rem;
    width: 100%;
    max-width: 400px;
    margin: 0 auto;
    transition: border-color 0.3s ease;
}

.login-container:hover {
    border-color: #404040;
}

/* Header */
.header {
    text-align: center;
    margin-bottom: 2.5rem;
}

.welcome-icon {
    width: 60px;
    height: 60px;
    background: #ffffff;
    border-radius: 12px;
    margin: 0 auto 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    color: #000000;
    font-weight: 600;
    transition: transform 0.3s ease;
}

.welcome-icon:hover {
    transform: scale(1.05);
}

.title {
    font-family: 'Inter', sans-serif;
    font-size: 1.875rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.5rem;
    letter-spacing: -0.025em;
}

.subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.9375rem;
    color: #a3a3a3;
    font-weight: 400;
    line-height: 1.5;
}

/* Form Elements */
.stTextInput > label {
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.stTextInput > div > div > input {
    background: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 0.875rem 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.9375rem;
    color: #ffffff;
    font-weight: 400;
    transition: all 0.2s ease;
    width: 100%;
}

.stTextInput > div > div > input:focus {
    border-color: #525252;
    outline: none;
    background: #222222;
    box-shadow: 0 0 0 3px rgba(82, 82, 82, 0.1);
}

.stTextInput > div > div > input::placeholder {
    color: #737373;
}

/* Button */
.stButton > button {
    background: #ffffff;
    color: #000000;
    border: none;
    border-radius: 8px;
    padding: 0.875rem 1.5rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.9375rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.2s ease;
    margin-top: 1.5rem;
    letter-spacing: 0.025em;
}

.stButton > button:hover {
    background: #f5f5f5;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.1);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Form Container */
.stForm {
    background: transparent;
    border: none;
    padding: 0;
}

/* Messages */
.stInfo {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid #3b82f6;
    border-radius: 6px;
    color: #60a5fa;
    padding: 0.75rem;
    font-size: 0.875rem;
    margin-bottom: 1.5rem;
}

.stSuccess {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid #22c55e;
    border-radius: 6px;
    color: #4ade80;
    padding: 0.75rem;
    font-size: 0.875rem;
}

.stError {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid #ef4444;
    border-radius: 6px;
    color: #f87171;
    padding: 0.75rem;
    font-size: 0.875rem;
}

/* Field Spacing */
.stTextInput {
    margin-bottom: 1rem;
}

/* Loading Spinner */
.stSpinner {
    text-align: center;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 1px solid #2a2a2a;
}

.footer-text {
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    color: #a3a3a3;
}

.footer-link {
    color: #ffffff;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s ease;
}

.footer-link:hover {
    color: #d4d4d4;
    text-decoration: underline;
}

/* Demo Credentials */
.demo-section {
    background: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1.5rem;
}

.demo-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.8125rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.025em;
}

.demo-item {
    font-family: 'Inter', sans-serif;
    font-size: 0.8125rem;
    color: #a3a3a3;
    margin: 0.25rem 0;
    display: flex;
    justify-content: space-between;
}

.demo-label {
    font-weight: 500;
}

.demo-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: #ffffff;
    background: #2a2a2a;
    padding: 0.125rem 0.375rem;
    border-radius: 4px;
    font-size: 0.75rem;
}

/* Quick Actions */
.quick-actions {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
}

.quick-btn {
    background: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 0.5rem;
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    flex: 1;
    text-align: center;
}

.quick-btn:hover {
    background: #2a2a2a;
    border-color: #525252;
}

/* Responsive */
@media (max-width: 480px) {
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 100%;
    }
    
    .login-container {
        padding: 2rem 1.5rem;
        border-radius: 8px;
    }
    
    .title {
        font-size: 1.5rem;
    }
    
    .welcome-icon {
        width: 50px;
        height: 50px;
        font-size: 1.25rem;
    }
}

/* Hide Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}

/* Page Link Styling */
.stPageLink > a {
    color: #ffffff !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    transition: color 0.2s ease !important;
}

.stPageLink > a:hover {
    color: #d4d4d4 !important;
    text-decoration: underline !important;
}
</style>
""", unsafe_allow_html=True)

# Main content
# st.markdown("""
# <div class="login-container">
#     <div class="header">
#         <div class="welcome-icon">⚡</div>
#         <h1 class="title">Welcome Back</h1>
#         <p class="subtitle">Sign in to access your AI workspace</p>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# Demo credentials section
st.markdown("""
<div class="demo-section">
    <div class="demo-title">Demo Credentials</div>
    <div class="demo-item">
        <span class="demo-label">Username:</span>
        <span class="demo-value">shareen</span>
    </div>
    <div class="demo-item">
        <span class="demo-label">Password:</span>
        <span class="demo-value">infosys@123</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Clean, professional form
with st.form("login_form", clear_on_submit=False):
    
    username = st.text_input(
        "Username",
        placeholder="Enter your username",
        value=""
    )
    
    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter your password",
        value=""
    )
    
    # Quick action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.form_submit_button("🚀 Sign In", use_container_width=True):
            submitted = True
        else:
            submitted = False
    
    with col2:
        demo_login = st.form_submit_button("⚡ Demo Login", use_container_width=True)
    
    # Handle demo login
    if demo_login:
        username = "demo"
        password = "demo123"
        submitted = True
    
    # Form processing
    if submitted:
        if username and password:
            with st.spinner("Authenticating..."):
                user = validate_user(username, password)
                
            if user:
                st.session_state.logged_in = True
                st.session_state.username = user["username"]
                st.success("✅ Login successful!")
                st.info("🔄 Redirecting to dashboard...")
                import time
                time.sleep(1.5)
                st.switch_page("pages/dashboard.py")
            else:
                st.error("❌ Invalid credentials. Please check your username and password.")
        else:
            st.error("❌ Please fill in all fields.")

# Footer with signup link
st.markdown("""
<div class="footer">
    <p class="footer-text">
        Don't have an account? 
        <a href="/signup" class="footer-link">Create one here</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Alternative: Use Streamlit's page link if the above doesn't work
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.page_link("pages/signup.py", label="")