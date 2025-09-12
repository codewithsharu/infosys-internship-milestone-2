import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.db_connection import create_user

# Page Configuration
st.set_page_config(
    page_title="Create Account - AI Studio",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Premium Black & White UI Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
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
    padding: 2rem 1rem;
    max-width: 100%;
    background: transparent;
}

/* Main Container */
.signup-wrapper {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem 1rem;
    background: #000000;
    position: relative;
}

.signup-container {
    background: #111111;
    border: 2px solid #333333;
    padding: 3rem 2.5rem;
    border-radius: 20px;
    max-width: 480px;
    width: 100%;
    position: relative;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 20px 60px rgba(255, 255, 255, 0.05);
}

.signup-container:hover {
    border-color: #555555;
    transform: translateY(-3px);
    box-shadow: 0 25px 80px rgba(255, 255, 255, 0.08);
}

/* Header Section */
.signup-header {
    text-align: center;
    margin-bottom: 3rem;
}

.signup-logo {
    width: 70px;
    height: 70px;
    background: #ffffff;
    color: #000000;
    border-radius: 16px;
    margin: 0 auto 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    font-weight: 700;
    transition: all 0.3s ease;
}

.signup-logo:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 25px rgba(255, 255, 255, 0.2);
}

.signup-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
    line-height: 1.2;
}

.signup-subtitle {
    color: #cccccc;
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 2.5rem;
    line-height: 1.5;
}

/* Form Elements */
.stTextInput > div > div {
    margin-bottom: 1rem;
}

.stTextInput > div > div > input {
    background-color: #1a1a1a;
    border: 2px solid #333333;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-size: 1rem;
    color: #ffffff;
    font-weight: 400;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-family: 'Inter', sans-serif;
}

.stTextInput > div > div > input:focus {
    border-color: #666666;
    background-color: #222222;
    outline: none;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.1);
}

.stTextInput > div > div > input::placeholder {
    color: #888888;
    font-weight: 400;
}

/* Form Labels */
.stTextInput > label {
    color: #ffffff;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
    display: block;
}

/* Button Styling */
.stButton > button {
    background: #ffffff;
    color: #000000;
    border: none;
    border-radius: 12px;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    font-weight: 700;
    width: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    margin-top: 1rem;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    background: #f0f0f0;
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(255, 255, 255, 0.2);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Form Container */
.stForm {
    border: none;
    background: transparent;
    padding: 0;
}

/* Success/Error Messages */
.stSuccess {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid #22c55e;
    border-radius: 12px;
    color: #22c55e;
    padding: 1rem;
    font-weight: 500;
    margin: 1rem 0;
}

.stError {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid #ef4444;
    border-radius: 12px;
    color: #ef4444;
    padding: 1rem;
    font-weight: 500;
    margin: 1rem 0;
}

.stWarning {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid #f59e0b;
    border-radius: 12px;
    color: #f59e0b;
    padding: 1rem;
    font-weight: 500;
    margin: 1rem 0;
}

.stInfo {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid #3b82f6;
    border-radius: 12px;
    color: #3b82f6;
    padding: 1rem;
    font-weight: 500;
    margin: 1rem 0;
}

/* Loading Spinner */
.stSpinner {
    color: #ffffff;
}

/* Footer Link */
.footer-link {
    text-align: center;
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 1px solid #333333;
}

.footer-link p {
    color: #cccccc;
    font-size: 0.9rem;
    margin: 0;
}

.footer-link a {
    color: #ffffff;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s ease;
    margin-left: 0.5rem;
}

.footer-link a:hover {
    color: #cccccc;
    text-decoration: underline;
}

/* Field Requirements */
.field-help {
    color: #888888;
    font-size: 0.85rem;
    margin-top: 0.25rem;
    font-style: italic;
}

/* Form Progress Indicator */
.form-progress {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 2rem;
    justify-content: center;
}

.progress-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #333333;
    transition: background 0.3s ease;
}

.progress-dot.active {
    background: #ffffff;
}

/* Responsive Design */
@media (max-width: 640px) {
    .signup-container {
        padding: 2rem 1.5rem;
        margin: 1rem;
    }
    
    .signup-title {
        font-size: 1.8rem;
    }
    
    .signup-subtitle {
        font-size: 0.9rem;
    }
}

/* Hide Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}

/* Toast Notifications Enhancement */
.stToast {
    background: #111111;
    border: 1px solid #333333;
    color: #ffffff;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# # Main signup page content
# st.markdown("""
# <div class="signup-wrapper">
#     <div class="signup-container">
#         <div class="signup-header">
#             <div class="signup-logo">🚀</div>
#             <h1 class="signup-title">Create Account</h1>
#             <p class="signup-subtitle">Join our AI-powered platform and start transforming your text with advanced paraphrasing technology</p>
#         </div>
# """, unsafe_allow_html=True)

# Information message
st.info("🔐 Create Account")

# Registration form
with st.form("signup_form", clear_on_submit=False):
    # Form fields with enhanced styling
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input(
            "👤 Full Name", 
            placeholder="Enter your full name",
            help="Your display name on the platform"
        )
        
    with col2:
        username = st.text_input(
            "🏷️ Username", 
            placeholder="Choose a unique username",
            help="This will be your unique identifier"
        )
    
    email = st.text_input(
        "📧 Email Address", 
        placeholder="your.email@domain.com",
        help="We'll use this to verify your account"
    )
    
    password = st.text_input(
        "🔐 Password", 
        type="password",
        placeholder="Create a strong password",
        help="Minimum 8 characters recommended"
    )
    
    # Progress indicator
    filled_fields = sum([bool(name), bool(email), bool(username), bool(password)])
    st.markdown(f"""
    <div class="form-progress">
        <div class="progress-dot {'active' if filled_fields >= 1 else ''}"></div>
        <div class="progress-dot {'active' if filled_fields >= 2 else ''}"></div>
        <div class="progress-dot {'active' if filled_fields >= 3 else ''}"></div>
        <div class="progress-dot {'active' if filled_fields >= 4 else ''}"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Submit button
    submitted = st.form_submit_button("🚀 Create Account", use_container_width=True)
    
    # Form processing
    if submitted:
        if name and email and username and password:
            # Validation feedback
            if len(password) < 6:
                st.warning("⚠️ Password should be at least 6 characters long")
            elif "@" not in email:
                st.warning("⚠️ Please enter a valid email address")
            else:
                # Show loading state
                with st.spinner("🔄 Creating your account..."):
                    success = create_user(name, email, username, password)
                    
                if success:
                    st.success("✅ **Account created successfully!** Welcome to AI Studio")
                    st.balloons()
                    st.info("🔄 Redirecting to login page...")
                    # Add a small delay for better UX
                    import time
                    time.sleep(2)
                    st.switch_page("pages/login.py")
                else:
                    st.error("❌ **Registration failed** • Username or email may already exist")
                    st.info("💡 Try using a different username or email address")
        else:
            st.warning("⚠️ **All fields are required** • Please fill in all information")

# Close container div
st.markdown('</div></div>', unsafe_allow_html=True)

# Footer with login link
st.markdown("""
<div class="footer-link">
    <p>Already have an account? <a href="/login">Sign in here</a></p>
</div>
""", unsafe_allow_html=True)

# Additional JavaScript for enhanced UX (optional)
st.markdown("""
<script>
// Auto-focus first field
document.addEventListener('DOMContentLoaded', function() {
    const firstInput = document.querySelector('input[type="text"]');
    if (firstInput) {
        firstInput.focus();
    }
});

// Form validation hints
document.querySelectorAll('input').forEach(input => {
    input.addEventListener('blur', function() {
        if (this.value.length > 0) {
            this.style.borderColor = '#666666';
        }
    });
});
</script>
""", unsafe_allow_html=True)