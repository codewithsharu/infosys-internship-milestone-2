import streamlit as st
from transformers import pipeline
from backend import db_connection # Import db_connection

# -----------------------------
# Summarizer model
# -----------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# -----------------------------
# Functions
# -----------------------------
def signup():
    st.title("📝 Signup Page")
    # name = st.text_input("Enter your Name") # Removed name input
    email = st.text_input("Enter your Email")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")

    if st.button("Signup"):
        if email and username and password: # Removed 'name'
            # Use db_connection to create user
            if db_connection.create_user(email, username, password): # Removed 'name' argument
                st.success("✅ Account created! Please log in now.")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error("⚠️ An account with this username or email already exists. Please choose another.")
        else:
            st.error("⚠️ Please fill in all fields.")

def login():
    st.title("🔒 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Use db_connection to validate user
        user = db_connection.validate_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username = user['username'] # Store username in session
            st.session_state.user_role = user['role'] # Store user role in session
            st.success("✅ Login successful! Redirecting...")
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    if st.button("Go to Signup"):
        st.session_state.page = "signup"
        st.rerun()

def summarizer_page():
    st.title("📄 Text Summarizer")

    text = st.text_area("Enter the text you want to summarize:", height=200)

    if st.button("Summarize"):
        if text.strip():
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
            st.subheader("✨ Summary:")
            st.write(summary[0]['summary_text'])
        else:
            st.error("⚠️ Please enter some text to summarize.")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login" # Ensure it redirects to login after logout
        st.rerun()

# -----------------------------
# Main App
# -----------------------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "login"

    if st.session_state.page == "signup":
        signup()
    elif st.session_state.page == "login":
        login()
    elif st.session_state.page == "dashboard" and st.session_state.logged_in: # New dashboard page
        st.title("Welcome to Dashboard!") # Placeholder for dashboard content
        # In a real app, you'd import and call a dashboard function here
        if st.button("Go to Summarizer"): # Example navigation to summarizer
            st.session_state.page = "summarizer"
            st.rerun()
        if st.button("Logout"): # Logout from dashboard
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()
    elif st.session_state.page == "summarizer" and st.session_state.logged_in:
        summarizer_page()
    else:
        st.session_state.page = "login"
        login()

if __name__ == "__main__":
    main()