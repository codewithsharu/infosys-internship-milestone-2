import streamlit as st
from transformers.pipelines import pipeline
import textstat
import plotly.express as px
import plotly.graph_objects as go
import time
import pandas as pd
import csv
import os

# -----------------------------
# Page Configuration & Custom CSS
# -----------------------------
st.set_page_config(
    page_title="AI Paraphrasing Studio",
    page_icon="🖤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for elegant black and white theme
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        background: #000000;
        color: #ffffff;
        min-height: 100vh;
    }
    
    .stApp {
        background: #000000;
        color: #ffffff;
    }
    
    /* Remove default Streamlit styling */
    .stApp > header {
        background: transparent;
    }
    
    .stApp > div > div > div > div > section {
        background: #000000;
    }
    
    /* Main container styling */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Header styling */
    .header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .header p {
        color: #cccccc;
        font-size: 1.1rem;
        margin: 0;
    }

    /* Column styling */
    .column-header {
        background: #111111;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #333333;
    }
    
    .column-header h3 {
        color: #ffffff;
        margin: 0;
        font-size: 1.2rem;
        font-weight: 500;
    }

    /* Content boxes */
    .content-box {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1.5rem;
        height: 400px;
        overflow-y: auto;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    .placeholder-box {
        background: #111111;
        border: 2px dashed #333333;
        border-radius: 8px;
        padding: 2rem;
        height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #cccccc;
    }
    
    .placeholder-content {
        color: #cccccc;
    }
    
    /* Controls styling */
    .controls-section {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .controls-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: #ffffff;
        color: #000000;
        border: none;
        padding: 0.8rem 2.5rem;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background: #f0f0f0;
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(255,255,255,0.1);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Form elements */
    .stSelectbox > div > div {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 8px;
        color: #ffffff;
    }
    
    .stSelectbox > div > div > div {
        color: #ffffff;
    }
    
    .stRadio > div {
        background: #111111;
        border: 1px solid #333333;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .stRadio > div > label {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .stRadio > div > div {
        gap: 1rem;
    }
    
    /* Text areas */
    .stTextArea > div > div > textarea {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 12px;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #666666;
        outline: none;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #333333;
        color: #666666;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #333333;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555555;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Paraphraser with Loading State
# -----------------------------
@st.cache_resource
def load_paraphraser(model_name):
    return pipeline("text2text-generation", model=model_name)

# -----------------------------
# Helper Functions for Text Statistics
# -----------------------------
def calculate_reading_time(text):
    """Calculate estimated reading time (assuming 200 words per minute)"""
    word_count = len(text.split())
    return max(1, round(word_count / 200))

def get_text_stats(text):
    """Get basic statistics about the text"""
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    characters = len(text)
    return len(words), sentences, characters

def find_reference_paraphrase(input_text, csv_path='pages/paraphrase.csv'):
    """
    Search for the input_text in the paraphrase.csv file.
    If found, return (original_text, paraphrase) tuple.
    If not found, return None.
    """
    if not input_text or not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                if len(row) < 2:
                    continue
                original = row[0].strip().strip('"')
                paraphrase = row[1].strip().strip('"')
                # Exact match (ignoring leading/trailing whitespace and quotes)
                if input_text.strip() == original:
                    return (original, paraphrase)
    except Exception as e:
        st.error(f"Error reading paraphrase.csv: {e}")
        return None
    return None

# -----------------------------
# AI Loading Component
# -----------------------------
def show_ai_loader(text="Processing with AI"):
    return f"""
    <div class="ai-loader">
        <div class="loader-dots">
            <div class="loader-dot"></div>
            <div class="loader-dot"></div>
            <div class="loader-dot"></div>
        </div>
        <div class="loader-text">{text}</div>
    </div>
    """

# -----------------------------
# Main Application
# -----------------------------
def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>🖤 AI Paraphrasing Studio</h1>
        <p>Transform your text with precision AI technology and comprehensive readability analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for loading
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Vamsi/T5_Paraphrase_Paws" # Default model

    # Check if the selected model has changed
    if st.session_state.get("model_selector") and st.session_state.selected_model != st.session_state.model_selector:
        st.session_state.model_loaded = False
        st.session_state.selected_model = st.session_state.model_selector
        st.rerun()

    # Load model with loading screen
    if not st.session_state.model_loaded:
        st.markdown(show_ai_loader("Initializing AI Models"), unsafe_allow_html=True)
        selected_model = st.session_state.selected_model # Get the selected model from the selectbox
        paraphraser = load_paraphraser(selected_model)
        st.session_state.paraphraser = paraphraser
        st.session_state.model_loaded = True
        time.sleep(1)
        st.rerun()
    
    paraphraser = st.session_state.paraphraser

    # Controls Section
    st.markdown("""
    <div class="controls-section">
        <div class="controls-title">Configuration Settings</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model and parameter controls in a single row
    control_col1, control_col2 = st.columns([3, 1])
    
    with control_col1:
        selected_model = st.selectbox(
            "Select AI Model",
            ["Vamsi/T5_Paraphrase_Paws", "tuner007/pegasus_paraphrase", "MBZUAI/LaMini-Flan-T5-248M"],
            index=["Vamsi/T5_Paraphrase_Paws", "tuner007/pegasus_paraphrase", "MBZUAI/LaMini-Flan-T5-248M"].index(st.session_state.selected_model),
            help="Choose the AI model for paraphrasing",
            key="model_selector"
        )
    st.session_state.selected_model = selected_model

    with control_col2:
        generate_button = st.button(
            "🚀 Generate Paraphrase",
            type="primary",
            use_container_width=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main three-column layout
    input_col, paraphrase_col, reference_col = st.columns([1, 1, 1])
    
    # Input Column
    with input_col:
        st.markdown("""
        <div class="column-header">
            <h3>📝 Input Text</h3>
        </div>
        """, unsafe_allow_html=True)
        
        user_text = st.text_area(
            "", 
            value="", 
            height=300,
            placeholder="""Enter your text here for paraphrasing...

Examples:
• Rephrase sentences for clarity
• Vary sentence structure
• Improve vocabulary
• Adjust tone and style
• Avoid plagiarism""",
            key="user_text_input"
        )
        
        # Display text statistics if there's input
        if user_text:
            word_count, sentence_count, char_count = get_text_stats(user_text)
            reading_time = calculate_reading_time(user_text)
            
            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 1rem; background: #111111; border-radius: 6px;">
                <small style="color: #cccccc;">
                    <strong>Stats:</strong> {word_count:,} words • {sentence_count} sentences • {reading_time} min read
                </small>
            </div>
            """, unsafe_allow_html=True)
    
    # Paraphrased Output Column
    with paraphrase_col:
        st.markdown("""
        <div class="column-header">
            <h3>✨ Generated Paraphrase</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if generate button was clicked and there's text to process
        if generate_button and user_text.strip():
            # Show processing status
            loader_placeholder = st.empty()
            loader_placeholder.markdown(show_ai_loader("Analyzing and transforming text"), unsafe_allow_html=True)
            
            # Processing parameters
            # Dynamically set max_len based on input length, ensuring it's at least 150 and at most 750
            max_len = min(750, max(150, int(len(user_text.split()) * 2.68)))
            num_return = 1  # Default number of return sequences

            # Generate paraphrased versions
            # The paraphraser is already loaded with the selected model.
            # We no longer have 'style' or 'level' controls.
            
            try:
                result = paraphraser(
                    user_text, # Pass user_text directly without style modification
                    max_length=max_len,
                    num_return_sequences=num_return,
                    min_length=max(10, int(len(user_text.split()) * 1.4)), # Ensure at least 40% more than input, min 10 words
                    do_sample=True, # Enable sampling for more diverse paraphrases
                    temperature=1.5 # Adjust for creativity
                )
                
                # Clear loader
                loader_placeholder.empty()
                
                paraphrased_output = result[0]['generated_text']
                
                # Display the paraphrased output
                st.markdown(f"""
                <div class="content-box">
                    <p style="line-height: 1.6; margin: 0;">{paraphrased_output}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Paraphrase statistics
                if result:
                    first_paraphrase = result[0]['generated_text']
                    para_word_count, para_sentence_count, para_char_count = get_text_stats(first_paraphrase)
                    original_words = len(user_text.split())
                    
                    st.markdown(f"""
                    <div style="margin-top: 1rem; padding: 1rem; background: #111111; border-radius: 6px; border-left: 4px solid #90EE90;">
                        <small style="color: #cccccc;">
                            <strong>Paraphrase:</strong> {para_word_count} words • {para_sentence_count} sentences
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download button
                    st.download_button(
                        "📥 Download Paraphrase",
                        paraphrased_output,
                        file_name="ai_paraphrase.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            except Exception as e:
                loader_placeholder.empty()
                st.error(f"⚠️ Processing error: {str(e)}")
                
        elif generate_button and not user_text.strip():
            st.error("⚠️ Please enter text to paraphrase.")
        
        else:
            # Placeholder when no paraphrase is generated
            st.markdown("""
            <div class="placeholder-box">
                <div class="placeholder-content">
                    <h4 style="color: #666666; margin-bottom: 1rem;">Ready to Generate Paraphrase</h4>
                    <p style="margin-bottom: 1rem;">Enter your text and click "Generate Paraphrase"</p>
                    <p style="font-size: 0.9em; margin: 0;">
                        ✓ Multiple Variants<br>
                        ✓ Customizable Style<br>
                        ✓ Instant Processing<br>
                        ✓ Export Options
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Reference Column
    with reference_col:
        st.markdown("""
        <div class="column-header">
            <h3>📚 Reference & Sources</h3>
        </div>
        """, unsafe_allow_html=True)
        
        reference = None
        if user_text:
            reference = find_reference_paraphrase(user_text)

        if reference:
            reference_text, reference_paraphrase = reference
            st.markdown(f"""
            <div class="content-box">
                <h4 style="color: #6c757d; margin-bottom: 0.5rem;">Original Text (from dataset):</h4>
                <p style="line-height: 1.4; margin-bottom: 1rem; color: #ffffff;">{reference_text}</p>
                <h4 style="color: #6c757d; margin-bottom: 0.5rem;">Paraphrase from Dataset:</h4>
                <p style="line-height: 1.4; margin: 0; color: #ffffff;">{reference_paraphrase}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="placeholder-box">
                    <div class="placeholder-content">
                        <h4 style="color: #666666; margin-bottom: 1rem;">References & Citations</h4>
                        <p style="margin-bottom: 1rem;">Source documents and references will appear here</p>
                        <p style="font-size: 0.9em; margin: 0;">
                            📄 Source Documents<br>
                            🔗 External Links<br>
                            📊 Data Sources<br>
                            📝 Citations
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>AI Paraphrasing Studio</strong> | Powered by Advanced Language Models</p>
        <p><em>Professional text paraphrasing for research, business, and academic use</em></p>
    </div>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
