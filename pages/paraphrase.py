import streamlit as st
from transformers.pipelines import pipeline
import textstat
import plotly.express as px
import plotly.graph_objects as go
import time
import pandas as pd

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
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #000000 100%);
        border: 2px solid #333333;
        padding: 4rem 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 20%, rgba(255,255,255,0.05) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(255,255,255,0.03) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem; /* Reduced from 4rem */
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.8rem; /* Reduced from 1rem */
        position: relative;
        z-index: 1;
        letter-spacing: -1.5px; /* Adjusted letter-spacing */
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem; /* Reduced from 1.3rem */
        font-weight: 400;
        color: #cccccc;
        text-align: center;
        margin-bottom: 1.5rem; /* Reduced from 2rem */
        position: relative;
        z-index: 1;
        line-height: 1.5;
    }
    
    /* Section Cards */
    .section-card {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 16px;
        padding: 1.5rem 2rem; /* Reduced top/bottom padding from 2rem */
        margin: 1.5rem 0; /* Reduced margin from 2rem */
        transition: all 0.3s ease;
        position: relative;
    }
    
    .section-card:hover {
        border-color: #555555;
        background: #151515;
        transform: translateY(-2px);
    }
    
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem; /* Reduced from 1.5rem */
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem; /* Reduced from 1.5rem */
        display: flex;
        align-items: center;
        gap: 0.8rem;
        letter-spacing: -0.5px;
    }
    
    /* AI Loading Animation */
    .ai-loader {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.5rem;
        padding: 3rem;
        background: #111111;
        border: 1px solid #333333;
        border-radius: 16px;
        margin: 2rem 0;
    }
    
    .loader-dots {
        display: flex;
        gap: 0.5rem;
    }
    
    .loader-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #ffffff;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    .loader-dot:nth-child(2) { animation-delay: 0.3s; }
    .loader-dot:nth-child(3) { animation-delay: 0.6s; }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.2); }
    }
    
    .loader-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #ffffff;
        font-weight: 500;
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
    
    /* File uploader */
    .stFileUploader > div {
        background: #111111;
        border: 2px dashed #333333;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 0 !important; /* Added to remove space below uploader */
        margin-top: 2rem; /* Added to create space above uploader */
    }
    
    .stFileUploader + .stAlert {
        margin-top: 0 !important; /* Added to remove space above alert when it follows uploader */
    }
    
    .stFileUploader > div:hover {
        border-color: #666666;
        background: #151515;
    }
    
    /* Metrics styling */
    .metric-card {
        background: #111111;
        border: 1px solid #333333;
        padding: 1.5rem 1rem; /* Reduced padding from 2rem 1.5rem */
        border-radius: 10px; /* Slightly reduced border-radius */
        text-align: center;
        margin: 0.4rem 0; /* Reduced margin from 0.5rem */
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-card:hover {
        border-color: #555555;
        background: #151515;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem; /* Reduced from 2.5rem */
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem; /* Reduced from 0.9rem */
        color: #cccccc;
        margin-top: 0.4rem; /* Reduced from 0.5rem */
        font-weight: 500;
    }
    
    /* Comparison cards */
    .comparison-card {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 12px; /* Slightly reduced border-radius */
        padding: 1.5rem; /* Reduced padding from 2rem */
        margin: 0.8rem 0; /* Reduced margin from 1rem */
        border-left: 4px solid #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .comparison-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.8rem; /* Reduced margin from 1rem */
        font-size: 1.1rem; /* Reduced from 1.2rem */
    }
    
    .comparison-content {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
        line-height: 1.5; /* Slightly reduced line-height */
        font-size: 0.9rem; /* Reduced from 0.95rem */
    }
    
    /* Success/Error messages */
    .stAlert {
        background: #111111;
        border: 1px solid #333333;
        color: #ffffff;
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
    
    /* Chart container */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-item {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .stat-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .stat-label {
        font-family: 'Inter', sans-serif;
        color: #cccccc;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Paraphraser with Loading State
# -----------------------------
@st.cache_resource
def load_paraphraser():
    return pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

# -----------------------------
# Function to calculate readability
# -----------------------------
def calculate_readability(text):
    return {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Gunning Fog": textstat.gunning_fog(text),
        "SMOG Index": textstat.smog_index(text),
        "Automated Readability Index": textstat.automated_readability_index(text),
        "Dale-Chall": textstat.dale_chall_readability_score(text),
    }

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
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title" style="font-size: 2.5rem; font-weight: 300;">AI Paraphrasing Studio</h1>
        <p class="hero-subtitle" style="font-size: 0.8rem; font-weight: 300;">Transform your text with precision AI technology and comprehensive readability analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for loading
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    # Load model with loading screen
    if not st.session_state.model_loaded:
        st.markdown(show_ai_loader("Initializing AI Models"), unsafe_allow_html=True)
        paraphraser = load_paraphraser()
        st.session_state.paraphraser = paraphraser
        st.session_state.model_loaded = True
        time.sleep(1)
        st.rerun()
    
    paraphraser = st.session_state.paraphraser

    # Input Section
    # st.markdown("""
    # <div class="section-card">
    #     <h2 class="section-title">📄 Text Input</h2>
    #     <p class="section-description">Upload a file or enter text directly for AI-powered paraphrasing and analysis</p>
    # </div>
    # """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File Upload
        uploaded_file = st.file_uploader(
            "Upload Text File (.txt)", 
            type=["txt"], 
            help="Select a text file to automatically load content"
        )
        
        input_text = ""
        if uploaded_file is not None:
            input_text = uploaded_file.read().decode("utf-8")
            st.success(f"✓ File loaded • {len(input_text)} characters")

    with col2:
        # Real-time Statistics
        if input_text:
            st.markdown(f"""
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-number">{len(input_text.split())}</div>
                    <div class="stat-label">Words</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Text Input Area
    user_text = st.text_area(
        "Enter your text:", 
        value=input_text, 
        height=200,
        placeholder="Type or paste your text here for AI processing...",
        help="Input text for paraphrasing and analysis"
    )

    # Configuration Section
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">⚙️ Processing Configuration</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        level = st.radio(
            "Complexity Level", 
            ["Beginner", "Intermediate", "Advanced"],
            help="Select processing complexity and number of variants"
        )

    with col2:
        style = st.selectbox(
            "Output Style",
            ["Fluency", "Formal", "Creative", "Concise"],
            help="Choose the writing style for paraphrased output"
        )

    with col3:
        variants = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{variants[level]}</div>
            <div class="metric-label">Variants</div>
        </div>
        """, unsafe_allow_html=True)

    # Processing Section
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">🚀 AI Processing</h2>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Process Text", help="Generate AI-powered paraphrased versions"):
        if user_text.strip():
            # Show processing loader
            loader_placeholder = st.empty()
            loader_placeholder.markdown(show_ai_loader("Analyzing and transforming text"), unsafe_allow_html=True)
            
            # Processing parameters
            if level == "Beginner":
                num_return = 1
                max_len = 100
            elif level == "Intermediate":
                num_return = 2
                max_len = 150
            else:  # Advanced
                num_return = 3
                max_len = 200

            # Generate paraphrased versions
            style_prompt = f"Paraphrase in a {style.lower()} style: {user_text}"
            
            try:
                result = paraphraser(
                    style_prompt,
                    max_length=max_len,
                    num_return_sequences=num_return,
                    do_sample=True,
                    temperature=1.5
                )
                
                # Clear loader
                loader_placeholder.empty()
                
                # Results Section
                st.markdown("""
                <div class="section-card">
                    <h2 class="section-title">📋 Comparison Results</h2>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="comparison-card">
                        <h3 class="comparison-title">Original Text</h3>
                        <div class="comparison-content">
                    """, unsafe_allow_html=True)
                    st.write(user_text)
                    st.markdown("</div></div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="comparison-card">
                        <h3 class="comparison-title">AI Enhanced ({style})</h3>
                        <div class="comparison-content">
                    """, unsafe_allow_html=True)
                    for i, r in enumerate(result):
                        st.write(f"**Version {i+1}:** {r['generated_text']}")
                    st.markdown("</div></div>", unsafe_allow_html=True)

                # Readability Analysis Section
                st.markdown("""
                <div class="section-card">
                    <h2 class="section-title">📊 Readability Analysis</h2>
                </div>
                """, unsafe_allow_html=True)

                # Calculate readability scores
                original_scores = calculate_readability(user_text)
                paraphrased_text = result[0]['generated_text']
                paraphrased_scores = calculate_readability(paraphrased_text)

                # Create comparison metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                metrics = [
                    ("Flesch Reading Ease", "📈"),
                    ("Gunning Fog", "🌫️"),
                    ("SMOG Index", "📏"),
                    ("Automated Readability Index", "🤖"),
                    ("Dale-Chall", "📚")
                ]

                for i, (col, (metric, icon)) in enumerate(zip([col1, col2, col3, col4, col5], metrics)):
                    with col:
                        orig_score = original_scores[metric]
                        para_score = paraphrased_scores[metric]
                        improvement = para_score - orig_score
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                            <div class="metric-value">{para_score:.1f}</div>
                            <div class="metric-label">{metric}</div>
                            <div style="font-size: 0.85rem; margin-top: 0.8rem; color: {'#90EE90' if improvement > 0 else ('#FF6347' if improvement < 0 else '#cccccc')}; font-weight: 600;">
                                <span style="margin-right: 5px;">{'▲' if improvement > 0 else ('▼' if improvement < 0 else '━')}</span> {improvement:+.1f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Interactive Chart
                data = []
                for metric in original_scores.keys():
                    data.append({"Metric": metric.replace(" ", "\n"), "Text": "Original", "Score": original_scores[metric]})
                    data.append({"Metric": metric.replace(" ", "\n"), "Text": "Enhanced", "Score": paraphrased_scores[metric]})

                fig = px.bar(
                    data, 
                    x="Metric", 
                    y="Score", 
                    color="Text", 
                    barmode="group",
                    title=f"Readability Comparison: Original vs Enhanced ({style})",
                    color_discrete_map={"Original": "#666666", "Enhanced": "#ffffff"}
                )
                
                fig.update_layout(
                    font=dict(family="Inter, sans-serif", size=12, color="#ffffff"),
                    title_font=dict(family="Inter, sans-serif", size=16, color="#ffffff"),
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    xaxis=dict(gridcolor='#333333'),
                    yaxis=dict(gridcolor='#333333'),
                )
                
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.markdown("#### Understanding Readability Scores")
                with st.expander("Learn more about these metrics"):
                    st.markdown('''
                    - **Flesch Reading Ease**: A higher score means easier to read. Scores of 60-70 are generally considered plain English.
                    - **Gunning Fog**: A lower score means easier to read. A score of 8 is good for general readership.
                    - **SMOG Index**: A lower score means easier to read. Estimates the years of education needed to understand a text.
                    - **Automated Readability Index (ARI)**: A lower score means easier to read. Based on characters per word and words per sentence.
                    - **Dale-Chall Readability Score**: A lower score means easier to read. Based on a list of familiar words.
                    ''')

            except Exception as e:
                loader_placeholder.empty()
                st.error(f"Processing error: {str(e)}")
                
        else:
            st.warning("Please enter text to process.")

    # Footer
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0 2rem 0; color: #666666; font-family: 'Inter', sans-serif; font-size: 0.9rem;">
        <p>Powered by Advanced AI Technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()