import streamlit as st
from transformers import pipeline
from rouge_score import rouge_scorer
import textstat
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Text Evaluation Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Black & White Theme
# -----------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #111111;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #ffffff 0%, #cccccc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #cccccc;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a1a, #0a0a0a);
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .comparison-card {
        background: linear-gradient(145deg, #1a1a1a, #0a0a0a);
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
    }
    
    /* Text areas */
    .stTextArea textarea {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #1a1a1a;
        border: 2px dashed #666666;
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ffffff, #e0e0e0);
        color: #000000;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #f0f0f0, #d0d0d0);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }
    
    /* Radio buttons and selectbox */
    .stRadio > div, .stSelectbox > div > div {
        background-color: #1a1a1a;
        border-radius: 8px;
    }
    
    /* Metrics display */
    .metric-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 2rem 0;
    }
    
    .metric-item {
        background: linear-gradient(145deg, #222222, #111111);
        border: 1px solid #444444;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem;
        text-align: center;
        min-width: 150px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-label {
        color: #cccccc;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        border-bottom: 2px solid #333333;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Progress indicator */
    .progress-container {
        background-color: #1a1a1a;
        border-radius: 20px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Summarizer Loader (cached)
# -----------------------------
@st.cache_resource
def load_summarizer(model_name):
    return pipeline("summarization", model=model_name)

# -----------------------------
# Calculate ROUGE
# -----------------------------
def calculate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(reference, generated)

# -----------------------------
# Calculate readability
# -----------------------------
def get_readability_scores(text):
    return {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "SMOG Index": textstat.smog_index(text),
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
        "Gunning Fog": textstat.gunning_fog(text),
        "Automated Readability": textstat.automated_readability_index(text),
        "Coleman-Liau": textstat.coleman_liau_index(text),
        "Dale-Chall": textstat.dale_chall_readability_score(text),
    }

# -----------------------------
# Enhanced Plotly Theme
# -----------------------------
def create_readability_chart(orig_scores, sum_scores):
    categories = list(orig_scores.keys())
    
    fig = go.Figure()
    
    # Add bars for original text
    fig.add_trace(go.Bar(
        name='Original',
        x=categories,
        y=list(orig_scores.values()),
        marker_color='#ffffff',
        marker_line_color='#cccccc',
        marker_line_width=2,
        hovertemplate='<b>%{x}</b><br>Original: %{y:.2f}<extra></extra>'
    ))
    
    # Add bars for summary
    fig.add_trace(go.Bar(
        name='Summary',
        x=categories,
        y=list(sum_scores.values()),
        marker_color='#666666',
        marker_line_color='#444444',
        marker_line_width=2,
        hovertemplate='<b>%{x}</b><br>Summary: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Readability Analysis Comparison',
        title_font_size=20,
        title_font_color='#ffffff',
        xaxis_title='Readability Metrics',
        yaxis_title='Scores',
        barmode='group',
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font_color='#ffffff',
        xaxis=dict(
            gridcolor='#333333',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#333333',
            zeroline=False
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#666666',
            borderwidth=1
        )
    )
    
    return fig

def create_rouge_chart(rouge_scores):
    metrics = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for metric, result in rouge_scores.items():
        metrics.append(metric.upper())
        precision_scores.append(result.precision)
        recall_scores.append(result.recall)
        f1_scores.append(result.fmeasure)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Precision',
        x=metrics,
        y=precision_scores,
        marker_color='#ffffff',
        hovertemplate='<b>%{x}</b><br>Precision: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Recall',
        x=metrics,
        y=recall_scores,
        marker_color='#aaaaaa',
        hovertemplate='<b>%{x}</b><br>Recall: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='F1-Score',
        x=metrics,
        y=f1_scores,
        marker_color='#666666',
        hovertemplate='<b>%{x}</b><br>F1-Score: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='ROUGE Evaluation Metrics',
        title_font_size=20,
        title_font_color='#ffffff',
        xaxis_title='ROUGE Metrics',
        yaxis_title='Scores',
        barmode='group',
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font_color='#ffffff',
        xaxis=dict(
            gridcolor='#333333',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#333333',
            zeroline=False,
            range=[0, 1]
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#666666',
            borderwidth=1
        )
    )
    
    return fig

# -----------------------------
# Main Application
# -----------------------------
def main():
    # Main title
    st.markdown('<h1 class="main-title" style="font-size: 2rem; font-weight: 500;">⭐ Text Evaluation </h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle" style="font-size: 1rem; font-weight: 300;">Advanced AI-powered text summarization with comprehensive evaluation metrics</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "🤖 Summarization Model",
            ["facebook/bart-large-cnn", "t5-small", "sshleifer/distilbart-cnn-12-6"],
            help="Choose the AI model for text summarization"
        )
        
        # Summary length
        length_choice = st.radio(
            "📏 Summary Length",
            ["Short", "Medium", "Long"],
            help="Select the desired length of the generated summary"
        )
        
        if length_choice == "Short":
            max_len, min_len = 60, 20
        elif length_choice == "Medium":
            max_len, min_len = 120, 40
        else:  # Long
            max_len, min_len = 200, 80
            
        st.markdown(f"**Length Range:** {min_len}-{max_len} words")
    
    # File upload section
    st.markdown('<div class="section-header">📂 Document Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload your text document for analysis",
        type=["txt"],
        help="Upload a .txt file containing the text you want to summarize and evaluate"
    )
    
    if uploaded_file is not None:
        # Read and display file info
        text = uploaded_file.read().decode("utf-8")
        word_count = len(text.split())
        char_count = len(text)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-item"><div class="metric-value">{word_count:,}</div><div class="metric-label">Words</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-item"><div class="metric-value">{char_count:,}</div><div class="metric-label">Characters</div></div>', unsafe_allow_html=True)
        with col3:
            estimated_time = max(1, word_count // 100)
            st.markdown(f'<div class="metric-item"><div class="metric-value">{estimated_time}</div><div class="metric-label">Est. Minutes</div></div>', unsafe_allow_html=True)
        
        # Generate button
        st.markdown("---")
        if st.button("🚀 Generate Summary & Analyze", use_container_width=True):
            with st.spinner("🔄 Processing your text... This may take a few moments"):
                try:
                    # Load model and generate summary
                    summarizer = load_summarizer(model_choice)
                    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
                    
                    # Success message
                    st.success("✅ Analysis completed successfully!")
                    
                    # Text comparison section
                    st.markdown('<div class="section-header">📖 Text Comparison</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
                        st.markdown("**📄 Original Text**")
                        st.text_area("", text, height=300, disabled=True, key="original")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
                        st.markdown("**✨ Generated Summary**")
                        st.text_area("", summary, height=300, disabled=True, key="summary")
                        
                        # Summary stats
                        summary_words = len(summary.split())
                        compression_ratio = (1 - summary_words / word_count) * 100
                        st.markdown(f"**Compression:** {compression_ratio:.1f}% reduction ({summary_words} words)")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Readability Analysis
                    st.markdown('<div class="section-header">📊 Readability Analysis</div>', unsafe_allow_html=True)
                    
                    orig_scores = get_readability_scores(text)
                    sum_scores = get_readability_scores(summary)
                    
                    readability_fig = create_readability_chart(orig_scores, sum_scores)
                    st.plotly_chart(readability_fig, use_container_width=True)
                    
                    # ROUGE Evaluation
                    st.markdown('<div class="section-header">📈 ROUGE Evaluation</div>', unsafe_allow_html=True)
                    
                    rouge_scores = calculate_rouge(text, summary)
                    
                    # ROUGE metrics cards
                    cols = st.columns(3)
                    rouge_metrics = ["rouge1", "rouge2", "rougeL"]
                    rouge_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
                    
                    for i, (metric, name) in enumerate(zip(rouge_metrics, rouge_names)):
                        with cols[i]:
                            result = rouge_scores[metric]
                            st.markdown(f'''
                            <div class="metric-card">
                                <h3 style="text-align: center; margin-bottom: 1rem;">{name}</h3>
                                <div style="text-align: center;">
                                    <div><strong>Precision:</strong> {result.precision:.3f}</div>
                                    <div><strong>Recall:</strong> {result.recall:.3f}</div>
                                    <div><strong>F1-Score:</strong> {result.fmeasure:.3f}</div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    # ROUGE chart
                    rouge_fig = create_rouge_chart(rouge_scores)
                    st.plotly_chart(rouge_fig, use_container_width=True)
                    
                    # Performance summary
                    st.markdown('<div class="section-header">🎯 Performance Summary</div>', unsafe_allow_html=True)
                    
                    avg_f1 = sum(result.fmeasure for result in rouge_scores.values()) / len(rouge_scores)
                    if avg_f1 >= 0.5:
                        performance = "Excellent"
                        color = "#00ff00"
                    elif avg_f1 >= 0.3:
                        performance = "Good"
                        color = "#ffff00"
                    else:
                        performance = "Needs Improvement"
                        color = "#ff6666"
                    
                    st.markdown(f'''
                    <div style="text-align: center; padding: 2rem; background: linear-gradient(145deg, #1a1a1a, #0a0a0a); border-radius: 12px; margin: 1rem 0;">
                        <h2 style="color: {color};">Overall Performance: {performance}</h2>
                        <p style="font-size: 1.2rem;">Average F1-Score: {avg_f1:.3f}</p>
                        <p style="color: #cccccc;">Model: {model_choice} | Length: {length_choice}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during processing: {str(e)}")
                    st.info("💡 Try with a shorter text or different model if the issue persists.")
    
    else:
        # Welcome message when no file is uploaded
        st.markdown('''
        <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(145deg, #1a1a1a, #0a0a0a); border-radius: 12px; margin: 2rem 0;">
            <h2>🚀 Ready to Analyze Your Text?</h2>
            <p style="font-size: 1.1rem; color: #cccccc; margin: 1rem 0;">Upload a text file to get started with AI-powered summarization and comprehensive evaluation.</p>
            <div style="margin-top: 2rem;">
                <p><strong>Features:</strong></p>
                <p>✨ Advanced AI Summarization | 📊 ROUGE Score Analysis | 📈 Readability Metrics | 🎯 Performance Insights</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()