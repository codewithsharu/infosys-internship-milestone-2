import streamlit as st
import time
import csv
import os

# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Professional CSS styling
st.markdown("""
<style>
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
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .header p {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Column styling */
    .column-header {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .column-header h3 {
        color: #495057;
        margin: 0;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    /* Content boxes */
    .content-box {
        background: #000000; /* Black background */
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        height: 400px;
        overflow-y: auto;
        color: #ffffff; /* White text */
    }
    
    .placeholder-box {
        background: #000000; /* Black background */
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        padding: 2rem;
        height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #ffffff; /* White text */
    }
    
    .placeholder-content {
        color: #ffffff; /* White text */
    }
    
    /* Controls styling */
    .controls-section {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .controls-title {
        color: #495057;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: #007bff;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #0056b3;
        transform: translateY(-1px);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border: 1px solid #ced4da;
        border-radius: 6px;
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        border: 1px solid #ced4da;
        border-radius: 6px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

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

def simulate_summarization(text, model_name, summary_length):
    """Simulate the summarization process with realistic delay"""
    time.sleep(2)  # Simulate API processing time
    
    words = text.split()
    target_words = min(summary_length, len(words))
    
    if target_words > 0:
        summary_words = words[:target_words]
        summary = " ".join(summary_words)
        
        # Add appropriate ending
        if not summary.endswith(('.', '!', '?')):
            summary += "..."
            
        return summary
    return "Unable to generate summary with current parameters."

def find_reference_summary(input_text, csv_path='pages/summary.csv'):
    """
    Search for the input_text in the summary.csv file.
    If found, return (original_text, summary) tuple.
    If not found, return None.
    """
    if not input_text or not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                original = row[0].strip().strip('"')
                summary = row[1].strip().strip('"')
                # Exact match (ignoring leading/trailing whitespace and quotes)
                if input_text.strip() == original:
                    return (original, summary)
    except Exception as e:
        return None
    return None

def app():
    # Header
    st.markdown("""
    <div class="header">
        <h1>📝 AI Text Summarizer</h1>
        <p>Professional text summarization with advanced AI models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Controls Section
    st.markdown("""
    <div class="controls-section">
        <div class="controls-title">Configuration Settings</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model and parameter controls in a single row
    control_col1, control_col2, control_col3, control_col4 = st.columns([2, 2, 2, 2])
    
    with control_col1:
        available_models = ["BART Large CNN", "T5 Small", "Pegasus XSUM"]
        selected_model = st.selectbox(
            "AI Model",
            available_models,
            index=0
        )
    
    with control_col2:
        summary_length = st.number_input(
            "Summary Length (words)",
            min_value=0,
            max_value=1000, # Increased max_value for flexibility
            value=150,
            step=1,
            key="summary_length_input"
        )
    
    with control_col3:
        st.write("") # Spacer for alignment
    
    with control_col4:
        generate_button = st.button(
            "🚀 Generate Summary",
            type="primary",
            use_container_width=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main three-column layout
    input_col, summary_col, reference_col = st.columns([1, 1, 1])
    
    # Input Column
    with input_col:
        st.markdown("""
        <div class="column-header">
            <h3>📝 Input Text</h3>
        </div>
        """, unsafe_allow_html=True)
        
        text_input = st.text_area(
            "",
            height=350,
            placeholder="""Enter your text here for summarization...

Examples:
• Research papers and articles
• News reports and stories
• Blog posts and web content
• Business documents
• Technical documentation

The AI will analyze your content and create a concise summary.""",
            key="text_input"
        )
        
        # Display text statistics if there's input
        if text_input:
            word_count, sentence_count, char_count = get_text_stats(text_input)
            reading_time = calculate_reading_time(text_input)
            
            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 1rem; background: #000000; border-radius: 6px;">
                <small style="color: #ffffff;">
                    <strong>Stats:</strong> {word_count:,} words • {sentence_count} sentences • {reading_time} min read
                </small>
            </div>
            """, unsafe_allow_html=True)
    
    # Summary Column
    with summary_col:
        st.markdown("""
        <div class="column-header">
            <h3>✨ Generated Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if generate button was clicked and there's text to process
        if generate_button and text_input and len(text_input.split()) >= 10:
            # Show processing status
            with st.spinner("Processing text..."):
                try:
                    summary = simulate_summarization(text_input, selected_model, summary_length)
                    
                    # Display the summary
                    st.markdown(f"""
                    <div class="content-box">
                        <p style="line-height: 1.6; margin: 0;">{summary}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Summary statistics
                    if summary:
                        summary_stats = get_text_stats(summary)
                        original_words = len(text_input.split())
                        compression = round((1 - summary_stats[0] / original_words) * 100, 1) if original_words > 0 else 0
                        
                        st.markdown(f"""
                        <div style="margin-top: 1rem; padding: 1rem; background: #000000; border-radius: 6px; border-left: 4px solid #28a745;">
                            <small style="color: #ffffff;">
                                <strong>Summary:</strong> {summary_stats[0]} words • {compression}% compression • Model: {selected_model}
                            </small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Download button
                        st.download_button(
                            "📥 Download Summary",
                            summary,
                            file_name="ai_summary.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                
                except Exception as e:
                    st.error("⚠️ Unable to process text. Please try again.")
        
        elif generate_button and not text_input:
            st.error("⚠️ Please enter text to summarize")
        
        elif generate_button and len(text_input.split()) < 10:
            st.warning("⚠️ Please provide at least 10 words for summarization")
        
        else:
            # Placeholder when no summary is generated
            st.markdown("""
            <div class="placeholder-box">
                <div class="placeholder-content">
                    <h4 style="color: #6c757d; margin-bottom: 1rem;">Ready to Generate Summary</h4>
                    <p style="margin-bottom: 1rem;">Enter your text and click "Generate Summary"</p>
                    <p style="font-size: 0.9em; margin: 0;">
                        ✓ Multiple AI Models<br>
                        ✓ Customizable Length<br>
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
        
        # Reference logic: If input text matches a row in summary.csv, show its summary
        reference = None
        if text_input:
            reference = find_reference_summary(text_input)
        
        if reference:
            reference_text, reference_summary = reference
            st.markdown(f"""
            <div class="content-box">
                <h4 style="color: #6c757d; margin-bottom: 0.5rem;">Original Text (from dataset):</h4>
                <p style="line-height: 1.4; margin-bottom: 1rem; color: #ffffff;">{reference_text}</p>
                <h4 style="color: #6c757d; margin-bottom: 0.5rem;">Summary from Dataset:</h4>
                <p style="line-height: 1.4; margin: 0; color: #ffffff;">{reference_summary}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="placeholder-box">
                    <div class="placeholder-content">
                        <h4 style="color: #6c757d; margin-bottom: 1rem;">References & Citations</h4>
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
        <p><strong>AI Text Summarizer</strong> | Powered by Advanced Language Models</p>
        <p><em>Professional text summarization for research, business, and academic use</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    app()