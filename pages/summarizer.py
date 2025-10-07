import streamlit as st
import time
import csv
import os
import evaluate
import textstat
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline, MarianMTModel, MarianTokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Load GPT-2 model and tokenizer for perplexity calculation
@st.cache_resource
def get_perplexity_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

perplexity_tokenizer, perplexity_model = get_perplexity_model()

# Load T5 Small model and tokenizer for summarization
@st.cache_resource
def get_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

t5_tokenizer, t5_model = get_t5_model()

@st.cache_resource
def get_translation_models():
    # Dictionary to store tokenizers and models for different languages
    translation_resources = {}
    
    # English to French
    translation_resources["French"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr"),
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    }
    
    # English to Hindi
    translation_resources["Hindi"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi"),
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    }

    # English to Telugu (Checking for availability, using a common one if specific not found)
    # Note: "Helsinki-NLP/opus-mt-en-te" is a common pattern, but actual availability should be verified.
    # If not available, consider a more general multilingual model or fallback.
    translation_resources["Telugu"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ine"), # Changed to a broader Indo-European model
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ine") # Changed to a broader Indo-European model
    }

    # English to Spanish
    translation_resources["Spanish"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es"),
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    }

    # English to German
    translation_resources["German"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de"),
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    }

    # English to Italian
    translation_resources["Italian"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it"),
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-it")
    }

    # Add more languages as needed
    
    return translation_resources

# Initialize translation models
translation_models = get_translation_models()

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
        border: 1px solid #dee2e6; /* Changed to match content-box/placeholder-box */
        border-radius: 8px;
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-size: 14px;
        line-height: 1.5;
        background: #000000; /* Added for black background */
        color: #ffffff; /* Added for white text */
        padding: 1.5rem; /* Added to match content-box padding */
        min-height: 400px; /* Ensure minimum height */
        box-sizing: border-box; /* Include padding and border in the element's total width and height */
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

    .score-box {
        background: #111111; /* Darker background than main content boxes */
        border-left: 4px solid #90EE90; /* Green accent bar on the left */
        border-radius: 8px; /* Slightly rounded corners */
        padding: 1rem 1.2rem; /* Ample padding */
        margin-top: 0.75rem; /* Space between metric boxes */
        color: #ffffff; /* White text for readability */
        font-family: 'Inter', sans-serif; /* Consistent font */
        display: flex; /* Use flexbox for alignment */
        justify-content: space-between; /* Space out content within the box */
        align-items: center; /* Vertically align content */
        box-shadow: 0 2px 8px rgba(0,0,0,0.2); /* Subtle shadow for depth */
        transition: all 0.2s ease-in-out;
    }

    .score-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3); /* Enhanced shadow on hover */
    }

    .score-box small {
        color: #cccccc; /* Lighter grey for small text */
        font-size: 0.95rem; /* Slightly larger font for metrics */
    }

    .score-box strong {
        color: #ffffff; /* White color for values */
    }

    .score-box.reference {
        border-left: 5px solid #ff9800; /* Distinct vibrant left border for reference scores */
    }

    .score-box.reference strong {
        color: #ff9800; /* Highlight reference score labels */
    }

    .score-icon {
        margin-right: 0.75rem;
        font-size: 1.2em;
        line-height: 1;
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

@st.cache_data
def calculate_bleu(reference, candidate):
    """Calculate BLEU score using HuggingFace evaluate library"""
    if not reference or not candidate:
        return 0.0
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=[candidate], references=[[reference]])
    return round(results["bleu"] * 100, 2)

@st.cache_data
def calculate_perplexity(text):
    """Calculate GPT-2 perplexity score"""
    if not text.strip():
        return 0.0
    
    try:
        encodings = perplexity_tokenizer(text, return_tensors='pt')
        seq_len = encodings.input_ids.size(1)
        
        if seq_len == 0:
            return 0.0
            
        with torch.no_grad():
            outputs = perplexity_model(encodings.input_ids, labels=encodings.input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            
        return round(perplexity, 2)
    except RuntimeError as e:
        # Handle cases where input might be too long or other torch-related errors
        return 0.0
    except Exception as e:
        return 0.0

@st.cache_data
def calculate_readability_scores(text):
    """Calculate Flesch-Kincaid Grade Level"""
    if not text.strip():
        return 0.0
    try:
        # Flesch-Kincaid Grade Level
        return round(textstat.flesch_kincaid_grade(text), 2)
    except Exception as e:
        return 0.0

def translate_text(text, target_language="fr"): # Default to French for now
    """Translate text using the loaded MarianMT model"""
    if not text.strip():
        return ""
    try:
        # The model is `Helsinki-NLP/opus-mt-en-fr`, so it translates from English to French.
        # If we want to support other languages, we'd need to load different models.
        translated_tokens = translation_models[target_language]["model"].generate(**translation_models[target_language]["tokenizer"](text, return_tensors="pt"))
        translated_text = translation_models[target_language]["tokenizer"].decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        return f"Translation error: {e}"

def generate_radar_chart(fluency, semantic_similarity, coherence, diversity, fact_preservation):
    categories = ['Fluency', 'Semantic Similarity', 'Coherence', 'Diversity', 'Fact Preservation']
    values = [fluency, semantic_similarity, coherence, diversity, fact_preservation]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'skyblue', alpha=0.4)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', size=10)
    ax.tick_params(axis='x', pad=10)
    ax.set_facecolor('#0E1117') # Set background to match Streamlit dark theme
    fig.patch.set_facecolor('#0E1117')

    # Set grid and spine colors to white for better visibility on dark background
    ax.grid(color='white', linestyle='--', alpha=0.7)
    ax.spines['polar'].set_color('white')
    ax.spines['outer'].set_color('white')
    
    return fig

def simulate_summarization(text, model_name, summary_length):
    """Simulate the summarization process with realistic delay"""
    # Use T5 Small for summarization
    if model_name == "T5 Small":
        inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = t5_model.generate(inputs, max_length=summary_length, min_length=int(summary_length * 0.5), length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

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
    # st.markdown("""
    # <div class="header">
    #     <h1>📝 AI Text Summarizer</h1>
    #     <p>Professional text summarization with advanced AI models</p>
    # </div>
    # """, unsafe_allow_html=True)
    
    # # Controls Section
    # st.markdown("""
    # <div class="controls-section">
    #     <div class="controls-title">Configuration Settings</div>
    # </div>
    # """, unsafe_allow_html=True)
    
    # Model and parameter controls in a single row
    control_col1, control_col2, control_col3, control_col4 = st.columns([2, 2, 2, 2])
    
    with control_col1:
        available_models = ["BART Large CNN", "T5 Small", "Pegasus XSUM"]
        selected_model = st.selectbox(
            "AI Model",
            available_models,
            index=available_models.index("T5 Small") # Set T5 Small as default
        )
    
    with control_col2:
        summary_length = st.number_input(
            "Summary Length (words)",
            min_value=0,
            max_value=1000, # Increased max_value for flexibility
            value=10,
            step=1,
            key="summary_length_input"
        )
    
    with control_col3:
        # Language selection for translation
        target_language = st.selectbox(
            "Target Language",
            list(translation_models.keys()), # Dynamically get keys from the loaded models
            index=list(translation_models.keys()).index("Hindi"),
            key="translation_language"
        )
    
    with control_col4:
        generate_button = st.button(
            "🚀 Generate Summary",
            type="primary",
            use_container_width=True
        )
        translate_button = st.button(
            "🌍 Translate",
            type="secondary",
            use_container_width=True,
            key="translate_button"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main four-column layout
    input_col, summary_col, reference_col, translate_col = st.columns([1, 1, 1, 1])
    
    # Input Column
    with input_col:
        st.markdown("""
        <div class="column-header">
            <h3>📝 Input Text</h3>
        </div>
        """, unsafe_allow_html=True)
        
        text_input = st.text_area(
            "",
            height=400,
            placeholder="""Enter your text here for summarization...

Examples:
• Research papers and articles
• News reports and stories
• Blog posts and web content
• Business documents
• Technical documentation

The AI will analyze your content and create a concise summary.""",
            key="text_input",
            label_visibility="collapsed" # Added to remove the default label spacing
        )
        
        # Display text statistics if there's input
        if text_input:
            word_count, sentence_count, char_count = get_text_stats(text_input)
            reading_time = calculate_reading_time(text_input)
            
            with st.expander("View Input Text Metrics"):
                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 1rem; background: #000000; border-radius: 6px;">
                    <small style="color: #ffffff;">
                        <strong>Stats:</strong> {word_count:,} words • {sentence_count} sentences • {reading_time} min read
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate and display metrics for the input text
                input_perplexity = calculate_perplexity(text_input)
                input_readability = calculate_readability_scores(text_input)

                st.markdown(f"""
                <div class="score-box">
                    <span class="score-icon">🧠</span> <small><strong>Perplexity (Input):</strong> {input_perplexity}</small>
                </div>
                <div class="score-box">
                    <span class="score-icon">🧠</span> <small><strong>Readability (Input):</strong> {input_readability}</small>
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
                    
                    st.session_state["summary_output"] = summary # Store summary in session state

                    # Find and store reference summary if available
                    reference = find_reference_summary(text_input)
                    st.session_state["reference_output"] = reference # Store reference in session state

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
                        
                        # Calculate scores
                        bleu_score = calculate_bleu(text_input, summary)
                        perplexity_score = calculate_perplexity(summary)
                        original_readability = calculate_readability_scores(text_input)
                        summary_readability = calculate_readability_scores(summary)
                        readability_delta = round(original_readability - summary_readability, 2)

                        st.markdown("""
                        <div style="display: flex; justify-content: space-around; margin-top: 1rem; padding: 0.5rem; background: #111111; border-radius: 6px;">
                            <button style="background: none; border: none; color: #ffffff; font-size: 1.2rem; cursor: pointer;">
                                <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M9.56055 2C11.1381 2.00009 12.3211 3.44332 12.0117 4.99023L11.6094 7H13.8438C15.5431 7 16.836 8.52594 16.5566 10.2021L15.876 14.2842C15.6148 15.8513 14.2586 17 12.6699 17H4.5C3.67157 17 3 16.3284 3 15.5V9.23828C3.00013 8.57996 3.4294 7.99838 4.05859 7.80469L5.19824 7.4541L5.33789 7.40723C6.02983 7.15302 6.59327 6.63008 6.89746 5.9541L8.41113 2.58984L8.48047 2.46094C8.66235 2.17643 8.97898 2.00002 9.32324 2H9.56055ZM7.80957 6.36523C7.39486 7.2867 6.62674 7.99897 5.68359 8.3457L5.49219 8.41016L4.35254 8.76074C4.14305 8.82539 4.00013 9.01904 4 9.23828V15.5C4 15.7761 4.22386 16 4.5 16H12.6699C13.7697 16 14.7087 15.2049 14.8896 14.1201L15.5703 10.0381C15.7481 8.97141 14.9251 8 13.8438 8H11C10.8503 8 10.7083 7.9331 10.6133 7.81738C10.5184 7.70164 10.4805 7.54912 10.5098 7.40234L11.0312 4.79395C11.2167 3.86589 10.507 3.00009 9.56055 3H9.32324L7.80957 6.36523Z"></path></svg>
                            </button>
                            <button style="background: none; border: none; color: #ffffff; font-size: 1.2rem; cursor: pointer;">
                                <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M12.6699 3C14.2586 3 15.6148 4.14871 15.876 5.71582L16.5566 9.79785C16.836 11.4741 15.5431 13 13.8438 13H11.6094L12.0117 15.0098C12.3211 16.5567 11.1381 17.9999 9.56055 18H9.32324C8.97898 18 8.66235 17.8236 8.48047 17.5391L8.41113 17.4102L6.89746 14.0459C6.59327 13.3699 6.02983 12.847 5.33789 12.5928L5.19824 12.5459L4.05859 12.1953C3.4294 12.0016 3.00013 11.42 3 10.7617V4.5C3 3.67157 3.67157 3 4.5 3H12.6699ZM4.5 4C4.22386 4 4 4.22386 4 4.5V10.7617C4.00013 10.981 4.14305 11.1746 4.35254 11.2393L5.49219 11.5898L5.68359 11.6543C6.62674 12.001 7.39486 12.7133 7.80957 13.6348L9.32324 17H9.56055C10.507 16.9999 11.2167 16.1341 11.0312 15.2061L10.5098 12.5977C10.4805 12.4509 10.5184 12.2984 10.6133 12.1826C10.7083 12.0669 10.8503 12 11 12H13.8438C14.9251 12 15.7481 11.0286 15.5703 9.96191L14.8896 5.87988C14.7087 4.79508 13.7697 4 12.6699 4H4.5Z"></path></svg>
                            </button>
                            <button style="background: none; border: none; color: #ffffff; font-size: 1.2rem; cursor: pointer;">
                                <span>&#x21BB;</span> <!-- Refresh icon -->
                            </button>
                            <button style="background: none; border: none; color: #ffffff; font-size: 1.2rem; cursor: pointer;">
                                <span>&#x2026;</span> <!-- Ellipsis icon -->
                            </button>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("View Summary Metrics"):
                            st.markdown(f"""
                            <div style="margin-top: 1rem; padding: 1rem; background: #000000; border-radius: 6px; border-left: 4px solid #28a745;">
                                <small style="color: #ffffff;">
                                    <strong>Summary:</strong> {summary_stats[0]} words • {compression}% compression • Model: {selected_model}
                                </small>
                            </div>
                            <div class="score-box">
                                <span class="score-icon">📊</span> <small><strong>BLEU Score:</strong> {bleu_score}</small>
                            </div>
                            <div class="score-box">
                                <span class="score-icon">🧠</span> <small><strong>Perplexity:</strong> {perplexity_score}</small>
                            </div>
                            <div class="score-box">
                                <span class="score-icon">🧠</span> <small><strong>Readability Delta:</strong> {readability_delta} (Original: {original_readability}, Summary: {summary_readability})</small>
                            </div>
                            """, unsafe_allow_html=True)
                
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
        
        # Reference logic: If a summary has been generated and a reference is found, show it
        if st.session_state.get("summary_output") and st.session_state.get("reference_output"):
            reference_text, reference_summary = st.session_state["reference_output"]
            st.markdown(f"""
            <div class="content-box">
                <h4 style="color: #6c757d; margin-bottom: 0.5rem;">Dataset:</h4>
                <p style="line-height: 1.4; margin: 0; color: #ffffff;">{reference_summary}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate scores for reference summary
            ref_bleu_score = calculate_bleu(reference_text, reference_summary)
            ref_perplexity_score = calculate_perplexity(reference_summary)
            ref_original_readability = calculate_readability_scores(reference_text)
            ref_summary_readability = calculate_readability_scores(reference_summary)
            ref_readability_delta = round(ref_original_readability - ref_summary_readability, 2)
            
            with st.expander("View Reference Summary Metrics"):
                st.markdown(f"""
                <div class="score-box reference">
                    <span class="score-icon">📊</span> <small><strong>BLEU Score (Reference):</strong> {ref_bleu_score}</small>
                </div>
                <div class="score-box reference">
                    <span class="score-icon">🧠</span> <small><strong>Perplexity (Reference):</strong> {ref_perplexity_score}</small>
                </div>
                <div class="score-box reference">
                    <span class="score-icon">🧠</span> <small><strong>Readability Delta (Reference):</strong> {ref_readability_delta} (Original: {ref_original_readability}, Summary: {ref_summary_readability})</small>
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
    
    with translate_col:
        st.markdown("""
        <div class="column-header">
            <h3>🌐 Translated Text</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholder for translated text
        translated_text_placeholder = st.empty()

        if translate_button and st.session_state.get("summary_output"):
            with st.spinner(f"Translating to {target_language}..."):
                summary_to_translate = st.session_state.get("summary_output")
                translated_summary = translate_text(summary_to_translate, target_language)
                translated_text_placeholder.markdown(f"""
                <div class="content-box">
                    <p style="line-height: 1.6; margin: 0;">{translated_summary}</p>
                </div>
                """, unsafe_allow_html=True)
        elif translate_button and not st.session_state.get("summary_output"):
            translated_text_placeholder.error("⚠️ Please generate a summary first before translating.")
        else:
            # Placeholder when no translation is generated
            translated_text_placeholder.markdown("""
                <div class="placeholder-box">
                    <div class="placeholder-content">
                        <h4 style="color: #6c757d; margin-bottom: 1rem;">Translate Your Summary</h4>
                        <p style="margin-bottom: 1rem;">Select a language and click translate</p>
                        <p style="font-size: 0.9em; margin: 0;">
                            ⚙️ Multiple Languages<br>
                            🚀 Fast Translation<br>
                            💡 AI-Powered
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