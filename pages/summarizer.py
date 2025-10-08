import streamlit as st
import evaluate
import textstat
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, MarianMTModel, MarianTokenizer
import torch
import numpy as np # For radar chart, if you decide to use it.
import matplotlib.pyplot as plt # For radar chart, if you decide to use it.
import os # For file operations
import csv # For CSV operations


# --- Model Loading ---
@st.cache_resource
def get_perplexity_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

perplexity_tokenizer, perplexity_model = get_perplexity_model()

@st.cache_resource
def get_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

t5_tokenizer, t5_model = get_t5_model()

@st.cache_resource
def get_translation_models():
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

    # English to Telugu (Changed to a broader Indo-European model)
    translation_resources["Telugu"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ine"), 
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ine")
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
    
    return translation_resources

translation_models = get_translation_models()

# --- Core Logic Functions ---
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
        return 0.0
    except Exception as e:
        return 0.0

def simulate_summarization(text, model_name, summary_length):
    """Simulate the summarization process with realistic delay"""
    if model_name == "T5 Small":
        inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = t5_model.generate(inputs, max_length=summary_length, min_length=int(summary_length * 0.5), length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    # Placeholder for other models or generic summarization if T5 Small is not selected
    # This part would need actual model calls for BART/Pegasus if you intend to use them.
    return "Unable to generate summary with current parameters."


def translate_text(text, target_language): 
    """Translate text using the loaded MarianMT model"""
    if not text.strip():
        return ""
    try:
        translated_tokens = translation_models[target_language]["model"].generate(**translation_models[target_language]["tokenizer"](text, return_tensors="pt"))
        translated_text = translation_models[target_language]["tokenizer"].decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        return f"Translation error: {e}"

# You will also need the helper functions like calculate_bleu, calculate_readability_scores, get_text_stats, calculate_reading_time
# and potentially generate_radar_chart if you are using it in your UI.
# For example:
@st.cache_data
def calculate_bleu(reference, candidate):
    """Calculate BLEU score using HuggingFace evaluate library"""
    if not reference or not candidate:
        return 0.0
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=[candidate], references=[[reference]])
    return round(results["bleu"] * 100, 2)

@st.cache_data
def calculate_readability_scores(text):
    """Calculate Flesch-Kincaid Grade Level"""
    if not text.strip():
        return 0.0
    try:
        return round(textstat.flesch_kincaid_grade(text), 2)
    except Exception as e:
        return 0.0

def get_text_stats(text):
    """Get basic statistics about the text"""
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    characters = len(text)
    return len(words), sentences, characters

def calculate_reading_time(text):
    """Calculate estimated reading time (assuming 200 words per minute)"""
    word_count = len(text.split())
    return max(1, round(word_count / 200))

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
            next(reader)  # Skip header row
            for row in reader:
                if len(row) < 2:
                    continue
                original = row[0].strip().strip('"')
                summary = row[1].strip().strip('"')
                # Exact match (ignoring leading/trailing whitespace and quotes)
                if input_text.strip() == original:
                    return (original, summary)
    except Exception as e:
        st.error(f"Error reading {csv_path}: {e}")
        return None
    return None

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
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')

    ax.grid(color='white', linestyle='--', alpha=0.7)
    ax.spines['polar'].set_color('white')
    ax.spines['outer'].set_color('white')
    
    return fig

# -----------------------------
# Page Configuration & Custom CSS
# -----------------------------
st.set_page_config(
    page_title="AI Summarization Studio",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for elegant black and white theme
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    /* Removed specific Streamlit container styles to rely on Streamlit defaults for top spacing */
    
    /* .main { */
    /*     background: #000000; */
    /*     color: #ffffff; */
    /*     min-height: 100vh; */
    /* } */
    
    /* .stApp { */
    /*     background: #000000; */
    /*     color: #ffffff; */
    /* } */
    
    /* .stApp > header { */
    /*     background: transparent; */
    /* } */
    
    /* .stApp > div > div > div > div > section { */
    /*     background: #000000; */
    /* } */
    
    /* Main container styling */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem; /* Aligned with summarizer.py */
    }
    
    /* Header styling */
    .header {
        text-align: center;
        margin-bottom: 3rem; /* Aligned with summarizer.py */
        padding: 2rem 0; /* Aligned with summarizer.py */
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
        background: #f8f9fa; /* Changed to white background */
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef; /* Changed border color */
    }
    
    .column-header h3 {
        color: #495057; /* Changed text color to dark grey */
        margin: 0;
        font-size: 1.2rem;
        font-weight: 500;
    }

    /* Content boxes */
    .content-box {
        background: #000000; /* Changed to black background */
        border: 2px dashed #ffffff; /* Vibrant white dashed border */
        border-radius: 8px;
        padding: 1.5rem;
        height: 400px; /* Ensure consistent height */
        overflow-y: auto;
        color: #ffffff; /* White text for contrast */
        font-family: 'Inter', sans-serif;
    }
    
    .placeholder-box {
        background: #000000; /* Changed to black background */
        border: 2px dashed #ffffff; /* Vibrant white dashed border */
        border-radius: 8px;
        padding: 2rem;
        height: 400px; /* Ensure consistent height */
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #ffffff; /* White text for contrast */
    }
    
    .placeholder-content {
        color: #ffffff; /* White text for contrast */
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
        background: #000000; /* Changed to black background */
        border: 2px dashed #ffffff; /* Vibrant white dashed border */
        border-radius: 8px;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        line-height: 1.6;
        height: 300px; /* Set height to match other boxes */
        padding: 1.5rem; /* Added padding to match content-box */
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #666666;
        outline: none;
    }

    /* Specific style for feedback text area */
    textarea[aria-label="Additional Feedback (Optional)"] {
        height: 20px !important; /* Force height for feedback text area */
        min-height: 20px !important;
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

    /* New styles for metric boxes */
    .metric-box {
        background: #111111; /* Darker background than main content boxes */
        border-left: 4px solid #90EE90; /* Green accent bar on the left */
        border-radius: 8px; /* Slightly rounded corners */
        padding: 1rem 1.2rem; /* Ample padding */
        margin-bottom: 0.75rem; /* Space between metric boxes */
        color: #ffffff; /* White text for readability */
        font-family: 'Inter', sans-serif; /* Consistent font */
        display: flex; /* Use flexbox for alignment */
        justify-content: space-between; /* Space out content within the box */
        align-items: center; /* Vertically align content */
        box-shadow: 0 2px 8px rgba(0,0,0,0.2); /* Subtle shadow for depth */
    }

    .metric-box small {
        color: #cccccc; /* Lighter grey for small text */
        font-size: 0.95rem; /* Slightly larger font for metrics */
    }

    .metric-value {
        font-weight: 600; /* Make the metric values stand out */
        color: #ffffff; /* White color for values */
    }
</style>
""", unsafe_allow_html=True)

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

def submit_feedback():
    if 'original_text_for_scores' in st.session_state and 'generated_summary_for_scores' in st.session_state:
        original_text = st.session_state['original_text_for_scores']
        output_text = st.session_state['generated_summary_for_scores']
        is_thumbs_up = True if st.session_state.feedback_radio == "👍" else False
        feedback_text = st.session_state.feedback_text_area

        feedback_payload = {
            "inputText": original_text,
            "outputText": output_text,
            "feedback": {
                "isThumbsUp": is_thumbs_up,
                "feedbackText": feedback_text
            }
        }
        try:
            import requests # Import requests here if not already imported globally
            response = requests.post("http://localhost:5000/api/summarize", json=feedback_payload)
            if response.status_code in [200, 201]:
                st.success("Feedback submitted successfully!")
            else:
                st.error(f"Failed to submit feedback: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Connection error: Could not connect to the feedback API. Please ensure the backend is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please generate a summary first before submitting feedback.")


# -----------------------------
# Main Application
# -----------------------------
def main():
    st.markdown("""
    
    """, unsafe_allow_html=True)

    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'selected_summarizer_model' not in st.session_state:
        st.session_state.selected_summarizer_model = "T5 Small"
    if 'summarized_output_for_translation' not in st.session_state:
        st.session_state.summarized_output_for_translation = ""
    if 'selected_translation_language' not in st.session_state:
        st.session_state.selected_translation_language = "Hindi"
    if 'original_text_for_scores' not in st.session_state:
        st.session_state['original_text_for_scores'] = ""
    if 'generated_summary_for_scores' not in st.session_state:
        st.session_state['generated_summary_for_scores'] = ""
    if 'summary_length' not in st.session_state:
        st.session_state['summary_length'] = 10 # Default summary length

    if not st.session_state.model_loaded:
        st.markdown(show_ai_loader("Initializing AI Models"), unsafe_allow_html=True)
        # In a real application, you'd load the T5 model here if it's not already loaded via @st.cache_resource
        # For now, we just mark it as loaded
        st.session_state.model_loaded = True
        # time.sleep(1) # Removed time.sleep for faster loading
        st.rerun()

    model_display_names = {
        "T5 Small": "T5 Small (Default)",
        # Add other summarization models here if you implement them
        # "BART Large CNN": "BART Large CNN",
        # "Pegasus XSUM": "Pegasus XSUM"
    }

    display_to_model_names = {v: k for k, v in model_display_names.items()}

    control_col1, control_col2 = st.columns([3, 1])

    with control_col1:
        current_model_display_name = model_display_names.get(st.session_state.selected_summarizer_model, "T5 Small")
        selected_model_display_name = st.selectbox(
            "Select AI Model",
            list(model_display_names.values()),
            index=list(model_display_names.values()).index(current_model_display_name),
            help="Choose the AI model for summarization",
            key="summarizer_model_selector"
        )
    
    actual_selected_model = display_to_model_names.get(selected_model_display_name, "T5 Small")
    if st.session_state.selected_summarizer_model != actual_selected_model:
        st.session_state.model_loaded = False # Consider reloading or re-initializing if models are heavy
        st.session_state.selected_summarizer_model = actual_selected_model
        st.rerun()

    input_col, summary_col, translate_col, metrics_col = st.columns([1, 1, 1, 1])

    with input_col:
        st.markdown("""
        <div class="column-header">
            <h3>📝 Input Text</h3>
        </div>
        """, unsafe_allow_html=True)
        
        user_text = st.text_area(
            "Enter your text for summarization:", 
            value="", 
            placeholder="""Enter your text here for summarization...

Examples:
• Summarize articles
• Condense long reports
• Extract key information
• Create executive summaries""",
            key="user_text_input_summarizer",
            height=300
        )
        
        with st.expander("Summarization Settings"):
            st.session_state['summary_length'] = st.slider(
                "Summary Length (words)",
                min_value=5,
                max_value=500,
                value=st.session_state['summary_length'],
                step=10,
                help="Adjust the desired length of the summary."
            )
            
        generate_button = st.button(
            "🚀 Generate Summary",
            type="primary",
            use_container_width=True
        )

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

    with summary_col:
        st.markdown("""
        <div class="column-header">
            <h3>✨ Generated Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if generate_button and user_text.strip():
            loader_placeholder = st.empty()
            loader_placeholder.markdown(show_ai_loader("Generating summary"), unsafe_allow_html=True)
            
            try:
                summary_output = simulate_summarization(
                    user_text,
                    st.session_state.selected_summarizer_model,
                    st.session_state['summary_length']
                )
                
                loader_placeholder.empty()
                
                st.session_state['generated_summary_for_scores'] = summary_output
                st.session_state['original_text_for_scores'] = user_text
                st.session_state['summarized_output_for_translation'] = summary_output
            
            except Exception as e:
                loader_placeholder.empty()
                st.error(f"⚠️ Processing error: {str(e)}")
                st.session_state['summarized_output_for_translation'] = ""
                st.session_state['generated_summary_for_scores'] = ""
                st.session_state['original_text_for_scores'] = ""
                
        elif generate_button and not user_text.strip():
            st.error("⚠️ Please enter text to summarize.")
        
        if st.session_state['generated_summary_for_scores']:
            summary_output = st.session_state['generated_summary_for_scores']
            st.markdown(f"""
            <div class="content-box">
                <p style="line-height: 1.6; margin: 0;">{summary_output}</p>
            </div>
            """, unsafe_allow_html=True)

            # Recalculate stats based on stored session state
            summary_word_count, _, _ = get_text_stats(summary_output)
            original_word_count, _, _ = get_text_stats(user_text)
            
            # Calculate scores
            bleu_score = calculate_bleu(user_text, summary_output)
            perplexity_score = calculate_perplexity(summary_output)
            original_readability = calculate_readability_scores(user_text)
            summary_readability = calculate_readability_scores(summary_output)
            readability_delta = round(original_readability - summary_readability, 2)

            # Horizontal radio buttons above the text area
            st.radio(
                "",
                ["👍", "👎"],
                key="feedback_radio",
                index=0,
                horizontal=True # Set to horizontal
            )
            
            st.text_area(
                "Additional Feedback (Optional)",
                key="feedback_text_area",
                placeholder="e.g., 'The summary is too short.'"
            )
            
            st.button(
                "Submit Feedback",
                key="submit_feedback_button",
                type="secondary",
                use_container_width=True,
                on_click=submit_feedback
            )
            
            # Moved Summary Metrics expander here
            with st.expander("Summary Metrics"):
                st.markdown(f"""
                <div style="margin-top: 0rem; padding: 1rem; background: #111111; border-radius: 6px;">
                    <small style="color: #cccccc;">
                        <strong>Summary:</strong> {summary_word_count} words ({round((summary_word_count/original_word_count)*100, 2) if original_word_count > 0 else 0}%)
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <small>BLEU Score:</small>
                    <span class="metric-value">{bleu_score}</span>
                </div>
                <div class="metric-box">
                    <small>Perplexity:</small>
                    <span class="metric-value">{perplexity_score}</span>
                </div>
                <div class="metric-box">
                    <small>Readability Delta:</small>
                    <span class="metric-value">{readability_delta}</span>
                </div>
                """, unsafe_allow_html=True)

                # Placeholder for radar chart, actual values would come from more advanced metrics
                fluency, semantic_similarity, coherence, diversity, fact_preservation = 70, 85, 80, 60, 90 # Example values
                # fig = generate_radar_chart(fluency, semantic_similarity, coherence, diversity, fact_preservation)
                # st.pyplot(fig) # Uncomment if you want to display the radar chart

        else:
            st.markdown("""
                <div class="placeholder-box">
                    <div class="placeholder-content">
                        <h4 style="color: #666666; margin-bottom: 1rem;">Ready to Generate Summary</h4>
                        <p style="margin-bottom: 1rem;">Enter your text and click "Generate Summary"</p>
                        <p style="font-size: 0.9em; margin: 0;">
                            ✓ Multiple Lengths<br>
                            ✓ Customizable Models<br>
                            ✓ Instant Processing<br>
                            ✓ Export Options
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    with translate_col:
        st.markdown("""
        <div class="column-header">
            <h3>🌐 Translated Summary</h3>
        </div>
        """, unsafe_allow_html=True)

        translated_text_placeholder = st.empty()

        if 'translated_summary_display' not in st.session_state:
            st.session_state['translated_summary_display'] = ""

        if st.session_state['summarized_output_for_translation']:
            target_language = st.selectbox(
                "Target Language",
                list(translation_models.keys()),
                index=list(translation_models.keys()).index(st.session_state.get('selected_translation_language', "Hindi")) if 'selected_translation_language' in st.session_state else list(translation_models.keys()).index("Hindi"),
                key="translation_language_selector_summary"
            )
        else:
            target_language = st.selectbox(
                "Target Language",
                list(translation_models.keys()),
                index=list(translation_models.keys()).index("Hindi"),
                key="translation_language_selector_disabled_summary",
                disabled=True
            )

        translate_button = st.button(
            "🌍 Translate Summary",
            type="secondary",
            use_container_width=True,
            key="translate_button_summary"
        )

        if translate_button:
            if st.session_state.get("summarized_output_for_translation"):
                with st.spinner(f"Translating to {target_language}..."):
                    try:
                        summary_to_translate = st.session_state.get("summarized_output_for_translation")
                        translated_summary = translate_text(summary_to_translate, target_language)
                        st.session_state['translated_summary_display'] = translated_summary
                        st.session_state['selected_translation_language'] = target_language
                    except Exception as e:
                        st.error(f"⚠️ Translation error: {str(e)}")
                        st.session_state['translated_summary_display'] = ""
            else:
                st.error("⚠️ Please generate a summary first before translating.")
                st.session_state['translated_summary_display'] = ""
        
        if st.session_state['translated_summary_display']:
            translated_text_placeholder.markdown(f"""
            <div class="content-box">
                <p style="line-height: 1.6; margin: 0;">{st.session_state['translated_summary_display']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            translated_text_placeholder.markdown("""
                <div class="placeholder-box">
                    <div class="placeholder-content">
                        <h4 style="color: #666666; margin-bottom: 1rem;">Translate Your Summary</h4>
                        <p style="margin-bottom: 1rem;">Select a language and click translate</p>
                        <p style="font-size: 0.9em; margin: 0;">
                            ⚙️ Multiple Languages<br>
                            🚀 Fast Translation<br>
                            💡 AI-Powered
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with metrics_col:
        st.markdown("""
        <div class="column-header">
            <h3>📚 Reference & Sources</h3>
        </div>
        """, unsafe_allow_html=True)
        
        reference = None
        if st.session_state['original_text_for_scores']:
            reference = find_reference_summary(st.session_state['original_text_for_scores'])

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

            # Calculate scores for reference summary
            ref_bleu_score = calculate_bleu(reference_text, reference_summary)
            ref_perplexity_score = calculate_perplexity(reference_summary)
            ref_original_readability = calculate_readability_scores(reference_text)
            ref_summary_readability = calculate_readability_scores(reference_summary)
            ref_readability_delta = round(ref_original_readability - ref_summary_readability, 2)

            with st.expander("View Reference Summary Metrics"):
                st.markdown(f"""
                <div class="metric-box">
                    <small>BLEU Score (Reference):</small>
                    <span class="metric-value">{ref_bleu_score}</span>
                </div>
                <div class="metric-box">
                    <small>Perplexity (Reference):</small>
                    <span class="metric-value">{ref_perplexity_score}</span>
                </div>
                <div class="metric-box">
                    <small>Readability Delta (Reference) (Original: {ref_original_readability}, Summary: {ref_summary_readability}):</small>
                    <span class="metric-value">{ref_readability_delta}</span>
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

    st.markdown("""
    <div class="footer">
        <p><strong>AI Summarization Studio</strong> | Powered by Advanced Language Models</p>
        <p><em>Condense and clarify text for various applications</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
