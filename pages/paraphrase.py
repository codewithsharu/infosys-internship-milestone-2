import streamlit as st
from transformers.pipelines import pipeline
import textstat
import plotly.express as px
import plotly.graph_objects as go
import time
import pandas as pd
import csv
import os
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianMTModel, MarianTokenizer
import torch
import requests

# -----------------------------
# Page Configuration & Custom CSS
# -----------------------------
st.set_page_config(
    page_title="AI Paraphrasing Studio",
    page_icon="🖤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load GPT-2 model and tokenizer for perplexity calculation
@st.cache_resource
def get_perplexity_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

perplexity_tokenizer, perplexity_model = get_perplexity_model()

@st.cache_resource
def get_translation_models():
    translation_resources = {}
    
    translation_resources["French"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr"),
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    }
    
    translation_resources["Hindi"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi"),
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    }

    translation_resources["Telugu"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ine"),
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ine")
    }

    translation_resources["Spanish"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es"),
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    }

    translation_resources["German"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de"),
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    }

    translation_resources["Italian"] = {
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it"),
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-it")
    }
    
    return translation_resources

translation_models = get_translation_models()

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
        return 0.0
    except Exception as e:
        return 0.0

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

@st.cache_data
def calculate_readability_scores(text):
    """Calculate Flesch-Kincaid Grade Level"""
    if not text.strip():
        return 0.0
    try:
        return round(textstat.flesch_kincaid_grade(text), 2)
    except Exception as e:
        return 0.0

def translate_text(text, target_language="French"):
    if not text.strip():
        return ""
    try:
        # Get tokenizer and model for the target language
        tokenizer = translation_models[target_language]["tokenizer"]
        model = translation_models[target_language]["model"]
        
        translated_tokens = model.generate(**tokenizer(text, return_tensors="pt"))
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        return f"Translation error: {e}"

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
    if 'original_text_for_scores' in st.session_state and 'generated_paraphrase_for_scores' in st.session_state:
        original_text = st.session_state['original_text_for_scores']
        output_text = st.session_state['generated_paraphrase_for_scores']
        is_thumbs_up = True if st.session_state.feedback_radio == "👍 Like" else False
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
            response = requests.post("http://localhost:5000/api/paraphrase", json=feedback_payload)
            if response.status_code in [200, 201]:
                st.success("Feedback submitted successfully!")
            else:
                st.error(f"Failed to submit feedback: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Connection error: Could not connect to the feedback API. Please ensure the backend is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please generate a paraphrase first before submitting feedback.")


# -----------------------------
# Main Application
# -----------------------------
def main():
    # Header
    # st.markdown("""
    # <div class="header">
    #     <h1>🖤 AI Paraphrasing Studio</h1>
    #     <p>Transform your text with precision AI technology and comprehensive readability analysis</p>
    # </div>
    # """, unsafe_allow_html=True)

    # Initialize session state for loading
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Vamsi/T5_Paraphrase_Paws" # Default model (actual model ID)
    if 'paraphrased_output_for_translation' not in st.session_state:
        st.session_state.paraphrased_output_for_translation = ""
    if 'selected_translation_language' not in st.session_state:
        st.session_state.selected_translation_language = "Hindi" # Default translation language changed to Hindi
    if 'original_text_for_scores' not in st.session_state:
        st.session_state['original_text_for_scores'] = ""
    if 'generated_paraphrase_for_scores' not in st.session_state:
        st.session_state['generated_paraphrase_for_scores'] = ""

    # Removed the initial block that checked for model_selector change, as it was interfering
    # with the model name mapping. The model change logic is now handled after the selectbox.

    # Load model with loading screen
    if not st.session_state.model_loaded:
        st.markdown(show_ai_loader("Initializing AI Models"), unsafe_allow_html=True)
        # Ensure we load the actual model ID, not the display name
        model_to_load = st.session_state.selected_model 
        paraphraser = load_paraphraser(model_to_load)
        st.session_state.paraphraser = paraphraser
        st.session_state.model_loaded = True
        time.sleep(1)
        st.rerun()
    
    paraphraser = st.session_state.paraphraser

    # Model name mapping for user-friendly display
    model_display_names = {
        "Vamsi/T5_Paraphrase_Paws": "T5 Paraphrase ",
        "tuner007/pegasus_paraphrase": "Pegasus Paraphrase",
        "MBZUAI/LaMini-Flan-T5-248M": "LaMini Flan T5"
    }
    
    # Reverse mapping for actual model loading
    display_to_model_names = {v: k for k, v in model_display_names.items()}

    # # Controls Section
    # st.markdown("""
    # <div class="controls-section">
    #     <div class="controls-title">Configuration Settings</div>
    # </div>
    # """, unsafe_allow_html=True)
    
    # Model and parameter controls in a single row
    control_col1, control_col2 = st.columns([3, 1])
    
    with control_col1:
        # Get the currently selected model's display name based on the actual model ID in session state
        current_model_display_name = model_display_names.get(st.session_state.selected_model, "T5 Paraphrase ")

        selected_model_display_name = st.selectbox(
            "Select AI Model",
            list(model_display_names.values()),
            index=list(model_display_names.values()).index(current_model_display_name),
            help="Choose the AI model for paraphrasing",
            key="model_selector"
        )
    
    # Update the actual selected_model in session state based on the display name
    actual_selected_model = display_to_model_names.get(selected_model_display_name, "Vamsi/T5_Paraphrase_Paws")
    # Check if the actual model ID has changed, then trigger reload
    if st.session_state.selected_model != actual_selected_model:
        st.session_state.model_loaded = False
        st.session_state.selected_model = actual_selected_model # Store the actual model ID
        st.rerun()

    # The generate button is placed in the input_col below.
    # The previous code in control_col2 for generate_button is commented out as it was a duplicate.
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main four-column layout
    input_col, paraphrase_col, reference_col, translate_col = st.columns([1, 1, 1, 1])
    
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
        
        # This is the correct placement for the generate button
        generate_button = st.button(
            "🚀 Generate Paraphrase",
            type="primary",
            use_container_width=True
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
            max_len = min(750, max(60, int(len(user_text.split()) * 4.0))) # Increased multiplier to 4.0 and min cap to 60
            num_return = 1  # Default number of return sequences

            # Generate paraphrased versions
            # The paraphraser is already loaded with the selected model.
            # We no longer have 'style' or 'level' controls.
            
            try:
                result = paraphraser(
                    user_text, # Pass user_text directly without style modification
                    max_length=max_len,
                    num_return_sequences=num_return,
                    min_length=max(40, int(len(user_text.split()) * 4.0)), # Increased multiplier to 4.0 and min cap to 40
                    do_sample=True, # Enable sampling for more diverse paraphrases
                    temperature=1.5 # Adjust for creativity
                )
                
                # Clear loader
                loader_placeholder.empty()
                
                paraphrased_output = result[0]['generated_text']
                
                # Store generated data in session state for score analysis
                st.session_state['generated_paraphrase_for_scores'] = paraphrased_output
                st.session_state['original_text_for_scores'] = user_text
                st.session_state['paraphrased_output_for_translation'] = paraphrased_output # Ensure this is always set
            
            except Exception as e:
                loader_placeholder.empty()
                st.error(f"⚠️ Processing error: {str(e)}")
                # Ensure session state is cleared or updated if an error occurs
                st.session_state['paraphrased_output_for_translation'] = ""
                st.session_state['generated_paraphrase_for_scores'] = "" # Clear on error
                st.session_state['original_text_for_scores'] = "" # Clear on error
                
        elif generate_button and not user_text.strip():
            st.error("⚠️ Please enter text to paraphrase.")
        
        # Display paraphrased output and metrics if available in session state
        if st.session_state['generated_paraphrase_for_scores']:
            paraphrased_output = st.session_state['generated_paraphrase_for_scores']
            user_text_original = st.session_state['original_text_for_scores']

            # Display the paraphrased output
            st.markdown(f"""
            <div class="content-box">
                <p style="line-height: 1.6; margin: 0;">{paraphrased_output}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Paraphrase statistics
            # Recalculate stats based on stored session state
            para_word_count, para_sentence_count, para_char_count = get_text_stats(paraphrased_output)
            original_words = len(user_text_original.split())
            
            # Calculate scores
            bleu_score = calculate_bleu(user_text_original, paraphrased_output)
            perplexity_score = calculate_perplexity(paraphrased_output)
            original_readability = calculate_readability_scores(user_text_original)
            paraphrase_readability = calculate_readability_scores(paraphrased_output)
            readability_delta = round(original_readability - paraphrase_readability, 2)

            # Download button (moved here, kept for consistency even if not active)
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
            
            with st.expander("View AI Paraphrase Metrics"):
                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 1rem; background: #111111; border-radius: 6px; border-left: 4px solid #90EE90;">
                    <small style="color: #cccccc;">
                        <strong>Paraphrase:</strong> {para_word_count} words • {para_sentence_count} sentences
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
                    <small>Readability Delta (Original: {original_readability}, Paraphrase: {paraphrase_readability}):</small>
                    <span class="metric-value">{readability_delta}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Feedback Section
        if st.session_state['generated_paraphrase_for_scores']:
            st.markdown("""
                <div style="margin-top: 2rem; padding: 1rem; background: #111111; border-radius: 8px;">
                    <h4 style="color: #ffffff; margin-bottom: 1rem;">Give Feedback</h4>
                </div>
            """, unsafe_allow_html=True)

            feedback_col1, feedback_col2 = st.columns([1, 4])
            with feedback_col1:
                like_status = st.radio(
                    "",
                    ["👍 Like", "👎 Dislike"],
                    key="feedback_radio",
                    index=0,
                    horizontal=False
                )
            with feedback_col2:
                feedback_text = st.text_area(
                    "Additional Feedback (Optional)",
                    key="feedback_text_area",
                    height=70,
                    placeholder="e.g., 'The paraphrase changed the meaning slightly.'"
                )
            
            feedback_button = st.button(
                "Submit Feedback",
                key="submit_feedback_button",
                type="secondary",
                use_container_width=True,
                on_click=submit_feedback
            )

        # Placeholder when no paraphrase is generated
        else:
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

            # Calculate scores for reference paraphrase
            ref_bleu_score = calculate_bleu(reference_text, reference_paraphrase)
            ref_perplexity_score = calculate_perplexity(reference_paraphrase)
            ref_original_readability = calculate_readability_scores(reference_text)
            ref_paraphrase_readability = calculate_readability_scores(reference_paraphrase)
            ref_readability_delta = round(ref_original_readability - ref_paraphrase_readability, 2)

            with st.expander("View Reference Paraphrase Metrics"):
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
                    <small>Readability Delta (Reference) (Original: {ref_original_readability}, Paraphrase: {ref_paraphrase_readability}):</small>
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

    with translate_col:
        st.markdown("""
        <div class="column-header">
            <h3>🌐 Translated Paraphrase</h3>
        </div>
        """, unsafe_allow_html=True)

        translated_text_placeholder = st.empty()

        # Initialize session state for translated paraphrase if not present
        if 'translated_paraphrase_display' not in st.session_state:
            st.session_state['translated_paraphrase_display'] = ""

        if st.session_state['paraphrased_output_for_translation']:
            target_language = st.selectbox(
                "Target Language",
                list(translation_models.keys()),
                index=list(translation_models.keys()).index(st.session_state.get('selected_translation_language', "Hindi")) if 'selected_translation_language' in st.session_state else list(translation_models.keys()).index("Hindi"),
                key="translation_language_selector_bottom"
            )
        else:
            target_language = st.selectbox(
                "Target Language",
                list(translation_models.keys()),
                index=list(translation_models.keys()).index("Hindi"),
                key="translation_language_selector_disabled_bottom",
                disabled=True
            )

        translate_button = st.button(
            "🌍 Translate Paraphrase",
            type="secondary",
            use_container_width=True,
            key="translate_button_final_bottom"
        )


        if translate_button:
            if st.session_state.get("paraphrased_output_for_translation"):
                with st.spinner(f"Translating to {target_language}..."):
                    try:
                        paraphrase_to_translate = st.session_state.get("paraphrased_output_for_translation")
                        translated_paraphrase = translate_text(paraphrase_to_translate, target_language)
                        st.session_state['translated_paraphrase_display'] = translated_paraphrase
                        st.session_state['selected_translation_language'] = target_language # Store selected language
                    except Exception as e:
                        st.error(f"⚠️ Translation error: {str(e)}")
                        if 'translated_paraphrase_display' not in st.session_state:
                            st.session_state['translated_paraphrase_display'] = ""
            else:
                st.error("⚠️ Please generate a paraphrase first before translating.")
                st.session_state['translated_paraphrase_display'] = ""
        
        # Always display the content from session state
        if st.session_state['translated_paraphrase_display']:
            translated_text_placeholder.markdown(f"""
            <div class="content-box">
                <p style="line-height: 1.6; margin: 0;">{st.session_state['translated_paraphrase_display']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Placeholder when no translation is generated or after clearing
            translated_text_placeholder.markdown("""
                <div class="placeholder-box">
                    <div class="placeholder-content">
                        <h4 style="color: #666666; margin-bottom: 1rem;">Translate Your Paraphrase</h4>
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
        <p><strong>AI Paraphrasing Studio</strong> | Powered by Advanced Language Models</p>
        <p><em>Professional text paraphrasing for research, business, and academic use</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
