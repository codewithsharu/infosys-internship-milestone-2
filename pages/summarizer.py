import streamlit as st
import requests # For API calls

st.set_page_config(layout="wide")

def app():
    st.title("Text Summarizer")

    st.markdown("Enter your text below to get a summarized version.")

    # Model selection
    available_models = {
        "BART Large CNN": "http://localhost:8000/summarize/bart",
        "T5 Small": "http://localhost:8000/summarize/t5-small",
        "Pegasus XSUM": "http://localhost:8000/summarize/pegasus"
    }
    selected_model = st.selectbox("Select Summarization Model", list(available_models.keys()))

    # Input text area
    text_input = st.text_area("", height=300, placeholder="Paste your text here...")

    # Summarization options
    col1, col2 = st.columns(2)
    with col1:
        summary_length = st.slider("Summary Length (words)", min_value=50, max_value=500, value=150, step=10)
    with col2:
        summary_ratio = st.slider("Summary Ratio (% of original)", min_value=10, max_value=90, value=30, step=5)

    summarize_button = st.button("Summarize")

    if summarize_button and text_input:
        with st.spinner("Summarizing..."):
            # Determine the API endpoint based on the selected model
            # model_endpoint = available_models[selected_model]
            # st.write(f"Using model: {selected_model} (Endpoint: {model_endpoint})") # For debugging

            # Placeholder for API call or model integration
            # In a real application, you would send text_input, summary_length, summary_ratio, and selected_model to a backend API
            # or a local summarization model.
            # For now, we'll just simulate a summary.

            # Example of calling a hypothetical local API (replace with your actual API endpoint)
            # try:
            #     response = requests.post(
            #         model_endpoint, # Use the selected model's endpoint
            #         json={
            #             "text": text_input,
            #             "length": summary_length,
            #             "ratio": summary_ratio,
            #             "model": selected_model # Pass the selected model to the backend
            #         }
            #     )
            #     if response.status_code == 200:
            #         summary = response.json().get("summary", "No summary returned.")
            #     else:
            #         summary = f"Error: {response.status_code} - {response.text}"
            # except requests.exceptions.ConnectionError:
            #     summary = "Could not connect to the summarization service. Please ensure it's running."

            # Simulated summary for demonstration
            words = text_input.split()
            num_words = min(summary_length, int(len(words) * (summary_ratio / 100)))
            summary = f"[Summarized by {selected_model}] " + " ".join(words[:num_words]) + "... [Simulated Summary]"

            st.subheader("Summary")
            st.write(summary)
    elif summarize_button and not text_input:
        st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    app()
