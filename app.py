import os
import gradio as gr
from phishing_detector import PhishingDetector
import re

# Initialize the detector
detector = PhishingDetector()

# Check if a pre-trained model exists
model_path = 'phishing_detector_model.pkl'
if os.path.exists(model_path):
    if detector.load_model(model_path):
        print("Model loaded successfully!")
    else:
        print("Failed to load model. Please train a model first.")
        exit(1)
else:
    print("No pre-trained model found. Please train a model first.")
    exit(1)

def extract_url_from_text(text):
    """Extract a URL from text if present"""
    url_pattern = r'(https?://\S+|www\.\S+|\S+\.\S+\/\S+)'
    match = re.search(url_pattern, text)
    return match.group(0) if match else "http://example.com"  # Default URL if none found

def detect_phishing(text, url=None):
    """Function to detect phishing for the web interface"""
    if not detector.vectorizer and not detector.pipeline:
        return "Error", "0.0%", "N/A", "N/A", "No trained model available. Please train a model first."
    
    if not text:
        return "No input provided", "0.0%", "N/A", "N/A", "Please provide email text or URL to analyze."
    
    # If no URL provided, try to extract one from text
    if not url:
        url = extract_url_from_text(text)
    
    # Make prediction
    result = detector.predict(text, url)
    
    # Extract values
    probability = result['phishing_probability']
    is_phishing = result['is_phishing']
    confidence = result['confidence']
    
    # Determine risk level
    if probability < 0.3:
        risk_level = "Low Risk"
    elif probability < 0.7:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
    
    # Generate explanation
    if is_phishing:
        explanation = "This message appears to be a phishing attempt. "
        if probability > 0.9:
            explanation += "It contains multiple strong indicators of phishing."
        else:
            explanation += "It contains some indicators of phishing."
    else:
        explanation = "This message appears to be legitimate. "
        if probability < 0.1:
            explanation += "It contains no indicators of phishing."
        else:
            explanation += "However, always verify sender identity and links before responding."
    
    # Format probability as percentage
    probability_pct = f"{probability*100:.1f}%"
    
    # Return individual values for Gradio outputs
    return (
        "Phishing" if is_phishing else "Legitimate",
        probability_pct,
        confidence,
        risk_level,
        explanation
    )

# Create Gradio interface
with gr.Blocks(title="AI-Enhanced Phishing Detector") as app:
    gr.Markdown("# AI-Enhanced Phishing Detector")
    gr.Markdown("Enter an email text or suspicious message to analyze for phishing indicators.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Email or Message Text",
                placeholder="Paste suspicious email or message text here...",
                lines=10
            )
            url_input = gr.Textbox(
                label="URL (Optional)",
                placeholder="Enter suspicious URL here (optional)"
            )
            submit_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            prediction_output = gr.Label(label="Prediction")
            probability_output = gr.Label(label="Phishing Probability")
            confidence_output = gr.Label(label="Confidence")
            risk_output = gr.Label(label="Risk Level")
            explanation_output = gr.Textbox(label="Explanation", lines=3)
    
    submit_btn.click(
        fn=detect_phishing,
        inputs=[text_input, url_input],
        outputs=[
            prediction_output,
            probability_output,
            confidence_output,
            risk_output,
            explanation_output
        ]
    )
    
    gr.Markdown("## How It Works")
    gr.Markdown("""
    This tool uses advanced Natural Language Processing (NLP) and machine learning techniques to analyze text for phishing indicators:
    
    1. Analyzes text content for suspicious phrases and patterns
    2. Checks URLs for common phishing characteristics
    3. Evaluates multiple risk factors to determine phishing probability
    
    **Note:** This is a prototype tool. Always exercise caution with suspicious messages.
    """)
    
    # Example inputs
    gr.Examples(
        [
            [
                "Dear valued customer, Your account has been locked due to suspicious activity. Please click http://secure-bank.com-login.xyz/update to verify your information.",
                "http://secure-bank.com-login.xyz/update"
            ],
            [
                "Congratulations! You've won a free iPhone 13. Claim your prize now at prize-winner.xyz/claim before it expires!",
                "prize-winner.xyz/claim"
            ],
            [
                "Hi Sarah, I'm sharing the project timeline document we discussed in yesterday's meeting. Let me know if you have any questions! Best, John",
                ""
            ]
        ],
        inputs=[text_input, url_input]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()