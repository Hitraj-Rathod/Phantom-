import os
import re
import json
import logging
from datetime import datetime
from transformers import pipeline
from collections import Counter
from functools import lru_cache
import torch 
# Summarization with GPU {change from ft1}
def init_summarizer():
    device = 0 if torch.cuda.is_available() else -1 
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# ----------------------
# Setup Logging
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',#contain log entry(time of log ,severity level)
    handlers=[
        logging.FileHandler('emergency_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------
# Load Configuration
# ----------------------
CONFIG_FILE = "config.json"

def load_config():
    """Load configuration from JSON file."""
    default_config = {
        "input_file": "conversation.txt",
        "output_file": "conversation_analysis.json",
        "emergency_keywords": {
            "help":1.0, "emergency":2.0, "urgent":2.0, "accident": 2.0, "danger": 1.5, "fire": 2.0, "injury": 1.5,
            "pain": 1.0, "critical": 2.0, "collapse": 1.5, "unconscious": 1.0, "bleeding": 2.0, "crash": 2.0, "burn": 2.0
        },
        "issue_keywords": {
            "Stroke": ["stroke", "drooping face", "face drooping", "slurred speech", "weakness", "numbness"],
            "Heart Attack": ["chest pain", "heart attack", "shortness of breath", "arm pain", "jaw pain"],
            "Car Accident": ["car accident", "crash", "collision", "hit by car", "wreck"],
            "Fire": ["fire", "burning", "smoke", "flames"],
            "Injury": ["bleeding", "cut", "broken bone", "fracture", "wound", "wounded"],
            "Unconscious": ["unconscious", "unresponsive", "passed out", "fainted"],
            "Other": ["fall", "choking", "seizure", "allergic reaction", "poisoning"]
        },
        "urgency_indicators": [
            "unconscious", "unresponsive", "bleeding", "shallow breathing", "chest pain",
            "drooping face", "slurred speech", "not moving", "severe pain", "burning",
            "crash", "fainted", "seizure", "choking"
        ],
        "thresholds": {
            "high": 5.0,
            "medium": 2.0,
            "urgency": 3.0
        },
        "summarizer": {
            "model": "facebook/bart-large-cnn",
            "max_length": 100,
            "min_length": 30
        },
        "name_extraction": "victim"
    }
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                default_config.update(config)
            logger.info("Configuration loaded from %s", CONFIG_FILE)
        else:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)
            logger.info("Default configuration created at %s", CONFIG_FILE)
        return default_config
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        return default_config

# ----------------------
# Initialize Summarizer
# ----------------------
@lru_cache(maxsize=1)
def init_summarizer(model_name):
    """Initialize and cache the summarization pipeline."""
    try:
        summarizer = pipeline("summarization", model=model_name)
        logger.info("Summarizer initialized with model %s", model_name)
        return summarizer
    except Exception as e:
        logger.error("Failed to initialize summarizer: %s", e)
        raise

# ----------------------
# Read Conversation
# ----------------------
def read_conversation(file_path):
    """Read conversation text from file."""
    try:
        if not os.path.exists(file_path):
            logger.error("Input file %s not found", file_path)
            raise FileNotFoundError(f"File {file_path} not found")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                logger.warning("Input file %s is empty", file_path)
                raise ValueError("Input file is empty")
            logger.info("Successfully read conversation from %s", file_path)
            return content
    except Exception as e:
        logger.error("Error reading conversation: %s", e)
        raise

# ----------------------
# Emergency Classification
# ----------------------
def classify_emergency(text, emergency_keywords, urgency_indicators, thresholds):
    """Classify the emergency level based on keywords and urgency indicators."""
    try:
        text_lower = text.lower()
        keyword_count = sum(text_lower.count(word) for word in emergency_keywords)
        urgency_count = sum(1 for indicator in urgency_indicators if indicator in text_lower)
        logger.info("Keyword count: %d, Urgency count: %d", keyword_count, urgency_count)

        if urgency_count >= thresholds['urgency'] or keyword_count >= thresholds['high']:
            return "High Emergency"
        elif keyword_count >= thresholds['medium']:
            return "Medium Emergency"
        else:
            return "Not an Emergency"
    except Exception as e:
        logger.error("Error classifying emergency: %s", e)
        return "Classification Error"

# spaCy extraction
def extract_info(text):
    info = {}
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        info['name'] = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Not Mentioned")
        info['location'] = next((ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]), "Unknown")
    except:
        name_match = re.search(r"([A-Z][a-z]+)", text)
        location_match = re.search(r"(?:in|at)\s+([A-Za-z\s]+)", text.lower())
        info['name'] = name_match.group(1).title() if name_match else "Not Mentioned"
        info['location'] = location_match.group(1).title() if location_match else "Unknown"
    return info

# ----------------------
# Extract Information
# ----------------------
def extract_info(text, issue_keywords, urgency_indicators, name_extraction_mode):
    """Extract key information from conversation text."""
    try:
        text_lower = text.lower()
        info = {}

        # Extract location with improved regex
        location_match = re.search(
            r"(?:in|at|near|on)\s+([a-zA-Z0-9\s]+(?:street|avenue|road|kitchen|room|building|shop|park|hospital)(?:\s*(?:near|at|by)\s*[a-zA-Z0-9\s]+)?)(?!\s*(crash|accident))",
            text, re.IGNORECASE
        )
        info['location'] = location_match.group(1).strip().title() if location_match else "Unknown"
        logger.info("Extracted location: %s", info['location'])

        # Extract name (prioritize victim if mode is 'victim')
        if name_extraction_mode == "victim":
            # Look for names near emergency-related terms or possessive phrases like "my friend"
            name_match = re.search(
                r"(?:my friend\s+|patient\s+|victim\s+)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b(?=.*(?:unconscious|bleeding|injured|crash|accident|wounded))",
                text, re.IGNORECASE
            )
            if not name_match:
                # Fallback to any proper name
                name_match = re.search(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b(?!\s*(street|avenue|road|park))", text, re.IGNORECASE)
        else:
            name_match = re.search(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b(?!\s*(street|avenue|road|park))", text, re.IGNORECASE)
        info['name'] = name_match.group(1).title() if name_match else "Not Mentioned"
        logger.info("Extracted name: %s (mode: %s)", info['name'], name_extraction_mode)

        # Determine issue
        detected_issues = [
            issue for issue, keywords in issue_keywords.items()
            if any(keyword in text_lower for keyword in keywords)
        ]
        info['issue'] = ', '.join(detected_issues) if detected_issues else "Unknown"
        logger.info("Detected issues: %s", info['issue'])

        # Extract urgency indicators
        found_indicators = [i for i in urgency_indicators if i in text_lower]
        info['urgency_indicators'] = ', '.join(set(found_indicators)) if found_indicators else "None"
        logger.info("Urgency indicators: %s", info['urgency_indicators'])

        # Check if help was contacted (always Yes for 911 calls)
        help_phrases = ["ambulance", "calling emergency", "called help", "911", "emergency services"]
        info['help_contacted'] = "Yes" if any(phrase in text_lower for phrase in help_phrases) or "911 operator" in text_lower else "No"
        logger.info("Help contacted: %s", info['help_contacted'])

        return info
    except Exception as e:
        logger.error("Error extracting information: %s", e)
        return {}

# ----------------------
# Summarize Text
# ----------------------
def summarize_text(text, summarizer, config):
    """Summarize the conversation text."""
    try:
        if not text.strip():
            logger.warning("No content to summarize")
            return "No conversation content to summarize."
        if len(text) > 1024:
            text = text[:1024]
            logger.info("Text truncated to 1024 characters for summarization")
        summary = summarizer(
            text,
            max_length=config['summarizer']['max_length'],
            min_length=config['summarizer']['min_length'],
            do_sample=False
        )
        summary_text = summary[0]['summary_text']
        # Clean up summary
        summary_text = re.sub(r"^Person 1:\s*|^Caller:\s*|^911 Operator:\s*", "", summary_text)
        # Fix common errors
        summary_text = re.sub(r"\bpail\b", "pale", summary_text, flags=re.IGNORECASE)
        summary_text = re.sub(r"\band\s+and\b", "and", summary_text, flags=re.IGNORECASE)
        summary_text = re.sub(r"\s+", " ", summary_text).strip().capitalize()
        if not summary_text.endswith('.'):
            summary_text += '.'
        logger.info("Summarized text: %s", summary_text)
        return summary_text
    except Exception as e:
        logger.error("Summarization error: %s", e)
        return "Summary could not be generated."

# ----------------------
# Format Human-Readable Output
# ----------------------
def format_readable_output(output_data):
    """Format output for human readability."""
    try:
        timestamp = output_data['timestamp']
        classification = output_data['emergency_classification']
        summary = output_data['summary']
        info = output_data['extracted_information']

        output = f"""
============================================================
Emergency Analysis Report
Generated on: {timestamp}
============================================================

Emergency Status: {classification}
------------------------------------------------------------

Summary:
{summary}
------------------------------------------------------------

Details:
- Location: {info['location']}
- Issue: {info['issue']}
- Urgency Indicators: {info['urgency_indicators']}
- Help Contacted: {info['help_contacted']}
============================================================
"""
        return output
    except Exception as e:
        logger.error("Error formatting readable output: %s", e)
        return "Error generating readable output."

# ----------------------
# Save Output
# ----------------------
def save_output(output_data, output_file):
    """Save analysis output to a JSON file and a readable text file."""
    try:
        # Save JSON output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        logger.info("JSON output saved to %s", output_file)

        # Save human-readable text output
        text_output_file = os.path.splitext(output_file)[0] + ".txt"
        with open(text_output_file, "w", encoding='utf-8') as f:
            f.write(format_readable_output(output_data))
        logger.info("Text output saved to %s", text_output_file)
    except Exception as e:
        logger.error("Error saving output: %s", e)
        raise

# ----------------------
# Main
# ----------------------
def main():
    """Main function to process conversation and generate analysis."""
    try:
        # Load configuration
        config = load_config()

        # Initialize summarizer
        summarizer = init_summarizer(config['summarizer']['model'])

        # Read conversation
        conversation_text = read_conversation(config['input_file'])

        # Classify emergency
        emergency_class = classify_emergency(
            conversation_text,
            config['emergency_keywords'],
            config['urgency_indicators'],
            config['thresholds']
        )

        # Summarize conversation
        summary = summarize_text(conversation_text, summarizer, config)

        # Extract information
        extracted_info = extract_info(
            conversation_text,
            config['issue_keywords'],
            config['urgency_indicators'],
            config['name_extraction']
        )

        # Prepare output
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "emergency_classification": emergency_class,
            "summary": summary,
            "extracted_information": extracted_info
        }

        # Save output (JSON and text)
        save_output(output_data, config['output_file'])

        # Print human-readable output
        print(format_readable_output(output_data))

    except Exception as e:
        logger.error("Main execution failed: %s", e)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
