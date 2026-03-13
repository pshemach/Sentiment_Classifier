import re

def clean_text(text):
    """Clean text by lowercasing, removing URLs, HTML, punctuation and extra whitespace.

    Args:
        text (str): Input text string to clean

    Returns:
        text: Cleaned text
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"<.*?>", "", text)         # remove HTML
    text = re.sub(r"[^a-zA-Z\s]", "", text)   # remove special chars
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces

    return text