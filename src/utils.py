import fitz  # PyMuPDF
import re


def read_pdf(file_path: str) -> str:
    """
    Read the text from a PDF file.

    :param file_path: Path to the PDF file.
    :return: Extracted text from the PDF.
    """
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
    return text


def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by removing special characters and lowercasing.

    :param text: The text to preprocess.
    :return: Cleaned text.
    """
    # Remove special characters and numbers
    text = re.sub(r'[^آ-یa-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text
