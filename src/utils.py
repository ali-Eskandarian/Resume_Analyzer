import fitz  # PyMuPDF
import re

def flatten_list(nested_list):
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist] if isinstance(nested_list, list) else [nested_list]

def cleanResume(resumeText):

    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
                        resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText


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
