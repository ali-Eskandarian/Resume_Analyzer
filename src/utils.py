import fitz  # PyMuPDF
import re
from name_detection.persian_names import extract_names
from skills import ds_keywords, web_keywords, android_keywords, ios_keywords, uiux_keywords, devops_keywords
import umap
import matplotlib.pyplot as plt

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


def extract_contact_info(text):
    """Extract contact information from the resume text."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{11}\b'

    email = re.search(email_pattern, text)
    phone = re.search(phone_pattern, text)
    name = extract_names(text)

    contact_info = {
        'email': email.group() if email else None,
        'phone': phone.group() if phone else None,
        'name': name
    }

    return contact_info


def extract_age(text):
    """Extract age from the resume text."""
    age_pattern = r'\b([\d۰-۹]{1,2})/([\d۰-۹]{4})\b'  # Matches formats like 8/2000, 12/1375, or ۸/۲۰۰۰, ۱۲/۱۳۷۵
    year_pattern = r'\b([\d۰-۹]{4})\b'  # Matches 4-digit years like 1403 or ۱۳۸۲

    age_match = re.search(age_pattern, text)
    year_matches = re.findall(year_pattern, text)  # Find all 4-digit numbers

    current_year = 1403  # Replace with the current year in Shamsi

    if age_match:
        age = int(age_match.group(1))  # Extracted age
        birth_year = int(age_match.group(2))  # Extracted birth year
        if 18 < age < 40:  # Check if age is within the specified range
            return age

    elif year_matches:
        birth_year = min([int(year) for year in year_matches])  # Find the lowest year
        calculated_age = current_year - birth_year  # Calculate age from birth year
        if 18 < calculated_age < 40:  # Check if calculated age is within the specified range
            return calculated_age

    return None  # Return None if no valid age is found


def extract_quality(text, skill):
    """Extract the quality of a skill from the text."""
    # Define patterns to search for quality indicators
    quality_patterns = r'(?<=\b' + re.escape(
        skill) + r'\b).*?(\d|کم|مبتدی|متوسط|زیاد|پیشرفته|beginner|intermediate|advance)'

    quality_mapping = {
        'کم': 1, 'مبتدی': 1, 'beginner': 1,
        'متوسط': 2, 'intermediate': 2, 'advance': 3,
        'زیاد': 3, 'پیشرفته': 3
    }

    match = re.search(quality_patterns, text)
    if match:
        quality_str = match.group(0).strip()
        # Check if the quality is numeric or a keyword
        for key, value in quality_mapping.items():
            if key in quality_str:
                return value
        if quality_str.isdigit():
            return int(quality_str)

    return None  # Return None if no quality found


def extract_skills(text):
    """Extract skills from the resume text based on predefined categories."""
    skills = {
        'Data Science': [],
        'Web Development': [],
        'Android Development': [],
        'iOS Development': [],
        'UI/UX Design': [],
        'DevOps': []
    }

    skills_quality = {}  # Dictionary to hold skills and their quality

    # Convert text to lowercase
    text = text.lower()

    # Check for each category of skills
    for keyword in ds_keywords:
        if keyword.lower() in text:
            skills['Data Science'].append(keyword)
            # Check for quality
            quality = extract_quality(text, keyword)
            if quality:
                skills_quality[keyword] = quality

    for keyword in web_keywords:
        if keyword.lower() in text:
            skills['Web Development'].append(keyword)
            quality = extract_quality(text, keyword)
            if quality:
                skills_quality[keyword] = quality

    for keyword in android_keywords:
        if keyword.lower() in text:
            skills['Android Development'].append(keyword)
            quality = extract_quality(text, keyword)
            if quality:
                skills_quality[keyword] = quality

    for keyword in ios_keywords:
        if keyword.lower() in text:
            skills['iOS Development'].append(keyword)
            quality = extract_quality(text, keyword)
            if quality:
                skills_quality[keyword] = quality

    for keyword in uiux_keywords:
        if keyword.lower() in text:
            skills['UI/UX Design'].append(keyword)
            quality = extract_quality(text, keyword)
            if quality:
                skills_quality[keyword] = quality

    for keyword in devops_keywords:
        if keyword.lower() in text:
            skills['DevOps'].append(keyword)
            quality = extract_quality(text, keyword)
            if quality:
                skills_quality[keyword] = quality

    return skills, skills_quality


def visualize_data_with_umap(data, output_filename='umap_visualization.png'):
    # Visualize the data using UMAP and save the plot as a PNG file.

    # If necessary, you can drop non-numeric columns or encode them
    numeric_data = data.select_dtypes(include=[float, int])

    # Initialize UMAP
    reducer = umap.UMAP()

    # Fit and transform the data
    embedding = reducer.fit_transform(numeric_data)

    # Create a scatter plot of the UMAP embedding
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)

    # Set labels and title
    plt.title('UMAP Visualization of Data')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # Save the plot as a PNG file
    plt.savefig(output_filename)
    plt.close()  # Close the plot to free memory
