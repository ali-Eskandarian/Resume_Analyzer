import fitz
import re
from name_detection.persian_names import extract_names
from skills import ds_keywords, web_keywords, android_keywords, ios_keywords, uiux_keywords, devops_keywords
import umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def flatten_list(nested_list):
    """
    Flatten a nested list.

    :param nested_list: The input list
    :return: flatten the list
    """
    return [item for sublist in nested_list for item in sublist] if isinstance(nested_list, list) else [nested_list]


def cleanResume(resumetext):
    """
    Clean the text

    :param resumetext: The input text containing information
    :return: A clean text ready to use
    """
    resumetext = re.sub('http\S+\s*', ' ', resumetext)  # remove URLs
    resumetext = re.sub('RT|cc', ' ', resumetext)  # remove RT and cc
    resumetext = re.sub('#\S+', '', resumetext)  # remove hashtags
    resumetext = re.sub('@\S+', '  ', resumetext)  # remove mentions
    resumetext = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
                        resumetext)  # remove punctuations
    resumetext = re.sub(r'[^\x00-\x7f]', r' ', resumetext)
    resumetext = re.sub('\s+', ' ', resumetext)  # remove extra whitespace
    return resumetext


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
    """
    Extract contact information from the resume text.

    :param text: The input text containing skill information
    :return: A dictionary containing the information
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{11}\b'

    email = re.search(email_pattern, text)
    phone = re.search(phone_pattern, text)
    name  = extract_names(text)

    contact_info = {
        'email': email.group() if email else None,
        'phone': phone.group() if phone else None,
        'name' : name
    }

    return contact_info


def extract_age(text):
    """
    Extract age from the resume text

    :param text: The input text containing skill information
    :return: An integer as age
    """
    age_pattern      = r'\b([\d۰-۹]{1,2})/([\d۰-۹]{4})\b'  # Matches formats like 8/2000, 12/1375, or ۸/۲۰۰۰, ۱۲/۱۳۷۵
    year_pattern     = r'\b([\d۰-۹]{4})\b'  # Matches 4-digit years like 1403 or ۱۳۸۲
    age_year_pattern = r'\b([\d۰-۹]{2})\b'  # Matches 4-digit years like 1403 or ۱۳۸۲

    age_match                = re.search(age_pattern, text)
    year_matches             = re.findall(year_pattern, text)  # Find all 4-digit numbers
    age_year_pattern_matches = re.findall(age_year_pattern, text)  # Find all 2-digit numbers

    current_year = 1403
    calculated_age = None
    if age_match:
        calculated_age = int(age_match.group(1))  # Extracted age
        if 18 < calculated_age < 40:  # Check if age is within the specified range
            return calculated_age
        else:
            calculated_age = None

    elif year_matches:
        birth_year = min([int(year) for year in year_matches])  # Find the lowest year
        calculated_age = current_year - birth_year  # Calculate age from birth year
        if 18 < calculated_age < 40:  # Check if calculated age is within the specified range
            return calculated_age
        else:
            calculated_age = None

    if calculated_age is None:
        age_candidates = [int(year) for year in age_year_pattern_matches]
        for calculated_age in age_candidates:
            if 40 > calculated_age > 18:
                return calculated_age

    return calculated_age


def extract_quality(text, skill):
    """
    Extract the quality of a skill from the text.

    :param text: The input text containing skill information
    :param skill: The specific skill to search for
    :return: An integer representing the skill quality (1: beginner, 2: intermediate, 3: advanced)
    """
    # Normalize the text by replacing Persian numbers with Arabic numbers
    text = text.replace('۰', '0').replace('۱', '1').replace('۲', '2').replace('۳', '3').replace('۴', '4') \
        .replace('۵', '5').replace('۶', '6').replace('۷', '7').replace('۸', '8').replace('۹', '9')

    # Define quality mappings
    quality_mapping = {
        'مبتدی'  : 1, 'کم'          : 1, 'اشنا'  : 1, 'آشنا'   : 1, 'beginner': 1,
        'متوسط'  : 2, 'intermediate': 2,
        'پیشرفته': 3, 'حرفه ای'     : 3, 'حرفه‌ای': 3, 'advance': 3, 'advanced': 3
    }

    # Create a pattern to match the skill and its quality
    pattern = rf'\b{re.escape(skill)}[\s|]*([^\n|]*)'
    match   = re.search(pattern, text, re.IGNORECASE)

    if match:
        quality_str = match.group(1).strip()

        # Check for percentage
        percentage_match = re.search(r'(\d+)%', quality_str)
        if percentage_match:
            percentage = int(percentage_match.group(1))
            if percentage <= 33:
                return 1
            elif percentage <= 66:
                return 2
            else:
                return 3

        # Check for quality keywords
        for key, value in quality_mapping.items():
            if key in quality_str:
                return value

        # If no quality is found, assume it's intermediate
        return 2

    # If the skill is not found, return None
    return None


def extract_skills(text):
    """Extract skills from the resume text based on predefined categories."""
    skills = {
        'Data Science'       : [],
        'Web Development'    : [],
        'Android Development': [],
        'iOS Development'    : [],
        'UI/UX Design'       : [],
        'DevOps'             : []
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



def visualize_data_with_umap(data, output_filename='../umap_visualization.png'):
    """
    Calculate a UMAP and save

    :param data: A pandas dataframe
    :param output_filename: The dir to save
    :return: ---
    """
    # Calculate bin ranges based on the maximum value in the 'score' column
    max_score = data['score'].max()
    bins      = [0, int(max_score/3), int(2*max_score/3), int(max_score)]

    # Encode 'score' column into grades A, B, C based on the calculated score ranges
    data['grades'] = pd.cut(data['score'], bins=bins, labels=['A', 'B', 'C'])

    # If necessary, you can drop non-numeric columns or encode them
    numeric_data = data.select_dtypes(include=[float, int])

    # Initialize UMAP with adjusted parameters (not optimized)
    reducer = umap.UMAP(metric='cosine',n_neighbors=5, min_dist=0.1)
    embedding = reducer.fit_transform(numeric_data)

    # Create a scatter plot of the UMAP embedding with color-coded grades
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=pd.factorize(data['grades'])[0], cmap='viridis', s=50, alpha=0.5)

    plt.title('UMAP Visualization of Data with Grades')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    cb = plt.colorbar(scatter)
    cb.set_ticks(np.arange(3))
    cb.set_ticklabels(['C', 'B', 'A'])

    legend_labels = ['0 - {:.2f}'.format(max_score/3), '{:.2f} - {:.2f}'.format(max_score/3, 2*max_score/3), '{:.2f} - {:.2f}'.format(2*max_score/3, max_score)]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Score Ranges', loc='upper right')
    plt.savefig(output_filename)
    plt.close()
