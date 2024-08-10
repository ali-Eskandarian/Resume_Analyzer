import fitz  # PyMuPDF for PDF reading
import re
from hazm import Normalizer, WordTokenizer, stopwords_list, Lemmatizer
from skills import ds_keywords, web_keywords, android_keywords, ios_keywords, uiux_keywords, devops_keywords
from name_detection.persian_names import extract_names
from collections import Counter
class ResumeReader:
    def __init__(self, resume_path):
        self.resume_path = resume_path
        self.normalizer = Normalizer()
        self.lemmatizer = Lemmatizer()
        self.word_tokenizer = WordTokenizer()
        self.stopwords = set(stopwords_list())
        self.quality_mapping = {
            'کم': 1, 'مبتدی': 1, 'beginner': 1,
            'متوسط': 2, 'intermediate': 2,
            'زیاد': 3, 'پیشرفته': 3, 'advance': 3
        }
    def extract_text_from_pdf(self):
        """Extract text from a PDF file located at resume_path."""
        document = fitz.open(self.resume_path)
        text = "".join(page.get_text() for page in document)
        return text

    def get_data_raw(self):
        """Get raw data from the resume."""
        text = self.extract_text_from_pdf()
        return text

    def processed_data(self):
        """Get processed data from the resume."""
        text = self.extract_text_from_pdf()
        text = re.sub('http\S+\s*', ' ', text)  # remove URLs
        text = re.sub('RT|cc', ' ', text)  # remove RT and cc
        text = re.sub('#\S+', '', text)  # remove hashtags
        text = re.sub('@\S+', '  ', text)  # remove mentions
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
                            text)  # remove punctuations
        text = self.normalizer.normalize(text)
        text = self.lemmatizer.lemmatize(text)
        return text

    def extract_skills(self, text):
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
                quality = self.extract_quality(text, keyword)
                if quality:
                    skills_quality[keyword] = quality

        for keyword in web_keywords:
            if keyword.lower() in text:
                skills['Web Development'].append(keyword)
                quality = self.extract_quality(text, keyword)
                if quality:
                    skills_quality[keyword] = quality

        for keyword in android_keywords:
            if keyword.lower() in text:
                skills['Android Development'].append(keyword)
                quality = self.extract_quality(text, keyword)
                if quality:
                    skills_quality[keyword] = quality

        for keyword in ios_keywords:
            if keyword.lower() in text:
                skills['iOS Development'].append(keyword)
                quality = self.extract_quality(text, keyword)
                if quality:
                    skills_quality[keyword] = quality

        for keyword in uiux_keywords:
            if keyword.lower() in text:
                skills['UI/UX Design'].append(keyword)
                quality = self.extract_quality(text, keyword)
                if quality:
                    skills_quality[keyword] = quality

        for keyword in devops_keywords:
            if keyword.lower() in text:
                skills['DevOps'].append(keyword)
                quality = self.extract_quality(text, keyword)
                if quality:
                    skills_quality[keyword] = quality

        return skills, skills_quality

    def extract_quality(self, text, skill):
        """Extract the quality of a skill from the text."""
        # Define patterns to search for quality indicators
        quality_patterns = r'(?<=\b' + re.escape(
            skill) + r'\b).*?(\d|کم|مبتدی|متوسط|زیاد|پیشرفته|beginner|intermediate|advance)'

        match = re.search(quality_patterns, text)
        if match:
            quality_str = match.group(0).strip()
            # Check if the quality is numeric or a keyword
            for key, value in self.quality_mapping.items():
                if key in quality_str:
                    return value
            if quality_str.isdigit():
                return int(quality_str)

        return None  # Return None if no quality found

    def extract_contact_info(self, text):
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

    def extract_age(self, text):
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

    def full_features(self):
        # Extract skills from resume
        resume_data = self.get_data_raw()
        resume_skills, skills_quality = self.extract_skills(resume_data)

        # Extract contact information from resume
        resume_contact_info = self.extract_contact_info(resume_data)

        # Extract age from resume
        resume_age = self.extract_age(resume_data)

        return resume_skills, skills_quality, resume_contact_info, resume_age



