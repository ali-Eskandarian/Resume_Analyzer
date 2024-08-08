# from PyPDF2 import PdfReader
# from bidi.algorithm import get_display
#
# # reader = PdfReader("../resumes/3.pdf")
#
# def extract_headings_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     headings = []
#
#     for page in reader.pages:
#         text = page.extract_text()
#         for line in text.split('\n'):
#             if line.isupper():  # Assuming headings are in uppercase
#                 headings.append(line)
#
#     return headings
#
#
# pdf_path = '../resumes/3.pdf'
# headings = extract_headings_from_pdf(pdf_path)
#
# # print(get_display(page.extract_text()))
# print(headings)
# print(type(page.extract_text()))

import fitz  # PyMuPDF

path = '../resumes/2.pdf'


def extract_headings_and_content(file_path: str) -> dict:
    """
    Extract headings and their content from a PDF file and return them as a dictionary.

    :param file_path: Path to the PDF file.
    :return: Dictionary with headings as keys and their content as values.
    """
    headings_content = {}
    current_heading = None

    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                # Extract text blocks from the page
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if not text:
                                    continue

                                # Identify headings by larger font size
                                if span["size"] > 12:
                                    current_heading = text
                                    if current_heading not in headings_content:
                                        headings_content[current_heading] = []
                                elif current_heading:
                                    headings_content[current_heading].append(text)

    except Exception as e:
        print(f"An error occurred: {e}")

    # Convert lists of strings into single strings for each heading
    for heading in headings_content:
        headings_content[heading] = ' '.join(headings_content[heading])

    return headings_content


headings_dict = extract_headings_and_content(path)
print(headings_dict)

import re
from utils import read_pdf, preprocess_text


class ResumeReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text = read_pdf(file_path)
        self.data = self.extract_information()

    def extract_information(self) -> dict:
        """
        Extract key information from the resume.

        :return: Dictionary with extracted information.
        """
        data = {
            "name": self.extract_name(),
            "email": self.extract_email(),
            "location": self.extract_location(),
            "skills": self.extract_skills(),
            # Add more fields as needed
        }
        return data

    def extract_name(self) -> str:
        """
        Extract the name from the resume.

        :return: Extracted name.
        """
        # Assuming the name is the first uppercase text in the resume
        match = re.search(r'\b[آ-ی]+\b', self.text)
        return match.group(0) if match else ""

    def extract_email(self) -> str:
        """
        Extract the email from the resume.

        :return: Extracted email.
        """
        match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', self.text)
        return match.group(0) if match else ""

    def extract_location(self) -> str:
        """
        Extract the location from the resume.

        :return: Extracted location.
        """
        # Add logic to extract location based on known patterns or keywords
        return ""

    def extract_skills(self) -> dict:
        """
        Extract skills and their levels from the resume.

        :return: Dictionary of skills and their levels.
        """
        skills = {}
        # Assuming skills are listed in a specific section
        skill_section = re.search(r'(?<=مهارتها)(.*?)(?=\n\n|\Z)', self.text, re.DOTALL)
        if skill_section:
            skill_lines = skill_section.group(0).split('\n')
            for line in skill_lines:
                parts = line.split(':')
                if len(parts) == 2:
                    skill, level = parts
                    skills[skill.strip()] = level.strip()
                else:
                    skills[line.strip()] = "Unknown"
        return skills

    def get_data(self) -> dict:
        """
        Get the extracted data.

        :return: Extracted data dictionary.
        """
        return self.data
