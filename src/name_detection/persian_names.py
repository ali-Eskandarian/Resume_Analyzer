import os
import re
import fitz
from hazm import Normalizer

# Define the path to the directory containing the data files
path = os.path.dirname(__file__)
normalizer = Normalizer()
files = ['male_fa.txt', 'female_fa.txt', 'male_en.txt', 'female_en.txt']


# Preload names from files into lists
def load_names(filename):
    with open(os.path.join(path, 'data', filename), 'r', encoding='utf8') as f:
        return f.read().splitlines()


# Load names from the corresponding files
male_names_fa = load_names(files[0])
male_names_fa = [normalizer.normalize(name) for name in male_names_fa]
female_names_fa = load_names(files[1])
female_names_fa = [normalizer.normalize(name) for name in female_names_fa]
male_names_en = load_names(files[2])
male_names_en = [normalizer.normalize(name) for name in male_names_en]
female_names_en = load_names(files[3])
female_names_en = [normalizer.normalize(name) for name in female_names_en]
names = female_names_en + male_names_en + male_names_fa + female_names_fa

# Define Persian prefixes and suffixes
some_prefixes_fa = [
    'میر', 'پیر', 'یار', 'آقا', 'ابو', 'پور', 'نور', 'نصر', 'زند', 'سید', "اس", "اح",
    'امیر', 'عزیز', 'صیاد', 'زاهد', 'شاه', 'نیک', 'حاج', 'حاجی', 'صوفی', 'وزین', "اف",
    'افضل', 'فاضل', 'شیخ', 'میرزا', 'استاد', 'خواجه', 'ملک', 'خان', 'بیگ', 'عرب', 'منش'
]
some_prefixes_fa = [normalizer.normalize(name) for name in some_prefixes_fa]

some_suffixes_fa = [
    'پور', 'زاده', 'فر', 'فرد', 'ان', 'کیا', 'راد', 'زند', 'خواه', 'نیا', "نی", "دی",
    'مهر', 'آذر', 'صدر', 'کهن', 'نژاد', 'بیات', 'یکتا', 'ثابت', 'آزاد', "کار","یان",
    'زارع', 'مقدم', 'روشن', 'تبار', 'راشد', 'دانا', 'زادگان', 'منش', 'یار', 'لو'
]
some_suffixes_fa = [normalizer.normalize(name) for name in some_suffixes_fa]


def is_valid_name(name1, name2):
    """
    Check if the provided names are valid Persian first or last names.

    :param name1: str - The first name to check.
    :param name2: str - The second name to check.
    :return: str or int - The normalized valid name if found, otherwise 0.
    """
    if normalizer.normalize(name1) in names:
        return normalizer.normalize(name1)
    elif normalizer.normalize(name2) in names:
        return normalizer.normalize(name2)
    else:
        return 0


def is_valid_lastname(name1, name2):
    """
    Check if the provided names are valid Persian last names.

    :param name1: str - The first name to check.
    :param name2: str - The second name to check.
    :return: str or int - The valid last name if found, otherwise 0.
    """
    name1, name2 = normalizer.normalize(name1), normalizer.normalize(name2)
    if any(name in name1 for name in names if len(name1) > len(name)):
        return name1
    elif any(name in name2 for name in names if len(name2) > len(name)):
        return name2
    elif any(name1.endswith(suffix) for suffix in some_suffixes_fa) or any(
            name1.startswith(prefixes) for prefixes in some_prefixes_fa) or name1:
        return name1
    elif any(name2.endswith(suffix) for suffix in some_suffixes_fa) or any(
            name2.startswith(prefixes) for prefixes in some_prefixes_fa):
        return name2
    else:
        return 0


def remove_english_and_numbers(text):
    """
    Remove all English letters and numbers from the input text.

    :param text: str - The input text from which English letters and numbers will be removed.
    :return: str - The cleaned text with English letters and numbers removed.
    """
    cleaned_text = re.sub(r'[A-Za-z0-9!@$_%]+', '', text)
    return cleaned_text


def extract_names(text):
    """
    Extract first and last names from the given text using defined prefixes and suffixes.

    :param text: str - The input text from which names will be extracted.
    :return: str - A string containing the recognized first and last names, or "Not Recognized" if no names are found.
    """
    text = remove_english_and_numbers(text)
    words = text.split()

    for i in range(len(words) - 1):

        first_name = words[i]
        last_name = words[i + 1]

        _first_name = is_valid_name(first_name, last_name)
        _last_name = is_valid_lastname(first_name, last_name)

        if _first_name != 0 and _last_name != 0:
            return f"{_first_name} {_last_name}"
    return "Not Recognized"


document = fitz.open('/home/alioto/PycharmProjects/Resume_Analyzer/resumes/علی_اسکندریان_Persian_Resume.pdf')
text1 = "".join(page.get_text() for page in document)

# print(extract_names(text1))
