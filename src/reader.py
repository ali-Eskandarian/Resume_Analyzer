import json
import fitz
import re
from hazm import Normalizer, WordTokenizer, stopwords_list, Lemmatizer
from utils import extract_age, extract_contact_info, extract_skills, flatten_list
import os
from models import SimilarityCalculator, KeywordExtractor
import pandas as pd
from tqdm import tqdm


class ResumeReader:
    def __init__(self, resume_path):
        """
        Initialize the ResumeReader with the path to the resume.

        :param resume_path: str - The file path to the resume PDF.
        """
        self.resume_path    = resume_path
        self.normalizer     = Normalizer()
        self.lemmatizer     = Lemmatizer()
        self.word_tokenizer = WordTokenizer()
        self.stopwords      = set(stopwords_list())

    def extract_text_from_pdf(self):
        """
        Extract text from a PDF file located at resume_path.

        :return: str - The extracted text from the PDF.
        """
        document = fitz.open(self.resume_path)
        text = "".join(page.get_text() for page in document)
        return text

    def get_data_raw(self):
        """
        Get raw data from the resume.

        :return: str - The raw text data extracted from the resume.
        """
        text = self.extract_text_from_pdf()
        return text

    def processed_data(self):
        """
        Get processed data from the resume.

        :return: str - The cleaned and processed text data from the resume.
        """
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

    def full_features(self):
        """
        Extract full features from the resume, including skills, contact information, and age.

        :return: tuple - A tuple containing:
            - resume_skills (list): A list of skills extracted from the resume.
            - skills_quality (list): A list indicating the quality of the extracted skills.
            - resume_contact_info (dict): A dictionary containing contact information extracted from the resume.
            - resume_age (int): The age extracted from the resume.
        """
        # Extract skills from resume
        resume_data = self.get_data_raw()
        resume_skills, skills_quality = extract_skills(resume_data)

        # Extract contact information from resume
        resume_contact_info = extract_contact_info(resume_data)

        # Extract age from resume
        resume_age = extract_age(resume_data)

        return resume_skills, skills_quality, resume_contact_info, resume_age


class ClusteringResumeReader(ResumeReader):
    def __init__(self, job_description_path, resumes_directory, nums=100):
        """
        Initialize the ClusteringResumeReader with the job description path, resumes directory, and the number of keywords.

        :param job_description_path: str - The file path to the job description.
        :param resumes_directory: str - The directory path containing the resumes.
        :param nums: int - The number of keywords to process (default is 100).
        """
        self.job_description_path = job_description_path
        self.resumes_directory = resumes_directory
        self.nums = nums
        super().__init__(None)  # Call parent constructor with no resume path

    def read_job_description(self):
        """
        Read the job description from the specified path.

        :return: str - The job description text.
        """
        with open(self.job_description_path, 'r', encoding='utf-8') as f:
            job_description = f.read()
        return job_description

    def load_resumes(self):
        """
        Load all resumes from the specified directory and return as ResumeReader objects.

        :return: list - A list of ResumeReader objects, each representing a resume.
        """
        resumes = []
        for filename in os.listdir(self.resumes_directory):
            if filename.endswith('.pdf'):
                resume_ = os.path.join(self.resumes_directory, filename)
                resumes.append(resume_)
        return resumes

    def process_resumes(self):
        """
        Process each resume to create the final data for clustering.

        :return: df - A pandas dataframe.
        """
        resumes = self.load_resumes()
        data = []

        # Extract keywords from job description
        job_description = self.read_job_description()
        job_keyword_extractor = KeywordExtractor(Lemmatizer().lemmatize(Normalizer().normalize(job_description)))
        skills_job, _ = extract_skills(job_description)
        skills_job = set(flatten_list(list(skills_job.values())))

        for resume_path in tqdm(resumes):
            resume_reader = ResumeReader(resume_path)
            resume_skills, skills_quality, _, _ = resume_reader.full_features()
            resume_skills = flatten_list(list(resume_skills.values()))
            resume_data_processed = resume_reader.processed_data()

            # Extract keywords from resume
            resume_keyword_extractor = KeywordExtractor(resume_data_processed)
            resume_keywords = resume_keyword_extractor.extract_keywords(self.nums)
            added_resume_keyword = resume_keywords + resume_skills
            job_description_keywords = job_keyword_extractor.extract_keywords(len(added_resume_keyword))

            # Calculate similarity
            similarity_calculator = SimilarityCalculator(added_resume_keyword, job_description_keywords)
            cosine_sim = similarity_calculator.calculate_cosine_similarity()
            jaccard_sim = similarity_calculator.calculate_jaccard_similarity()

            # Prepare the row for the dataset
            row = [os.path.splitext(os.path.basename(resume_path))[0], cosine_sim, jaccard_sim]

            score = cosine_sim*60+jaccard_sim*10
            # Add skills presence as additional columns
            for skill in skills_job:
                val = 1
                if skill in resume_skills:
                    if skill in skills_quality.keys():
                        val = skills_quality[skill]
                        score += val * 15 / len(skills_job)
                    else:
                        score += 15 / len(skills_job)

                row.append(val if skill in resume_skills else 0)
            row.append(score)
            data.append(row)


        # Create a DataFrame with the specified columns
        columns = ['resume', 'cosine_sim', 'jaccard_sim'] + list(skills_job) + ['score']
        df = pd.DataFrame(data, columns=columns)

        return df


class SingleResumeReader:
    def __init__(self, job_description_path, resume_path, nums=100):
        """
        Initialize the SingleResumeReader with the job description path, resume directory, and the number of keywords.

        :param job_description_path: str - The file path to the job description.
        :param resume_path: str - The directory path containing the resumes.
        :param nums: int - The number of keywords to process (default is 100).
        """
        self.job_description_path = job_description_path
        self.resume_path = resume_path
        self.nums = nums
        self.resume_reader = ResumeReader(resume_path)

    def read_job_description(self):
        """
        Read the job description from the specified path.

        :return: str - The job description text.
        """
        with open(self.job_description_path, 'r', encoding='utf-8') as f:
            job_description = f.read()
        return job_description

    def get_resume_features_as_json(self):
        """
        Get features of the resume and similarities as a JSON object.

        :return: json file - A Json config.
        """
        resume_skills, skills_quality, resume_contact_info, resume_age = self.resume_reader.full_features()
        resume_skills = flatten_list(list(resume_skills.values()))
        resume_data_processed = self.resume_reader.processed_data()

        # Extract keywords from job description
        job_description = self.read_job_description()
        job_keyword_extractor = KeywordExtractor(Lemmatizer().lemmatize(Normalizer().normalize(job_description)))
        skills_job, _ = extract_skills(job_description)
        skills_job = flatten_list(list(skills_job.values()))

        # Extract keywords from resume
        resume_keyword_extractor = KeywordExtractor(resume_data_processed)
        resume_keywords = resume_keyword_extractor.extract_keywords(self.nums)
        added_resume_keyword = resume_keywords + resume_skills
        job_description_keywords = job_keyword_extractor.extract_keywords(len(added_resume_keyword))

        # Calculate similarity
        similarity_calculator = SimilarityCalculator(added_resume_keyword, job_description_keywords)
        cosine_sim = similarity_calculator.calculate_cosine_similarity()
        jaccard_sim = similarity_calculator.calculate_jaccard_similarity()

        # Calculate score
        score = cosine_sim*90+jaccard_sim*10
        for skill in skills_job:
            if skill in resume_skills:
                if skill in skills_quality.keys():
                    score += skills_quality[skill] * 15 / len(skills_job)
                else:
                    score += 15/len(skills_job)

        resume_data = {
            'skills': resume_skills,
            'skills_quality': skills_quality,
            'contact_info': resume_contact_info,
            'age': resume_age,
            'cosine_similarity': float(cosine_sim),
            'jaccard_similarity': jaccard_sim,
            'final_score(%)': 100 if score >= 100 else score
        }
        return json.dumps(resume_data, ensure_ascii=False)