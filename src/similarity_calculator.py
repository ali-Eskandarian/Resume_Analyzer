from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess_text


class SimilarityCalculator:
    def __init__(self, resume_data: dict, job_description: str):
        self.resume_data = resume_data
        self.job_description = job_description

    def calculate_cosine_similarity(self) -> float:
        """
        Calculate the cosine similarity between the resume and job description.

        :return: Cosine similarity score.
        """
        resume_text = ' '.join([value for key, value in self.resume_data.items() if isinstance(value, str)])
        job_text = preprocess_text(self.job_description)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])

        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]

    def calculate_jaccard_similarity(self) -> float:
        """
        Calculate the Jaccard similarity between the resume and job description.

        :return: Jaccard similarity score.
        """
        resume_text = ' '.join([value for key, value in self.resume_data.items() if isinstance(value, str)])
        job_text = preprocess_text(self.job_description)

        resume_set = set(resume_text.split())
        job_set = set(job_text.split())

        intersection = resume_set.intersection(job_set)
        union = resume_set.union(job_set)
        return len(intersection) / len(union) if union else 0
