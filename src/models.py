from sklearn.metrics.pairwise import cosine_similarity
from utils import flatten_list
from hazm import Normalizer, WordTokenizer, stopwords_list, Lemmatizer
from collections import Counter


class SimilarityCalculator:
    def __init__(self, resume_keywords, job_description_keywords):
        self.resume_keywords = flatten_list(resume_keywords)
        self.job_description_keywords = flatten_list(job_description_keywords)

    def calculate_cosine_similarity(self):
        """Calculate cosine similarity between resume and job description keywords."""
        # all_keywords = list(set(self.resume_keywords + self.job_description_keywords))
        all_keywords = list(set(self.resume_keywords) | set(self.job_description_keywords))
        resume_vector = [1 if keyword in self.resume_keywords else 0 for keyword in all_keywords]
        job_desc_vector = [1 if keyword in self.job_description_keywords else 0 for keyword in all_keywords]
        return cosine_similarity([resume_vector], [job_desc_vector])[0][0]

    def calculate_jaccard_similarity(self):
        """Calculate Jaccard similarity between resume and job description keywords."""
        intersection = len(set(self.resume_keywords) & set(self.job_description_keywords))
        union = len(set(self.resume_keywords) | set(self.job_description_keywords))
        return intersection / union if union != 0 else 0


class KeywordExtractor:
    def __init__(self, text):
        self.text = text
        self.normalizer = Normalizer()
        self.word_tokenizer = WordTokenizer()
        self.stopwords = set(stopwords_list())

    def extract_keywords(self, nums):
        """Extract keywords from the text using frequency analysis."""
        words = self.word_tokenizer.tokenize(self.text)
        words = [word for word in words if word not in self.stopwords]
        word_counts = Counter(words)
        keywords = [word for word, _ in word_counts.most_common(nums)]  # Get top 150 keywords
        return keywords




