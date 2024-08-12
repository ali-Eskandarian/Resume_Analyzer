from sklearn.metrics.pairwise import cosine_similarity
from utils import flatten_list
from hazm import Normalizer, WordTokenizer, stopwords_list
from collections import Counter
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.externals import joblib
import numpy as np
import joblib


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


class ClusterModel:
    def __init__(self, min_cluster_size=5):
        self.min_cluster_size = min_cluster_size
        self.models = {
            'KMeans': KMeans(),
            'HDBSCAN': HDBSCAN(min_cluster_size=min_cluster_size)
        }
        self.clustered_data = {}
        self.data_model = None

    def forward(self, data):
        """Fit the clustering models to the provided data."""
        self.data_model = data.drop('resume', axis=1).values

        for model_name, model in self.models.items():
            labels_ = model.fit_predict(self.data_model)
            data[f'{model_name}'] = labels_
            self.clustered_data[model_name] = labels_

        # Save the updated DataFrame
        data.to_csv('clustered_data.csv', index=False)
        return data

    def save(self, directory):
        """Save all fitted models to the specified directory."""
        for model_name, model in self.models.items():
            joblib.dump(model, f"{directory}/{model_name}_model.pkl")

    def predict(self, new_data):
        """Predict the cluster for a new data point using all models."""
        predictions = {}
        for model_name, model in self.models.items():
            if model_name == 'KMeans':
                predictions[model_name] = model.predict(new_data.reshape(1, -1))[0]
            else:
                predictions[model_name] = model.predict([new_data])[0]
        return predictions