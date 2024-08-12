from sklearn.metrics.pairwise import cosine_similarity
from hazm import Normalizer, WordTokenizer, stopwords_list, WordEmbedding
from collections import Counter
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
import numpy as np
import joblib


class SimilarityCalculator:
    def __init__(self, resume_keywords, job_description_keywords, embedding_method='model'):
        self.resume_keywords = resume_keywords
        self.job_description_keywords = job_description_keywords
        if embedding_method=='model':
            self.word_embedding = WordEmbedding(model_type='fasttext')
            self.word_embedding.load_model('../saved_model/word2vec_model.bin')
            self.embedding_method = embedding_method
    def vectorize_keywords(self, keywords):
        """Get the word vectors for each keyword in a 1xn shape."""
        vectors = []
        for keyword in keywords:
            try:
                vector = self.word_embedding.get_normal_vector(keyword)  # Get the normalized vector for the keyword
                vector = np.expand_dims(vector, axis=0)  # Extend the vector to a 1xn shape
                vectors.append(vector)
            except KeyError:
                continue  # Skip keywords not in the vocabulary
        return vectors

    def calculate_cosine_similarity(self):
        """Calculate cosine similarity between resume and job description keywords."""

        if self.embedding_method == 'model':
            resume_vector = self.vectorize_keywords(self.resume_keywords)
            job_desc_vector = self.vectorize_keywords(self.job_description_keywords)

            resume_vector_flat = np.concatenate([arr.flatten() for arr in resume_vector]).reshape(1, -1)
            job_desc_vector_flat = np.concatenate([arr.flatten() for arr in job_desc_vector]).reshape(1, -1)

            # Replace NaN values with 0
            resume_vector_flat = np.nan_to_num(resume_vector_flat, nan=0.00001)
            job_desc_vector_flat = np.nan_to_num(job_desc_vector_flat, nan=0.00001)

            # Calculate cosine similarity
            similarity = cosine_similarity(resume_vector_flat,job_desc_vector_flat)
            return similarity[0][0]
        else:
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
    def __init__(self, min_cluster_size=4):
        self.min_cluster_size = min_cluster_size
        self.models = {
            'KMeans': KMeans(self.min_cluster_size),
            'HDBSCAN': HDBSCAN(min_cluster_size=min_cluster_size)
        }
        self.clustered_data = {}
        self.data_model = None

    def forward(self, data):
        """Fit the clustering models to the provided data."""
        # self.data_model = data.drop('resume', axis=1).values
        self.data_model = data[['cosine_sim', 'jaccard_sim','score']]

        for model_name, model in self.models.items():
            labels_ = model.fit_predict(self.data_model)
            data[f'{model_name}'] = labels_
            self.clustered_data[model_name] = labels_

        # Save the updated DataFrame
        data.to_csv('clustered_data.csv', index=False)
        return data

    def fit_predict(self, data):
        final_data = self.forward(data)
        kmeans_scores,    hdbscan_scores    = {}, {}

        # Calculate average similarity scores and best rows for KMeans clusters
        for cluster in final_data['KMeans'].unique():
            cluster_data = final_data[final_data['KMeans'] == cluster]
            avg_score = cluster_data['score'].mean()
            kmeans_scores[cluster] = avg_score

        # Find the rows with the highest overall similarity score in the cluster
        best_kmeans_rows = final_data[final_data['KMeans']==max(kmeans_scores, key=kmeans_scores.get)]

        # Calculate average similarity scores and best rows for HDBSCAN clusters
        for cluster in final_data['HDBSCAN'].unique():
            cluster_data = final_data[final_data['HDBSCAN'] == cluster]
            avg_score = cluster_data['score'].mean()
            hdbscan_scores[cluster] = avg_score

        best_hdbscan_rows = final_data[final_data['KMeans']==max(hdbscan_scores, key=hdbscan_scores.get)]

        return kmeans_scores, hdbscan_scores, best_kmeans_rows, best_hdbscan_rows

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