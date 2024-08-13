from sklearn.metrics.pairwise import cosine_similarity
from hazm import *
from collections import Counter
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
import numpy as np
import joblib
from utils  import extract_candidates, equalize_keyword_lengths

class SimilarityCalculator:
    def __init__(self, resume_keywords, job_description_keywords, embedding_method='model'):
        """
        Initialize the SimilarityCalculator with the resume keywords, job description keywords, and embedding method.

        :param resume_keywords: list - A list of keywords extracted from the resume.
        :param job_description_keywords: list - A list of keywords extracted from the job description.
        :param embedding_method: str - The method to use for word embedding (default is 'model').
        """
        self.resume_keywords, self.job_description_keywords = equalize_keyword_lengths(
            resume_keywords,
            job_description_keywords
        )
        if embedding_method=='model':
            self.word_embedding = WordEmbedding(model_type='fasttext')
            self.word_embedding.load_model('../saved_model/word2vec_model.bin')
            self.embedding_method = embedding_method
    def vectorize_keywords(self, keywords):
        """
        Get the word vectors for each keyword in a 1xn shape

        :param keywords: string word
        :return: vectors: array embedding
        """
        vectors = []
        for keyword in keywords:
            try:
                vector = self.word_embedding[keyword]  # Get the normalized vector for the keyword
                vector = np.expand_dims(vector, axis=0)  # Extend the vector to a 1xn shape
                vectors.append(vector)
            except KeyError:
                continue  # Skip keywords not in the vocabulary
        return vectors

    def calculate_cosine_similarity(self):
        """
        Calculate cosine similarity between resume and job description keywords

        :return: similarity: float
        """

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
        """
        Calculate Jaccard similarity between resume and job description keywords

        :return: similarity: float
        """
        intersection = len(set(self.resume_keywords) & set(self.job_description_keywords))
        union = len(set(self.resume_keywords) | set(self.job_description_keywords))
        return intersection / union if union != 0 else 0


class KeywordExtractor:
    def __init__(self, text, embedding_method='model'):
        """
        Initialize the KeywordExtractor with the input text.

        :param text: str - The text from which keywords will be extracted.
        :param embedding_method: str - The method to use for word embedding (default is 'model').
        """
        self.text           = text
        self.normalizer     = Normalizer()
        self.word_tokenizer = WordTokenizer()
        self.stopwords      = set(stopwords_list())
        if embedding_method=='model':
            self.word_embedding   = WordEmbedding(model_type='fasttext')
            self.word_embedding.load_model('../saved_model/word2vec_model.bin')
            self.embedding_method = embedding_method
            self.tagger           =  POSTagger(model='../saved_model/pos_tagger.model')
    def extract_keywords(self, nums):
        """
        Extract keywords from the text using frequency analysis

        :param nums: number nums
        :return: keywords: list keywords
        """
        words = self.word_tokenizer.tokenize(self.text)
        if self.embedding_method != 'model':
            words_vectors = [word for word in words if word not in self.stopwords]
            word_counts   = Counter(words)
            keywords      = [word for word, _ in word_counts.most_common(nums)]  # Get top 150 keywords
            return keywords
        else:
            try:
                return self.text_rank_nlp(nums)
            except:
                words_vectors = [word for word in words if word not in self.stopwords]
                word_counts = Counter(words)
                keywords = [word for word, _ in word_counts.most_common(nums)]  # Get top 150 keywords
                return keywords


    def text_rank_nlp(self, keyword_count):
        grammers = [
            """
            NP:
                    {<NOUN,EZ>?<NOUN.*>}    # Noun(s) + Noun(optional)
            """,
            """
            NP:
                    {<NOUN.*><ADJ.*>?}    # Noun(s) + Adjective(optional)
            """
        ]
        tokenize_text = [word_tokenize(txt) for txt in sent_tokenize(self.text)]
        token_tag_list = self.tagger.tag_sents(tokenize_text)
        all_candidates = set()
        for grammer in grammers:
            all_candidates.update(extract_candidates(token_tag_list, grammer))
        all_candidates = list(all_candidates)
        all_candidates_vectors = [self.word_embedding[candidate] for candidate in all_candidates]
        candidates_concatinate = ' '.join(all_candidates)
        whole_text_vector = self.word_embedding[candidates_concatinate]
        candidates_sim_whole = cosine_similarity(all_candidates_vectors, whole_text_vector.reshape(1, -1))
        candidate_sim_candidate = cosine_similarity(all_candidates_vectors)
        candidates_sim_whole_norm = candidates_sim_whole / np.max(candidates_sim_whole)
        candidates_sim_whole_norm = 0.5 + (
                candidates_sim_whole_norm - np.average(candidates_sim_whole_norm)) / np.std(
            candidates_sim_whole_norm)
        np.fill_diagonal(candidate_sim_candidate, np.NaN)
        candidate_sim_candidate_norm = candidate_sim_candidate / np.nanmax(candidate_sim_candidate, axis=0)
        candidate_sim_candidate_norm = 0.5 + (
                candidate_sim_candidate_norm - np.nanmean(candidate_sim_candidate_norm, axis=0)) / np.nanstd(
            candidate_sim_candidate_norm, axis=0)
        beta = 0.82
        N = min(len(all_candidates), keyword_count)
        selected_candidates = []
        unselected_candidates = [i for i in range(len(all_candidates))]
        best_candidate = np.argmax(candidates_sim_whole_norm)
        selected_candidates.append(best_candidate)
        unselected_candidates.remove(best_candidate)

        for i in range(N - 1):
            selected_vec = np.array(selected_candidates)
            unselected_vec = np.array(unselected_candidates)

            unselected_candidate_sim_whole_norm = candidates_sim_whole_norm[unselected_vec, :]

            dist_between = candidate_sim_candidate_norm[unselected_vec][:, selected_vec]

            if dist_between.ndim == 1:
                dist_between = dist_between[:, np.newaxis]

            best_candidate = np.argmax(
                beta * unselected_candidate_sim_whole_norm - (1 - beta) * np.max(dist_between, axis=1).reshape(-1,
                                                                                                               1))
            best_index = unselected_candidates[best_candidate]
            selected_candidates.append(best_index)
            unselected_candidates.remove(best_index)
        return [all_candidates[i] for i in selected_candidates]



class ClusterModel:
    def __init__(self, min_cluster_size=4):
        """
        Initialize the ClusterModel with the minimum cluster size.

        :param min_cluster_size: int - The minimum number of data points required to form a cluster (default is 4).
        """
        self.min_cluster_size = min_cluster_size
        self.models = {
            'KMeans': KMeans(self.min_cluster_size),
            'HDBSCAN': HDBSCAN(min_cluster_size=min_cluster_size)
        }
        self.clustered_data = {}
        self.data_model = None

    def forward(self, data):
        """
        Fit the clustering models to the provided data

        :param data: pandas dataframe
        :return: data: pandas dataframe add clusters
        """

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
        """
        Fit the clustering models to the provided data and get the best clusters.

        :param data: pandas DataFrame containing the data to be clustered
        :return: A tuple containing:
            - kmeans_scores (dict): A dictionary with cluster labels as keys and their average similarity scores as values.
            - hdbscan_scores (dict): A dictionary with cluster labels as keys and their average similarity scores as values.
            - best_kmeans_rows (DataFrame): A DataFrame containing the rows with the highest overall similarity score in the best KMeans cluster.
            - best_hdbscan_rows (DataFrame): A DataFrame containing the rows with the highest overall similarity score in the best HDBSCAN cluster.
        """
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
        """
        Save all fitted models to the specified directory

        :param directory: saving directory
        """
        for model_name, model in self.models.items():
            joblib.dump(model, f"{directory}/{model_name}_model.pkl")

    def predict(self, new_data):
        """
        Predict the cluster for a new data point using all models

        :param new_data: pandas dataframe data
        :return predictions: dictionary of predicted cluster
        """
        predictions = {}
        for model_name, model in self.models.items():
            if model_name == 'KMeans':
                predictions[model_name] = model.predict(new_data.reshape(1, -1))[0]
            else:
                predictions[model_name] = model.predict([new_data])[0]
        return predictions