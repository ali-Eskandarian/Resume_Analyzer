import pandas as pd
from  Embedding_nlp import concatenate_pdfs_and_description
from hazm import *
from reader import SingleResumeReader, ClusteringResumeReader
from models import ClusterModel
from utils import visualize_data_with_umap

def main(train=False):
    resume_path = '../resumes/علی_اسکندریان_Persian_Resume.pdf'
    resume_dir = '../resumes/'
    job_description_path = "../description_position.txt"

    if train:
        concatenate_pdfs_and_description(resume_dir)
        word_embedding = WordEmbedding(model_type='fasttext')
        word_embedding.train(dataset_path='../full_text.txt', workers=4, vector_size=32, epochs=100000,
                            min_count=1, fasttext_type='cbow', dest_path='../saved_model/word2vec_model.bin')

    resume_analyzer = SingleResumeReader(job_description_path, resume_path)
    print(resume_analyzer.get_resume_features_as_json())
    #
    dataset_cluster = ClusteringResumeReader(job_description_path, resume_dir)
    data = dataset_cluster.process_resumes()
    data.to_csv("output_1.csv")

    data = pd.read_csv("output_1.csv")
    visualize_data_with_umap(data, output_filename="../umap_visualization.png")

    # Initialize the clustering model
    cluster_model = ClusterModel()

    kmeans_scores, hdbscan_scores, best_kmeans_rows, best_hdbscan_rows = cluster_model.fit_predict(data)
    # Save the models
    cluster_model.save('../saved_model')

    print("DONE")


if __name__ == "__main__":
    # main(train=True)
    main(train=False)
