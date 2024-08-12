import pandas as pd

from reader import SingleResumeReader, ClusteringResumeReader
from models import ClusterModel
from flask import Flask, request, jsonify
from utils import visualize_data_with_umap

# app = Flask(__name__)

# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'pdf'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/main', methods=['POST'])
def main():
    # if 'directory' not in request.form:
        # return jsonify({'error': 'No directory specified'}), 400

    # directory_path = request.form['directory']
    
    resume_path = '../resumes/نرجس_محسنی پور_Persian_Resume.pdf'
    resume_dir = '../resumes/'
    job_description_path = "../description_position.txt"

    resume_analyzer = SingleResumeReader(job_description_path, resume_path)
    print(resume_analyzer.get_resume_features_as_json())
    #
    dataset_cluster = ClusteringResumeReader(job_description_path, resume_dir)
    data = dataset_cluster.process_resumes()
    data.to_csv("output_1.csv")

    data = pd.read_csv("output_1.csv")
    visualize_data_with_umap(data)

    # Initialize the clustering model
    cluster_model = ClusterModel()

    # Fit the model with the data
    # cluster_model.forward(data)
    kmeans_scores, hdbscan_scores, best_kmeans_rows, best_hdbscan_rows = cluster_model.fit_predict(data)
    # Save the models
    cluster_model.save('../saved_model')

    print(1)


if __name__ == "__main__":
    main()
