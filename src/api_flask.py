import os
from flask import Flask, request, jsonify
from reader import SingleResumeReader, ClusteringResumeReader
from models import ClusterModel
from utils import visualize_data_with_umap

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Resume Analyzer API! Use the /analyze endpoint to submit your data.'})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json

    # Get job description path
    job_description_path = data.get('job_description')

    # Check if a single PDF or a directory is provided
    pdf_path = data.get('pdf_path')  # Path for a single PDF if given
    resume_dir = data.get('resume_folder')  # Directory for resumes if given

    if pdf_path:
        # Process a single PDF
        if not os.path.isfile(pdf_path):
            return jsonify({'error': 'The specified PDF file does not exist.'}), 400

        resume_analyzer = SingleResumeReader(job_description_path, pdf_path)
        features_json = resume_analyzer.get_resume_features_as_json()
        return jsonify({'type': 'single_pdf', 'features': features_json})

    elif resume_dir:
        # Process resumes in a directory
        if not os.path.isdir(resume_dir):
            return jsonify({'error': 'The specified directory does not exist.'}), 400

        dataset_cluster = ClusteringResumeReader(job_description_path, resume_dir)
        data = dataset_cluster.process_resumes()

        # Visualize data with UMAP
        visualize_data_with_umap(data)

        # Ask if wants to train
        if data.get('train_model', False):
            cluster_model = ClusterModel()
            kmeans_scores, hdbscan_scores, best_kmeans_rows, best_hdbscan_rows = cluster_model.fit_predict(data)

            # Ask for output format
            output_format = data.get('output_format', 'json')  # Default to JSON
            if output_format == 'csv':
                best_kmeans_rows.to_csv('best_kmeans_rows.csv', index=False)
                return jsonify({'message': 'Best KMeans rows saved to CSV.'})
            else:
                return jsonify(best_kmeans_rows.to_json())

            # Save model
            cluster_model.save('../saved_model')
            return jsonify({'message': 'Model saved and training completed.'})

        return jsonify({'message': 'No training performed.'})

    else:
        return jsonify({'error': 'No valid PDF or directory provided.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
