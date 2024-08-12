from hazm import *
import fitz
import os


def concatenate_pdfs_and_description(directory):
    """Concatenate text from all PDFs and a description file into a single text file."""
    full_text = ""

    # Read all PDF files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            document = fitz.open(pdf_path)
            text = "".join(page.get_text() for page in document)
            full_text += text + "\n"  # Add a newline after each PDF's text
            document.close()  # Close the document

    # Read the description_position.txt file
    description_path = os.path.join(directory, '../description_position.txt')
    if os.path.exists(description_path):
        with open(description_path, 'r', encoding='utf-8') as file:
            description_text = file.read()
        full_text += description_text + "\n"  # Add a newline after the description

    # Save the combined text to full_text.txt
    output_path = '../full_text.txt'
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(full_text)

    print(f"Full text saved to {output_path}")

directory_path = '../resumes'  # Replace with your directory path
concatenate_pdfs_and_description(directory_path)
wordEmbedding = WordEmbedding(model_type = 'fasttext')
wordEmbedding.train(dataset_path =  '../full_text.txt' , workers = 4, vector_size = 16, epochs = 50000,
                    min_count = 1, fasttext_type = 'cbow', dest_path = '../saved_model/word2vec_model.bin')
