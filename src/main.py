from hazm import Normalizer, Lemmatizer
from models import SimilarityCalculator, KeywordExtractor
from reader import ResumeReader


def main():
    resume_path = '../resumes/نرجس_محسنی پور_Persian_Resume.pdf'
    job_description_path = "../description_position.txt"


    # Read job description
    with open(job_description_path, 'r', encoding='utf-8') as f:
        job_description = f.read()

    # Extract information from the resume
    resume_reader = ResumeReader(resume_path)
    resume_skills, skills_quality, resume_contact_info, resume_age = resume_reader.full_features()
    resume_data_processed = resume_reader.processed_data()

    # Extract keywords from resume
    resume_keyword_extractor = KeywordExtractor(resume_data_processed)
    resume_keywords = resume_keyword_extractor.extract_keywords(100)
    added_resume_keyword = resume_keywords+list(resume_skills.values())

    # Extract keywords from job description
    job_keyword_extractor = KeywordExtractor(Lemmatizer().lemmatize(Normalizer().normalize(job_description)))
    job_description_keywords = job_keyword_extractor.extract_keywords(len(added_resume_keyword))

    # Calculate similarity
    similarity_calculator = SimilarityCalculator(added_resume_keyword, job_description_keywords)
    cosine_sim = similarity_calculator.calculate_cosine_similarity()
    jaccard_sim = similarity_calculator.calculate_jaccard_similarity()

    print(f"Resume Keywords: {resume_keywords}")
    print(f"Job Description Keywords: {job_description_keywords}")
    print(f"Resume Skills: {resume_skills}")
    print(f"Resume Skills Quality: {skills_quality}")
    print(f"Resume Contact Info: {resume_contact_info}")
    print(f"Resume Age: {resume_age}")
    print(f"Cosine Similarity: {cosine_sim}")
    print(f"Jaccard Similarity: {jaccard_sim}")

if __name__ == "__main__":
    main()
