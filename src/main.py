# Import necessary classes
from resume_reader import ResumeReader
from similarity_calculator import SimilarityCalculator

# Path to the resume PDF
resume_path = '../resumes/2.pdf'

# Job description string
job_description = "توضیحات شغلی به زبان فارسی"

# Extract information from the resume
resume_reader = ResumeReader(resume_path)
resume_data = resume_reader.get_data()

# Calculate similarity
similarity_calculator = SimilarityCalculator(resume_data, job_description)
cosine_sim = similarity_calculator.calculate_cosine_similarity()
jaccard_sim = similarity_calculator.calculate_jaccard_similarity()

print(f"Cosine Similarity: {cosine_sim}")
print(f"Jaccard Similarity: {jaccard_sim}")
