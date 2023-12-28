from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

# Download stopwords
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load dataset
data = pd.read_csv('./test.csv', delimiter='|')

# Fungsi preprocessing dengan menggunakan stopwords
def preprocess_text(text):
    stop_words_indonesian = set(stopwords.words('indonesian'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words_indonesian]
    return ' '.join(words)

# Preprocessing teks pekerjaan
data['cleaned_text'] = data['job_description'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['cleaned_text'])

# Hitung cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi pencarian pekerjaan
def search_jobs(query, cosine_sim, data):
    # Preprocess query
    query = preprocess_text(query)

    # Transform query menjadi vektor TF-IDF
    query_vector = vectorizer.transform([query])

    # Hitung similarity antara query dan dokumen
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Ambil 5 pekerjaan dengan similarity tertinggi
    top_jobs_indices = similarity_scores.argsort()[0][-5:][::-1]

    # Tampilkan hasil beserta similarity score
    results = []
    for index in top_jobs_indices:
        job_title = data.loc[index, 'job_title']
        location = data.loc[index, 'location']
        salary_currency = data.loc[index, 'salary_currency']
        career_level = data.loc[index, 'career_level']
        experience_level = data.loc[index, 'experience_level']
        education_level = data.loc[index, 'education_level']
        employment_type = data.loc[index, 'employment_type']
        job_function = data.loc[index, 'job_function']
        job_benefits = data.loc[index, 'job_benefits']
        company_process_time = data.loc[index, 'company_process_time']
        company_size = data.loc[index, 'company_size']
        company_industry = data.loc[index, 'company_industry']
        job_description = data.loc[index, 'job_description']
        salary = int(data.loc[index, 'salary'])  # Convert to int

        similarity_score = similarity_scores[0][index]

        results.append({
            'Job Title': job_title,
            'Location': location,
            'Salary Currency': salary_currency,
            'Career Level': career_level,
            'Experience Level': experience_level,
            'Education Level': education_level,
            'Employment Type': employment_type,
            'Job Function': job_function,
            'Job Benefits': job_benefits,
            'Company Process Time': company_process_time,
            'Company Size': company_size,
            'Company Industry': company_industry,
            'Job Description': job_description,
            'Salary': salary,
            'Similarity Score': similarity_score
        })

    return results


# Endpoint untuk pencarian pekerjaan
@app.route('/search', methods=['POST'])
def search_endpoint():
    try:
        # Menerima data JSON dari POST request
        request_data = request.get_json()

        # Ambil nilai query dari data JSON
        query = request_data.get('query', '')

        # Lakukan pencarian
        search_results = search_jobs(query, cosine_sim, data)

        # Kirim respons dalam bentuk JSON
        if not search_results:
            return jsonify({'message': 'No results found for the given query.'}), 404
        else:
            return jsonify({'results': search_results})
    except Exception as e:
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500

if __name__ == "__main__":
    # Use the PORT environment variable if available, or default to 8080
    port = int(os.environ.get("PORT", 60))
    app.run(host="0.0.0.0", port=port)
