from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
with open("doc1.txt", "r", encoding="utf-8") as file:
    doc1 = file.read()
with open("doc2.txt", "r", encoding="utf-8") as file:
    doc2 = file.read()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
# Get the individual TF-IDF vectors for each document
tfidf_doc1 = tfidf_matrix[0]
tfidf_doc2 = tfidf_matrix[1]
# Reshape the vectors to be 2D arrays
tfidf_doc1 = tfidf_doc1.reshape(1, -1)
tfidf_doc2 = tfidf_doc2.reshape(1, -1)
# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_doc1, tfidf_doc2)
print("Cosine Similarity between doc1 and doc2:")
print(cosine_sim[0][0])



    



