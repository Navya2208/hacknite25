# 1. Imports and Data Loading
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('/Users/sahitipotini/Desktop/movie_rec/netflix_processed.csv')

# 2. Build "soup" feature (if not already in processed file)
df['soup'] = (
    df['title'].fillna('') + ' ' +
    df['director'].fillna('') + ' ' +
    df['cast'].fillna('') + ' ' +
    df['listed_in'].fillna('') + ' ' +
    df['description'].fillna('')
).str.replace(',', ' ')

# 3. TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['soup'])

# 4. Compute Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 5. Recommendation Function
indices = pd.Series(df.index, index=df['title'].str.lower())

def recommend(title, n=5):
    idx = indices[title.lower()]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    rec_indices = [i[0] for i in sim_scores]
    return df[['title', 'listed_in']].iloc[rec_indices]

# 6. Example Usage
recommend('Kota Factory')

data=df