import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = None
        self.tfidf_matrix = None
        self.df = None
        self.indices = None

    def fit(self, df):
        """
        Expects a DataFrame with a 'soup' column (combined text features).
        """
        self.df = df.reset_index(drop=True)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['soup'].fillna(''))
        self.indices = pd.Series(self.df.index, index=self.df['title'].str.lower())
        return self

    def recommend(self, title, n=5):
        """
        Returns top n similar titles to the given title.
        """
        idx = self.indices.get(title.lower())
        if idx is None:
            return pd.DataFrame()  # Title not found
        sim_scores = list(enumerate(cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        rec_indices = [i[0] for i in sim_scores]
        return self.df.iloc[rec_indices][['title', 'type', 'listed_in', 'description']]
