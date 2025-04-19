import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.sim_matrix = None

    def fit(self, ratings_df):
        """
        Expects a DataFrame with columns: user_id, show_id, rating.
        """
        # Pivot to user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(index='user_id', columns='show_id', values='rating').fillna(0)
        self.sim_matrix = cosine_similarity(self.user_item_matrix)
        return self

    def recommend(self, user_id, n=5):
        """
        Returns top n recommended show_ids for the given user_id.
        """
        if user_id not in self.user_item_matrix.index:
            return []
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        sim_scores = self.sim_matrix[user_idx]
        # Find most similar users (excluding self)
        similar_users = sim_scores.argsort()[::-1][1:]
        # Aggregate ratings from similar users
        recs = self.user_item_matrix.iloc[similar_users].mean().sort_values(ascending=False)
        # Remove shows already rated by the user
        watched = self.user_item_matrix.loc[user_id]
        recs = recs[watched == 0]
        return recs.head(n).index.tolist()
