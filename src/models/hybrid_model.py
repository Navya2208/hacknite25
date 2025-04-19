class HybridRecommender:
    def __init__(self, content_model, collab_model):
        self.content_model = content_model
        self.collab_model = collab_model

    def recommend(self, user_id, title, n=5):
        """
        Returns a hybrid recommendation list: half from collaborative, half from content-based.
        """
        recs_content = self.content_model.recommend(title, n=n//2)
        recs_collab = self.collab_model.recommend(user_id, n=n - n//2)
        # Combine and deduplicate
        recs_combined = pd.concat([
            recs_content[['title']],
            pd.DataFrame({'title': recs_collab})
        ]).drop_duplicates().head(n)
        return recs_combined
