import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def precision_at_k(recommended_items, relevant_items, k=5):
    """
    Calculate precision@k for recommendation results.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of actually relevant item IDs
        k (int): Number of recommendations to consider
        
    Returns:
        float: Precision@k score
    """
    if len(recommended_items) == 0:
        return 0.0
        
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(relevant_items))
    return hits / min(k, len(recommended_k))

def recall_at_k(recommended_items, relevant_items, k=5):
    """
    Calculate recall@k for recommendation results.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of actually relevant item IDs
        k (int): Number of recommendations to consider
        
    Returns:
        float: Recall@k score
    """
    if len(relevant_items) == 0:
        return 0.0
        
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(relevant_items))
    return hits / len(relevant_items)

def ndcg_at_k(recommended_items, relevant_items, k=5):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of actually relevant item IDs
        k (int): Number of recommendations to consider
        
    Returns:
        float: NDCG@k score
    """
    if len(recommended_items) == 0 or len(relevant_items) == 0:
        return 0.0
        
    # Get top k recommendations
    recommended_k = recommended_items[:k]
    
    # Calculate DCG
    dcg = 0
    for i, item in enumerate(recommended_k):
        if item in relevant_items:
            # Using binary relevance (1 if relevant, 0 if not)
            # Position i+1 because enumerate starts from 0
            dcg += 1 / np.log2(i + 2)  # log2(i+2) because i is 0-indexed
    
    # Calculate IDCG (ideal DCG)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant_items))))
    
    # Avoid division by zero
    if idcg == 0:
        return 0.0
        
    return dcg / idcg

def mean_average_precision(recommended_items_list, relevant_items_list, k=5):
    """
    Calculate Mean Average Precision (MAP) at k.
    
    Args:
        recommended_items_list (list of lists): List of lists containing recommended item IDs for each user
        relevant_items_list (list of lists): List of lists containing actually relevant item IDs for each user
        k (int): Number of recommendations to consider
        
    Returns:
        float: MAP@k score
    """
    if len(recommended_items_list) == 0:
        return 0.0
        
    ap_sum = 0
    for recommended, relevant in zip(recommended_items_list, relevant_items_list):
        ap_sum += average_precision_at_k(recommended, relevant, k)
    
    return ap_sum / len(recommended_items_list)

def average_precision_at_k(recommended_items, relevant_items, k=5):
    """
    Calculate Average Precision at k.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of actually relevant item IDs
        k (int): Number of recommendations to consider
        
    Returns:
        float: AP@k score
    """
    if len(recommended_items) == 0 or len(relevant_items) == 0:
        return 0.0
        
    recommended_k = recommended_items[:k]
    
    hits = 0
    sum_precs = 0
    
    for i, item in enumerate(recommended_k):
        if item in relevant_items:
            hits += 1
            sum_precs += hits / (i + 1)
    
    if hits == 0:
        return 0.0
        
    return sum_precs / min(len(relevant_items), k)

def evaluate_content_recommendations(recommender, test_titles, ground_truth, k=5):
    """
    Evaluate a content-based recommender system using various metrics.
    
    Args:
        recommender: The recommender model with a recommend method
        test_titles (list): List of movie/show titles to get recommendations for
        ground_truth (dict): Dictionary mapping titles to relevant items
        k (int): Number of recommendations to consider
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    precisions = []
    recalls = []
    ndcgs = []
    
    for title in test_titles:
        # Get recommendations
        recommendations = recommender.recommend(title, n=k)
        recommended_ids = recommendations['show_id'].tolist() if not recommendations.empty else []
        
        # Get ground truth
        relevant_ids = ground_truth.get(title, [])
        
        # Calculate metrics
        precisions.append(precision_at_k(recommended_ids, relevant_ids, k))
        recalls.append(recall_at_k(recommended_ids, relevant_ids, k))
        ndcgs.append(ndcg_at_k(recommended_ids, relevant_ids, k))
    
    return {
        f'precision@{k}': np.mean(precisions),
        f'recall@{k}': np.mean(recalls),
        f'
