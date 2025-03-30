def generate_recommendation(performance_metrics, sentiment_score):
    """
    Generate investment recommendation based on performance and sentiment.
    
    :param performance_metrics: Dictionary containing performance metrics.
    :param sentiment_score: Average sentiment score from news articles.
    :return: Recommendation string.
    """
    recommendation = "HOLD"
    if performance_metrics['Total Return'] > 0 and sentiment_score > 0:
        recommendation = "BUY"
    elif performance_metrics['Total Return'] < 0 and sentiment_score < 0:
        recommendation = "SELL"
    
    return recommendation 