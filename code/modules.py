def clean_tweets(tweet):
    """
    Function to clean tweets
    """

    clean_tweet = tweet.replace('\n', '') 
    return(clean_tweet)