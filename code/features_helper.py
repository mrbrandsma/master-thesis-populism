import nltk
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#nltk.download('punkt')

### TF-IDF #####################
def remove_words(path):
    """
    Takes a .txt-file of words and keeps them out of the analysis.
    """
    word_list = []

    # Open ignore-words file
    with open(path) as f:
        words = f.readlines()
    
    for word in words:
        clean_word = word.lower().strip('\n')
        word_list.append(clean_word)
    
    return(word_list)


def get_stop_words(data, 
              freq_threshold):
    """
    Returns a sublist of words from the data, dependent on the frequency of the words.
    Creates a list to compare tweets to at a later stage.
    """
    word_list = []

    # Collect all tokens
    all_tokens = []
    for tweet in data:
        clean_tweet = tweet.lower()
        clean_tweet = clean_tweet = re.sub(r'[^\w\s]', '', clean_tweet)
        token_list = nltk.tokenize.word_tokenize(clean_tweet, language='dutch')

        for token in token_list:
            all_tokens.append(token)
    
    freq_word_counter = Counter(all_tokens)

    for word, count in freq_word_counter.items():
        if count > freq_threshold:
            word_list.append(word)
    
    return(word_list)


def bag_of_words(data_list, min_freq_threshold, stopwords):
    """
    Returns a bag of words of the given data list.
    """
    # Create the settings
    utterance_vec = CountVectorizer(min_df=min_freq_threshold, 
                                    tokenizer=nltk.word_tokenize, 
                                    stop_words=stopwords
                                   )

    # Create the dictionary and bag-of-words vector representations
    count_vectors = utterance_vec.fit_transform(data_list)

    return(utterance_vec, count_vectors)


def tfidf_transform(data_vector):
    """
    Changes vector to TF-IDF values.
    Returns a vector with TF-IDF weights.
    """
    # Create a transformer
    tfidf_transformer = TfidfTransformer()
    tfidf_vectors = tfidf_transformer.fit_transform(data_vector)
    
    # Return the weighted vectors
    return(tfidf_vectors)

#################################

def get_features(train_data, features_to_extract):
    """
    Extracts the features and adds them to a vector
    """

    # For every instance in the data, retrieve the features
    for i, instance in enumerate(train_data):
        feature_dict = dict()

        # FEATURE 1: TF-IDF
        if features_to_extract['tokens'] == True:
            print('Extract feature: ')

def get_tfidf(train_data, test_data, min_frequency, stopwords):
    """
    Gets TF-IDF vectors for both train and test data
    """
    print("Extracting feature: TF-IDF")
    # Get the bag of words of dataset
    train_tweet_vec, training_count_vectors = bag_of_words(train_data, min_frequency, stopwords)

    # Transform test data
    test_count_vectors = train_tweet_vec.transform(test_data)

    # Get TF-IDF vectors
    train_tfidf_vectors = tfidf_transform(training_count_vectors)
    test_tfidf_vectors = tfidf_transform(test_count_vectors)

    return(train_tfidf_vectors, test_tfidf_vectors)