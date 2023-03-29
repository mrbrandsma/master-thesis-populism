import analysis_helper as helper
from features_helper import remove_words
import pandas as pd
from nltk.corpus import stopwords
# nltk.download('stopwords')

# Define what files to use
outfile = 'large_data'        # dataset
data_type_all = ['train', 'dev', 'test']

for data_type in data_type_all:
    # Import the data as list of dicts
    data = helper.get_data(outfile, data_type)

    # Deep clean the tweets and store as cleaned dataset
    for tweet in data:
        tweet['Tweet'] = helper.deep_clean(tweet['Tweet']) 

    data_columns = ['Party', 'Year', 'Date', 'Tweet', 'Populist', 'Role', 'Left/right', "Prog/cons"]
    tweet_df = pd.DataFrame(data, columns=data_columns)
    tweet_df.to_csv('../data/' + outfile + '/' + 'clean_' + data_type + '.csv', index=True, sep=',')

    #remove_words = remove_words('../data/ignore_words.txt')

    dutch_stopwords = stopwords.words('dutch')
    vocabulary_dict = helper.get_frequency_of_words(data, left_out_words=[], stop_words=dutch_stopwords)
    most_frequent = helper.get_most_frequent_words(vocabulary_dict)
    #helper.get_frequency_plot(most_frequent)