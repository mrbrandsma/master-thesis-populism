import helper_analysis as helper
import pandas as pd
from nltk.corpus import stopwords

# Define what files to use
outfile = 'large_data'        # dataset
data_type_all = ['train', 'dev', 'test']

for data_type in data_type_all:
    # Import the data as list of dicts
    data = helper.get_data(outfile, data_type)

    # Deep clean the tweets and store as cleaned dataset
    for tweet in data:
        tweet['Tweet'] = helper.deep_clean(tweet['Tweet']) 
    
    # Remove short tweet
    min_tweet_length = 5
    data = helper.clean_short_tweets(data, min_tweet_length)

    # Get the graphs
        # Hardcoded Dutch parties and their HEX-codes:
    party_colour_list = {
                        "D66": "#6eb1a5",  ##6eb1a5
                        "SGP": "#dbdada", #bb4a0b
                        "SP": "#dbdb92", #dbdb92
                        "VVD": "#d99344", #d99344
                        "FvD": "#d9abc2", #d9abc2
                        "PvdA": "#b7b7b7", #b7b7b7
                        "GroenLinks": "#92bc4b", #92bc4b
                        "PVV": "#9b619c", #9b619c
                        "CU": "#aac8a4" #aac8a4
                        }
    label_colour_list = {
                        "yes": "#d86154", #d86154
                        "no": "#6191b1" #6191b1
                        }
    helper.get_graphs(data, data_type, outfile, party_colour_list, label_colour_list)

    data_columns = ['Party', 'Year', 'Date', 'Tweet', 'Populist', 'Role', 'Left/right', "Prog/cons"]
    tweet_df = pd.DataFrame(data, columns=data_columns)
    tweet_df.to_csv('../data/' + outfile + '/clean/' + 'clean_' + data_type + '.csv', index=True, sep=',')