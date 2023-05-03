import helper_analysis as helper
import pandas as pd
from nltk.corpus import stopwords

# Define what files to use
outfile = 'test'        # dataset
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
                        "D66": "#2e862c",  #2e862c
                        "SGP": "#bb4a0b", #bb4a0b
                        "SP": "#b40816", #milanored
                        "VVD": "#c87800", #c87800
                        "FvD": "#6a1213", #cloudburst
                        "PvdA": "#b80000", #b80000
                        "GroenLinks": "#018b35", #1f2223
                        "PVV": "#1d385c", #1d385c
                        "CU": "#0084b9" #0084b9
                        }
    label_colour_list = {
                        "yes": "#ff7f0e",
                        "no": "#1f77b4"
                        }
    helper.get_graphs(data, data_type, outfile, party_colour_list, label_colour_list)

    data_columns = ['Party', 'Year', 'Date', 'Tweet', 'Populist', 'Role', 'Left/right', "Prog/cons"]
    tweet_df = pd.DataFrame(data, columns=data_columns)
    tweet_df.to_csv('../data/' + outfile + '/clean/' + 'clean_' + data_type + '.csv', index=True, sep=',')