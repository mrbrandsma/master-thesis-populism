from helper_data import get_tweets, get_data_description

# Settings
party_data_path = '../data/party_data_small.json'       # json-file with information about target twitter accounts
name_data = 'test'                                # name of data file, will determine the name of stored files
limit_per_party = 20                                  # Set limit of amount of tweets per party, if no limit wanted: use 0.
since = '2017-10-26'                                    # Set begin date of tweet collection
until = '2023-03-29'                                    # Set end date of tweet collection

# Collect tweets
data_list, data_labels = get_tweets(party_data_path, limit_per_party, name_data, since, until)

# Get basic description of data (written to files)
get_data_description(data_list, data_labels, name_data)