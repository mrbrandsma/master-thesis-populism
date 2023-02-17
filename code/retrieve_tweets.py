from helper import get_tweets

# For date:
    # Last election: 2021-03-26
    # Before-last election: 2015-03-15

# Change settings here
party_data_path = '../data/party_data.json'
outfile = '../data/coalition_large.csv'
# If no limit wanted: use 0
limit_per_party = 0
since = '2015-03-15'
until = '2023-02-17'

# Collect tweets
get_tweets(party_data_path, limit_per_party, outfile, since, until)