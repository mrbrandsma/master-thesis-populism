import snscrape.modules.twitter as sntwitter
import pandas as pd
import csv
import json

from modules import clean_tweets

# Set data, empty list, and a tweet limit per party if necessary
party_data = json.load(open('../data/party_data.json', 'r'))
tweets_data = []
limit_per_party = 500

# Scroll through the list of parties
for party in party_data['parties']:
    
    # Counter
    party_counter = 0

    # Keep user up to date
    print("Now retrieving tweets from ", party['party'])
    
    # Set query settings
    # query =  "(from:" + party + ") until:2023-02-16 since:2015-03-15 lang:nl"
    query =  "(from:" + party['twitter_handle'] + ") until:2023-02-16 since:2021-03-26 lang:nl"

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        #print(vars(tweet))
        #break
        party_counter += 1
        if party_counter == limit_per_party:
            break
        else:
            original_tweet = tweet.content
            clean_tweet = clean_tweets(tweet.content)

            tweets_data.append([
                party['party'],
                tweet.date, 
                clean_tweet,
                party['considered_populist'],
                party['role'],
                party['ideology_l_r'],
                party['ideology_p_c']])

# Keep user updated
print("Finished retrieving party tweets. Now writing to file.")

# Store tweets in Pandas dataframe
df = pd.DataFrame(tweets_data, columns=['Party', 'Date', 'Tweet', 'Populist', 'Role', 'Left/right', "Prog/cons"])

# Store dataframe in file
df.to_csv('../data/last_election_data.csv', index=True, sep=',')

