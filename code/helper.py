import snscrape.modules.twitter as sntwitter
import pandas as pd
import json

def clean_tweets(tweet):
    """
    Function to clean tweets
    """
    # Remove line breaks
    clean_tweet = tweet.replace('\n', '') 

    return(clean_tweet)


def clean_date (date):
    """
    Cleans data format by removing the time
    Returns year and date
    """
    date_string = str(date)
    splitted = date_string.split()

    # Return year and date
    return(splitted[0][:4], splitted[0])


def get_tweets(party_path, limit_per_party, outfile, since_date, until_date):
    """
    Get Tweets.
    """


    # Set data, empty list, and a tweet limit per party if necessary
    party_data = json.load(open(party_path, 'r'))
    tweets_data = []

    # Scroll through the list of parties
    for party in party_data['parties']:
        
        # Counter
        party_counter = 0

        # Keep user up to date
        print("Now retrieving tweets from ", party['party'])
        
        # Set query settings
        # query =  "(from:" + party + ") until:2023-02-16 since:2015-03-15 lang:nl"
        query =  "(from:" + party['twitter_handle'] + ") until:" + until_date + " since:" + since_date + " lang:nl"

        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            # Keep track of amount of tweets per party
            party_counter += 1
            if party_counter == limit_per_party:
                break
            
            # Clean tweets and store them in dataset
            else:
                clean_tweet = clean_tweets(tweet.content)
                year, date = clean_date(tweet.date)

                tweets_data.append([
                    party['party'],
                    year, 
                    date,
                    clean_tweet,
                    party['considered_populist'],
                    party['role'],
                    party['ideology_l_r'],
                    party['ideology_p_c']])

    # Keep user updated
    print("Finished retrieving party tweets. Now writing to file.")

    # Store tweets in Pandas dataframe
    df = pd.DataFrame(tweets_data, columns=['Party', 'Year', 'Date', 'Tweet', 'Populist', 'Role', 'Left/right', "Prog/cons"])

    # Store dataframe in file
    df.to_csv(outfile, index=True, sep=',')