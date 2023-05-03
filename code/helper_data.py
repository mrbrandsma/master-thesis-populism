import json
from matplotlib import pyplot as plt
import numpy
import pandas as pd
import random
import re
import snscrape.modules.twitter as sntwitter

def clean_tweets(tweet):
    """
    Function to clean tweets
    """
    # Remove line breaks
    clean_tweet = tweet.replace('\n', '')
    # Change usernames into @USER
    clean_tweet = re.sub('@[^\s]+','@USER', clean_tweet)
    # Change links into (LINK)
    clean_tweet = re.sub('http[^\s]+', '(LINK)', clean_tweet)
    return(clean_tweet)


def clean_date (date):
    """
    Cleans data format by removing the time.
    Returns year and date separately.
    """
    date_string = str(date)
    splitted = date_string.split()

    # Return year and date
    return(splitted[0][:4], splitted[0])


def get_data_description(data_list, data_labels, outfile):
    """
    Gives a basic description of data file
    """
    for i, data in enumerate(data_list):
        data = data.values.tolist()
        with open('../data/' + outfile + '/' + data_labels[i] + '_description.txt', 'w') as f:
            f.write("Total data size: " + str(len(data)) + '\n'+ '\n')
            f.write("Description per party:" + '\n')

            parties = []
            amount_dict = dict()
            for item in data:
                if item[0] not in parties:
                    parties.append(item[0])
                    amount_dict[item[0]] = 0
                amount_dict[item[0]] += 1

            for item in amount_dict:
                f.write('- ' + str(item).strip("'") + ': ' + str(amount_dict[item]) + '\n') 

            # Division populism
            amount_dict_pop = dict()
            for item in data:
                if item[4] not in amount_dict_pop:
                    amount_dict_pop[item[4]] = 0
                amount_dict_pop[item[4]] += 1
            f.write("\nDivision populism:\n")
            f.write("- Non-populist: " + str(amount_dict_pop['no']) + '\n')
            f.write("- Populist: " + str(amount_dict_pop['yes']))
            amount_list_pop = []
            amount_list_pop.append(amount_dict_pop['no'])
            amount_list_pop.append(amount_dict_pop['yes'])
            
        # Pie chart for division data party
        amount_list = []
        for item in amount_dict:
            amount_list.append(amount_dict[item])
        fig = plt.figure(figsize = (10,7))
        plt.pie(amount_list, labels = parties)
        plt.savefig('../data/' + outfile + '/' + data_labels[i] + '_party_division.png')

        # Pie chart and amounts for division data populist/non-populist data
        pop_list = ['non-populist', 'populist']
        fig = plt.figure(figsize = (10,7))
        plt.pie(amount_list_pop, labels = pop_list)
        plt.savefig('../data/' + outfile + '/' + data_labels[i] + '_pop_division.png')


def get_tweets(party_path, limit_per_party, outfile, since_date, until_date):
    """
    Collects tweets through the SNScrape module. Stores the data in a .json-file.
    Input:
    - party_path: path to .json file containing data about targeted parties/accounts
    - limit_per_party: amount of tweets that should be collected per party
    - outfile: path to the .json file to write the data to
    - since_date: start date for collecting tweets
    - until_date: end date for collecting tweets
    Returns:
    - tweets_data: the dataset in list form
    - columns: list of what data is stored in what position in tweets_data
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
            if party_counter + 1 == limit_per_party:
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

    # Split data into train, dev and test data
    random.shuffle(tweets_data)

    # Store tweets in Pandas dataframe
    # When adding to the data_columns, do so at the end so that the index structure stays the same
    data_columns = ['Party', 'Year', 'Date', 'Tweet', 'Populist', 'Role', 'Left/right', "Prog/cons"]
    tweet_df = pd.DataFrame(tweets_data, columns=data_columns)

    # Store dataframe in file
    tweet_df.to_csv('../data/' + outfile + '/' + outfile + '_complete.csv', index=True, sep=',')

    # Divide data into train, dev and test
    chunked_df = numpy.array_split(tweet_df, 100)
    train_data = pd.concat(chunked_df[0:70])
    train_data.to_csv('../data/' + outfile + '/train.csv')
    dev_data = pd.concat(chunked_df[70:85])
    dev_data.to_csv('../data/' + outfile + '/dev.csv')
    test_data = pd.concat(chunked_df[85:100])
    test_data.to_csv('../data/' + outfile + '/test.csv')

    data_labels = ['train', 'dev', 'test']

    # Return list
    return([train_data, dev_data, test_data], data_labels)