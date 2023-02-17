import snscrape.modules.twitter as sntwitter
import pandas as pd
import json

from helper import get_tweets

party_data_path = '../data/party_data.json'
outfile = '../data/last_election_small.csv'
limit_per_party = 500
since = '2021-03-26'
until = '2023-02-17'

get_tweets(party_data_path, limit_per_party, outfile, since, until)