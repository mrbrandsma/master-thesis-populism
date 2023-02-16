import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "(from:vvd) until:2023-02-16 since:2015-03-15"
tweets = []

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    #print(vars(tweet))
    #break
    tweets.append([tweet.date,tweet.user.username, tweet.content])

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
print(df)