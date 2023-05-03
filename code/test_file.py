from helper_features import lemmatizer
import spacy
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

language_pipeline = spacy.load("nl_core_news_sm")
tweets = ["TNO zei dat er 72.000 banen bij zouden komen door het klimaatakkoord. Dat blijkt niet waar: het baneneffect van de honderden miljarden is op de lange termijn verwaarloosbaar. TNO komt erop terug. Wanneer komt het kabinet terug op dit desastreuze beleid? (LINK) (LINK)", "@USER @USER Een nieuwe kerncentrale is heel erg duur en de bouw duurt super lang. Té lang om de klimaatdoelstellingen te halen. Tegelijkertijd kunnen wij als een land van wind en water onze doelen wél halen met echte alternatieven: wind en zon! ^Liz"]

# Preprocessing
lemmatized_tweets = lemmatizer(tweets, [], language_pipeline)

cleaned_tweets = []
for tweet in tweets:
    clean = tweet.lower()
    clean = re.sub(r'[^\w\s]', '', clean)
    cleaned_tweets.append(clean)

# Do things
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(tweets)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns = feature_names)    
print(df)