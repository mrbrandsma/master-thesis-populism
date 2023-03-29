from classifier_helper import classifier
from features_helper import get_features
from nltk.corpus import stopwords

data_set = 'large_data'
target_label = 'Populist'
min_frequency = 50
max_frequency = 1000
iterations = 3
dev = True

# Define what features to extract
feature_dict = dict()
feature_dict['tf-idf'] = True
feature_dict['feature_2'] = False

stopwords = stopwords.words('dutch')

classifier(data_set, target_label, min_frequency, max_frequency, iterations, feature_dict, stopwords, dev)