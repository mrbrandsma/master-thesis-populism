from helper_classifier import classifier
from helper_features import get_function_words
from nltk.corpus import stopwords

data_set = 'large_data'
target_label = 'Populist'
outfile = 'test'
min_frequency = 50
max_frequency = 1000
iterations = 3
dev = True

# Define what features to extract
feature_dict = dict()
# Vocabulary features
feature_dict['tf-idf'] = True
# Readability features
feature_dict['flesch-kincaid'] = False
feature_dict['smog'] = False
feature_dict['word_amount'] = False
feature_dict['sentence_amount'] = False
feature_dict['syllable_amount'] = False
feature_dict['polysyllable_amount'] = False

stopwords = stopwords.words('dutch')
list_of_function_words = ['articles', 'prepositions', 'quantifiers', 'conjunctions', 
                        'pronouns', 'auxiliary verbs', 'adverbs', 'modifiers', 'interjections']
function_words_to_remove = get_function_words(wanted_types=[
                        'articles', 
                        'prepositions', 
                        'quantifiers', 
                        'conjunctions', 
                        'pronouns', 
                        'auxiliary verbs', 
                        'adverbs', 
                        'modifiers', 
                        'interjections'
                        ])
data_specific_words = ['!', '#', '%', '&', "'", '(', ')', '+', ',', '-', '.', '...', '1', '1/2', '10', '100', '11', '12', '13', '14', '15', '16', '17', '19.30', '2', '20', '20.00', '2018', '2019', '2020', '2021', '20:00', '21', '23', '25', '3', '30', '4', '5', '50', '6', '7', '9', ':', ';', '=', '?', '@', '^bas', '^evi', '^ivo', '^kaj', '^kim', '^liz', '^luuk', '^noa', '^r', '^sb', '^tom', '_', '_^luuk', '``', 'baudet',  'forum', 'fvd', 'geert', 'hans', 'hoekstra', 'jesse', 'kaag', 'lilian', 'lilianne', 'paul', 'ploumen', 'rob', 'ronald', 'rutte', 'sigrid', 'thierry', 'user', 'wilders', '||', 'écht', 'échte', 'één', '\u200d', '‘', '’', '“', '”', '…', '\u2066', '✅', '✊', '✍', '❤', '➡', '⤵️', '⬇', '️', '🇪', '🇺', '🌈', '🌍', '🌹', '🍀', '🍅', '🎉', '🎥', '🏡', '🏼', '👀', '👇', '👉', '👍', '👩', '💙', '💚', '💪', '📺', '📻', '🔥', '🔴', '🗳', '😉', '🙌', '🤯', '🧡']

stopwords_complete = stopwords + function_words_to_remove + data_specific_words

classifier(data_set, target_label, min_frequency, max_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)