from helper_classifier import classifier
from helper_features import get_function_words
from nltk.corpus import stopwords

data_set = 'large_data'
target_label = 'Prog/cons'
min_frequency = 50
max_frequency = 1000
iterations = 1000
dev = False
outfile = 'readability_progcons'

# Define what features to extract
feature_dict = dict()
# Vocabulary features
feature_dict['tf-idf'] = False
# Readability features
feature_dict['flesch-kincaid'] = True
feature_dict['smog'] = True
feature_dict['word_amount'] = True
feature_dict['sentence_amount'] = True
feature_dict['syllable_amount'] = True
feature_dict['polysyllable_amount'] = True

stopwords = stopwords.words('dutch')
list_of_function_words = ['articles', 'prepositions', 'quantifiers', 'conjunctions', 
                        'pronouns', 'auxiliary verbs', 'adverbs', 'modifiers', 'interjections']
list_of_topic_words = ['islam', 'kartel', 'islamitisch', 'asielzoeker', 'gelukzoeker', 'lockdown', 
                       'immigratie', 'toeslagenschandaal', 'klimaatakkoord', 'zorgverlener', 'aow', 
                       'kringlooplandbouw', 'klimaatcrisis', 'duurzaam', 'abortus', 'klimaatverandering', 
                       'progressief', 'uitstoot', 'groen', 'green', 'europees', 'leraar', 'student', 'klimaat', 'schoon', 
                       'vluchteling']
data_specific_words = ['ha', 'hoi', 'hi', 'party-kamerlid', 'partykamerlid', 'inside', '!', '#', '%', '&', "'", '(', ')', '+', ',', '-', '.', '...', '1', '1/2', '10', '100', 
                       '11', '12', '13', '14', '15', '16', '17', '19.30', '2', '20', '20.00', '2018', '2019', 
                       '2020', '2021', '20:00', '21', '23', '25', '3', '30', '4', '5', '50', '6', '7', '9', 
                       ':', ';', '=', '?', '@', '^bas', '^evi', '^ivo', '^kaj', '^kim', '^liz', '^luuk', '^noa', 
                       '^r', '^sb', '^tom', '_', '_^luuk', '``', 'baudet',  'forum', 'fvd', 'geert', 'hans', 
                       'hoekstra', 'jesse', 'mark', 'kaag', 'lilian', 'lilianne', 'paul', 'ploumen', 'rob', 
                       'ronald', 'rutte', 'sigrid', 'thierry', 'user', 'wilders', '||', 'écht', 'échte', 'één', 
                       '\u200d', '‘', '’', '“', '”', '…', '\u2066', '✅', '✊', '✍', '❤', '➡', '⤵️', '⬇', 
                       '️', '🇪', '🇺', '🌈', '🌍', '🌹', '🍀', '🍅', '🎉', '🎥', '🏡', '🏼', '👀', '👇', '👉', 
                       '👍', '👩', '💙', '💚', '💪', '📺', '📻', '🔥', '🔴', '🗳', '😉', '🙌', '🤯', '🧡']

f = open(f'../results/{data_set}/descriptives_{outfile}.txt', 'w', encoding="utf-8")
f.write('READABILITY\n')
f.write('\n\n\n')

# Experiment 1: all features included

print('EXPERIMENT 1: all features')
f.write('\n\nEXPERIMENT 1: all features')
f.close()
stopwords_complete = []
classifier(data_set, target_label, min_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)


# Abblation study
counter = 1
targeted_features = ['flesch-kincaid', 'smog', 'word_amount', 'sentence_amount', 'syllable_amount', 'polysyllable_amount']
for feature in targeted_features:
    counter += 1
    f = open(f'../results/{data_set}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
    print('EXPERIMENT ', counter,': ', feature, ' removed')
    f.write(f'\n\nEXPERIMENT {counter}: {feature} removed')
    f.close()
    feature_dict[feature] = False
    classifier(data_set, target_label, min_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)
    feature_dict[feature] = True