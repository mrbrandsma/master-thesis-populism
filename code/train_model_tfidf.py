from helper_classifier import classifier
from helper_features import get_function_words
from nltk.corpus import stopwords

data_set = 'test'
target_label = 'Populist'
min_frequency = 50
max_frequency = 1000
iterations = 3
dev = False
#feature_test = "ablation"
#outfile = 'tfidf_ablation'
feature_test = "addition"
outfile = 'tfidf_addition'

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
f.write('TF-IDF AS FEATURE\n')
f.write('\n\n\n')

# Experiment 1: all words included
print('EXPERIMENT 1: all words')
f.write('\n\nEXPERIMENT 1: all words')
f.close()
stopwords_complete = stopwords + data_specific_words
classifier(data_set, target_label, min_frequency, max_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)

# Experiment 2: only content words included
f = open(f'../results/{data_set}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
print('EXPERIMENT 2: only content words')
f.write('\n\nEXPERIMENT 2: only content words')
f.close()
function_words = get_function_words(wanted_types=list_of_function_words)
stopwords_complete = stopwords + data_specific_words + function_words
classifier(data_set, target_label, min_frequency, max_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)

# Experiment 2: only content words included + topic words removed
f = open(f'../results/{data_set}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
print('EXPERIMENT 3: only content words, topic words removed')
f.write('\n\nEXPERIMENT 2: only content words, topic words removed')
f.close()
function_words = get_function_words(wanted_types=list_of_function_words)
stopwords_complete = stopwords + data_specific_words + function_words + list_of_topic_words
classifier(data_set, target_label, min_frequency, max_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)

counter = 3
for function_word_type in list_of_function_words:
    counter += 1
    f = open(f'../results/{data_set}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
    if feature_test == 'addition':
        print('EXPERIMENT ', counter,': ', function_word_type, ' added')
        f.write(f'\n\nEXPERIMENT {counter}: {function_word_type} added')
        f.close()
        temporary_list = list_of_function_words.copy()
        temporary_list.remove(function_word_type)
        function_words = get_function_words(temporary_list)
        stopwords_complete = stopwords + data_specific_words + function_words + list_of_topic_words
        classifier(data_set, target_label, min_frequency, max_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)
    else:
        print('EXPERIMENT ', counter,': ', function_word_type, ' removed')
        f.write(f'\n\nEXPERIMENT {counter}: {function_word_type} removed')
        f.close()
        function_words = get_function_words([function_word_type])
        stopwords_complete = stopwords + data_specific_words + function_words + list_of_topic_words
        classifier(data_set, target_label, min_frequency, max_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)