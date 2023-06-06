from helper_classifier import classifier
from helper_features import get_function_words
from nltk.corpus import stopwords

data_set = 'large_data'
target_label = 'Prog/cons'
min_frequency = 50
iterations = 1000
dev = False
#feature_test = "ablation"
#outfile = 'tfidf_ablation'
feature_test = "none"
outfile = 'tfidf_progcons'

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
# Sentiment analysis


stopwords = stopwords.words('dutch')
list_of_function_words = ['articles', 'prepositions', 'quantifiers', 'conjunctions', 
                        'pronouns', 'auxiliary verbs', 'adverbs', 'modifiers', 'interjections']
old_list_of_topic_words = ['islam', 'kartel', 'islamitisch', 'asielzoeker', 'gelukzoeker', 'lockdown', 
                       'immigratie', 'toeslagenschandaal', 'klimaatakkoord', 'zorgverlener', 'aow', 
                       'kringlooplandbouw', 'klimaatcrisis', 'duurzaam', 'abortus', 'klimaatverandering', 
                       'progressief', 'uitstoot', 'groen', 'green', 'europees', 'leraar', 'student', 'klimaat', 'schoon', 
                       'vluchteling', 'grens', 'koopkracht', 'groninger', 'natuur', 'onderwijs', 'permanent', 'permanente', 
                       'coronawet', 'oorlog']
list_of_topic_words = ['islam', 'kartel', 'islamitisch', 'asielzoeker', 'gelukzoeker', 'lockdown', 'ondernemer'
                       'immigratie', 'toeslagenschandaal', 'klimaatakkoord', 'zorgverlener', 'aow', 
                       'kringlooplandbouw', 'klimaatcrisis', 'duurzaam', 'abortus', 'klimaatverandering', 
                       'progressief', 'uitstoot', 'groen', 'green', 'europees', 'leraar', 'student', 'klimaat', 'schoon', 
                       'vluchteling', 'grens', 'koopkracht', 'groninger', 'natuur', 'onderwijs', 'permanent', 'permanente', 
                       'coronawet', 'oorlog', 'huurder', 'wooncrisis', 'minimumloon', 'kernenergie', 'multinational', 'horeca']
data_specific_words = ['ha', 'hoi', 'hi', 'party-kamerlid', 'partykamerlid', 'inside', '!', '#', '%', '&', "'", '(', ')', '+', ',', '-', '.', '...', '1', '1/2', '10', '100', 
                       '11', '12', '13', '14', '15', '16', '17', '19.30', '2', '20', '20.00', '2018', '2019', 
                       '2020', '2021', '20:00', '21', '23', '25', '3', '30', '4', '5', '50', '6', '7', '9', 
                       ':', ';', '=', '?', '@', '^bas', '^evi', '^ivo', '^kaj', '^kim', '^liz', '^luuk', '^noa', 'youtube-kanaal', 'donderdag', 'vrijdag', 'npo1', 'ers', 
                       '^r', '^sb', '^tom', '_', '_^luuk', '``', 'baudet',  'forum', 'fvd', 'geert', 'hans', 
                       'hoekstra', 'raak', 'jesse', 'mark', 'kaag', 'lilian', 'lilianne', 'paul', 'ploumen', 'rob', 
                       'ronald', 'rutte', 'sigrid', 'thierry', 'user', 'wilders', '||', '\u200d', '‘', '’', '“', '”', 
                       '…', '\u2066', '✅', '✊', '✍', '❤', '➡', '⤵️', '⬇', 'luistertip', 'kijktip', 'ticket', 'luister'
                       'socialer', 'leiderschap', 'studio', 'uitzending', 'vd', 'nl', 'oa', 'app',
                       '️', '🇪', '🇺', '🌈', '🌍', '🌹', '🍀', '🍅', '🎉', '🎥', '🏡', '🏼', '👀', '👇', '👉', 
                       '👍', '👩', '💙', '💚', '💪', '📺', '📻', '🔥', '🔴', '🗳', '😉', '🙌', '🤯', '🧡']
old_data_specific_words = ['ha', 'hoi', 'hi', 'party-kamerlid', 'partykamerlid', 'inside', '!', '#', '%', '&', "'", '(', ')', '+', ',', '-', '.', '...', '1', '1/2', '10', '100', 
                       '11', '12', '13', '14', '15', '16', '17', '19.30', '2', '20', '20.00', '2018', '2019', 
                       '2020', '2021', '20:00', '21', '23', '25', '3', '30', '4', '5', '50', '6', '7', '9', 
                       ':', ';', '=', '?', '@', '^bas', '^evi', '^ivo', '^kaj', '^kim', '^liz', '^luuk', '^noa', 
                       '^r', '^sb', '^tom', '_', '_^luuk', '``', 'baudet',  'forum', 'fvd', 'geert', 'hans', 
                       'hoekstra', 'raak', 'jesse', 'mark', 'kaag', 'lilian', 'lilianne', 'paul', 'ploumen', 'rob', 
                       'ronald', 'rutte', 'sigrid', 'thierry', 'user', 'wilders', '||', '\u200d', '‘', '’', '“', '”', 
                       '…', '\u2066', '✅', '✊', '✍', '❤', '➡', '⤵️', '⬇', 'luistertip', 'kijktip', 'ticket',
                       'socialer', 'leiderschap', 'studio', 'uitzending', 'vd', 'nl', 'oa', 'app',
                       '️', '🇪', '🇺', '🌈', '🌍', '🌹', '🍀', '🍅', '🎉', '🎥', '🏡', '🏼', '👀', '👇', '👉', 
                       '👍', '👩', '💙', '💚', '💪', '📺', '📻', '🔥', '🔴', '🗳', '😉', '🙌', '🤯', '🧡']

f = open(f'../results/{data_set}/descriptives_{outfile}.txt', 'w', encoding="utf-8")
f.write(f'TF-IDF AS FEATURE FOR {target_label}\n')
f.write('\n\n\n')

# Experiment 1: all words included
print('EXPERIMENT 1: all words')
f.write('\n\nEXPERIMENT 1: all words')
f.close()
stopwords_complete = stopwords + data_specific_words
classifier(data_set, target_label, min_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)

# Experiment 2: only content words included
f = open(f'../results/{data_set}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
print('EXPERIMENT 2: only content words')
f.write('\n\nEXPERIMENT 2: only content words')
f.close()
function_words = get_function_words(wanted_types=list_of_function_words)
stopwords_complete = stopwords + data_specific_words + function_words
classifier(data_set, target_label, min_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)

# Experiment 3: only topic words removed
f = open(f'../results/{data_set}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
print('EXPERIMENT 3: only topic words removed')
f.write('\n\nEXPERIMENT 3: only topic words removed')
f.close()
stopwords_complete = stopwords + data_specific_words + list_of_topic_words
classifier(data_set, target_label, min_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)

# Experiment 4: only content words included + topic words removed
f = open(f'../results/{data_set}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
print('EXPERIMENT 4: only content words, topic words removed')
f.write('\n\nEXPERIMENT 4: only content words, topic words removed')
f.close()
function_words = get_function_words(wanted_types=list_of_function_words)
stopwords_complete = stopwords + data_specific_words + function_words + list_of_topic_words
classifier(data_set, target_label, min_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)

counter = 4
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
        classifier(data_set, target_label, min_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)
    elif feature_test == 'ablation':
        print('EXPERIMENT ', counter,': ', function_word_type, ' removed')
        f.write(f'\n\nEXPERIMENT {counter}: {function_word_type} removed')
        f.close()
        function_words = get_function_words([function_word_type])
        stopwords_complete = stopwords + data_specific_words + function_words + list_of_topic_words
        classifier(data_set, target_label, min_frequency, iterations, feature_dict, stopwords_complete, dev, outfile)
    else:
        break