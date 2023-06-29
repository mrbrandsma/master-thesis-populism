from helper_classifier import classifier
import spacy

# Change settings
data_set = 'large_data'
target_label = 'Populist'
#target_label = 'Role'
#target_label = "Prog/cons"
#target_label = "Left/right"
min_frequency = 50
max_frequency = 1000
iterations = 1000
dev = False
outfile = 'readability_populist'
#outfile = 'readability_role'
#outfile = 'readability_progcons'
#outfile = 'readability_leftright'
nl_spacy = spacy.load('nl_core_news_sm')

# Define what features to extract, don't change
feature_dict = dict()
# Vocabulary features
feature_dict['tf-idf'] = False
# Readability features
feature_dict['w/s'] = True
feature_dict['Leesindex A'] = True
feature_dict['flesch'] = True
feature_dict['flesch-douma'] = True

f = open(f'../results/{data_set}/descriptives_{outfile}.txt', 'w', encoding="utf-8")
f.write('READABILITY\n')
f.write('\n\n\n')

# Experiment 1: all features included

print('EXPERIMENT 1: all features')
f.write('\n\nEXPERIMENT 1: all features')
f.close()
stopwords_complete = []
classifier(data_set, target_label, min_frequency, iterations, feature_dict, stopwords_complete, dev, outfile, nl_spacy, write_features_to_file=True)

# Abblation study
counter = 1
targeted_features = ['w/s', 'Leesindex A', 'flesch', 'flesch-douma']
for feature in targeted_features:
    counter += 1
    f = open(f'../results/{data_set}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
    print('EXPERIMENT ', counter,': ', feature, ' removed')
    f.write(f'\n\nEXPERIMENT {counter}: {feature} removed')
    f.close()
    feature_dict[feature] = False
    classifier(data_set, target_label, min_frequency, iterations, feature_dict, stopwords_complete, dev, outfile, write_features_to_file=False)
    feature_dict[feature] = True