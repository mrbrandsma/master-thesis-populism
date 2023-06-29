import helper_analysis as helper
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

outfile = 'large_data'        # dataset

# Change settings here
#data_type = 'large_data_complete'
file = 'progcons_features'
target_label = 'Prog/cons'
#data_type = 'dev'
data_type = 'train'
#data_type = 'test'

get_dict = dict()
get_dict['tfidf'] = False
get_dict['readability'] = True
get_dict['sentiment'] = False
get_dict['general'] = False


# Get data
data = helper.get_data(outfile, data_type)

if get_dict['readability'] == True:
    df_readability = pd.read_csv(f"../data/large_data/clean/clean_{data_type}_readability_{file}.csv")

    readability_list = ['w/s', 'Leesindex A', 'Flesch', 'Flesch-Douma']
    
    for item in readability_list:
        print(f'-------{item}-------')
        print(f"Mean {item} scores of target label {target_label}")
        averages = df_readability.groupby([target_label])[item].mean()
        print(averages)
        print('\n')
        print(f"Std {item} scores of target label {target_label}")
        std = df_readability.groupby([target_label])[item].std()
        print(std)
        print('\n')


if get_dict['tfidf'] == True:
    # Get mean of words of populist tweets
    train_tfidf = pd.read_csv("../analysis/large_data/train/tfidf")
    test_data = helper.get_data(outfile, "test")
    train_data = helper.get_data(outfile, "train")
    populist_tfidf = dict()
    non_populist_tfidf = dict()

    # Add all words to dictionary
    tfidf_labels = []
    for col in train_tfidf.columns:
        tfidf_labels.append(col)
        populist_tfidf[col] = []
        non_populist_tfidf[col] = []

    # Add scores to corresponding words
    test_score = 0
    for instance, tfidf in zip(data, train_tfidf.values.tolist()):
        # Store scores in the right dictionary
        if instance['Populist'] == 'yes':
            counter = 0
            for item in tfidf:
                # Get the current word
                current_word = tfidf_labels[counter]
                if item != 0:
                    populist_tfidf[current_word].append(item)
                counter += 1
            test_score += 1

        elif instance['Populist'] == 'no':
            counter = 0
            for item in tfidf:
                # Get the current word
                current_word = tfidf_labels[counter]
                if item != 0:
                    non_populist_tfidf[current_word].append(item)
                counter += 1
            test_score += 1

    # Get mean
    populist_mean_tfidf = dict()
    non_populist_mean_tfidf = dict()
    for word in populist_tfidf:
        populist_mean_tfidf[word] = np.mean(populist_tfidf[word])
        non_populist_mean_tfidf[word] = np.mean(non_populist_tfidf[word])

    # Get the most frequent terms for both
    populist_mean = Counter(populist_mean_tfidf)
    pop_highest = populist_mean.most_common(20)

    non_populist_mean = Counter(non_populist_mean_tfidf)
    non_pop_highest = non_populist_mean.most_common(20)
    frequent_word_list = []
    populist = []
    non_populist = []
    for instance in pop_highest:
        if instance[0] not in frequent_word_list and instance[0] != "Unnamed: 0":
            frequent_word_list.append(instance[0])
            populist.append(populist_mean[instance[0]])
            non_populist.append(non_populist_mean[instance[0]])
    for instance in non_pop_highest:
        if instance[0] not in frequent_word_list and instance[0] != "Unnamed: 0":
            frequent_word_list.append(instance[0])
            populist.append(populist_mean[instance[0]])
            non_populist.append(non_populist_mean[instance[0]])

    df = pd.DataFrame({'Populist': populist,
                        'Non-populist': non_populist},
                        index = frequent_word_list)

    ax = df.plot.barh(rot=0, xlim=[0.3, 0.7])
    plt.show()
    plt.close("all")

    # Differences in TF-IDF
    differences = dict()
    for pop_score, non_pop_score in zip(populist_mean_tfidf, non_populist_mean_tfidf):
        difference = populist_mean_tfidf[pop_score] - non_populist_mean_tfidf[non_pop_score]
        differences[pop_score] = dict()
        differences[pop_score]["Populist"] = populist_mean_tfidf[pop_score]
        differences[pop_score]["Non-populist"] = non_populist_mean_tfidf[non_pop_score]
        differences[pop_score]["Difference"] = difference

    differences_list = list(differences.items())
    sorted_differences = sorted(differences_list, key=lambda x: x[1]['Difference'], reverse=True)
    sorted_differences = dict(sorted_differences)

    amount_of_words = 10
    counter = 0

    for word in sorted_differences:
        if counter < amount_of_words:
            print(word, "(difference: ", sorted_differences[word]['Difference'], ")")
            print("- Populist score: ", sorted_differences[word]['Populist'])
            print("- Non-populist score: ", sorted_differences[word]['Non-populist'])
            counter += 1
        else:
            break

if get_dict['general'] == True:
    # Count average word and character amount per party
    party_dict = dict()

    # Create a dictionary with general information
    for instance in data:
        # Check whether party exists
        if instance['Party'] not in party_dict: 
            party_dict[instance['Party']] = dict()
            party_dict[instance['Party']]['amount_of_tweets'] = 0
            party_dict[instance['Party']]['amount_of_words'] = 0
            party_dict[instance['Party']]['amount_of_chars'] = 0

        # Increase amount of tweets
        party_dict[instance['Party']]['amount_of_tweets'] += 1
        
        # Calculate average words
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(instance['Tweet'])
        party_dict[instance['Party']]['amount_of_words'] += len(tokens)

        # Calculate average characters
        party_dict[instance['Party']]['amount_of_chars'] += len(instance['Tweet'])
    
    # Print information
    for party in party_dict:
        print(f"for {party}:")
        print(f"Average amount of words p.t.: {party_dict[party]['amount_of_words'] / party_dict[party]['amount_of_tweets']}")
        print(f"Average amount of characters p.t.: {party_dict[party]['amount_of_chars'] / party_dict[party]['amount_of_tweets']}")