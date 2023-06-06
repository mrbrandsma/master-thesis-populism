import helper_analysis as helper
import helper_features as features
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

outfile = 'large_data'        # dataset

data_type = 'large_data_complete'
#data_type = 'dev'
#data_type = 'train'
#data_type = 'test'

get_dict = dict()
get_dict['tfidf'] = False
get_dict['readability'] = False
get_dict['sentiment'] = False
get_dict['general'] = True

# Get data
data = helper.get_data(outfile, data_type)

if get_dict['readability'] == True:
    # Get average scores for:
    populist_readability = {'flesch_kincaid': [], 'SMOG': [], 'words': [], 'sentences': [], 
                        'syllables': [], 'polysyllables': []}
    non_populist_readability = {'flesch_kincaid': [], 'SMOG': [], 'words': [], 'sentences': [], 
                        'syllables': [], 'polysyllables': []}
    party_data = dict()
#    Separate parties for later
    pop_party_list = []
    non_pop_party_list = []

    # Add info on readability to data
    for instance in data:
        detailed_data, flesch_kincaid, smog = features.get_readability_scores(instance['Tweet'])

        if instance['Populist'] == 'yes':
            if instance['Party'] not in pop_party_list:
                pop_party_list.append(instance['Party'])
            populist_readability['words'].append(detailed_data['words'])
            populist_readability['sentences'].append(detailed_data['sentences'])
            populist_readability['syllables'].append(detailed_data['syllables'])
            populist_readability['polysyllables'].append(detailed_data['polysyllables'])
            populist_readability['flesch_kincaid'].append(flesch_kincaid)
            populist_readability['SMOG'].append(smog)
        else:
            if instance['Party'] not in non_pop_party_list:
                non_pop_party_list.append(instance['Party'])
            non_populist_readability['words'].append(detailed_data['words'])
            non_populist_readability['sentences'].append(detailed_data['sentences'])
            non_populist_readability['syllables'].append(detailed_data['syllables'])
            non_populist_readability['polysyllables'].append(detailed_data['polysyllables'])
            non_populist_readability['flesch_kincaid'].append(flesch_kincaid)
            non_populist_readability['SMOG'].append(smog)

        if instance['Party'] not in party_data:
            party_data[instance['Party']] = {'flesch_kincaid': [], 'SMOG': [], 'words': [], 'sentences': [], 
                                            'syllables': [], 'polysyllables': []}
        party_data[instance['Party']]['words'].append(detailed_data['words'])
        party_data[instance['Party']]['sentences'].append(detailed_data['sentences'])
        party_data[instance['Party']]['syllables'].append(detailed_data['syllables'])
        party_data[instance['Party']]['polysyllables'].append(detailed_data['polysyllables'])
        party_data[instance['Party']]['flesch_kincaid'].append(flesch_kincaid)
        party_data[instance['Party']]['SMOG'].append(smog)

    # Calculate score details
    # Create a list of details to find
    detail_list = ['words', 'sentences', 'syllables', 'polysyllables', 'flesch_kincaid', 'SMOG']
    scores = dict()
    scores['Populist'] = dict()
    scores['Non-populist'] = dict()
    for party in party_data:
        scores[party] = dict()
        for detail in detail_list:
            scores[party][detail] = {'average': np.average(party_data[party][detail]), 'sd': np.std(party_data[party][detail]), 'variance': np.var(party_data[party][detail])}

    for detail in detail_list:
        scores['Populist'][detail] = {'average': np.average(populist_readability[detail]), 'sd': np.std(populist_readability[detail]), 'variance': np.var(populist_readability[detail])}
        scores['Non-populist'][detail] = {'average': np.average(non_populist_readability[detail]), 'sd': np.std(non_populist_readability[detail]), 'variance': np.var(non_populist_readability[detail])}

    # Write scores to .txt file
    with open('../analysis/' + outfile + '/' + data_type + '/readability.txt', 'w') as f:
        f.write("In this file you can find the readability scores for populist parties and non-populist parties.")
        for detail in detail_list:
            f.write('-------------------------' + detail + '-------------------------\n')
            f.write('\t \t \t Average \t SD \t Variance \n')
            f.write('POPULIST'+ '\t' + '\t' + str(round(scores['Populist'][detail]['average'], 2)) + '\t' + str(round(scores['Populist'][detail]['sd'], 2)) + '\t' + str(round(scores['Populist'][detail]['variance'], 2)) + '\n')
            for party in party_data:
                if len(party) > 6:
                    f.write('- ' + party + '\t' + str(round(scores[party][detail]['average'], 2)) + '\t' + str(round(scores[party][detail]['sd'], 2)) + '\t' + str(round(scores[party][detail]['variance'], 2)) + '\n')
                else:
                    f.write('- ' + party + '\t' + '\t' + '\t' + str(round(scores[party][detail]['average'], 2)) + '\t' + str(round(scores[party][detail]['sd'], 2)) + '\t' + str(round(scores[party][detail]['variance'], 2)) + '\n')
            f.write('NON-POPULIST'+ '\t' + str(round(scores['Non-populist'][detail]['average'], 2)) + '\t' + str(round(scores['Non-populist'][detail]['sd'], 2)) + '\t' + str(round(scores['Non-populist'][detail]['variance'], 2)) + '\n')
            for party in party_data:
                if party in non_pop_party_list:
                    if len(party) > 6:
                        f.write('- ' + party + '\t' + str(round(scores[party][detail]['average'], 2)) + '\t' + str(round(scores[party][detail]['sd'], 2)) + '\t' + str(round(scores[party][detail]['variance'], 2)) + '\n')
                    else:
                        f.write('- ' + party + '\t' + '\t' + '\t' + str(round(scores[party][detail]['average'], 2)) + '\t' + str(round(scores[party][detail]['sd'], 2)) + '\t' + str(round(scores[party][detail]['variance'], 2)) + '\n')
            f.write('\n')


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