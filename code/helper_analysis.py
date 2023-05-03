import re
from collections import Counter
from matplotlib import pyplot as plt
import csv
import nltk

def get_data(outfile, data_type):
    """
    Takes data from csv file and turns it into a list of dictionaries.
    """
    with open('../data/' + outfile + '/' + data_type + '.csv', encoding='utf-8', newline='\n') as csv_file:
        rows = csv.reader(csv_file)
        label_index = 0
        data_list = []
        for row in rows:
            if label_index == 0:
                labels = row
                label_index += 1
                pass
            else:
                current_dict = dict()
                for i, item in enumerate(row):
                    current_dict[labels[i]] = item
                data_list.append(current_dict)
        return(data_list)


def get_frequency_of_words(data, left_out_words=[], stop_words=[]): 
    vocabulary_parties = dict()
    party_list = []
    for instance in data:
        if instance['Party'] not in party_list:
            party_list.append(instance['Party'])
            vocabulary_parties[instance['Party']] = []
        current_vocabulary = vocabulary_parties[instance['Party']]
        # Remove punctuality
        clean_tweet = re.sub(r'[^\w\s]', '', instance['Tweet'])
        word_list = clean_tweet.split()
        for word in word_list:
            if str(word).lower() not in left_out_words and str(word).lower() not in stop_words:
                current_vocabulary.append(str(word).lower())
        vocabulary_parties[instance['Party']] = current_vocabulary

    # Get a list of tuples, with word and frequency
    for party_data in vocabulary_parties:
        counts = Counter(vocabulary_parties[party_data])
        vocabulary_parties[party_data] = counts
    
    # Return dictionary of lists with words and their frequencies
    return(vocabulary_parties)


def get_most_frequent_words(vocabulary_dict, number_of_min_freq=1):
    """
    Get a list of the most frequent words. 
    - vocabulary_dict: dictionary of frequencies: {party: [word: 5, word: 3, word: 1]}
    - number_of_min_freq: the minimum amount of occurences in the text to include the word in the list. Standard frequentie of 1
    """
    most_frequent_dict = dict()
    for party in vocabulary_dict:
        frequency_numbers = []
        frequency_words = []
        frequencies = vocabulary_dict[party]
        for item in frequencies:
            if frequencies[item] >= number_of_min_freq:
                frequency_words.append(item)
                frequency_numbers.append(frequencies[item])
                most_frequent_list = list(zip(frequency_words, frequency_numbers))
                most_frequent_dict[party] = most_frequent_list
    
    # Return dict of lists
    return(most_frequent_dict)


def get_frequency_plot(frequency_dict, max_words=20):
    """
    Prints plots with frequency of words
    """
    # https://www.tutorialspoint.com/frequency-plot-in-python-pandas-dataframe-using-matplotlib
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    for party in frequency_dict:
        frequency_dict[party].sort(key=lambda a: a[1], reverse=True)
        number_list = []
        word_list = []
        for word in frequency_dict[party]:
            number_list.append(word[1])
            word_list.append(word[0])

        if len(number_list) > max_words:
            number_list = number_list[:max_words]
            word_list = word_list[:max_words]

        plt.bar(word_list, number_list)
        plt.suptitle('Words frequency of ' + party)
        plt.show()

def deep_clean(tweet):
    """
    Cleans the given tweet
    """
    # Change hashtags into #HASHTAG
    clean_tweet = re.sub('#[^\s]+','#HASHTAG', tweet)

    # Replace party names with PARTY
    party_name_list = ['VVD', 'CDA', 'D66', 'ChristenUnie', 'CU', 'SP', 'PvdA', 'GroenLinks', 
                       'PvdD', 'FvD', 'JA21', 'SGP', 'DENK', 'Volt', 'BBB', 'BIJ1', 'PVV']
    for party in party_name_list:
        clean_tweet = re.sub(party, 'PARTY', clean_tweet)

    # Return the clean tweet
    return(clean_tweet)


def get_feature_averages(data):
    """
    
    """

def get_graphs(data, data_type, outfile, party_colours, label_colours):
    """
    Get graphs that give insight on the data
    """ 
    # Get label order for later
    parties = []
    pop_non_pop = []
    # Get empty count dicts
    amount_dict_parties = dict()
    amount_dict_pop = dict()

    for instance in data:
        # Party division
        if instance['Party'] not in parties and instance['Party'] != 'ChristenUnie':
            parties.append(instance['Party'])
            amount_dict_parties[instance['Party']] = 0
        elif instance['Party'] == 'ChristenUnie' and 'CU' not in parties:
            parties.append('CU')
            amount_dict_parties['CU'] = 0
        elif instance['Party'] == 'ChristenUnie':
            amount_dict_parties['CU'] += 1
        else:
            amount_dict_parties[instance['Party']] += 1

        # Pop/non-pop division:
        if instance['Populist'] not in amount_dict_pop:
            pop_non_pop.append(instance['Populist'])
            amount_dict_pop[instance['Populist']] = 0
        amount_dict_pop[instance['Populist']] += 1

    # Pie chart for division data party
    amount_list = []
    colours = []
    for party in amount_dict_parties:
        amount_list.append(amount_dict_parties[party])
        colours.append(party_colours[party])

    fig = plt.figure(figsize = (5,5))
    plt.pie(amount_list, labels=parties, colors=colours, autopct='%1.1f%%')
    plt.savefig('../data/' + outfile + '/clean/' + data_type + '_clean_party_division.png')
    plt.close(fig)

    # Pie chart and amounts for division data populist/non-populist data
    amount_list = []
    label_colour = []
    for item in amount_dict_pop:
        amount_list.append(amount_dict_pop[item])
        label_colour.append(label_colours[item])
    for i in range(len(pop_non_pop)):
        if pop_non_pop[i] == 'no':
            pop_non_pop[i] = "Not populist"
        elif pop_non_pop[i] == 'yes':
            pop_non_pop[i] = "Populist"

    fig = plt.figure(figsize = (5,5))
    plt.pie(amount_list, labels = pop_non_pop, autopct='%1.1f%%', colors=label_colour)
    plt.savefig('../data/' + outfile + '/clean/' + data_type + '_clean_pop_division.png')
    plt.close(fig)

    plt.close("all")

    # Get written descriptions
    with open('../data/' + outfile + '/clean/' + data_type + '_clean_description.txt', 'w') as f:
        f.write("Total data size: " + str(len(data)) + '\n'+ '\n')
        f.write("Description per party:" + '\n')
        for item in amount_dict_parties:
            f.write('- ' + str(item).strip("'") + ': ' + str(amount_dict_parties[item]) + '\n')
        f.write("\nDivision populism:\n")
        f.write("- Non-populist: " + str(amount_dict_pop['no']) + '\n')
        f.write("- Populist: " + str(amount_dict_pop['yes']))

def clean_short_tweets(data, min_tweet_length):
    """
    Removes tweets that are smaller than the given minimum length from the dataset.
    """
    clean_data = []
    for instance in data:
        clean_tweet = re.sub(r'[^\w\s]', '', instance['Tweet'])
        word_list = nltk.tokenize.word_tokenize(clean_tweet, language='dutch')
        word_amount = len(word_list)

        if word_amount >= min_tweet_length:
            clean_data.append(instance)
        
    return(clean_data)