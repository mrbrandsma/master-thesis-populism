import re
from collections import Counter
from matplotlib import pyplot as plt
import csv

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