import nltk
import numpy as np
import spacy
import math
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
#nltk.download('punkt')
#nltk.download('cmudict')

### TF-IDF #####################
def get_function_words(wanted_types=['articles', 'prepositions', 'quantifiers', 'conjunctions', 'pronouns', 'auxiliary verbs', 'adverbs', 'modifiers', 'interjections']):
    """
    Gets a list of function words. Based on: https://github.com/Yoast/javascript/blob/develop/packages/yoastseo/src/researches/dutch/functionWords.js
    """    
    function_words = {
    "articles": ["de", "het", "een", "der", "des", "den"],
    "prepositions": ["à", "aan", "aangaande", "achter", "behalve", "behoudens", "beneden", "benevens", "benoorden", "benoordoosten", "benoordwesten",
                "beoosten", "betreffende", "bewesten", "bezijden", "bezuiden", "bezuidoosten", "bezuidwesten", "bij", "binnen", "blijkens", "boven", "bovenaan",
                "buiten", "circa", "conform", "contra", "cum", "dankzij", "door", "gedurende", "gezien", "in", "ingevolge", "inzake", "jegens", "krachtens",
                "langs", "luidens", "met", "middels", "na", "naar", "naast", "nabij", "namens", "nevens", "niettegenstaande", "nopens", "om",
                "omstreeks", "omtrent", "onder", "onderaan", "ongeacht", "onverminderd", "op", "over", "overeenkomstig", "per", "plus", "post",
                "richting", "rond", "rondom", "spijts", "staande", "te", "tegen", "tegenover", "ten", "ter", "tijdens", "tot", "tussen",
                "uit", "van", "vanaf", "vanuit", "versus", "via", "vis-à-vis", "volgens", "voor", "voorbij", "wegens", "zijdens",
                "zonder", ],
    "quantifiers":  ["eén", "één", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen", "tien", "elf", "twaalf", "dertien",
	                "veertien", "vijftien", "zestien", "zeventien", "achttien", "negentien", "twintig", "honderd", "honderden", "duizend", "duizenden", "miljoen",
	                "miljoenen", "biljoen", "biljoenen", "eerste", "tweede", "derde", "vierde", "vijfde", "zesde", "zevende", "achtste", "negende",
	                "tiende", "elfde", "twaalfde", "dertiende", "veertiende", "vijftiende", "zestiende", "zeventiende",
	                "achttiende", "negentiende", "twinstigste", "alle", "sommige", "sommigen", "weinig", "weinige", "weinigen", "veel", "vele", "velen", "geen", "beetje",
	                "elke", "elk", "genoeg", "meer", "meest", "meeste", "meesten", "paar", "zoveel", "enkele", "enkelen", "zoveelste", "hoeveelste",
	                "laatste", "laatsten", "iedere", "allemaal", "zekere", "ander", "andere", "gene", "enig", "enige", "verscheidene",
	                "verschillende", "voldoende", "allerlei", "allerhande", "enerlei", "enerhande", "beiderlei", "beiderhande", "tweeërlei", "tweeërhande",
	                "drieërlei", "drieërhande", "velerlei", "velerhande", "menigerlei", "menigerhande", "enigerlei", "enigerhande", "generlei", "generhande"],
    "conjunctions": ["en", "alsmede", "of", "ofwel", "en/of", "zowel", "evenmin", "zomin", "hetzij", "vermits", "dewijl", "dorodien", "naardien", "nademaal", "overmits", "wijl", "eer",
	                "eerdat", "aleer", "vooraleer", "alvorens", "totdat", "zolang", "sinds", "sedert", "ingeval", "tenware", "alhoewel",
	                "hoezeer", "uitgezonderd", "zoverre", "zover", "naargelang", "naarmate", "alsof"],
    "pronouns": ["ik", "je", "jij", "hij", "ze", "we", "wij", "jullie", "zij", "u", "ge", "gij", "men", "mij", "jou", "hem", "haar", "hen", "hun", "uw", 
                "dit", "dat", "deze", "die", "zelf", "mijn", "mijne", "jouw", "jouwe", "zijne", "hare", "ons", "onze", "hunne", "uwe", "elkaars", "elkanders", 
                "zijn", "het", "mezelf", "mijzelf", "jezelf", "jouzelf", "zichzelf", "haarzelf", "hemzelf", "onszelf", "julliezelf",
	            "henzelf", "hunzelf", "uzelf", "zich", "mekaar", "elkaar", "elkander", "mekander", 
                "iedereen", "ieder", "eenieder", "alleman", "allen", "alles", "iemand", "niemand", "iets", "niets", "menigeen", "ieders", "aller", "iedereens", "eenieders", 
                "welke", "welk", "wat", "wie", "wiens", "wier"],
    "auxiliary verbs": ["word", "wordt", "werd", "werden", "ben", "bent", "is", "was", "waren", "worden", "zijn", "heb", "hebt", "heeft", "hadden", "had", "kun", "kan", "kunt", "kon", "konden", "mag",
	                "mocht", "mochten", "dien", "dient", "diende", "dienden", "moet", "moest", "moesten", "ga", "gaat",
	                "ging", "gingen", "hebben", "kunnen", "mogen", "dienen", "moeten", "gaan", "blijkt", "blijk", "bleek", "bleken", "gebleken", "dunkt", "dunk", "dunkte", "dunkten",
	                "gedunkt", "heet", "heette", "heetten", "geheten", "lijkt", "lijk", "geleken", "leek", "leken",
	                "schijn", "schijnt", "scheen", "schenen", "toescheen", "toeschijnt", "toeschijn", "toeschenen", "blijken", "dunken", "heten", "lijken", "schijnen", "toeschijnen"],
    "adverbs": ["hoe", "waarom", "waar", "hoezo", "hoeveel", "daaraan", "daarachter", "daaraf", "daarbij", "daarbinnen", "daarboven", "daarbuiten", "daardoorheen",
            "daarheen", "daarin", "daarjegens", "daarmede", "daarnaar", "daarnaartoe", "daaromtrent", "daaronder", "daarop", "daarover",
            "daaroverheen", "daarrond", "daartegen", "daartussen", "daartussenuit", "daaruit", "daarvan", "daarvandaan", "eraan", "erachter",
            "erachteraan", "eraf", "erbij", "erbinnen", "erboven", "erbuiten", "erdoor", "erdoorheen", "erheen", "erin", "erjegens", "ermede",
            "ermee", "erna", "ernaar", "ernaartoe", "ernaast", "erom", "eromtrent", "eronder", "eronderdoor", "erop", "eropaf", "eropuit", "erover",
            "eroverheen", "errond", "ertegen", "ertegenaan", "ertoe", "ertussen", "ertussenuit", "eruit", "ervan", "ervandaan", "ervandoor", "ervoor",
            "hieraan", "hierachter", "hieraf", "hierbij", "hierbinnen", "hierboven", "hierbuiten", "hierdoor", "hierdoorheen", "hierheen", "hierin",
            "hierjegens", "hierlangs", "hiermede", "hiermee", "hierna", "hiernaar", "hiernaartoe", "hiernaast", "hieromheen", "hieromtrent",
            "hieronder", "hierop", "hierover", "hieroverheen", "hierrond", "hiertegen", "hiertoe", "hiertussen", "hiertussenuit", "hieruit", "hiervan",
            "hiervandaan", "hiervoor", "vandaan", "waaraan", "waarachter", "waaraf", "waarbij", "waarboven", "waarbuiten", "waardoorheen",
            "waarheen", "waarin", "waarjegens", "waarmede", "waarna", "waarnaar", "waarnaartoe", "waarnaast", "waarop", "waarover", "waaroverheen",
            "waarrond", "waartegen", "waartegenin", "waartoe", "waartussen", "waartussenuit", "waaruit", "waarvan", "waarvandaan", "waarvoor",
            "daar", "hier", "ginder", "daarginds", "ginds", "ver", "veraf", "ergens", "nergens", "overal", "dichtbij",
            "kortbij", "af", "heen", "mee", "toe", "achterop", "onderin", "voorin", "bovenop",
	        "buitenop", "achteraan", "onderop", "binnenin", "tevoren"],
    "modifiers": ["zeer", "erg", "redelijk", "flink", "tikkeltje", "bijzonder", "ernstig", "enigszins",
            "zo", "tamelijk", "nogal", "behoorlijk", "zwaar", "heel", "hele", "reuze", "buitengewoon",
            "ontzettend", "vreselijk", "nieuw", "nieuwe", "nieuwer", "nieuwere", "nieuwst", "nieuwste", "oud", "oude", "ouder", "oudere",
            "oudst", "oudste", "vorig", "vorige", "goed", "goede", "beter", "betere", "best", "beste", "groot", "grote", "groter", "grotere",
            "grootst", "grootste", "makkelijk", "makkelijke", "makkelijker", "makkelijkere", "makkelijkst", "makkelijste", "gemakkelijk",
            "gemakkelijke", "gemakkelijker", "gemakkelijkere", "gemakkelijkst", "gemakkelijste", "simpel", "simpele", "simpeler", "simpelere",
            "simpelst", "simpelste", "snel", "snelle", "sneller", "snellere", "snelst", "snelste", "verre", "verder", "verdere", "verst",
            "verste", "lang", "lange", "langer", "langere", "langst", "langste", "hard", "harde", "harder", "hardere", "hardst", "hardste",
            "minder", "mindere", "minst", "minste", "eigen", "laag", "lage", "lager", "lagere", "laagst", "laagste", "hoog", "hoge", "hoger",
            "hogere", "hoogst", "hoogste", "klein", "kleine", "kleiner", "kleinere", "kleinst", "kleinste", "kort", "korte", "korter", "kortere",
            "kortst", "kortste", "herhaaldelijke", "directe", "ongeveer", "slecht", "slechte", "slechter", "slechtere", "slechtst",
            "slechtste", "zulke", "zulk", "zo'n", "zulks", "er", "extreem", "extreme", "bijbehorende", "bijbehorend", "niet"],
    "interjections": ["oh", "wauw", "hèhè", "hè", "hé", "au", "ai", "jaja", "welja", "jawel", "ssst", "heremijntijd", "hemeltjelief", "aha",
	            "foei", "hmm", "nou", "nee", "tja", "nja", "okido", "ho", "halt", "komaan", "komop", "verrek", "nietwaar", "brr", "oef",
	            "ach", "och", "bah", "enfin", "afijn", "haha", "hihi", "hatsjie", "hatsjoe", "hm", "tring", "vroem", "boem", "hopla"],
    }

    final_list = []
    for type in wanted_types:
        final_list = final_list + function_words[type]

    return(final_list)


def lemmatizer(data, stopwords, pipeline):
    """
    Returnes a lemmatized tweets without stopwords.
    """
    texts = [text.replace("\n", "").strip() for text in data]
    docs = pipeline.pipe(texts)
    cleaned_lemmas = [[t.lemma_ for t in doc if t.lemma_ not in stopwords] for doc in docs]

    cleaned_data = []
    for tweet in cleaned_lemmas:
        clean_tweet = ""
        for word in tweet:
            clean_tweet = clean_tweet + " " + word 
        cleaned_data.append(clean_tweet)

    return cleaned_data


def remove_words(path):
    """
    Takes a .txt-file of words and keeps them out of the analysis.
    """
    word_list = []

    # Open ignore-words file
    with open(path) as f:
        words = f.readlines()
    
    for word in words:
        clean_word = word.lower().strip('\n')
        word_list.append(clean_word)
    
    return(word_list)


def get_stop_words(data, 
              freq_threshold):
    """
    Returns a sublist of words from the data, dependent on the frequency of the words.
    Creates a list to compare tweets to at a later stage.
    """
    word_list = []

    # Collect all tokens
    all_tokens = []
    for tweet in data:
        clean_tweet = tweet.lower()
        clean_tweet = clean_tweet = re.sub(r'[^\w\s]', '', clean_tweet)
        token_list = nltk.tokenize.word_tokenize(clean_tweet, language='dutch')

        for token in token_list:
            all_tokens.append(token)
    
    freq_word_counter = Counter(all_tokens)

    for word, count in freq_word_counter.items():
        if count > freq_threshold:
            word_list.append(word)
    
    return(word_list)


def dataframe_coefficients(classifier, vect, dataset, outfile, top_features=20):
    """
    Based on Myrthe Reuver's code at https://github.com/myrthereuver/claims-reproduction/blob/main/analysis_reproduction/notebooks/SVM_UKPdata.ipynb
    """
    f = open(f'../results/{dataset}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
    f.write('\n--------------TF-IDF WORDS---------------\n')


    feature_names = vect.get_feature_names_out()
    coefs_with_fns = sorted(zip(classifier.coef_[0], feature_names)) 
    df=pd.DataFrame(coefs_with_fns)
    df.columns='coefficient','word'
    df.sort_values(by='coefficient')

    positive = "Non-populist"
    negative = "Populist"
    
#     df_negativeclass = df[df['coefficient'] < 0]
    df_negativeclass = df.sort_values(by='coefficient', ascending=False)
    df_negative_top = df_negativeclass[:top_features]
    
#     df_positiveclass = df[df['coefficient'] > 0]
    df_positiveclass = df.sort_values(by='coefficient')
    df_positive_top = df_positiveclass[:top_features]
    
    print(negative)
    print(df_negative_top)
    print("//////")
    print("######")
    print("\\\\\\")
    print(positive)
    print(df_positive_top)
    
    negative_as_string = df_negative_top.to_string(header=False, index=False)
    positive_as_string = df_positive_top.to_string(header=False, index=False)

    f.write('\n')
    f.write(negative)
    f.write('\n')
    f.write(negative_as_string)
    f.write('\n')
    f.write("//////\n")
    f.write("######\n")
    f.write("\\\\\\\n")
    f.write(positive)
    f.write('\n')
    f.write(positive_as_string)
    f.write('\n')

    f.write('\n----------------------------------------------\n')

    
    return df, df_positive_top, df_negative_top


def bag_of_words(data_list, min_freq_threshold, stopwords):
    """
    Returns a bag of words of the given data list. 

    Input:
    - data_list: a list of tweets
    - min_freq_threshold: the smallest amount in which a word has to occur to be 
    included in the bag-of-words
    - stopwords: a list of words that should be ignored
    
    Returns:
    - tweet_vectorizer: the structure that should be used to transform data
    """
    # Create the settings
    # Creates a numerical matrix where each row represents a document and each column
    # represents a unique word
    tweet_vectorizer = CountVectorizer(min_df=min_freq_threshold, 
                                    tokenizer=nltk.word_tokenize, 
                                    stop_words=stopwords
                                   )
    
    # Lemmatize data
    language_pipeline = spacy.load("nl_core_news_sm")
    clean_data = lemmatizer(data_list, stopwords, language_pipeline)

    # Change data into a bag-of-words representation
    bag_of_words = tweet_vectorizer.fit_transform(clean_data)

    return(tweet_vectorizer, bag_of_words)


def tfidf_transform(data_vector):
    """
    Changes vector to TF-IDF values.
    Returns a vector with TF-IDF weights.
    """
    # Create a transformer
    tfidf_transformer = TfidfTransformer()
    tfidf_vectors = tfidf_transformer.fit_transform(data_vector)
    
    # Return the weighted vectors
    return(tfidf_vectors)


def get_tfidf(train_data, test_data, min_frequency, stopwords):
    """
    Gets TF-IDF vectors for both train and test data
    """
    print("Extracting feature: TF-IDF")
    #tfidf = TfidfVectorizer(stop_words=stopwords)
    # Clean data: get lemmatized words
    language_pipeline = spacy.load("nl_core_news_sm")
    clean_train_data = lemmatizer(train_data, stopwords, language_pipeline)
    clean_test_data = lemmatizer(test_data, stopwords, language_pipeline)

    # Get the bag of words of dataset
    tweet_vectorizer, training_count_vectors = bag_of_words(clean_train_data, min_frequency, stopwords)

    # Transform test data
    test_count_vectors = tweet_vectorizer.transform(clean_test_data)

    # Get TF-IDF vectors
    train_tfidf_vectors = tfidf_transform(training_count_vectors)
    test_tfidf_vectors = tfidf_transform(test_count_vectors)

    # Get the terms with the highest frequencies
    feature_names = tweet_vectorizer.get_feature_names()
    
    # For train
    dense_representation = train_tfidf_vectors.todense()
    dense_list = dense_representation.tolist()
    tfidf_train_data = pd.DataFrame(dense_list, columns = feature_names)
    tfidf_train_data.to_csv("../analysis/large_data/train/tfidf")

    # For test
    dense_representation = test_tfidf_vectors.todense()
    dense_list = dense_representation.tolist()
    tfidf_test_data = pd.DataFrame(dense_list, columns = feature_names)
    tfidf_test_data.to_csv("../analysis/large_data/test/tfidf")
    
    # Taken from: https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score
    #train_feature_array = np.array(tweet_vectorizer.get_feature_names())
    #tfidf_sorting = np.argsort(train_tfidf_vectors.toarray()).flatten()[::-1]
    #n = 20
    #top_n = train_feature_array[tfidf_sorting][:n]
    #print(top_n)

    return(train_tfidf_vectors, test_tfidf_vectors, tweet_vectorizer)

#################################

def get_features(data, features_to_extract):
    """
    Extracts the features and adds them to a vector
    """

    # For every instance in the data, retrieve the features
    features = []
    for i, instance in enumerate(data):
        feature_dict = dict()

        # TAKES VERY LONG. DOES NOT WORK.
        #if features_to_extract['tf-idf'] == True:
            # TFIDF features
            #tfidf_dense = tfidf.toarray()
            #feature_dict['TF-IDF'] = tfidf_dense[i]

        # Readability features
        detailed_information, flesch_kincaid, smog = get_readability_scores(instance)

        if features_to_extract['flesch-kincaid'] == True:
            feature_dict['Flesch-Kincaid Formula'] = flesch_kincaid

        if features_to_extract['smog'] == True:
            feature_dict['SMOG Formula'] = smog
        
        if features_to_extract['word_amount'] == True:
            feature_dict['Word amount'] = detailed_information['words']
        
        if features_to_extract['sentence_amount'] == True:
            feature_dict['Word amount'] = detailed_information['sentences']

        if features_to_extract['syllable_amount'] == True:
            feature_dict['Syllable amount'] = detailed_information['syllables']

        if features_to_extract['polysyllable_amount'] == True:
            feature_dict['Polysyllable amount'] = detailed_information['polysyllables']
        
        # Append to data list
        features.append(feature_dict)
    
    return(features)


def get_readability_scores(tweet):
    # Calculate word amount
    clean_tweet = re.sub(r'[^\w\s]', '', tweet)
    word_list = nltk.tokenize.word_tokenize(clean_tweet, language='dutch')
    word_amount = len(word_list)

    # Calculate sentence amount
    sent_list = nltk.tokenize.sent_tokenize(tweet)
    sent_amount = len(sent_list)

    # Calculate (poly)syllable amount
    clean_word_list = word_list
    if 'HASHTAG' in clean_word_list:
        clean_word_list.remove('HASHTAG')
    if 'PARTY' in clean_word_list:
        clean_word_list.remove('PARTY')
    if 'USER' in clean_word_list:
        clean_word_list.remove('USER')
    if 'LINK' in clean_word_list:
        clean_word_list.remove('LINK')
    syl_amount = 0
    pol_syl_amount = 0
    for word in clean_word_list:
        if word.isalpha() == True:
            syl_amount_word = syllable_count(word)
            if syl_amount_word > 2:
                pol_syl_amount += 1
            syl_amount += syl_amount_word
        
    detailed_data = {'words': word_amount, 'sentences': sent_amount, 'syllables': syl_amount, 'polysyllables': pol_syl_amount}

    # Flesch-Kincaid Reading Ease
    flesch_kincaid = 206.835 - 1.015 * (word_amount / sent_amount) - 84.6 * (syl_amount / word_amount)

    # SMOG Index
    smog = 1.0430 * math.sqrt(pol_syl_amount * (30 / sent_amount)) + 3.1291

    return(detailed_data, flesch_kincaid, smog)

def syllable_count(word):
    """
    Counts the amount of syllables in a word.
    Works for Dutch.
    Inspired by https://stackoverflow.com/questions/46759492/syllable-count-in-python
    """
    word = word.lower()
    count = 0
    vowels = "aáäeéëiéëoóöuúüyýÿ"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if count == 0:
        count += 1
    return count


########################

def combine_sparse_and_dense_features(dense_vectors, # tfidf as vectors
                                      sparse_features # the traditional features as a list
                                      ):
    """
    Function that takes sparse and dense feature representations and appends their vector representation
    :return combined_vectors: list of arrays in which sparse and dense vectors are concatenated
    """
    combined_vectors = []

    # Turn the traditional features into an array
    sparse_vectors = np.array(sparse_features.toarray())

    # Combine the sparse and dense features
    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)
    
    # Return a combined vector of features
    return combined_vectors


def create_vectorizer(features # The features that the model should be trained on
                     ):
    """
    Transforms the data into a vector
    :return vec: the way to vectorize
    :return features_vectorized: the features as a vector
    """
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(features)
    
    return vec, features_vectorized