----WHAT IS THIS CODE FOR----
This code can be used to collect and analyse twitter data from Dutch political parties for research into linguistic variety, focusing on populism.

----WHAT FILES ARE INCLUDED----
This GitHub contains four different folders:
- Analysis: documents containing feature information that can be used to analyse these features.
- Code: all the code necessary to collect the data and to analyse it.
- Data: contains the datasets and the files necessary to collect the data.
- Results: contains .txt files with the results of the classification models.

CODE: 
1. analyse_data.py: can analyse (such as averages) features stored in the Analysis folder.
2. comparative_tfidf.py
3. function_words.py: contains a function that returns function words. The user can choose what type of function words to include.
4. helper_analysis.py: contains the helper functions for analyse_data.py.
5. helper_classifier.py: contains functions for training and evaluating a classifier. Heavily relies on helper_features.py.
6. helper_data.py: contains functions related to the importing and exporting of data.
7. helper_features.py: contains functions related to feature extraction.
8. process_data.py: code for preprocessing of the data.
9. retrieve_tweets.py: code to pull tweets from Twitter. This code does not work anymore since the Twitter API is now only available for paying members.
10. train_model_readability.py: the code to train a classification model on linguistic simplicity as a feature.
11. train_model_tfidf.py: the code to train a classification model on lexical choice as a feature.

DATA:
1. party_data.json: contains a .json file with information about the different parties. This data can be edited here if needed.
2. last_election_data_small: contains tweets from the current Dutch government, with up to 500 tweets per party up to the 17th of february 2023. The dataset contains 6666 tweets.
3. last_election_data_large: contains all tweets from the current Dutch government up to the 17th of february 2023. The dataset contains 13.791 tweets.
4. coalition_large: contains all tweets from the current coalition, which has been together since 2015. The dataset contains 105.986 tweets.

----WHAT IS THE OUTPUT----
retrieve_tweets: returns a dataset with tweets. The following information is included: index number, party name, date of posting, tweet content, whether the party is considered populist or not, it's role in the House of Representatives (opposition/coalition), horizontal ideology (left/right/middle), vertical ideology (progressive/conservative/middle).

----DATA USED----
The following twitter accounts have been mined: 
 - VVD              @VVD
 - CDA              @cdavandaag
 - D66              @D66
 - ChristenUnie     @christenunie
 - PVV              
 - SP               @SPnl
 - PvdA             @PvdA
 - GroenLinks       @groenlinks
 - PvdD             @PartijvdDieren
 - FvD              @fvdemocratie
 - JA21             @JuisteAntwoord
 - SGP              @SGPnieuws
 - DENK             @DenkNL
 - Volt             @VoltNederland
 - BBB              @BoerBurgerB
 - BIJ1             @PolitiekBIJ1

Working on adding politicians as well.

----NEEDED PACKAGES----
For retrieving the datasets:
- snscraper
- json
- csv


----CONTACT----
m.r.brandsma@student.vu.nl

Retrieve the data:
1. To retrieve the data, run data/retrieve_tweets.py. You can adjust settings here as well.
 
