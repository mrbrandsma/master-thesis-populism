# How to use the code

----WHAT IS THIS CODE FOR----
This code can be used to collect and analyse twitter data from Dutch political parties for research into linguistic variety, focusing on populism.

----WHAT FILES ARE INCLUDED----
This GitHub contains two different folders: a folder with Python code and a folder with datasets.

CODE: 
1. modules.py: contains helper modules: clean_tweets, clean_date, get_tweets.
2. retrieve_tweets.py: mines Twitter for tweets, no API keys necessary. Settings can be changed here.

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
 
