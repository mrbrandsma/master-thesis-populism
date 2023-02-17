# How to use the code

----WHAT IS THIS CODE FOR----
This code can be used to collect and analyse twitter data from Dutch political parties for research into linguistic variety, focusing on populism.

----WHAT FILES ARE INCLUDED----
This GitHub contains two different folders: a folder with Python code and a folder with datasets.

CODE: 
1. modules.py: contains helper modules
2. retrieve_tweets.py: mines Twitter for tweets, no API keys necessary. Will work on a simple way to change the settings. For now it's mixed in the code.
3. data_notes.txt: just some personal notes to use besides my coding. Will delete later.

DATA:
1. party_data.json: contains a .json file with information about the different parties. This data can be edited here if needed.
2. last_election_data_small: contains tweets from the current Dutch government, with up to 500 tweets per party up to the 17th of february 2023. The dataset contains 6666 tweets.
3. last_election_data_large: contains all tweets from the current Dutch government up to the 17th of february 2023. The dataset contains 13.791 tweets.
4. coalition_large: contains all tweets from the current coalition, which has been together since 2015. The dataset contains x tweets.

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

One dataset contains tweets from the current government (since march 26th 2021). There is a small version and a large version.
The other dataset contains tweets from the previous government as well (since march 15th 2017). --> TO DO

----CONTACT----
m.r.brandsma@student.vu.nl

Retrieve the data:
1. To retrieve the data, run data/retrieve_tweets.py. You can adjust settings here as well.
 
