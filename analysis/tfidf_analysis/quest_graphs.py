import matplotlib.pyplot as plt
import numpy as np

# Code based on https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

# Data
pop_words = ("waanzin", "schandalig", "rechtvaardigheid", "verzet", "tuig", "massaal", "volk", "leugen", "fantastisch", "kamerdebat")
pop_words_label_means = {
    'Pop': [87, 88, 41, 92, 79, 64, 82, 89, 13, 28],
    'Non-pop': [13, 12, 59, 8, 21, 36, 18, 11, 87, 72],
    }

non_pop_words = ("europarlementariër", "twijfel", "thema", "bijdrage", "check", "manifest", "wetsvoorstel", "bijvoorbeeld", "keuze", "rechts")
non_pop_words_label_means = {
    'Pop': [32, 29, 25, 14, 27, 63, 25, 33, 27, 46], 
    'Non-pop': [68, 71, 75, 86, 73, 38, 75, 67, 73, 54],
    }

# Graph settings populist words
width = 0.6  # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
bottom = np.zeros(10)

for type, pop_words_label in pop_words_label_means.items():
    if type == 'Pop':
       color = "#d86154"
    else:
       color = "#6191b1"
    p = ax.bar(pop_words, pop_words_label, width, label=type, bottom=bottom, color=color)
    bottom += pop_words_label
    ax.bar_label(p, label_type='center')

ax.legend(loc="upper right")
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.show()
plt.clf()

# Graph settings non-populist words
width = 0.6  # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
bottom = np.zeros(10)

for type, non_pop_words_label in non_pop_words_label_means.items():
    if type == 'Pop':
       color = "#d86154"
    else:
       color = "#6191b1"
    p = ax.bar(non_pop_words, non_pop_words_label, width, label=type, bottom=bottom, color=color)
    bottom += non_pop_words_label
    ax.bar_label(p, label_type='center')

ax.legend(loc="upper right")
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.show()