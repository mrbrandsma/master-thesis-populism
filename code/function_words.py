# These lists are based on
# https://github.com/Yoast/javascript/blob/develop/packages/yoastseo/src/researches/dutch/functionWords.js


def get_function_words(wanted_types=['articles', 'prepositions', 'quantifiers', 'conjunctions', 'pronouns', 'auxiliary verbs', 'adverbs', 'modifiers', 'interjections']):
    """
    This function returns a list of function words to remove.
    The user can specify what types of function words they want returned.
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