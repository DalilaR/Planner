import warnings
import operator
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for word in test_set.wordlist:
        word_id = test_set.wordlist.index(word)
        test_X, test_lengths = test_set.get_item_Xlengths(word_id)
        try:
            logL = models[word].score(test_X,test_lengths)
        except:
            logL = float("-inf")
        tmp = {word:logL}
        for word2, model in models.items():
            try:
                if word != word2:
                    logL2 = model.score(test_X,  test_lengths)
                    try:
                        tmp[word2]= logL2
                    except:
                        tmp[word2] = float("-inf")
            except:
                tmp[word2] = float("-inf")
                continue
        probabilities.append(tmp)
        best_guess = max(tmp.items(), key=lambda k: k[1])[0]
        guesses.append(best_guess)
    return probabilities, guesses
