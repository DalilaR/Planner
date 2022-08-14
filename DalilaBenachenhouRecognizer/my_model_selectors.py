import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        '''Initial state occupation probabilities = numStates
        
        Transition probabilities = numStates*(numStates - 1)

        Emission probabilities = numStates*numFeatures*2 = numMeans+numCovars
        #n_components = number of states
        #n_features = number of featues
        #startprob_.size = initial state'''
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        n_states = self.min_n_components
        best_model = self.base_model(n_states)
        try:
            logL1 = best_model.score(self.X,self.lengths)
            p = n_states**2+(2*n_states*(best_model.n_features))
            BIC_best_model = -2 * logL1 + p * np.log(len(self.X))
            for i in range(self.min_n_components+1,self.max_n_components):

                try:
                    model2 = self.base_model(i)
                    logL2 = model2.score(self.X,self.lengths)
                    p = i**2 + (2*i * (model2.n_features))
                    BIC2 = -2 * logL2+ p * np.log(len(self.X))
                    if BIC2 < BIC_best_model:
                        best_model = model2
                except:
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, i))
                    #return None
                    return best_model
        except:
            return None
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        best_BIC = -1000000000

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_model = self.base_model(self.min_n_components)
        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                model1 = self.base_model(i)
                logL1 = model1.score(self.X, self.lengths)
                all_but_this_word = 0
                word_count = 0
                for word in self.hwords:
                    if word != self.this_word:
                        X, lengths = self.hwords[word]
                        #model2 = self.base_model(i)
                        all_but_this_word += model1.score(X, lengths)
                        word_count += 1
                DIC = logL1 - (all_but_this_word/ ( word_count -1))
                if DIC > best_BIC:
                    best_BIC = DIC
                    best_model = model1
            except:
                #warnings.filterwarnings("ignore", category=DeprecationWarning)
                #warnings.filterwarnings("ignore", category=RuntimeWarning)
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, i))
                return best_model
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_model = self.base_model(self.min_n_components)
        word_sequences = self.sequences
        l = len(word_sequences)
        if l > 2:
            split_method = KFold()
            best_value = -10000000
            try:
                for i in range(self.min_n_components,self.max_n_components+1):
                    average_log = 0
                    k = 0
                    for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                        X_train, lengths_train = combine_sequences(cv_train_idx, word_sequences)
                        X_test,lengths_test = combine_sequences(cv_test_idx, word_sequences)
                        model = GaussianHMM(n_components=i, n_iter=1000, covariance_type="diag",
                                    random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        average_log += model.score(X_test,lengths_test)
                        k += 1
                    average_log /= k
                    if best_value < average_log:
                        best_value = average_log
                        best_model = self.base_model(i)
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, i))
                    return best_model
        else:
            l = len(self.sequences)
            best_value = -10000000
            if l == 2:
                X_train, lengths_train = combine_sequences([0], word_sequences)
                X_test, lengths_test = combine_sequences([1], word_sequences)
                try:
                    for i in range(self.min_n_components,self.max_n_components+1):
                        model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        model2 = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X_test, lengths_test)
                        average_log = (model.score(X_test,lengths_test)+ model2.score(X_train, lengths_train))/2
                        if average_log > best_value:
                            best_value = average_log
                            best_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, i))
                        return best_model
            if l == 1:
                for i in range(self.min_n_components,self.max_n_components+1):
                    try:
                        model = self.base_model(i)
                        average_log = model.score(self.X,self.lengths)
                        if best_value < average_log:
                            best_model = model
                            best_value = average_log
                    except:
                        if self.verbose:
                            print("failure on {} with {} states".format(self.this_word, i))
                            return best_model
        return best_model
