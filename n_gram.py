from math import log, exp
import numpy as np

# Define a vocabulary based on the entire corpus
def vocabulary(train_corpus, test_corpus):
    vocab = set()
    
    # Iterate through the training corpus
    for id in train_corpus:
        for token in train_corpus[id]:
            vocab.add(token) 
    
    # Iterate through the test corpus
    for id in test_corpus:
        for token in test_corpus[id]:
            vocab.add(token)
    vocab.add("<s>")  # Special start of sentence token
    vocab.add("</s>")  # Special end of sentence token
    return vocab

# Function for counting the occurrence of n-grams in the training corpus
def count_n_gram(corpus, n):
    
    ngram_counts = {}
    for id in corpus:
        if (n == 1):
            tokens = ["<s>"]
            tokens += corpus[id] + ["</s>"]
        else:
            tokens = ["<s>"] * (n - 1)
            tokens += corpus[id] + ["</s>"] * (n - 1)

        N = len(tokens)
        for i in range(0, N - n + 1):
            str = ''
            for j in range(0, n):
                if (j > 0):
                    str += ' '
                str += tokens[i + j] 
            if(str not in ngram_counts):
                ngram_counts[str] = 0
            ngram_counts[str] += 1
    return ngram_counts

# Function to compute Nc (counts of counts) for smoothing
def Nc(count_dict):
    Nc_ = [0]*(max(list(count_dict.values()))+1)
    for key in count_dict:
        Nc_[count_dict[key]-1] += 1
    return Nc_

# Function to remove null values from an array
def remove_null(lst):
    arr_ = np.array(lst)
    nz_indices = np.nonzero(arr_)[0]  
    if nz_indices[0] != 0:
        nz_indices = np.insert(nz_indices, 0, 0)
    arr_[:-1] = np.interp(np.arange(len(arr_)-1), nz_indices, arr_[nz_indices])
    diff = arr_[-2] - arr_[nz_indices[-2]]
    arr_[-1] = arr_[-2] + diff 
    arr_ = list(arr_)
    return arr_

# Function to compute new counts for Good Turing smoothing
def new_count(Nc, c):
    c_star = ((c+1) * Nc[c]) / (Nc[c-1])
    return c_star

# Function for training the n-gram model with or without smoothing
def train_n_gram(corpus, n, vocab=0, smoothing=False, how='Laplace', k=1):
    """
    how: Types of smoothing = {'Laplace','Add_k','Good Turing'}

    - Must specify k with how='Add_k' or it will implement Laplace smoothing
    """
    prob_n_words = {}
    ngram_counts = count_n_gram(corpus, n)
    
    if how == 'Good Turing':
        Nc_ = Nc(ngram_counts)
        Nc_ = remove_null(Nc_)
        C_star = [Nc_[0]]
        for c in range(1, len(Nc_)):
            C_star.append(new_count(Nc_, c))
        N_train_GT = sum(list(ngram_counts.values()))
        for key in ngram_counts:
            P_GT = (C_star[ngram_counts[key]]) / N_train_GT
            prob_n_words[key] = P_GT
        prob_n_words['0*#'] = C_star[0] / N_train_GT  # Probability for words with count=0 in test but not in train
        
    else:
        if how == 'Laplace':
            k = 1
        if n > 1:
            n1gram_counts = count_n_gram(corpus, n - 1)
            unigram_counts = count_n_gram(corpus, 1)
            
            for key in ngram_counts:
                words = key.split(" ")
                prev = ''
                for j in range(0, n - 1):
                    if (j > 0):
                        prev += ' '
                    prev += words[j] 
                
                if prev in n1gram_counts:
                    if smoothing is False:
                        prob_n_words[key] = ngram_counts[key] / n1gram_counts[prev]
                    else:
                        prob_n_words[key] = (ngram_counts[key] + k) / (n1gram_counts[prev] + k * vocab)
                elif n > 2:
                    if smoothing is False:
                        prob_n_words[key] = ngram_counts[key] / unigram_counts[words[n - 2]]
                    else:
                        prob_n_words[key] = (ngram_counts[key] + k) / (unigram_counts[words[n - 2]] + k * vocab)
                else:
                    if smoothing is False:
                        prob_n_words[key] = 0
                    else:
                        prob_n_words[key] = 1 / vocab
        else:
            N_train = 0
            for key in ngram_counts:
                N_train += ngram_counts[key]

            for key in ngram_counts:
                if smoothing is False:
                    prob_n_words[key] = ngram_counts[key] / N_train
                else:
                    prob_n_words[key] = (ngram_counts[key] + k) / (N_train + k * vocab)

    return prob_n_words

# Function for testing the n-gram model with or without smoothing
def test_n_gram(test_data, n, prob_words, epsilon=1e-15, Vocabulary=0, smoothing=False, how='Laplace', k=1, processed_corpus=None):
    """
    how: Types of smoothing = {'Laplace','Add_k','Good Turing'}

    - Must specify k with how='Add_k' or it will implement Laplace smoothing
    """
    perplexity = {}
    if how == 'Good Turing':
        for id in test_data:
            if n == 1:
                tokens = ["<s>"]
            else: 
                tokens = ["<s>"] * (n - 1)
            tokens += test_data[id]
            if n == 1:
                tokens += ["</s>"]
            else:
                tokens += ["</s>"] * (n - 1)
            data_len = len(tokens)
            log_p = 0
            for i in range(0, data_len - n + 1):
                str = ""
                for j in range(i, i + n - 1):
                    str += tokens[j] + " "
                str += tokens[i + n - 1]
                if str in prob_words:
                    log_p += log(prob_words[str])
                else:
                    log_p += log(prob_words['0*#'])
            log_p = (-log_p) / data_len
            perplexity[id] = exp(log_p)
        avg_perplexity = np.mean(list(perplexity.values()))

    else:
        if how
