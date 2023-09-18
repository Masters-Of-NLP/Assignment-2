from math import log,exp
import numpy as np

def vocabulary(train_corpus, test_corpus):
    vocab = set()
    
    for id in train_corpus:
        for token in train_corpus[id]:
            vocab.add(token) 
    
    for id in test_corpus:
        for token in test_corpus[id]:
            vocab.add(token)
    vocab.add("<s>")
    vocab.add("</s>")
    return vocab

def count_n_gram(corpus, n):
    
    ngram_counts = {}
    for id in corpus:
        if (n == 1):
            tokens = ["<s>"]
            tokens += corpus[id] + ["</s>"]
        else:
            tokens = ["<s>"]*(n - 1)
            tokens += corpus[id] + ["</s>"]*(n-1)

        N = len(tokens)
        for i in range(0, N-n+1):
            str = ''
            for j in range(0, n):
                if (j >0):
                    str += ' '
                str += tokens[i + j] 
            if(str not in ngram_counts):
                ngram_counts[str] = 0
            ngram_counts[str] += 1
    return ngram_counts



def train_n_gram(corpus, n, vocab = 0, smoothing = False, how='Laplace', k=1):
    """
    how: Types of smoothing = {'Laplace','Add_k','Good Turing'}

    - Must specify k with how='Add_k' or it will implement Laplace smoothing
    """
    prob_n_words = {}
    ngram_counts = count_n_gram(corpus, n)
    if(how=='Laplace'):
        k=1
    if(n > 1):
        n1gram_counts = count_n_gram(corpus, n-1)
        unigram_counts = count_n_gram(corpus, 1)
        
        for key in ngram_counts:
            words = key.split(" ")
            prev = ''
            for j in range(0, n-1):
                if (j >0):
                    prev += ' '
                prev += words[j] 
            
            if (prev in n1gram_counts):
                if(smoothing == False):
                    prob_n_words[key] = ngram_counts[key] / n1gram_counts[prev]
                else:
                    prob_n_words[key] = (ngram_counts[key] + k) / (n1gram_counts[prev] + k*vocab)
            elif (n >2):
                if(smoothing == False):
                    prob_n_words[key] = ngram_counts[key] / unigram_counts[words[n-2]]
                else:
                    prob_n_words[key] = (ngram_counts[key] + k) / (unigram_counts[words[n-2]] + k*vocab)
            else:
                if(smoothing == False):
                    prob_n_words[key] = 0
                else:
                    prob_n_words[key] = 1/vocab
    else:
        N_train = 0
        for key in ngram_counts:
            N_train += ngram_counts[key]

        for key in ngram_counts:
            if(smoothing == False):
                prob_n_words[key] = ngram_counts[key] / N_train
            else:
                prob_n_words[key] = (ngram_counts[key] + k) / (N_train + k*vocab)
    return prob_n_words

def test_n_gram(test_data, n, prob_words, epsilon=1e-15, Vocabulary=0,smoothing=False,how='Laplace',k=1,processed_corpus=None):
    """
    how: Types of smoothing = {'Laplace','Add_k','Good Turing'}

    - Must specify k with how='Add_k' or it will implement Laplace smoothing
    """
    perplexity={}
    if(how=='Laplace'):
        k=1
    if(smoothing==False):
        for id in test_data:
            if(n==1):
                tokens=["<s>"]
            else: 
                tokens=["<s>"]*(n-1)
            tokens+=test_data[id]
            if(n==1):
                tokens+=["</s>"]
            else:
                tokens+=["</s>"]*(n-1)
            data_len = len(tokens)
            log_p=0
            for i in range(0,data_len-n+1):
                str=""
                for j in range(i,i+n-1):
                    str+=tokens[j]+" "
                str+=tokens[i+n-1]
                if str in prob_words:
                    log_p += log(prob_words[str])
                else:
                    log_p += log(epsilon)
            log_p=(-log_p)/data_len
            perplexity[id]=exp(log_p)
        
        avg_perplexity=np.mean(list(perplexity.values()))

    else:
        n1gram_counts=count_n_gram(processed_corpus,n-1)
        unigram_counts=count_n_gram(processed_corpus,1)

        for id in test_data:
            if(n==1):
                tokens=["<s>"]
            else: 
                tokens=["<s>"]*(n-1)
            tokens+=test_data[id]
            if(n==1):
                tokens+=["</s>"]
            else:
                tokens+=["</s>"]*(n-1)
            data_len = len(tokens)
            log_p=0
            for i in range(0,data_len-n+1):
                prev=""
                for j in range(i,i+n-1):
                    prev+=tokens[j]+" "
                str=prev+tokens[i+n-1]
                if str in prob_words:
                    log_p += log(prob_words[str])
                else:
                    if(prev in n1gram_counts):
                        log_p += log(k/(n1gram_counts[prev]+k*Vocabulary))
                    elif(n>1 and tokens[i+n-2]=="<s>"):
                        log_p += log(k/(unigram_counts["<s>"]+k*Vocabulary))
                    else:
                        log_p += log(1/Vocabulary)
            log_p=(-log_p)/data_len
            perplexity[id]=exp(log_p)
        
        avg_perplexity=np.mean(list(perplexity.values()))
    return perplexity,avg_perplexity