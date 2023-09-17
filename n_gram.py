def vocabulary(train_corpus, test_corpus):
    vocab = set()
    
    for id in train_corpus:
        for token in train_corpus[id]:
            vocab.add(token) 
    
    for id in test_corpus:
        for token in test_corpus[id]:
            vocab.add(token)
        
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



def train_n_gram(corpus, n, vocab = 0, smoothing = False):
    prob_n_words = {}
    ngram_counts = count_n_gram(corpus, n)

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
                    prob_n_words[key] = (ngram_counts[key] + 1) / (n1gram_counts[prev] + vocab)
            elif (n >2):
                if(smoothing == False):
                    prob_n_words[key] = ngram_counts[key] / unigram_counts[words[n-2]]
                else:
                    prob_n_words[key] = (ngram_counts[key] + 1) / (unigram_counts[words[n-2]] + vocab)
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
                prob_n_words[key] = (ngram_counts[key] + 1) / (N_train + vocab)
    return prob_n_words