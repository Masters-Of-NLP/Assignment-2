from n_gram import n_gram_counts as count_n_gram

def train_n_gram(corpus, n):
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
                prob_n_words[key] = ngram_counts[key] / n1gram_counts[prev]
            elif (n >2):
                prob_n_words[key] = ngram_counts[key] / unigram_counts[words[n-2]]
            else:
                prob_n_words[key] = 0
    else:
        N_train = len(corpus)
        for key in ngram_counts:
            if (prev in n1gram_counts):
                prob_n_words[key] = ngram_counts[key] / N_train
            else:
                prob_n_words[key] = 0
    return prob_n_words