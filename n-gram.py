import nltk
import numpy as np
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize


def preprocess_text(corpus):
    # Sentence segmentation
    sentences = sent_tokenize(corpus)

    # Tokenization, punctuation handling
    tokens = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        tokens.extend([w for w in words if w.isalnum() or w in [".", "!", "?"]])

    return tokens


def extract_frequencies(tokens):
    unigrams = nltk.FreqDist(tokens)
    bigrams = nltk.FreqDist(ngrams(tokens, 2))
    return unigrams, bigrams


def format_output(ngram_dist):
    # Sort tokens alphabetically
    sorted_tokens = sorted(ngram_dist.keys())
    # Get maximum token length for right alignment
    max_len = max(len(str(token)) for token in sorted_tokens)

    output = f"{'Word':^{max_len+2}} Count"
    # Create rows with aligned numbers
    for token in sorted_tokens:
        count = ngram_dist[token]
        output += f"\n{str(token):^{max_len+2}} {count:>7}"
    return output


def create_bigram_matrix(bigrams, tokens):
    matrix = np.zeros((len(tokens), len(tokens)))
    token_index = {token: i for i, token in enumerate(tokens)}

    for bigram, count in bigrams.items():
        i = token_index[bigram[0]]
        j = token_index[bigram[1]]
        matrix[i, j] = count

    return matrix


# Example usage
corpus = """ """

tokens = preprocess_text(corpus)
unigrams, bigrams = extract_frequencies(tokens)

print("\nUnigrams:")
print(format_output(unigrams))

print("\nBigram Matrix:")
bigram_matrix = create_bigram_matrix(bigrams, tokens)
print(bigram_matrix)
print(bigram_matrix.shape)


def laplace_smoothing(matrix):
    # return (matrix + 1) / (matrix.sum(axis=1)[:, None] + matrix.shape[0])
    return matrix + 1


add_one = laplace_smoothing(bigram_matrix)

print("\nAdd-One Matrix:\n", add_one)
print(add_one / add_one.sum(axis=1) + 2)
