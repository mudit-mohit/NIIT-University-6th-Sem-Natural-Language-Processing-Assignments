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


def generate_word_bigram_dict(words):
    word_bigram_dict = {}

    for word in words:
        if len(word) > 2:
            # Generate alphabet bigrams for the word
            char_bigrams = [word[i : i + 2] for i in range(len(word) - 1)]
            # Store in the dictionary
            word_bigram_dict[word] = char_bigrams

    return word_bigram_dict


# Example usage
corpus = "Sky blue sunsets, mountains whisper, ocean waves bloom. Trees embrace, rivers meander, nature's symphony. Life's journey, moments unfold, love moms blossoms, stars witness"
tokens = preprocess_text(corpus)

training_word_bigram_dict = generate_word_bigram_dict(tokens)

for a, b in training_word_bigram_dict.items():
    print(f"{a}:{b}")

# Given misspelled word
misspelled_word = "bloms"

# Generate character bigrams for the misspelled word
test_word_bigrams = [
    misspelled_word[i : i + 2] for i in range(len(misspelled_word) - 1)
]
print("\n************************************ \n")
print(
    f"Character Bigrams for the {misspelled_word} has length {len(test_word_bigrams)}:",
    test_word_bigrams,
)


# # Calculate MED for each training word
# med_results = {}
# for training_word, training_bigrams in training_word_bigram_dict.items():
#     med_distance = calculate_med(test_word_bigrams, training_bigrams)
#     med_results[training_word] = med_distance
# # Sort the results by minimum edit distance
# sorted_results = sorted(med_results.items(), key=lambda x: x[1])
# print(" by Minimum Edit Distance:")
# for result in sorted_results:
#     print(result)
def count_matching_bigrams(test_bigrams, training_bigrams):
    return sum(1 for bigram in test_bigrams if bigram in training_bigrams)


candidate_list = []
for training_word, training_bigrams in training_word_bigram_dict.items():
    matching_bigrams = count_matching_bigrams(test_word_bigrams, training_bigrams)
    candidate_list.append((training_word, matching_bigrams))

sorted_candidates = sorted(candidate_list, key=lambda x: x[1], reverse=True)


top_10_candidates = sorted_candidates[:10]

print("\nTop 10 Candidate List by Matching Bigrams:")
for candidate in top_10_candidates:
    print(f"{candidate[0]}: {candidate[1]}")


def edit_distance(word1, word2):
    m = len(word1) + 1
    n = len(word2) + 1
    dp = [[0] * n for _ in range(m)]

    # Initialize the matrix with base cases
    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j

    # Fill in the rest of the matrix
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,
            )  # substitution

    return dp[-1][-1]


def min_edit_distance(misspelled_word, correct_words):
    min_distance = float("inf")
    closest_words = []

    for word in correct_words:
        distance = edit_distance(misspelled_word, word)
        if distance < min_distance:
            min_distance = distance
            closest_words = [word]
        elif distance == min_distance:
            closest_words.append(word)

    return closest_words, min_distance


correct_words = [word for word, _ in top_10_candidates]

print("\n************************************ \n")
print(f"Closest words to {misspelled_word} are:")
closest_words, min_distance = min_edit_distance(misspelled_word, correct_words)

print(f"Closest words: {closest_words}")
print(f"Minimum Edit Distance: {min_distance}")


def edit_distance_with_bigrams(
    word1, word2, matching_bigrams, matching_bigram_bonus=0.1
):
    m = len(word1) + 1
    n = len(word2) + 1
    dp = [[0] * n for _ in range(m)]

    # Initialize the matrix with base cases
    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j

    # Fill in the rest of the matrix
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

            # If the current bigram in word1 is in the matching bigrams, reduce the cost
            if word1[i - 1 : i + 1] in matching_bigrams:
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] - matching_bigram_bonus)

    return dp[-1][-1]


def min_edit_distance_with_bigrams(misspelled_word, correct_words, matching_bigrams):
    min_distance = 99999
    closest_words = []

    for word in correct_words:
        distance = edit_distance_with_bigrams(misspelled_word, word, matching_bigrams)
        if distance < min_distance:
            min_distance = distance
            closest_words = [word]
        elif distance == min_distance:
            closest_words.append(word)

    return closest_words, min_distance


# Example usage
print("\n************************************ \n")
print(f"Closest words to {misspelled_word} are:")
closest_words, min_distance = min_edit_distance_with_bigrams(
    misspelled_word, correct_words, test_word_bigrams
)

print(f"Closest words: {closest_words}")
print(f"Minimum Edit Distance with Bigrams: {min_distance}")
