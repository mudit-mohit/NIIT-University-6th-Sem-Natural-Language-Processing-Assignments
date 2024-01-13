import nltk
nltk.download('punkt')
import string
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import matplotlib.pyplot as plt
# Read the text from the file
with open("C:\\Dev\\NLP\\001 Welcome Challenge!.html", "r", encoding="utf-8") as file:
    text = file.read()
# Tokenization including punctuation
tokens_with_punctuation = word_tokenize(text)
# Tokenization excluding punctuation
tokens_without_punctuation = [token for token in tokens_with_punctuation if token.isalnum()]
# Analysis
freq_dist_with_punctuation = FreqDist(tokens_with_punctuation)
freq_dist_without_punctuation = FreqDist(tokens_without_punctuation)
# Display results
print("Tokenization Including Punctuation:")
print(tokens_with_punctuation[:10])  # Displaying the first 10 tokens
print("\nTokenization Excluding Punctuation:")
print(tokens_without_punctuation[:10])  # Displaying the first 10 tokens
# Plot frequency distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
#freq_dist_with_punctuation.plot(20, title='Frequency Distribution (Including Punctuation)')
plt.subplot(1, 2, 2)
freq_dist_without_punctuation.plot(20, title='Frequency Distribution (Excluding Punctuation)')
plt.tight_layout()
plt.show()
