import nltk

# Define a grammar for chart parsing
grammar = nltk.CFG.fromstring("""
    S -> NP VP
    PP -> P NP
    NP -> Det N | NP PP
    VP -> V NP | VP PP
    Det -> 'the' | 'a'
    N -> 'cat' | 'dog' | 'house'
    V -> 'chased' | 'caught'
    P -> 'in' | 'on' | 'under'
""")

# Create a chart parser
parser = nltk.ChartParser(grammar)

# New sample sentence for parsing
sentence = "The cat chased the dog in the house"

# Manually split the sentence into words
words = sentence.split()

# Tokenize the words and convert them to lowercase
tokens = [word.lower() for word in words]

# Parse the sentence
for tree in parser.parse(tokens):
    print("Parse Tree:")
    tree.pretty_print()




