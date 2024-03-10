import nltk
import time

def generate_trees(sentence, num_trees):
  # Define a grammar for top-down parsing
  grammar = nltk.CFG.fromstring("""
        S -> NP VP  # Removed comment, not allowed within grammar rule
        NP -> Det Noun | ProperNoun
        # Noun phrase can be definite or proper noun
        Det -> "the" | "a" | "an"
        Noun -> "fox" | "dog" | "car" | "table"
        ProperNoun -> "Fox" | "Fido"
        VP -> Verb NP PP | Verb
        # Verb phrase can have an object and optional prepositional phrase
        Verb -> "jumps" | "runs" | "sleeps"
        PP -> Prep NP
        Prep -> "over" | "under" | "on"
  """)
  
  # Create a top-down parser
  parser = nltk.ChartParser(grammar)
  
  # Tokenize the input sentence
  tokens = nltk.word_tokenize(sentence)
  
  # Parse the sentence multiple times
  for i in range(num_trees):
    start_time = time.time()
    trees = list(parser.parse(tokens))
    end_time = time.time()
    
    if trees:
      print(f"Time taken for Tree {i+1}: {end_time - start_time:.6f} seconds")
      for tree in trees:
        print("Tree:")
        tree.pretty_print()
        print()
    else:
      print(f"No trees found for Tree {i+1}")

sentence = "The quick brown fox jumps over the lazy dog"
num_trees = 5  # Number of trees to generate for the sentence

generate_trees(sentence, num_trees)












