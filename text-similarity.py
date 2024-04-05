from collections import Counter
import string
import pandas as pd
import numpy as np
from numpy.linalg import norm

sentences = """The cat and the dog play in the garden shed.
The dog loudly chases the cat.
Cat and dog run after the ball.
The cat sleeps in the garden."""

X = sentences.lower().split() # the list of words
X = [sent.strip(".") for sent in X]
Y = sentences.split('\n')
Z = Counter(X)
M = sorted(Z.keys()) # unique words

def text2vector(sentence, voc):
    cleantext = "".join(char for char in sentence if char not in string.punctuation)
    words = cleantext.lower().split()
    vector = [words.count(w) for w in voc]
            # Counter(words) returns a dictionary of num of occurences
    return vector

sent2vec = [text2vector(sent, M) for sent in Y]
print(sent2vec)
df = pd.DataFrame(sent2vec, columns=M,
                  index = [f"doc_{i+1}" for i in range(len(Y))])

def cosine_sim(vec1, vec2):
    """Calculate the cosine similarity between two vectors"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2)/(norm(v1)*norm(v2))

#print(cosine_sim(df.iloc[0], df.iloc[2]))