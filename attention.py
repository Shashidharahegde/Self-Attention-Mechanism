import numpy as np;

sentence = "The cat sat on the mat"
#Tokenisation
tokens = sentence.split()

#Step 1
#First, words are converted/assigned vectors using 'word embedding'. Because Computers only understand numbers.
np.random.seed(33)
word_embeddings= np.random.rand(len(tokens),4)
#creates a 6×4 matrix because "The", "cat", "sat", "on", "the", "mat" id 6 words => Each word gets a row assigned to it.
#The assignment is implicit => the order in words[] is same order in word_embeddings[]


for i, tokens in enumerate(tokens):
    print(f"{tokens}: {word_embeddings[i]}")

# Step 2
# Creation of weights Wq, Wk, Wv
Wq = np.random.rand(4,4)
Wk = np.random.rand(4,4)
Wv = np.random.rand(4,4)

print("Wq:\n", Wq, "\nWk:\n", Wk, "\nWv:\n", Wv)

#Why Create them:
    # So that Attention can tweak itself and 'learn' the relationships in the sentence
    # If we dont multiply weights, there is noflexibility
    # Otherwise Q,K,V will be fundamentally same!

#Step 3
#Assigning Query(Q),Key(K), Value(V) Matrices

#In Self Attention, each word has 3 versions of itself.

# We do dot product 
Q = np.dot(word_embeddings, Wq)
K = np.dot(word_embeddings, Wk)
V = np.dot(word_embeddings, Wv)

print("----------------------------------------------------------------")

print("Q:\n" , Q)

#Step 4:
#Calculating Attention

attention_scores = np.dot(Q,K.T)

#You take transpose to match dimentionality

print("--------------------------------------------------------------------")

print("\nRaw Attention Scores:\n", attention_scores)

#Step 5:
#Applying Softmax funtion to normalise attention scores

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims =True)
    
attention_weights = softmax(attention_scores)

print("----------------------------------------------------------------")

print("\nAttention Weights (after Softmax):\n", attention_weights)


#Step 6
#Final word representations

# Muktiplying V ensures that words are influenced by imp words abd are context-aware

output_vectors = np.dot(attention_weights, V)

print("--------------------------------------------------------------")

print("\nFinal Output Vectors:\n", output_vectors)

