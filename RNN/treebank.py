"""
treebank.py
Stanford Sentiment Treebank Parser

Written by Alex Bain (https://github.com/convexquad/treebank)
"""
import numpy
import re

D = 100     # Dimensionality of our word model
R = 0.0001  # Range for the uniform dist to initialize the words

# Wrapper class for the Treebank datasets. Uses the "all" dataset to build the
# vocabulary.
class Treebank:
    def __init__(self, dev, test, train, all):
        self.dev = dev
        self.test = test
        self.train = train
        self.word_label_map = self.read_word_labels(all)
        self.vocabulary = self.word_label_map.keys()
        self.word_vector_map = self.build_word_vector_map(self.vocabulary)

    # Builds a uni(-R, R) vector for each word in the vocabulary
    def build_word_vector_map(self, vocabulary):
       word_vector_map = dict()
       for word in vocabulary:
           word_vector = numpy.random.uniform(-R, R, D)
           word_vector_map[word] = word_vector
       return word_vector_map

    # Function to get the words and labels for a set of trees from Treebank.
    def read_word_labels(self, trees):
        word_label_map = dict()
        for tree in trees:
            self.read_word_labels_tree(tree, word_label_map)
        return word_label_map

    # Helper function to add each (word, sentiment) pair in Treebank to a map.
    def read_word_labels_tree(self, tree, word_label_map):
        if tree.is_leaf():
            if word_label_map.has_key(tree.word):
                # Make a check to see that Treebank always labels the same word
                # with the same sentiment label.
                if word_label_map[tree.word] != tree.sentiment:
                    print "Found word {0} with label {1} and {2}".format(tree.word, word_label_map[tree.word], tree.sentiment)
            else:
                word_label_map[tree.word] = tree.sentiment
        else:
            self.read_word_labels_tree(tree.left, word_label_map)
            self.read_word_labels_tree(tree.right, word_label_map)

# Simple Node class for Treebank sentences.
class Node:
    def __init__(self):
        self.sentiment = None
        self.word = None
        self.left = None
        self.right = None

    def is_leaf(self):
        return (self.word != None)

    # With the str conversion we can test our code by printing out the parsed
    # Treebank and diffing our version against the input file.
    def __str__(self):
        if self.is_leaf():
            return "({0} {1})".format(self.sentiment, self.word)
        else:
            return "({0} {1} {2})".format(self.sentiment, self.left, self.right)

# Reads all the sentences of a Treebank file our Node representation.
def read_treebank_file(file_path):
    with open(file_path, "rU") as open_file:
        lines = open_file.readlines()
        trees = []
        for line in lines:
            root = tokenize_sentence(line)
            trees.append(root)
        return trees

# Tokenizes an individual Treebank sentence into our Node representation.
def tokenize_sentence(line):
   tokens = re.split(r'(\(|\)|\s)\s*', line)
   stack = []
   current = None
   for token in tokens:
       if token.strip() == "":
           continue
       elif token == "(":
           if current != None:
               stack.append(current)
           current = Node()
       elif token == ")":
           if len(stack) == 0:
               return current
           parent = stack.pop()
           if parent.left == None:
               parent.left = current
           else:
               parent.right = current
           current = parent
       else:
           if current.sentiment == None:
               current.sentiment = int(token)
           else:
               current.word = token

# Helper function to build the Treebank object.
def build_standard_treebank():
    dev = read_treebank_file("/Users/abain/RNN/trees/dev.txt")
    tst = None # read_treebank_file("/Users/abain/RNN/trees/test.txt")
    trn = None # read_treebank_file("/Users/abain/RNN/trees/train.txt")
    all = read_treebank_file("/Users/abain/RNN/trees/dev.txt")
    return Treebank(dev, tst, trn, all)

# Main function for testing the Treebank parser. Note that we can easily unit
# test this code by simply diffing the printed output against the original
# file. Thus, I won't bother with specific unit tests for this file.
def main():
    treebank = build_standard_treebank()
    # for tree in treebank.dev:
    #   print tree
    print len(treebank.vocabulary)

if __name__ == "__main__":
    main()
