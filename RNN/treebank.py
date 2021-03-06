"""
treebank.py
Stanford Sentiment Treebank Parser

Written by Alex Bain (https://github.com/convexquad/treebank)
"""
import numpy
import re

C = 5  # Number of Treebank label classes

# Wrapper class for the Treebank datasets.
class Treebank:
    def __init__(self, dev, test, train):
        self.dev = dev
        self.test = test
        self.train = train

        word_label_map_dev = self.read_word_labels(dev)
        word_label_map_trn = self.read_word_labels(train)
        word_label_map_tst = self.read_word_labels(test)

        self.word_label_map = word_label_map_dev.copy()
        self.word_label_map.update(word_label_map_trn)
        self.word_label_map.update(word_label_map_tst)

        self.vocabulary_dev = set(word_label_map_dev.keys())
        self.vocabulary_trn = set(word_label_map_trn.keys())
        self.vocabulary_tst = set(word_label_map_tst.keys())
        self.vocabulary = set(self.word_label_map.keys())

        self.word_index_map = self.build_word_index_map(self.vocabulary)
        self.add_word_index_trees(self.dev)
        self.add_word_index_trees(self.train)
        self.add_word_index_trees(self.test)

    # Annotate each leaf node in a tree with an index that identifies its word
    def add_word_index_trees(self, trees):
        for tree in trees:
            self.add_word_index_tree(tree)

    # Helper function to annotate the leaf nodes in a tree with an idnex that
    # identifies its word
    def add_word_index_tree(self, tree):
        if tree.is_leaf():
            word_index = self.word_index_map[tree.word]
            assert word_index >= 0
            tree.word_index = word_index
        else:
            self.add_word_index_tree(tree.left)
            self.add_word_index_tree(tree.right)

    # Builds a map of each word to an index identifying that word
    def build_word_index_map(self, vocabulary):
        word_index_map = dict()
        current_index = 0
        for word in vocabulary:
            word_index_map[word] = current_index
            current_index = current_index + 1
            assert len(word_index_map) == current_index
        return word_index_map

    # Function to get the words and labels for a set of trees from Treebank.
    def read_word_labels(self, trees):
        word_label_map = dict()
        for tree in trees:
            self.read_word_labels_tree(tree, word_label_map)
        return word_label_map

    # Helper function to add each (word, label) pair in Treebank to a map.
    def read_word_labels_tree(self, tree, word_label_map):
        if tree.is_leaf():
            if word_label_map.has_key(tree.word):
                # Make a check to see that Treebank always labels the same word
                # with the same sentiment label.
                assert word_label_map[tree.word] == tree.label
            else:
                word_label_map[tree.word] = tree.label
        else:
            self.read_word_labels_tree(tree.left, word_label_map)
            self.read_word_labels_tree(tree.right, word_label_map)

# Simple Node class for Treebank sentences.
class Node:
    def __init__(self):
        self.word = None
        self.word_index = None
        self.label = None
        self.left = None
        self.right = None
        self.t_k = numpy.zeros(C)  # Vector t_k for the label at node k
        self.y_k = None            # Vector y_k for the softmax prediction node k
        self.a_k = None            # Vector a_k at node k
        self.x_k = None            # Vector x_k at node k
        self.z_k = None            # Vector z_k at node k
        self.J_ce_k = 0            # Softmax cross-entropy cost at node k
        self.delta_sm_k = None     # Softmax error delta at node k
        self.delta_in_k = None     # "Incoming" error delta at span node k
        self.delta_k = None        # Full error delta at span node k = delta_in_k + delta_sm_k

    def is_leaf(self):
        return (self.word != None)

    # With the str conversion we can test our code by printing out the parsed
    # Treebank and diffing our version against the input file.
    def __str__(self):
        if self.is_leaf():
            return "({0} {1})".format(self.label, self.word)
        else:
            return "({0} {1} {2})".format(self.label, self.left, self.right)

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
           if current.label == None:
               current.label = int(token)
               current.t_k[current.label] = 1.0
           else:
               current.word = token

# Helper function to build the Treebank object.
def build_standard_treebank():
    dev = read_treebank_file("/Users/abain/treebank/trees/dev.txt")
    tst = read_treebank_file("/Users/abain/treebank/trees/test.txt")
    trn = read_treebank_file("/Users/abain/treebank/trees/train.txt")
    return Treebank(dev, tst, trn)

# Main function for testing the Treebank parser. Note that we can easily unit
# test this code by simply diffing the printed output against the original
# file. Thus, I won't bother with specific unit tests for this file.
def main():
    treebank = build_standard_treebank()
    # for tree in treebank.dev:
    #    print tree
    print len(treebank.vocabulary)

if __name__ == "__main__":
    main()
