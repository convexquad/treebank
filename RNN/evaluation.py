"""
evaluation.py
Evaluation of RNN for Treebank sentiment

Written by Alex Bain (https://github.com/convexquad/treebank)
"""
from rnn import RNN, softmax
import numpy
import treebank

# Fine-grained / all nodes accuracy results (Table 1 Socher 2013).
def show_node_results(rnn, trees):
    node_count = 0
    correct_pr = 0
    for tree in trees:
        (node_c, corr_c) = show_node_results_tree(tree)
        node_count = node_count + node_c
        correct_pr = correct_pr + corr_c
    print "Total nodes: {}, correct predictions: {}, accuracy: {}".format(node_count, correct_pr, correct_pr / float(node_count))

# Helper function for fine-grained / all nodes accuracy results (Table 1 Socher 2013).
def show_node_results_tree(node):
    softmax_pr = node.y_k
    prediction = numpy.argmax(softmax_pr)
    correct_pr = 1 if node.label == prediction else 0
    if node.is_leaf():
        return (1, correct_pr)
    else:
        (node_l, corr_l) = show_node_results_tree(node.left)
        (node_r, corr_r) = show_node_results_tree(node.right)
        return (node_l + node_r + 1, corr_l + corr_r + correct_pr)

# Fine-grained / root accuracy results (Table 1 Socher 2013).
def show_root_results(rnn, trees):
    root_count = 0
    correct_pr = 0
    for tree in trees:
        softmax_pr = tree.y_k
        prediction = numpy.argmax(softmax_pr)
        likelihood = softmax_pr[tree.label]
        root_count = root_count + 1
        correct_pr = correct_pr + 1 if tree.label == prediction else correct_pr
        # print "ROOT label: {}, prediction: {}, likelihood: {}".format(root.label, prediction, likelihood)
    print "Total root nodes: {}, correct predictions: {}, accuracy: {}".format(root_count, correct_pr, correct_pr / float(root_count))

def show_word_results(rnn, vocabulary):
    word_count = 0
    correct_pr = 0
    for word in vocabulary:
        word_index = rnn.tbank.word_index_map[word]
        word_label = rnn.tbank.word_label_map[word]
        word_vectr = rnn.L[:, word_index]
        softmax_pr = softmax(rnn.Ws.dot(word_vectr))
        prediction = numpy.argmax(softmax_pr)
        likelihood = softmax_pr[word_label]
        word_count = word_count + 1
        correct_pr = correct_pr + 1 if word_label == prediction else correct_pr
        # print "WORD: {}, label: {}, prediction: {}, likelihood {}".format(word, word_label, prediction, likelihood)
    print "Total words: {}, correct predictions: {}, accuracy: {}".format(word_count, correct_pr, correct_pr / float(word_count))

# Main function for evaluating the Treebank RNN
def main():
    tbank = treebank.build_standard_treebank()
    rnn = RNN(tbank)
    rnn.load_models("/Users/abain/treebank/RNN/Models/Train/model_0.05_0.001_5")
    J = rnn.forward_pass_over(tbank.dev)

    print "Average cost per sentence: {}".format(J)
    show_word_results(rnn, tbank.vocabulary_dev)
    show_root_results(rnn, tbank.dev)
    show_node_results(rnn, tbank.dev)

if __name__ == "__main__":
    main()
