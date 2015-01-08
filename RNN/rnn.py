"""
rnn.py
Recursive Neural Network (RNN) for Treebank sentiment

Written by Alex Bain (https://github.com/convexquad/treebank)
"""
import numpy
import treebank

# Simple RNN class for Treebank sentiment analysis.
class RNN:
   def __init__(self, tbank):
       self.tbank = tbank
       self.word_vector_map = tbank.word_vector_map
       self.W = numpy.zeros((D, 2*D))
       self.b = numpy.zeros(D)

   # Makes a forward evaluation pass for a Treebank sentence.
   def forward_pass(self, tree):
       if tree.is_leaf():
           return self.word_vector_map[tree.word]
       else:
           p1 = self.forward_pass(tree.left)
           p2 = self.forward_pass(tree.right)
           return numpy.tanh(self.W.dot(numpy.concatenate((p1, p2))) + self.b)

def main():
    tbank = treebank.build_standard_treebank()
    rnn = RNN(tbank)
    for tree in tbank.dev:
        rnn.forward_pass(tree)
   
    """ 
    lr = build_softmax_classifier(word_map)
    (accuracy, predictions, labels) = test_softmax_classifier(lr, word_map)
    classes = [0 for x in range(5)]
    for predict in predictions:
        classes[predict] = classes[predict] + 1
    print accuracy
    print classes
    """

if __name__ == "__main__":
    main()
