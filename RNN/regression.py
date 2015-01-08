"""
regression.py

Written by Alex Bain (https://github.com/convexquad/treebank)
"""
from sklearn.linear_model import LogisticRegression
import numpy
import treebank

def build_softmax_classifier(word_label_map, word_vector_map):
    X = []
    y = []
    for word in word_label_map.keys():
        X.append(word_vector_map[word])
        y.append(word_label_map[word])
    lr = LogisticRegression(C = 1e10, fit_intercept = True, tol=1e-5)
    lr.fit(X, y)
    return lr

def test_softmax_classifier(lr, word_label_map, word_vector_map):
    X = []
    y = []
    for word in word_label_map.keys():
        X.append(word_vector_map[word])
        y.append(word_label_map[word])
    accuracy = lr.score(X, y)
    predictions = lr.predict(X)
    return (accuracy, predictions, y)

def main():
    tbank = treebank.build_standard_treebank()
    lr = build_softmax_classifier(tbank.word_label_map, tbank.word_vector_map)

    (accuracy, predictions, labels) = test_softmax_classifier(lr, tbank.word_label_map, tbank.word_vector_map)
    classes = [0 for x in range(5)]
    for predict in predictions:
        classes[predict] = classes[predict] + 1
    print accuracy
    print classes

if __name__ == "__main__":
    main()    
