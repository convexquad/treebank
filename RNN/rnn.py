"""
rnn.py
Recursive Neural Network (RNN) for Treebank sentiment

Written by Alex Bain (https://github.com/convexquad/treebank)
"""
import numpy
import treebank

D = 30          # Dimensionality of our word model
R = 0.0001      # Range for the uniform dist to initialize the words
C = treebank.C  # Number of Treebank label classes
E = 1e-12       # Error epsilon for comparing float values to constants

# Fix numpy's random number generation for reproducible results
numpy.random.seed(10)

# Simple RNN class for Treebank sentiment analysis.
class RNN:
    def __init__(self, tbank, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs
        self.tbank = tbank
	self.L = numpy.random.uniform(-R, R, (D, len(tbank.vocabulary)))
        self.W = numpy.random.normal(0, 0.01, (D, 2*D + 1))
        self.Ws = numpy.random.normal(0, 0.01, (C, D))
        # self.W = numpy.zeros((D, 2*D + 1))
        # self.Ws = numpy.zeros((C, D))

    # Makes a forward evaluation pass for a Treebank sentence.
    def forward_pass(self, node):
        if node.is_leaf():
            node.x_k = self.L[:, node.word_index]
            assert len(node.x_k) == D

            node.y_k = softmax(self.Ws.dot(node.x_k))
            assert len(node.y_k) == C
            assert abs(node.y_k.sum() - 1) < E
            return node.x_k
        else:
            x_l = self.forward_pass(node.left)
            x_r = self.forward_pass(node.right)
            assert len(x_l) == D
            assert len(x_r) == D

            node.z_k = numpy.concatenate((x_l, x_r, numpy.ones((1))))
            node.a_k = self.W.dot(node.z_k)
            node.x_k = numpy.tanh(node.a_k)
            assert len(node.x_k) == D

            node.y_k = softmax(self.Ws.dot(node.x_k))
            assert len(node.y_k) == C
            assert abs(node.y_k.sum() - 1) < E
            return node.x_k

    # Top-level function to train the RNN.
    def train(self):
        for i in range(self.epochs):
            self.training_run(i + 1)

    # Makes an training run on the RNN for a particular training epoch.
    def training_run(self, epoch):
        J = 0
        for tree in self.tbank.dev:
            self.forward_pass(tree)
            J = J + self.compute_softmax_err(tree)

        print "Cost in epoch {} is J = {}; matrix norms are W = {}, Ws = {}, L = {}".format(
            epoch, J, numpy.linalg.norm(self.W), numpy.linalg.norm(self.Ws), numpy.linalg.norm(self.L))

        W_l = self.W[:, 0:D]
        W_r = self.W[:, D:2*D]
        assert W_l.shape == (D, D)
        assert W_r.shape == (D, D)

        # Compute the backprop deltas
        for tree in self.tbank.dev:
            self.compute_softmax_delta(tree)
            self.compute_incoming_delta(tree, W_l, W_r)

        # Compute the backprop gradients
        grad_L  = numpy.zeros(self.L.shape)
        grad_W  = numpy.zeros(self.W.shape)
        grad_Ws = numpy.zeros(self.Ws.shape)

        for tree in self.tbank.dev:
            self.compute_backprop_L(tree, grad_L, W_l, W_r)
            self.compute_backprop_W(tree, grad_W)
            self.compute_backprop_Ws(tree, grad_Ws)

        # Average the word gradients by the word occurrences
        for word in self.tbank.vocabulary:
            word_count = self.tbank.word_count_map[word]
            word_index = self.tbank.word_index_map[word]
            grad_L[:, word_index] /= word_count

        # Average the W gradients by the number of span nodes
        grad_W = (1.0 / self.tbank.span_node_count) * grad_W

        # Average the Ws gradients by the total number of nodes
        grad_Ws = (1.0 / self.tbank.total_node_count) * grad_Ws

        # Make the parameter update
        self.L  = self.L  - ((self.alpha / epoch) * grad_L)
        self.W  = self.W  - ((self.alpha / epoch) * grad_W)
        self.Ws = self.Ws - ((self.alpha / epoch) * grad_Ws)

    # When we compute the softmax cross-entropy cost, take advantage of the
    # fact that there is only one non-zero term in the sum.
    def compute_softmax_err(self, node):
        node.J_ce_k = -numpy.log2(node.y_k[node.label])
        assert node.J_ce_k >= 0
        if node.is_leaf():
            return node.J_ce_k
        else:
            J_ce_lk = self.compute_softmax_err(node.left)
            J_ce_rk = self.compute_softmax_err(node.right)
            return (node.J_ce_k + J_ce_lk + J_ce_rk)

    # Compute the softmax error deltas for nodes of the tree.
    def compute_softmax_delta(self, node):
        if not node.is_leaf():
            node.delta_sm_k = self.Ws.T.dot(node.y_k - node.t_k) * tanh_prime(node.a_k)
            assert len(node.delta_sm_k) == D
            self.compute_softmax_delta(node.left)
            self.compute_softmax_delta(node.right)

    # Compute the "incoming" error deltas for the nodes in this tree. There is
    # no incoming delta for the root node, so just recurse into the children.
    def compute_incoming_delta(self, root, W_l, W_r):
        root.delta_k = root.delta_sm_k
        if not root.is_leaf():
            self.compute_incoming_delta_child(root, root.left, W_l, W_l, W_r)
            self.compute_incoming_delta_child(root, root.right, W_r, W_l, W_r)

    # Compute the incoming error deltas for non-root nodes of the tree.
    def compute_incoming_delta_child(self, parent, node, W_k, W_l, W_r):
        if not node.is_leaf():
            node.delta_in_k = parent.delta_k.T.dot(W_k).T * tanh_prime(node.a_k)
            assert len(node.delta_in_k) == D
            node.delta_k = node.delta_in_k + node.delta_sm_k
            self.compute_incoming_delta_child(node, node.left, W_l, W_l, W_r)
            self.compute_incoming_delta_child(node, node.right, W_r, W_l, W_r)

    # Computes the contribution to partial J / partial L for a tree.
    def compute_backprop_L(self, node, grad_L, W_l, W_r):
        if node.is_leaf():
            return

        if node.left.is_leaf():
            grad_x_n = node.delta_k.T.dot(W_l).T
            assert len(grad_x_n) == D
            grad_L[:, node.left.word_index] += grad_x_n
        else:
            self.compute_backprop_L(node.left, grad_L, W_l, W_r)

        if node.right.is_leaf():
            grad_x_n = node.delta_k.T.dot(W_r).T
            assert len(grad_x_n) == D
            grad_L[:, node.right.word_index] += grad_x_n
        else:
            self.compute_backprop_L(node.right, grad_L, W_l, W_r)

    # Computes the contribution to partial J / partial W for a tree.
    def compute_backprop_W(self, node, grad_W):
        if node.is_leaf():
            return
        else:
            grad_k = numpy.outer(node.delta_k, node.z_k)
            assert grad_k.shape == self.W.shape
            grad_W += grad_k
            self.compute_backprop_W(node.left, grad_W)
            self.compute_backprop_W(node.right, grad_W)

    # Computes the contribution to partial J / partial Ws for a tree.
    def compute_backprop_Ws(self, node, grad_Ws):
        grad_k = numpy.outer((node.y_k - node.t_k), node.x_k)
        assert grad_k.shape == (C, D)
        grad_Ws += grad_k
        if not node.is_leaf():
            self.compute_backprop_Ws(node.left, grad_Ws)
            self.compute_backprop_Ws(node.right, grad_Ws)

    def show_word_results(self):
        for word in self.tbank.vocabulary:
            word_index = self.tbank.word_index_map[word]
            word_label = self.tbank.word_label_map[word]
            word_vectr = self.L[:, word_index]
            prediction = softmax(self.Ws.dot(word_vectr))
            print "WORD: {}, label: {}, prediction: {}".format(word, word_label, prediction[word_label])

"""
Note that the normal softmax function is susceptible to overflow. To avoid this,
subtract a const from each term of the input array before computing exp.

References:
http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression
http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.softmax
"""
def softmax(w):
    e_x = numpy.exp(w - w.max())
    dist = e_x / numpy.sum(e_x)
    return dist

# Derivative of the hyperbolic tangent function (applied elementwise).
def tanh_prime(x):
    return (1.0 - numpy.power(numpy.tanh(x), 2))

# Main function for the RNN.
def main():
    alpha = 10.0
    epochs = 30
    tbank = treebank.build_standard_treebank()
    rnn = RNN(tbank, alpha, epochs)
    rnn.train()
    rnn.show_word_results()
   
if __name__ == "__main__":
    main()
