"""
rnn.py
Recursive Neural Network (RNN) for Treebank sentiment

Written by Alex Bain (https://github.com/convexquad/treebank)
"""
import numpy
import os
import treebank

D = 30          # Dimensionality of our word model
R = 0.0001      # Range for the uniform dist to initialize the words
C = treebank.C  # Number of Treebank label classes
E = 1e-12       # Error epsilon for comparing float values to constants

# Epsilon for gradient checking. Value suggested by Spare Autoencoder notes
EPSILON = 1e-4

# Epsilon added for numerical stability of AdaGrad computation
ADAGRAD_EPSILON = 1e-6

# Fix numpy's random number generation for reproducible results
numpy.random.seed(10)

# Simple RNN class for Treebank sentiment analysis.
class RNN:
    def __init__(self, tbank):
        self.tbank = tbank
	self.L = numpy.random.uniform(-R, R, (D, len(tbank.vocabulary)))
        self.W = numpy.random.normal(0, 0.01, (D, 2*D + 1))
        self.W[:, 2*D] = 0  # Set the intercept column to zero
        self.Ws = numpy.random.normal(0, 0.01, (C, D))

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

    # Make a forward pass over all sentences in the given dataset and computes
    # the average cost per sentence J(theta) over these sentences.
    def forward_pass_over(self, trees):
        J = 0.0
        for tree in trees:
            self.forward_pass(tree)
            J = J + self.compute_softmax_err(tree)
        return (J / len(trees))

    # Load pre-trained matrices from the given base directory.
    def load_models(self, base_dir):
        base_dir = base_dir.rstrip("/")
        self.L = numpy.load("{}/L.npy".format(base_dir))
        self.W = numpy.load("{}/W.npy".format(base_dir))
        self.Ws = numpy.load("{}/Ws.npy".format(base_dir))

    # Save the trained matrices to the given base directory.
    def save_models(self, base_dir, J):
        base_dir = base_dir.rstrip("/")
        base_dir = "{}/model_{}_{}_{}".format(base_dir, self.alpha, self.llambda, self.epochs)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        numpy.save("{}/L.npy".format(base_dir), self.L)
        numpy.save("{}/W.npy".format(base_dir), self.W)
        numpy.save("{}/Ws.npy".format(base_dir), self.Ws)
        open("{}/avg_train_cost_{}.txt".format(base_dir, J), "w").close()

    # Top-level function to train the RNN. AdaGrad references:
    # http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
    # http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
    def train(self, trees, alpha, llambda, epochs, use_adagrad):
        self.alpha = alpha
        self.epochs = epochs
        self.llambda = llambda

        hist_grad_L  = numpy.zeros(self.L.shape)
        hist_grad_W  = numpy.zeros(self.W.shape)
        hist_grad_Ws = numpy.zeros(self.Ws.shape)

        for i in range(epochs):
            # In each training run, compute the cost and the gradients
            (J, grad_L, grad_W, grad_Ws) = self.training_run(trees, i + 1)

            # Adjust the gradients by the regularization terms
            grad_L  += llambda * self.L
            grad_W  += llambda * self.W
            grad_Ws += llambda * self.Ws

            if use_adagrad:
                hist_grad_L  += numpy.square(grad_L)
                hist_grad_W  += numpy.square(grad_W)
                hist_grad_Ws += numpy.square(grad_Ws)

                grad_L  = grad_L  / (ADAGRAD_EPSILON + numpy.sqrt(hist_grad_L))
                grad_W  = grad_W  / (ADAGRAD_EPSILON + numpy.sqrt(hist_grad_W))
                grad_Ws = grad_Ws / (ADAGRAD_EPSILON + numpy.sqrt(hist_grad_Ws))

            grad_norms = numpy.linalg.norm(grad_L) + numpy.linalg.norm(grad_W) + numpy.linalg.norm(grad_Ws)
            print "Sum of grad norms in epoch {} is {}".format(i, grad_norms)

            self.L  -= alpha * grad_L
            self.W  -= alpha * grad_W
            self.Ws -= alpha * grad_Ws

    # Makes an training run on the RNN for a particular training epoch.
    def training_run(self, trees, epoch):
        J = self.forward_pass_over(trees)

        print "Cost in epoch {} is J = {}; matrix norms are W = {}, Ws = {}, L = {}".format(
            epoch, J, numpy.linalg.norm(self.W), numpy.linalg.norm(self.Ws), numpy.linalg.norm(self.L))

        W_l = self.W[:, 0:D]
        W_r = self.W[:, D:2*D]
        assert W_l.shape == (D, D)
        assert W_r.shape == (D, D)

        # Compute the backprop deltas
        for tree in trees:
            self.compute_softmax_delta(tree)
            self.compute_incoming_delta(tree, W_l, W_r)

        # Compute the backprop gradients
        grad_L  = numpy.zeros(self.L.shape)
        grad_W  = numpy.zeros(self.W.shape)
        grad_Ws = numpy.zeros(self.Ws.shape)

        for tree in trees:
            self.compute_backprop_L(tree, grad_L, W_l, W_r)
            self.compute_backprop_W(tree, grad_W)
            self.compute_backprop_Ws(tree, grad_Ws)

        # Average the gradients by the number of sentences since we do this in
        # the cost function and add the regularization term to the gradient.
        M = len(trees)
        grad_L  /= M
        grad_W  /= M
        grad_Ws /= M

        # Check the gradients we computed against a finite difference estimate.
        # I verified this check already, so I'll comment this out now.
        # self.check_matrix_gradients(trees, self.L, grad_L, "grad_L")
        # self.check_matrix_gradients(trees, self.W, grad_W, "grad_W")
        # self.check_matrix_gradients(trees, self.Ws, grad_Ws, "grad_Ws")

        # Return the cost and the gradients
        return (J, grad_L, grad_W, grad_Ws)

    # Check the word gradients against a finite difference estimate
    def check_matrix_gradients(self, trees, matrix, grad, grad_name):
        (M, N) = matrix.shape
        for i in range(M):
            for j in range(N):
                Mij = matrix[i, j]
                Gij = grad[i, j]

                matrix[i, j] = Mij + EPSILON
                J_pos = self.forward_pass_over(trees)

                matrix[i, j] = Mij - EPSILON
                J_neg = self.forward_pass_over(trees)

                matrix[i, j] = Mij
                finite_est = (J_pos - J_neg) / (2.0 * EPSILON)
                print "{}[{}, {}]: {}, finite est: {} with difference: {}".format(grad_name, i, j, Gij, finite_est, abs(Gij - finite_est))

    # Computes the training cost J(theta), assuming forward passes have already
    # been made over all sentences.
    def collect_tree_costs(self, trees):
        J = 0.0
        for tree in trees:
            J = J + self.compute_softmax_err(tree)
        return (J / len(trees))

    # When we compute the softmax cross-entropy cost, take advantage of the
    # fact that there is only one non-zero term in the sum.
    def compute_softmax_err(self, node):
        node.J_ce_k = -numpy.log(node.y_k[node.label])
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
            node.delta_k = node.delta_in_k + node.delta_sm_k
            assert len(node.delta_k) == D
            self.compute_incoming_delta_child(node, node.left, W_l, W_l, W_r)
            self.compute_incoming_delta_child(node, node.right, W_r, W_l, W_r)

    # Computes the contribution to partial J / partial L for a tree.
    def compute_backprop_L(self, node, grad_L, W_l, W_r):
        if node.is_leaf():
            grad_x_n = self.Ws.T.dot(node.y_k - node.t_k)
            assert len(grad_x_n) == D
            grad_L[:, node.word_index] += grad_x_n
            return

        if node.left.is_leaf():
            grad_x_n = node.delta_k.T.dot(W_l).T
            assert len(grad_x_n) == D
            grad_L[:, node.left.word_index] += grad_x_n

        if node.right.is_leaf():
            grad_x_n = node.delta_k.T.dot(W_r).T
            assert len(grad_x_n) == D
            grad_L[:, node.right.word_index] += grad_x_n

        self.compute_backprop_L(node.left, grad_L, W_l, W_r)
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
    tbank = treebank.build_standard_treebank()
    rnn = RNN(tbank)
    alpha = 0.01
    llambda = 0.001
    epochs = 2000
    rnn.train(tbank.train, alpha, llambda, epochs, True)
    J = rnn.forward_pass_over(tbank.train)
    rnn.save_models("/Users/abain/treebank/RNN/Models/Train", J)

if __name__ == "__main__":
    main()
