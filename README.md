Treebank
========
Treebank sentiment RNN. Dataset is from http://nlp.stanford.edu/sentiment/.

Implementation is based off the paper "Recursive Deep Models for Semantic
Compositionality Over a Sentiment Treebank" by Socher et. al that is available
at http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf.

This implementation is done with Python and NumPy. It requires:
-- Python >= 2.7.9
-- NumPy  >= 1.9

Training parameters (the learning rate alpha; strength of regularization
lambda; and the number of training epochs) are specified in the main function
of rnn.py. To train, type "python rnn.py". This will save a model under the
folder RNN/Models for the given parameters.

Then update evaluation.py with this folder name and run "python evaluation.py"
to evaluate the learned model on the test set.
