import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    score = X[i,:].dot(W)
    score_e = np.e**score
    score_pro = score_e / np.sum(score_e)
    correct_pro = score_pro[y[i]]
    loss -= np.log(correct_pro)

    M = np.zeros(score.shape)
    M[y[i]] = 1
    left_loss = score_pro - M
    Xi = np.reshape(X[i,:], (1, X.shape[1]))
    left = np.reshape(left_loss, (1, W.shape[1]))
    dW += Xi.T.dot(left)

  loss = loss/num_train
  loss += 0.5*reg*np.sum(W*W)
  dW = dW/num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  score_max = np.reshape(np.max(score, axis=1), (num_train, 1))
  score_pro = np.exp(score - score_max) / np.sum(np.exp(score - score_max), axis=1, keepdims=True)
  correct_pro = score_pro[np.arange(X.shape[0]), y]
  loss = -np.sum(np.log(correct_pro+1e-6))
  loss = loss/num_train
  loss += 0.5*reg*np.sum(W*W)

  M = np.zeros(score.shape)
  M[np.arange(num_train), y] = 1.0
  left_loss = score_pro - M
  dW = X.T.dot(left_loss)/num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

