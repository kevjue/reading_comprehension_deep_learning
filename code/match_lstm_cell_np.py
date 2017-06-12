import numpy as np


def softmax(w):
    e = np.exp(np.array(w))
    dist = e / np.sum(e)
    return dist


H_q = \
np.array([[[0.6, 0.3, -0.3], [0.7, 0.1, 0.3]],
	  [[0.7, -0.4, 0.2], [0.7, 0.5, 0.1]]])


X = \
np.array([[0.4, 0.5, 0.6],
	  [0.3, -0.2, -0.1]])


h = \
np.array([[0.2, 0.5, 0.7],
	  [-0.3, -0.3, -0.3]])


vocab_dim = 3
q_len = 2

W_q=np.array(np.eye(vocab_dim, vocab_dim))
W_p=np.array(np.eye(vocab_dim, vocab_dim))
W_r=np.array(np.eye(vocab_dim, vocab_dim))
b_p=np.array(np.ones(vocab_dim))
w_a=np.array(np.eye(vocab_dim,1))
b_a=np.array(np.ones(1))


G = np.tanh(H_q_ + np.reshape(np.tile(X + h + b_p, [1, 2]), [-1, vocab_dim]))
a_t_ = np.reshape(np.matmul(G, w_a) + b_a, [-1, q_len, 1])
a_t = np.array([softmax(a_t_[0]), softmax(a_t_[1])])

H_q_a = np.sum(H_q * a_t, 1)

h_r = np.concatenate((H_q_a, X), axis = 1)
print h_r
