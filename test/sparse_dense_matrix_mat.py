import sys
import time
from scipy.sparse.coo import coo_matrix
from scipy.sparse.csr import csr_matrix
from tensorflow import sparse
import scipy.sparse as sp
import numpy as np
import pdb

#tf.sparse.sparse_dense_matmul working with first input as the adjacency matrix (adj), then a random second input (can just #be anything).
#For the random second input, you can just do (np.random.rand(?,2048))


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


number_of_nodes = int(sys.argv[1])
print("number_of_nodes: ", number_of_nodes)

adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("../data/reddit/")

print(features.shape)
adj = adj+adj.T

adj_test = adj[test_index, :][:, test_index]
adj_test = adj_test[:, :number_of_nodes][:number_of_nodes,:]

# csr = adj_test.tocsr()
# csr = csr[0:number_of_nodes]
# adj_test = adj_test.tocsr()
# csr_matrix.sort_indices(adj_test)
coo = adj_test.tocoo()
# coo = adj.tocoo()
indices = np.mat([coo.row, coo.col]).transpose()
adj_sp = sparse.SparseTensor(indices, coo.data, coo.shape)
adj_sp = sparse.transpose(adj_sp)
rand_array = np.random.rand(number_of_nodes, 256)
# rand_array = np.random.rand(232965, 232965)


pdb.set_trace()
x = sparse.sparse_dense_matmul(adj_sp, rand_array)
print(x.shape)
start = time.time()
y = sparse.sparse_dense_matmul(adj_sp, rand_array)
print(y.shape)
end = time.time()

total_time = end - start


print("Total Time: ", total_time)

