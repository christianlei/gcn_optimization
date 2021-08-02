import scipy.sparse as sp
import numpy as np




    # return adj, x, data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']
    # return adj, data['x'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(data_path)
    # labels = np.zeros(adj.shape[0])
    # labels[train_index]  = y_train
    # labels[val_index]  = y_val
    # labels[test_index]  = y_test
    # adj = adj + adj.T
    # train_adj = adj[train_index, :][:, train_index]
    # features = torch.FloatTensor(np.array(features))
    # features = (features-features.mean(dim=0))/features.std(dim=0)
    # adj_normalizer = fetch_normalization(normalization)
    # adj = adj_normalizer(adj)
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    # train_adj = adj_normalizer(train_adj)
    # train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    # labels = torch.LongTensor(labels)
    # if cuda:
    #     adj = adj.cuda()
    #     train_adj = train_adj.cuda()
    #     features = features.cuda()
    #     labels = labels.cuda()
    # return adj, train_adj, features, labels, train_index, val_index, test_index