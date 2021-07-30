import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
import scipy.sparse as sp
from spektral.layers import GCNConv
from tensorflow.keras.optimizers import Adam

command_line_arg = int(sys.argv[1])

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("../data/")
adj = adj+adj.T

adj_train = adj[train_index, :][:, train_index]

dropout = 0.2
F = 602
N = 232965
channels = 256
l2_reg = 5e-4
learning_rate = .001
num_classes = 41
epochs = 600

numNode_train = adj_train.shape[0]

checkpoint_path = "training_256/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#A = GCNConv.preprocess(adj).astype('f4')
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)

dropout_1 = Dropout(dropout)(X_in)
graph_conv_1 = GCNConv(channels,
                       activation='relu',
                       kernel_regularizer=l2(l2_reg),
                       use_bias=False)([dropout_1, fltr_in])

dropout_2 = Dropout(dropout)(graph_conv_1)
graph_conv_2 = GCNConv(num_classes,
                       activation='softmax',
                       use_bias=False)([dropout_2, fltr_in])

model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()
model.load_weights(checkpoint_path)

# Evaluate model
# (55334,)
test_features = features[test_index]
adj_test = adj[test_index, :][:, test_index]

# print("test_index_shape", test_index.shape)
M = test_features.shape[0]

tf.profiler.experimental.start('prediction_logs')
y_pred = model.predict([test_features, adj_test], batch_size=M)
tf.profiler.experimental.stop()