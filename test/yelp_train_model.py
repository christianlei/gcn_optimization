import sys
import os
import tensorflow as tf
import numpy as np
import scipy
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from spektral.layers import GCNConv
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import statistics


def create_node_degree_graph(figure_name, dictionary):

    node_list = []

    for key, value in dictionary.items():
        for i in range(value):
            node_list.append(int(key))
        # if degrees in node_degrees:
        #     node_degrees[degrees]+=1
        # else:
        #     node_degrees[degrees] = 1

    print("median: ", statistics.median(node_list))
    # fig = plt.figure()
    # fig.suptitle('Degrees of Nodes in Graph - YELP', fontsize=20)
    # plt.bar(list(dictionary.keys()), dictionary.values(), width=1.0, color='g')
    # plt.xlabel("Degrees")
    # plt.ylabel("Occurrences")
    # # plt.xlim(0,400)
    # plt.ylim(0,500)
    # plt.show()
    # plt.savefig(figure_name + '.png')

# command_line_arg = int(sys.argv[1])
# print("command_line: ", command_line_arg)

checkpoint_path = "training_256/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

dropout = 0.2
F = 602
N = 232965
# channels = command_line_arg
l2_reg = 5e-4
learning_rate = .001
num_classes = 41
epochs = 600

# adj_npz = np.load()
sparse_mat = scipy.sparse.load_npz('../data/yelp/yelp_adj.npz')
# yelp_feat = np.load('../data/yelp/yelp.npy')

row_dict = {}

for i in range(716847):
    try:
        count = sparse_mat.getrow(i).count_nonzero()
        row_dict[i] = count
    except:
            row_dict[i] = 0

# print(yelp_feat)


create_node_degree_graph('graphs/full_graph_of_degrees_yelp', row_dict)


# #A = GCNConv.preprocess(adj).astype('f4')
# X_in = Input(shape=(F, ))
# fltr_in = Input((N, ), sparse=True)

# dropout_1 = Dropout(dropout)(X_in)
# graph_conv_1 = GCNConv(channels,
#                        activation='relu',
#                        kernel_regularizer=l2(l2_reg),
#                        use_bias=False)([dropout_1, fltr_in])

# dropout_2 = Dropout(dropout)(graph_conv_1)
# graph_conv_2 = GCNConv(num_classes,
#                        activation='softmax',
#                        use_bias=False)([dropout_2, fltr_in])

# model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
# optimizer = Adam(lr=learning_rate)
# model.compile(optimizer=optimizer,
#               loss='categorical_crossentropy',
#               weighted_metrics=['acc'])
# model.summary()

# train_features = 
# val_features = 

# adj_train.sort_indices()
# adj_val.sort_indices()


# #y_train and y_val needs to be one hot 

# model.fit([train_features, adj_train],
#           y_train,
#           epochs=epochs,
#           validation_data=([val_features, adj_val],y_val),
#           validation_batch_size = len(val_features),
#           batch_size=len(train_features),
#           shuffle=False,
#           callbacks=[cp_callback])
