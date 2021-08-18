import scipy.sparse as sp
from spektral.layers import GCNConv
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np

def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels, label_encoder.classes_


def main():

    data_path = '../../.spektral/datasets/GraphSage/reddit/'
    data = np.load(data_path+"reddit.npz")
    # adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    adj = sp.csr_matrix(
            (data["adj_data"], (data["adj_row"], data["adj_col"])),
            shape=data["adj_shape"],
        )
    # ['x', 'adj_data', 'adj_row', 'adj_col', 'adj_shape', 'y', 'mask_tr', 'mask_va', 'mask_te']
    print(data.files)
    print("adj", adj.shape)
    print("x", data['x'].shape)  #feat
    print("y", data['y'].shape) #labels
    print(data['mask_tr'].shape)
    print(data['mask_va'].shape)

    log_dir = 'logs'
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    labels_encoded = data['y']


    X = data['x']
    F = data['x'].shape[1]
    N = data['x'].shape[0]
    num_classes = data['y'].shape[1]
    val_mask = data['mask_va']
    test_mask = data['mask_te']
    train_mask = data['mask_tr']
    
    # labels_encoded, classes = encode_label(labels)
    
    # Parameters
    channels = 160          # Number of channels in the first layer
    dropout = 0.5           # Dropout rate for the features
    l2_reg = 5e-4           # L2 regularization rate
    learning_rate = 1e-2    # Learning rate
    epochs = 10           # Number of training epochs
    es_patience = 10        # Patience for early stopping

    A = GCNConv.preprocess(adj).astype('f4')
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

    # Build model
    model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                weighted_metrics=['acc'])
    model.summary()

    tbCallBack_GCN = tf.keras.callbacks.TensorBoard(
        log_dir='./Tensorboard_GCN_reddit',
    )

    #_________________________________________-

    # Train model
    validation_data = ([X, A], labels_encoded, val_mask)
    print(type(X))
    print(type(A))
    print(type(labels_encoded))
    print(type(val_mask))

# <class 'numpy.ndarray'>
#<class 'scipy.sparse.csr.csr_matrix'>
#<class 'numpy.ndarray'>
#<class 'numpy.ndarray'>
    model.fit([X, A],
            labels_encoded,
            sample_weight=train_mask,
            epochs=epochs,
            batch_size=N,
            validation_data=validation_data,
            shuffle=False,
            callbacks=[
                EarlyStopping(patience=es_patience,  restore_best_weights=True),
                tbCallBack_GCN
            ])
    # Evaluate model
    X_te = X[test_mask]
    A_te = A[test_mask,:][:,test_mask]
    y_te = labels_encoded[test_mask]

    M = X_te.shape[0]
    # print("batch size:", N)

    y_pred = model.predict([X_te, A_te], batch_size=M, callbacks=[tb_callback])
    report = classification_report(np.argmax(y_te,axis=1), np.argmax(y_pred,axis=1)) #, target_names=classes
    print('GCN Classification Report: \n {}'.format(report))




if __name__ == "__main__":
    main()