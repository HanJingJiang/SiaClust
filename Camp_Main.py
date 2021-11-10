"""
This is a modified version of the Keras mnist example.
https://keras.io/examples/mnist_cnn/

Instead of using a fixed number of epochs this version continues to train until a stop criteria is reached.

A siamese neural network is used to pre-train an embedding for the network. The resulting embedding is then extended
with a softmax output layer for categorical predictions.

Model performance should be around 99.84% after training. The resulting model is identical in structure to the one in
the example yet shows considerable improvement in relative error confirming that the embedding learned by the siamese
network is useful.
"""

from __future__ import print_function
import warnings

warnings.filterwarnings("ignore")

import os, csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Concatenate
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Input, Flatten, Dense
import numpy as np
from sklearn.model_selection import train_test_split
from siamese import SiameseNetwork

def ReadMyCsv2(SaveList, fileName, mode=0):
    csv_reader = csv.reader(open(fileName, encoding="utf-8-sig"))
    for row in csv_reader:
        if not mode:
            row = [float(x) for x in row]
        else:
            row = int(row[0]) - 1
        SaveList.append(row)
    return


def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def storFile2(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(map(lambda x:[x],data))
    return
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName, encoding="utf-8-sig"))
    for row in csv_reader:
        SaveList.append(row)
    return

label = []
ReadMyCsv2(label, "camp1-Label.csv",1)
SMat = []
ReadMyCsv(SMat, "LogSigmoidcamp1.csv")
# Ze = np.zeros((268, 2))# deng
# Ze = np.zeros((430, 10)) #Patel
# Ze = np.zeros((124, 2)) #Goo
Ze = np.zeros((777, 3))
data = np.hstack((SMat, Ze))
mult = 10

batch_size = 3
num_classes = 7
epochs = 10
img_rows, img_cols = 60, 130 # Yan
# img_rows, img_cols = 22, 20 #patel
x, y = [], []
x = data
y = label
x, y = np.array(x).astype(np.float), np.array(y)

# p = r'GoolamLabel.csv'
# with open(p, encoding='utf-8') as f:
#     y = np.loadtxt(f, dtype=np.int, delimiter=",")
# np.savetxt("GoolamLabel.csv",y,delimiter=",", fmt='%d')

# label = label.astype(int)
x = np.concatenate([x] * mult, axis=1)
print(y)
print("x.shape y.shape:", x.shape, y.shape)

# the data, split between train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("1", type(x_train[0][0]))

# x_train = np.concatenate([x_train] * mult, axis=1)
# x_test = np.concatenate([x_test] * mult, axis=1)

# y_train = np.concatenate([y_train] * mult, axis=1)
print("2", x_train)
print("3", type(y_train))
print("4", type(y_train[0]))
x_train = np.concatenate([x_train] * mult, axis=0)
y_train = np.concatenate([y_train] * mult, axis=0)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    x = x.reshape(x.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x = x.reshape(x.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
print("x_train.shape", x_train.shape)
print("x_test.shape", x_test.shape)
print("y.shape",y.shape)
print("y_test.shape",y_test.shape)
print("type(y_test)", type(y_test))
print("type(y)",type(y))


def create_base_model(input_shape):
    model_input = Input(shape=input_shape)

    embedding = Conv2D(16, kernel_size=(3, 3), input_shape=input_shape)(model_input)

    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)
    embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
    embedding = Conv2D(8, kernel_size=(3, 3))(embedding)
    embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
    embedding = Conv2D(8, kernel_size=(3, 3))(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)
    # embedding = MaxPooling2D(pool_size=(1, 1))(embedding)


    embedding = Flatten()(embedding)
    embedding = Dense(32, name="dense-32")(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)

    return Model(model_input, embedding)


def create_head_model(embedding_shape):
    embedding_a = Input(shape=embedding_shape)
    embedding_b = Input(shape=embedding_shape)

    head = Concatenate(name='result')([embedding_a, embedding_b])
    head = Dense(8, name='Dense')(head)
    head = BatchNormalization()(head)
    head = Activation(activation='sigmoid')(head)

    head = Dense(1)(head)
    head = BatchNormalization()(head)
    head = Activation(activation='sigmoid')(head)

    return Model([embedding_a, embedding_b], head)


base_model = create_base_model(input_shape)
head_model = create_head_model(base_model.output_shape)
print(head_model)

# for i in range(1):
siamese_checkpoint_path = "./siamese_checkpoint.hdf5"

siamese_network = SiameseNetwork(base_model, head_model)
siamese_network.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])

siamese_callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(siamese_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=0)
]

siamese_network.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=siamese_callbacks)

siamese_network.load_weights(siamese_checkpoint_path)
embedding = base_model.outputs[-1]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Add softmax layer to the pre-trained embedding network
embedding = Dense(num_classes)(embedding)
embedding = BatchNormalization()(embedding)
embedding = Activation(activation='sigmoid')(embedding)

model = Model(base_model.inputs[0], embedding)
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])

model_checkpoint_path = "./model_checkpoint.hdf5"

model__callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(model_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=0)
]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=model__callbacks,
          validation_data=(x_test, y_test),
          verbose=1)


model.load_weights(model_checkpoint_path)
score = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



dense2_layer_model = Model(inputs=model.input,outputs=model.get_layer('dense-32').output)
MatrixDense = []
MatrixDense = dense2_layer_model.predict(x)
# print(dense2_layer_model.predict(x_test))
storFile(MatrixDense, 'Log-camp1-MatrixDense.csv')
# print(len(MatrixDense[0]))
# print(len(MatrixDense))
# print(y.shape)
# print(type(y_test))
# print(type(y))
storFile2(y,'Log-camp1-LabelDense.csv')