#!/usr/bin/python3
import pandas
import numpy
import matplotlib.pyplot as plt

import cv2
import pickle

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress all warnings from tensorflow

# tensorflow
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, MaxPooling2D, Activation, Conv2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# plotly
from plotly.graph_objects import Scatter
from plotly.subplots import make_subplots


## input data ##
IMDIR = 'data/images'
DF_STYLES = pandas.read_csv('data/styles.csv', error_bad_lines=False, warn_bad_lines=False)
Y_COLUMN = 'masterCategory'
Y_CLASSES = DF_STYLES[Y_COLUMN].unique()

## best params ##
SQUARED_IMSIZE = 50 # size the images are going to be represented
BEST_LR = 0.001 # from models comparision graph
BEST_BS = 256 # from models comparision graph


## output data ##
MODELS_DIR = 'models/'


# get rid of low occurencies
DF_STYLES.drop(DF_STYLES[DF_STYLES[Y_COLUMN] == 'Sporting Goods'].index, inplace=True)
DF_STYLES.drop(DF_STYLES[DF_STYLES[Y_COLUMN] == 'Home'].index, inplace=True)


def show_image_data(imageData):
    plt.imshow(imageData, cmap='gray')
    plt.show()
    print(imageData)


def load_parse_images():
    global DF_STYLES
    training_data = []
    for image in os.listdir(IMDIR):
        try:
            image_name = image.split('.')[0]
            imData = cv2.imread(os.path.join(IMDIR, image), cv2.IMREAD_GRAYSCALE)
            imData = cv2.resize(imData, (SQUARED_IMSIZE, SQUARED_IMSIZE))
            id = DF_STYLES.loc[DF_STYLES['id'].astype(int) == int(image_name)]
            if not id.empty:
                training_data.append([imData, id[Y_COLUMN].values[0]])
            # show_image_data(imData)
        except Exception as e:
            pass
    return training_data


def separate_training_data(data):
    X = []
    y = []
    for imData, labels in data:
        X.append(imData)
        y.append(labels)
    X = numpy.array(X).reshape(-1, SQUARED_IMSIZE, SQUARED_IMSIZE, 1)
    X = X/255
    y = LabelEncoder().fit_transform(numpy.array(y))
    y = to_categorical(y)
    return X, y

def save_data_to_pickle():
    X, y = separate_training_data(
        load_parse_images()
    )
    pickle_out = open("data/X_{0}x{0}.pickle".format(SQUARED_IMSIZE), 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open('data/y_{0}x{0}.pickle'.format(SQUARED_IMSIZE), 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()


def load_data_from_pickle():
    X_pickle = open('data/X_{0}x{0}.pickle'.format(SQUARED_IMSIZE), 'rb')
    y_pickle = open('data/y_{0}x{0}.pickle'.format(SQUARED_IMSIZE), 'rb')
    X = pickle.load(X_pickle)
    y = pickle.load(y_pickle)
    X_pickle.close()
    y_pickle.close()
    return X, y


def get_model(filters, kernel_size, pool_size, input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(filters, kernel_size, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())

    model.add(Dense(output_shape, activation='softmax'))

    return model


def train_with_different_params(model, X_train, y_train):
    global MODELS_DIR
    learning_rates = [0.001, 0.002, 0.003, 0.004, 0.005]
    batch_sizes = [8, 32, 256]
    for lr in learning_rates:
        for bs in batch_sizes:
            cnn_name = "cnn_lr-{0}_bs-{1}_{2}".format(lr, bs, int(time.time()))
            tensor_board = TensorBoard(log_dir='history/{0}'.format(cnn_name))
            model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(lr=lr),
                metrics=['accuracy'])
            model.fit(
                X_train,
                y_train,
                batch_size=bs,
                validation_split=0.3,
                epochs=5,
                callbacks=[tensor_board])
            save_model(
                model,
                '{0}/cnn_lr-{1}_bs-{2}.model'.format(MODELS_DIR, lr, bs),
                overwrite=True)
            time.sleep(30.0)


def get_existing_models(directory):
    all_models = []
    for file in os.listdir(directory):
        if file.endswith('.model'):
            all_models.append('{0}/{1}'.format(directory, file))

    return all_models


def compare_models(labels_of_models, accuracies, losses):
    global MODELS_DIR
    for i in range(0, len(labels_of_models)):
        labels_of_models[i] = labels_of_models[i].replace(MODELS_DIR, '')
        labels_of_models[i] = labels_of_models[i].replace('/', '')
        labels_of_models[i] = labels_of_models[i].replace('cnn_', '')
        labels_of_models[i] = labels_of_models[i].replace('.model', '')
    figure = make_subplots(rows=1, cols=2)
    figure.add_trace(
        Scatter(x=labels_of_models, y=accuracies, name='Accuracies', mode='lines'),
        row=1, col=1)
    figure.add_trace(
        Scatter(x=labels_of_models, y=losses, name='Losses', mode='lines'),
        row=1, col=2)
    figure.update_layout(title_text='Best LR and BS {lr : bs}')
    figure.show()


def evaluate_models_in_dir(models_dir, X_test, y_test):
    accs = []
    losses = []
    models = get_existing_models(models_dir)
    for model in models:
        reconstructed_model = load_model(model)
        eval = reconstructed_model.evaluate(X_test, y_test)
        losses.append(eval[0])
        accs.append(eval[1])

    compare_models(models, accs, losses)


def analyze_best_cnn(the_model, X_test, y_test):
    global Y_CLASSES
    eval = the_model.evaluate(X_test, y_test)
    print("Loss  of  the  best  fitted model: {0}".format(eval[0]))
    print("Accuracy of the best fitted model: {0}".format(eval[1]))

    y_pred = numpy.argmax(the_model.predict(X_test), axis=-1)
    y_test = numpy.argmax(y_test, axis=-1)

    classreport = classification_report(
        y_test,
        y_pred,target_names=Y_CLASSES)
    print(classreport)


# re-generate our pickled data in case we have changed some params
save_data_to_pickle()

# load pre-made data from pickles
X, y = load_data_from_pickle()

# split loaded data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# build a model for training
model = get_model(128, (3,3), (2,2), X_train.shape[1:], len(y_train[0]))

# train the same model but with different learning rate and batch size
train_with_different_params(model, X_train, y_train)

# get all .model files from specified directory and test them
evaluate_models_in_dir(MODELS_DIR, X_test, y_test)

# best_model = load_model(
#         '{0}/cnn_lr-{1}_bs-{2}.model'.format(
#             MODELS_DIR, BEST_LR, BEST_BS))

# analyze_best_cnn(best_model, X_test, y_test)
