print("Create env environment before proceeding in the training process.\n"
      "Run the following command in the terminal: python3 -m venv env\n"
      "Activate the environment by running: source env/bin/activate\n"

        "Install the required packages by running: pip install -r requirements.txt\n")

import kagglehub # Uploading from kagglehub the EMNIST dataset from crawford

import os # os and sys to chdir to the project working directory
import sys

import numpy as np # method to output images
import struct
from array import array

import random
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers, models, callbacks
from timeit import default_timer as timer

from delay_callback import EpochDelayCallback

# Download the dataset from Kaggle
path = kagglehub.dataset_download('crawford/emnist')

# Set working directory to current folder
print("Path to the downloaded dataset: ", path)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)
# print("Project root: ", project_root)
# print("System root: ", sys.path)

#
# EMNIST Data Loader Class
#
class EmnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []

        # Get labels for each datapoint
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            # if magic != 2049:
            #     raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())
        
        # Get images for each datapoint
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            # if magic != 2051:
            #     raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())

        images = [] # Images

        # Makes a list of 2D arrays(images) rows x cols with 0 value
        for i in range(size):
            images.append([0] * rows * cols)

        # Fill the images with the data read from the images file
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]) # Get data of image using slicing operations
            img = img.reshape(28, 28) # 28 x 28 format size
            img = img.transpose() # Flip horizontally and rotate 90 degrees counter-clockwise.
            images[i][:] = img # Replace the empty 2D array with the actual image data
         
        return np.array(images), np.array(labels)
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test) 
    
#
# Set file paths based on added EMNIST Datasets
#
source_files = "/emnist_source_files"
train_image = "/emnist-byclass-train-images-idx3-ubyte"
train_label = "/emnist-byclass-train-labels-idx1-ubyte"
test_image = "/emnist-byclass-test-images-idx3-ubyte"
test_label = "/emnist-byclass-test-labels-idx1-ubyte"

training_images_filepath = f"{path}{source_files}{train_image}"
training_labels_filepath = f"{path}{source_files}{train_label}"
test_images_filepath = f"{path}{source_files}{test_image}"
test_labels_filepath = f"{path}{source_files}{test_label}"

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 3
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(10,10))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.subplots_adjust(hspace=.5)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 10)
        index += 1

    plt.show()

#
# Show some random training and test images 
#
def sample_emnist():
    images_2_show = []
    titles_2_show = []
    for i in range(0, 9):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

    for i in range(0, 3):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])        
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

    show_images(images_2_show, titles_2_show)
    ''


### LeNet with Keras and TensorFlow ###
# Training module
def train_model(model=None, epoch=100, sleep=30):
    if not os.path.exists('training/emnist_model.keras'):
        print("Creating new model")
    else:
        print("Loading from existing")
        model = models.load_model('training/emnist_model.keras')
        # model.load_weights('training/emnist_model.weights.h5')
    
    start = timer()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) # Check what other metrics can be analayzed
    
    checkpoint_path = "training/emnist_model.weights.h5"
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            verbose=1)

    early_stopping = callbacks.EarlyStopping(monitor='val_accuracy',
                                            patience=10)
    history = model.fit(x_train, y_train,
                        epochs=epoch, # 100 epochs is 1 hour with RTX 4080
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping, cp_callback, EpochDelayCallback(delay_seconds=sleep)],
                        verbose=1)
    

    print(f"Total Time consumed for {epoch} epochs -->", timer()-start)

    model.save('training/emnist_model.keras')

# Testing module
def test_model(model=None, start_index=0, size=100):
    if not os.path.exists('training/emnist_model.keras'):
        print("Can't evaluate without saved model")
    else:
        print("Evaluating accuracy of training model")
        model = models.load_model('training/emnist_model.keras')

    x_rand_test = x_test[0:20]
    y_rand_test = y_test[0:20]
    print(x_rand_test)
    print(y_rand_test)
    
    result = model.predict(x_rand_test, 20, 1, )
    result = [np.argmax(ix) for ix in result]

    print(result)

    # Overall accuracy on test images
    model.evaluate(x_test, y_test)

emnist_dataloader = EmnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = emnist_dataloader.load_data()
# sample_emnist()

# Check for GPU available
available_GPU = "Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))
# print(available_GPU)
info_GPU = tf.config.list_physical_devices()
# print(info_GPU)

# LeNet - 5 modified for 62 outputs
model = models.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(strides=2))
model.add(layers.Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu')) 
model.add(layers.MaxPooling2D(strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(62, activation='softmax'))

# model.build()
# Show composition of model
# model.summary()

# Main Loop
while (True):
    print("Train or evaluate?")
    user_input = input("(T) or (E):")

    if user_input.upper() == 'T':
        epoch = int(input("Epochs (50 epochs = ~30min): "))
        sleep = int(input("Sleep Time Between Epochs(sec): "))
        train_model(model, epoch, sleep)
        break
    elif user_input.upper() == 'E':
        start_index = int(input("Starting index of test_images:"))
        size = int(input("Size of input batch:"))
        test_model(model, start_index, size)
        break
    else:
        print("Invalid input try another")

