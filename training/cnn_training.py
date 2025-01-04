#"Create env environment before proceeding in the training process.\n"
#"Run the following command in the terminal: python3 -m venv env\n"
#"Activate the environment by running: source env/bin/activate\n"
#"Install the required packages by running: pip install -r requirements.txt\n"

import kagglehub # Uploading from kagglehub the EMNIST dataset from crawford

import os # os and sys to chdir to the project working directory
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU usage instead of CUDA gpu
import sys

import numpy as np # method to output images
import struct
from array import array

import random
import matplotlib.pyplot as plt

import tensorflow as tf

from keras import layers, models, callbacks, optimizers
from timeit import default_timer as timer


from delay_callback import EpochDelayCallback

# Download the dataset from Kaggle
path = kagglehub.dataset_download('crawford/emnist')
# print("Path to the downloaded dataset: ", path)

# Set working directory to current folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)
# print("Project root: ", project_root)
# print("System root: ", sys.path)


# Replaced for pandas read csv
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

character_by_index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

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
        titles_2_show.append('training image [' + str(r) + '] = ' + character_by_index[y_train[r]])    

    for i in range(0, 3):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])        
        titles_2_show.append('test image [' + str(r) + '] = ' + character_by_index[y_test[r]])    

    show_images(images_2_show, titles_2_show)


### LeNet with Keras and TensorFlow ###
# Training module
def train_model(model=None, rounds=10, epoch=60, sleep=30, filename_model=None, filename_weights=None):
    if filename_model:
        if os.path.exists(f'training/{filename_model}'):
            print("Loading from existing")
            model = models.load_model(f'training/{filename_model}')
        if filename_weights:
            try:
                model.load_weights(f"training/{filename_weights}")
            except ValueError:
                print("Failed to load existing weights")
            # model.load_weights('training/model_weights.h5')
        else:
            print("No weights file provided")
    else:
        print("Creating new model")
    
    start = timer()

    # Check if recompiling on each session is neccessary or loss in performance
    adam = optimizers.Adam(learning_rate=5e-4)
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) # Check what other metrics can be analyzed
    
    checkpoint_path = "training/emnist_model.weights.h5"
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            save_best_only=True,
                                            verbose=1)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            verbose=1)
    
    # Learning rate annealer
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.2, min_lr=1e-6)

    for round in range(rounds):
        history = model.fit(x_train, y_train,
                            epochs=epoch, # 100 epochs is 1 hour with RTX 4080
                            validation_data=(x_test, y_test),
                            batch_size=16,
                            callbacks=[early_stopping, cp_callback, reduce_lr, EpochDelayCallback(delay_seconds=sleep)],
                            verbose=1)
        

        print(f"Completed round {round}. Results: ")
        print(history.history)
        evaluation = model.evaluate(x_test, y_test)
        print("test loss, test acc:", evaluation)
        
        model.save(f'training/emnist_model_{round}.keras') # Save between each round of epoch ()
        # keras.backend.clear_session() # Unecessary since we are maintaining the model used from the start
     

    print(f"Total Time consumed for {epoch} epochs -->", timer()-start)

    model.save('training/emnist_model.keras') # Final save after all operations

# Testing module
def test_model(model=None, start_index=0, size=100, filename_model=None):
    if not os.path.exists(f'training/{filename_model}'):
        print("Can't evaluate without saved model")
    else:
        print("Evaluating accuracy of training model")
        model = models.load_model(f'training/{filename_model}')

    x_rand_test = x_test[start_index:start_index + size]
    y_rand_test = y_test[start_index:start_index + size]
    print(x_rand_test)
    print(y_rand_test)
    
    result = model.predict(x_rand_test, size, 1, )
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
    print("(T)rain, (E)valuate, (S)ample EMNIST or (Q)uit?")
    user_input = input("Command: ")

    if user_input.upper() == 'T':
        default = input("(D)efault or (C)ustom?: ")
        if default.upper() == 'D':
            round = 1
            epoch = 60
            sleep = 30
            filename_model = 'emnist_model.keras'
            filenmae_weights = 'emnist_model.weights.h5'
            train_model(model, round, epoch, sleep, filename_model, filenmae_weights)
            break
        elif default.upper() == 'C':
            round = int(input("Rounds of epoch sets: ")) # We split epochs to ensure clearing sessions and no memory leak in the end.
            # epoch = int(input("Epochs (50 epochs = ~30min): "))
            epoch = 60 # This should be decided rather than inputted by user since the number can affect performance (underfitting if too little or overfitting if too high)
            sleep = int(input("Sleep Time Between Epochs(sec): "))
            filename_model = input("Model Filename (Create new if empty): ")
            filename_weights = input("Weights Filename (Empty if none): ")
            train_model(model, round, epoch, sleep, filename_model, filename_weights)
            break
        else:
            print("Invalid train input")
    elif user_input.upper() == 'E':
        start_index = int(input("Starting index of test_images:"))
        size = int(input("Size of input batch:"))
        filename_model = input("Model Filename: ")
        test_model(model, start_index, size, filename_model)
        break
    elif user_input.upper() == 'S':
        sample_emnist()
    elif user_input.upper() == 'Q':
        break
    else:
        print("Invalid input try another")

