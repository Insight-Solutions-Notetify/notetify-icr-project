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
from sklearn.model_selection import train_test_split

from keras import layers, models, callbacks, optimizers
from keras.api.layers import RandomZoom, Rescaling, RandomFlip, RandomRotation, RandomTranslation
from timeit import default_timer as timer

import asyncio
from alive_progress import alive_bar;
import time

from delay_callback import EpochDelayCallback

# Download the dataset from Kaggle
path = '/home/louisoporto/.cache/kagglehub/datasets/crawford/emnist/versions/3'
if not os.path.exists(path):
    try:
        path = kagglehub.dataset_download('crawford/emnist')
    except Exception as e:
        print("Failed to download dataset from Kaggle: ", e)

print("Path to the downloaded dataset: ", path)

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
    
    async def read_images_labels(self, images_filepath, labels_filepath):        
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
        print("Initializing images...")
        with alive_bar(size) as bar:
            for i in range(size):
                images.append([0] * rows * cols)
                bar()
        # images.append([0] * rows * cols)

        # Fill the images with the data read from the images file
        print("Loading images...")
        with alive_bar(size) as bar:
            for i in range(size):
                img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]) # Get data of image using slicing operations
                img = img.reshape(28, 28) # 28 x 28 format size
                img = img.transpose() # Flip horizontally and rotate 90 degrees counter-clockwise.
                images[i][:] = img # Replace the empty 2D array with the actual image data
                bar()
         
        return np.array(images), np.array(labels)
            
    async def load_data(self):
        print("Loading training data...")
        x_train, y_train = await self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)

        print("Loading test data...")
        x_test, y_test = await self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return [x_train, y_train, x_test, y_test] 
    
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
def sample_emnist(dataset = None):
    images_2_show = []
    titles_2_show = []
    for i in range(0, 9):
        r = random.randint(1, 10000)
        images_2_show.append(dataset[0][r])
        titles_2_show.append('training image [' + str(r) + '] = ' + character_by_index[dataset[1][r]])    

    for i in range(0, 3):
        r = random.randint(1, 10000)
        images_2_show.append(dataset[2][r])        
        titles_2_show.append('test image [' + str(r) + '] = ' + character_by_index[dataset[3][r]])    

    show_images(images_2_show, titles_2_show)

### NS4-23-training-function
# This is the entire training function of our CNN model. Just ensure that the model prepares the model and loads the data for training. Later on task will look over exacts of this function.
### LeNet with Keras and TensorFlow ###
# Training module
def train_model(model=None, dataset=None, rounds=10, epoch=60, sleep=30, filename_model=None, filename_weights=None):
    loss_index = -1
    if filename_model:
        if os.path.exists(f'training/{filename_model}'):
            print("Loading from existing")
            model = models.load_model(f'training/{filename_model}')
        if filename_weights:
            loss_index = filename_weights.find("loss") # Check if file is emnist_model_lossx.xx.weights.h5
            if loss_index != -1:
                prev_val_loss: float = float(filename_weights[loss_index + 4:loss_index + 8])
            try:
                model.load_weights(f"training/{filename_weights}")
            except ValueError:
                print("Failed to load existing weights")
            # model.load_weights('training/model_weights.h5')
        else:
            print("No weights file provided")
    else:

        ### NS4-20-CNN-sequential-layer
        ### Review the composition of the this layer sequence of the CNN model. Based on the LENET-5 model and additional use of data augmentation available during training and excluded during testing
        print("Creating new model")
        data_augmentation = models.Sequential([
        layers.RandomRotation(1./9),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomShear((0, 0.2), (0, 0.2)),
        layers.RandomZoom(0.2),
        ])
        # LeNet - 5 modified for 62 outputs
        model = models.Sequential()
        model.add(layers.Input(shape=(28, 28, 1)))
        model.add(data_augmentation)
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
    
    # Divide the dataset[0] and dataset[1] training data into training and validation data
    # 10% of the training data will be used for validation
    x_train, x_val, y_train, y_val = train_test_split(dataset[0], dataset[1], test_size=0.1, random_state=42)
    # x_val, x_train = dataset[0][:69793], dataset[0][69793:] # 10% of the training data
    # y_val, y_train = dataset[1][:69793], dataset[1][69793:] # 10% of the training data

    round_results = []
    start = timer()

    # Check if recompiling on each session is neccessary or loss in performance
    adam = optimizers.Adam(learning_rate=5e-4)
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) # Check what other metrics can be analyzed
    
    filepath = 'training/emnist_model_loss{val_loss:.2f}.weights.h5'
    # checkpoint_path = "training/emnist_model.weights.h5"
    cp_callback = callbacks.ModelCheckpoint(filepath=filepath,
                                            save_weights_only=True,
                                            save_best_only=True,
                                            verbose=1)
    if loss_index != -1:
        cp_callback.best = prev_val_loss # Continue previous weights from last call

    early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            verbose=1,
                                            min_delta=0.001)
    
    # Learning rate annealer
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.2, min_lr=1e-6)

    for round in range(rounds):
        if round != 0:
            model.load_weights(f'training/emnist_model_loss{prev_val_loss:.2f}.weights.h5') # Continue where previous weights were running
            cp_callback.best = prev_val_loss # Continue with previous weights from last call

        # Reset optimzier learning rate for fresh lense on sample data
        adam.learning_rate = 5e-4
        model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) # Check what other metrics can be analyzed)

        history = model.fit(x_train, y_train,
                            epochs=epoch, # 100 epochs is 1 hour with RTX 4080
                            validation_data=(x_val, y_val),
                            # batch_size=32,
                            callbacks=[early_stopping, cp_callback, reduce_lr, EpochDelayCallback(delay_seconds=sleep)],
                            verbose=1)
        
        prev_val_loss = cp_callback.best
        
        print(f"Completed round {round + 1}.\nResults: ")
        # print(history.history)
        evaluation = model.evaluate(dataset[2], dataset[3])
        print(f"test loss: {evaluation[0]:.04f}, test acc: {evaluation[1]:.04f}")

        results = f"Round: {round + 1} ----- test loss, test acc: {evaluation}"
        round_results.append(results)
        
        
        model.save_weights(f'training/emnist_model_{round + 1}.weights.h5') # Save weights instead of model.keras

    model.save('training/emnist_model.keras') # Final save after all operations

    print(f"====================================\n",
           "Total Time consumed for {epoch} epochs -->", timer()-start,
           "Results of all rounds:")
    for result in round_results:
        print(result)


# Testing module
def test_model(model=None, dataset=None, start_index=0, size=0, filename_model=None, filename_weights=None):
    if not os.path.exists(f'training/{filename_model}'):
        print("Can't evaluate without saved model")
    else:
        print("Evaluating accuracy of training model")
        model = models.load_model(f'training/{filename_model}')
        if filename_weights:
            try:
                model.load_weights(f"training/{filename_weights}")
            except ValueError:
                print("Failed to load existing weights")
            # model.load_weights('training/model_weights.h5')

    if size == 0:
        model.evaluate(dataset[2], dataset[3]) # Evaluate the model on the test data
    else:
    
        x_rand_test = dataset[2][start_index:start_index + size]
        y_rand_test = dataset[3][start_index:start_index + size]
        y_rand_test = [character_by_index[ix] for ix in y_rand_test]
        
        result = model.predict(x_rand_test, batch_size=32)
        printed_result = [character_by_index[np.argmax(ix)] for ix in result]

        print(y_rand_test)
        print(printed_result)

        correct = 0
        for i in range(size):
            if (printed_result[i] == y_rand_test[i]):
                correct += 1
        
        print(f"Accuracy: {correct/size}")

# NS4-18-implement-training-tool
# Just check that the loading of the module is good and the CLI interface can go through all possible options
async def main():
    emnist_dataloader = EmnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    dataset = asyncio.gather(emnist_dataloader.load_data()) # Dataset consist of [x_train, y_train, x_test, y_test] (x - images, y - labels)
    # sample_emnist()

    # Check for GPU available
    # available_GPU = "Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))
    # print(available_GPU)
    info_GPU = tf.config.list_physical_devices('GPU')
    # print(info_GPU)
    if info_GPU:
        try:
            for gpu in info_GPU:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            

    model = models.Sequential()

    # Main Loop
    while (True):
        print("(T)rain, (E)valuate, (S)ample EMNIST or (Q)uit?")
        user_input = input("Command: ")

        if user_input.upper() == 'T':
            default = input("(D)efault or (C)ustom?: ")
            if default.upper() == 'D':
                round = 4
                epoch = 60
                sleep = 30
                filename_model = 'emnist_model.keras'
                filename_weights = 'emnist_model.weights.h5'
                print(f"Starting training for {round} rounds and {epoch} epochs. Sleep: {sleep} secs. Saving to file: {filename_model} & {filename_weights}.")
                print("Loading dataset...")
                if type(dataset) != type([]):
                    await dataset
                    dataset = dataset.result()[0]
                train_model(model, dataset, round, epoch, sleep, filename_model, filename_weights)
            elif default.upper() == 'C':
                round = int(input("Rounds of epoch sets: ")) # We split epochs to ensure clearing sessions and no memory leak in the end.
                # epoch = int(input("Epochs (50 epochs = ~30min): "))
                epoch = 60 # This should be decided rather than inputted by user since the number can affect performance (underfitting if too little or overfitting if too high)
                sleep = int(input("Sleep Time Between Epochs(sec): "))
                filename_model = input("Model Filename (Create new if empty): ")
                filename_weights = input("Weights Filename (Empty if none): ")
                print(f"Starting training for {round} rounds and {epoch} epochs. Sleep: {sleep} secs. Saving to file: {filename_model} & {filename_weights}.")
                print("Loading dataset...")
                if type(dataset) != type([]):
                    await dataset
                    dataset = dataset.result()[0]
                train_model(model, dataset, round, epoch, sleep, filename_model, filename_weights)
            else:
                print("Invalid train input")
        elif user_input.upper() == 'E':
            start_index = 0
            print("Retrieving test images and labels...")
            if type(dataset) != type([]):
                await dataset
                dataset = dataset.result()[0]
            print("Total size of test images ", len(dataset[2]))
            size = int(input("Size of input batch(0 for all):"))
            if not size == 0:
                start_index = int(input("Starting index of test_images:"))
            filename_model = 'emnist_model.keras'
            filename_weights = 'emnist_model.weights.h5'
            # filename_model = input("Model Filename: ")
            # filename_weights = input("Weights Filename: ")
            if start_index + size > len(dataset[2]):
                print("Invalid combination of size and start index")
            else:
                print(f"Testing model with {size} images starting from index {start_index}")
                test_model(model, dataset, start_index, size, filename_model, filename_weights)
        elif user_input.upper() == 'S':
            print("Loading dataset")
            if type(dataset) != type([]):
                await dataset
                dataset = dataset.result()[0]
            sample_emnist(dataset)
        elif user_input.upper() == 'Q':
            break
        else:
            print("Invalid input try another")

        print("Leaving program")


if __name__ == '__main__':
    asyncio.run(main())