import time
import keras

class EpochDelayCallback(keras.callbacks.Callback):
    def __init__(self, delay_seconds = 5):
        super(EpochDelayCallback, self).__init__()
        self.delay_seconds = delay_seconds

    def on_epoch_end(self, epoch, logs=None):
        print(f"Sleeping for {self.delay_seconds} seconds")
        time.sleep(self.delay_seconds)