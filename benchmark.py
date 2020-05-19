import tensorflow as tf
import time
import numpy as np

def sysInfo():
    from tensorflow.python.client import device_lib
    deviceINFO = open(r"deviceSummary.txt","a")
    summary = device_lib.list_local_devices()
    print (summary,file=deviceINFO)


class TimeRecorder(tf.keras.callbacks.Callback):

    def on_train_begin(self,logs={}):
        self.times=[]

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

    def write_result(self):
        np.savetxt("result.csv", self.times, delimiter=",")
