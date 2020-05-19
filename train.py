import tensorflow as tf
#from frankModel import FrankNet  #supposedly tensorcore optimized
from frankModel_ts import FrankNet  # The model we are gonna use to train
from benchmark import TimeRecorder
from benchmark import sysInfo
from logReader import Reader
from sklearn.model_selection import train_test_split
import time
import numpy as np
import os


#! Training Configuration
EPOCHS = 15 #EPOCHS
INIT_LR = 1e-3   #LEARNING RATE
BS = 256        #Batch Size 
GPU_COUNT = 1    # Change this value if you are using multiple GPUs
MULTI_GPU = False #Change this to enable multi-GPU

#! Log Interpretation
STORAGE_LOCATION = "trained_models/behavioral_cloning"

#! Global training data storage
# TODO: This should be optimized?
observation = []
linear = []
angular = []

sysInfo()

def load_data():
    global observation, linear, angular
    reader = Reader('train.log')
    observation, linear, angular = reader.read()
    observation = np.array(observation)
    linear = np.array(linear)
    angular = np.array(angular)
    print('Observation Length: ', len(observation))
    print('Linear Length: ', len(linear))
    print('Angular Length: ', len(angular))
    # exit()
    return


# -----------------------------------------------------------------------------
# Define custom loss functions for regression in Keras
# -----------------------------------------------------------------------------
# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


#!================================================================
# 1. Load all the datas
load_data()
print('Load all complete')

# 2. Split training and testing
observation_train, observation_valid, linear_train, linear_valid, angular_train, angular_valid = train_test_split(
    observation, linear, angular, test_size=0.2, shuffle=True)

# 3. Build the model
single_model = FrankNet.build(200, 150)

# 4. Define the loss function and weight
losses = {
    "Linear": "mse",
    "Angular": "mse"
}
lossWeights = {"Linear": 0.001, "Angular": 10.0}

# 5. Select optimizer
opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# 6. Select loss metrics
metrics_list = ["mse", rmse, r_square]

# 7. Select if multi GPU training or single GPU training.
if MULTI_GPU:
    print("Currently using multiple GPUs")
    model = tf.keras.utils.multi_gpu_model(single_model, gpus=GPU_COUNT) #TODO: Fix using the tf.distribute
else:
    print("Currently using single GPUs")
    model = single_model

# 8. Compile model and plot to see
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
              metrics=metrics_list)

bencher = TimeRecorder()

callbacks_list = [bencher]

# 11. GO!
history = model.fit(observation_train,
                    {"Linear": linear_train,
                        "Angular": angular_train}, validation_data=(observation_valid, {
                            "Linear": linear_valid, "Angular": angular_valid}),
                    epochs=EPOCHS, callbacks=callbacks_list, verbose=1)

# 12. Write benchmark results
bencher.write_result()