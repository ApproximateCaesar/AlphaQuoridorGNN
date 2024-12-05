# ====================
# parameter update part
# ====================

from dual_network import DN_INPUT_SHAPE
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle

NUM_EPOCH = 100
BATCH_SIZE = 128

def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)
    
# Training the dual network
def train_network():
    # Loading training data
    history = load_data()
    s, p, v = zip(*history)

    # Reshaping the input data for training
    a, b, c = DN_INPUT_SHAPE
    s = np.array(s)
    s = s.reshape(len(s), c, a, b).transpose(0, 2, 3, 1)
    p = np.array(p)
    v = np.array(v)

    # Loading the best player's model
    model = load_model('./model/best.keras')

    # Compiling the model
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    # Learning rate
    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)

    # Output
    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch, logs: print('\rTraining model (epoch {}/{})'.format(epoch + 1, NUM_EPOCH), end='')
    )

    # Executing training
    model.fit(
        s, [p, v], batch_size=BATCH_SIZE , epochs=NUM_EPOCH, verbose=0, callbacks=[lr_decay, print_callback]
    )

    # Saving the latest player's model
    model.save('./model/latest.keras')

    # Clearing the model
    K.clear_session()
    del model

if __name__ == '__main__':
    train_network()
