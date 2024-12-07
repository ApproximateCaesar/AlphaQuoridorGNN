# ====================
# Creating the Dual Network
# ====================

# Importing packages
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

# Preparing parameters
from constants import BOARD_SIZE
DN_FILTERS  = 128  # Number of kernels in the convolutional layer (256 in the original version)
DN_RESIDUAL_NUM =  16  # Number of residual blocks (19 in the original version)
DN_INPUT_SHAPE = (BOARD_SIZE, BOARD_SIZE, 6)  # Input shape
DN_POLICY_OUTPUT_SIZE = BOARD_SIZE**2 + 2*(BOARD_SIZE-1)**2  # Number of possible actions = N^2 player positions + 2*(N-1)^2 wall placements

# Creating the convolutional layer
def conv(filters):
    return Conv2D(filters, 3, padding='same', use_bias=False,
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

# Creating the residual block
def residual_block():
    def f(x):
        sc = x
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation('relu')(x)
        return x
    return f

# Creating the dual network
def dual_network():
    # Do nothing if the model is already created
    if os.path.exists('./model/best.keras'):
        return

    # Input layer
    input = Input(shape=DN_INPUT_SHAPE)

    # Convolutional layer
    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks x 16
    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    # Pooling layer
    x = GlobalAveragePooling2D()(x)

    # Policy output
    p = Dense(DN_POLICY_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
              activation='softmax', name='pi')(x)

    # Value output
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation('tanh', name='v')(v)

    # Creating the model
    model = Model(inputs=input, outputs=[p, v])

    # Saving the model
    os.makedirs('./model/', exist_ok=True)  # Create folder if it does not exist
    model.save('./model/best.keras')  # Best player's model

    # Clearing the model
    K.clear_session()
    del model

# Running the function
if __name__ == '__main__':
    dual_network()
