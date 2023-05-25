from tensorflow.keras.layers import Input, Dense, Lambda,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.callbacks import TensorBoard, History,EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import regularizers , initializers
import tensorflow.keras


def AEModel( latent_dim, original_dim =84, alpha = 1e-2):
    use_bias = True
    dim1 = 20
    dim2 = 15
    latent_dim = latent_dim

    x = Input(shape=(original_dim,))
    
    reg = regularizers.l2(alpha)

    intit = initializers.RandomNormal(mean=0.0, stddev = 0.05, seed=None)
    
    enc = Dense(dim1, activation="elu", 
              use_bias=use_bias, kernel_regularizer=reg,
              kernel_initializer=intit)(x)
    #enc = BatchNormalization()(enc)
    
    enc = Dense(dim2, activation="elu",
                use_bias=use_bias, kernel_regularizer=reg,
                kernel_initializer=intit)(enc)
    #enc = BatchNormalization()(enc)

    z = Dense(latent_dim, activation="elu",
              use_bias=use_bias, kernel_regularizer=reg,
              kernel_initializer=intit)(enc)
    
    #z = BatchNormalization()(z)
    
    dec = Dense(dim2,activation="elu",
                use_bias=use_bias, kernel_regularizer=reg,
                kernel_initializer=intit)(z)
    #dec = BatchNormalization()(dec)

    dec = Dense(dim1,activation="elu",
                use_bias=use_bias, kernel_regularizer=reg,
                kernel_initializer=intit)(dec)
    
    #dec = BatchNormalization()(dec)

    xhat = Dense(original_dim,
                 kernel_initializer=intit)(dec)

    # Full autoencoder map data to bottleneck and reconstruct it
    model_AE = Model(x, xhat)
    
    # Encoder model map data to bottleneck
    model_enc = Model(x, z)
    
    # Decoder model reconstruct the data by bottleneck  
    model_dec = Model(z, xhat)
    
    return model_AE, model_enc, model_dec


