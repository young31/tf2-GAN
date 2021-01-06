import numpy as np
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, callbacks, layers, losses
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input, Reshape, Concatenat
from tensorflow.keras.models import Model, Sequential, load_model

warnings.filterwarnings('ignore')
tf.config.experimental.set_visible_devices([], 'GPU') # if cannot find gpu device

class BiGAN(keras.Model):
    def __init__(self, d_shape, z_dim):
        super(BiGAN, self).__init__()
        self.d_shape = d_shape
        self.z_dim = z_dim
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.encoder = self.build_encoder()
    
    def compile(self, g_optim, d_optim, e_optim, loss_fn):
        super(BiGAN, self).compile()
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.e_optim = e_optim
        self.loss_fn = loss_fn
        
    def build_encoder(self):
        activation = 'relu'
        
        inputs = Input(shape = (self.d_shape[1], ))
        
        x = Dense(512, kernel_initializer='he_normal')(inputs)
        x = Activation(activation)(x)
        x = Dense(256, kernel_initializer='he_normal')(x)
        x = Activation(activation)(x)
        
        outputs = Dense(self.z_dim)(x)
        
        return Model(inputs, outputs, name='encoder')
        
    def build_generator(self):
        activation = 'relu'
        inputs = Input(shape=(self.z_dim, ))
        
        h = Dense(256)(inputs)
        h = Activation(activation)(h)
        h = BatchNormalization(momentum=0.8)(h)
        
        h = Dense(512)(h)
        h = Activation(activation)(h)
        h = BatchNormalization(momentum=0.8)(h)
        
        h = Dense(1024)(h)
        h = Activation(activation)(h)
        
        outputs = Dense(self.d_shape[1], kernel_initializer='he_normal', activation='tanh')(h)
        return Model(inputs, outputs, name='generator')
    
    def build_discriminator(self):
        activation = leakyrelu
        
        r = Input(shape = (self.d_shape[1], )) # target
        z = Input(shape = (self.z_dim, )) # latent space
        inputs = Concatenate()([r, z])
        
        h = Dense(512)(inputs)
        h = Activation(activation)(h)
        
        h = Dense(256)(h)
        h = Activation(activation)(h)
        
        outputs = Dense(1)(h)
        
        return Model([r, z], outputs, name='dicriminator')
    
    def train_step(self, x):
        batch_size = tf.shape(x)[0]
        
        noise = self.sampler(batch_size)

        fake_labels = tf.ones((batch_size, 1))*0
        real_labels = tf.ones((batch_size, 1))
        labels = tf.concat([fake_labels, real_labels], 0)

        # disc
        with tf.GradientTape() as tape:
            # real 
            enc = self.encoder(x)
            preds_real = self.discriminator([x, enc])
            # fake
            preds_gen = self.generator(noise)
            preds_fake = self.discriminator([preds_gen, noise])
            # concat
            preds = tf.concat([preds_fake, preds_real], 0)
            
            d_loss = self.loss_fn(labels, preds)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optim.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # gen
        with tf.GradientTape() as tape:
            preds_gen = self.generator(noise)
            preds = self.discriminator([preds_gen, noise])
            g_loss = self.loss_fn(real_labels, preds)
            
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optim.apply_gradients(zip(grads, self.generator.trainable_weights))
            
        # enc
        with tf.GradientTape() as tape:
            enc = self.encoder(x)
            preds = self.discriminator([x, enc])
            e_loss = self.loss_fn(fake_labels, preds)
            
        grads = tape.gradient(e_loss, self.encoder.trainable_weights)
        self.e_optim.apply_gradients(zip(grads, self.encoder.trainable_weights))

        return {'d_loss': d_loss, 'g_loss': g_loss, 'e_loss': e_loss}
    
    def sampler(self, batch_size):
        return tf.random.normal(shape=(batch_size, self.z_dim))
