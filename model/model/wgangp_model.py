""" Create WGAN-GP model and its components """

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Add, BatchNormalization, Conv1D, Dense,
                                     Flatten, Input, LeakyReLU, Reshape)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.backend as K



from tensorflow.python.framework.ops import disable_eager_execution

from metrics import partial_loss, wasserstein_loss
from tensorflow import keras


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
disable_eager_execution()

class RandomWeightedAverage(Add):
    """ Weighted average between real and fake samples """

    def _merge_function(self, inputs):
        alpha = tf.random.uniform((32, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGPModel:
    def __init__(self, model, discriminator_param, critic_param, generator_param, adam=True):
        self.model = model
        self.critic_param = critic_param
        self.generator_param = generator_param
        self.sample_shape = (self.model.get("num_neurons"), 1)

        if adam:
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.model.get("lr"),
                             beta_1=self.model.get("beta_1"),
                             beta_2=self.model.get("beta_2"))
            
        else:
            optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=self.model.get("lr"))

        self.generator = self.build_generator()
        self.critic = self.build_critic()

        self.generator.trainable = False

        real_sample = Input(shape=self.sample_shape)
        z = Input(shape=(self.model.get("latent_dim"),))
        fake_sample = self.generator(z)
        fake = self.critic(fake_sample)
        
        valid = self.critic(real_sample)

        interpolated_sample = RandomWeightedAverage()([real_sample, fake_sample])
        validity_interpolated = self.critic(interpolated_sample)

        partial_gp_loss = partial_loss(interpolated_sample)

        self.critic_model = Model(inputs=[real_sample, z],
                                  outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(
            loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
            optimizer=optimizer,
            loss_weights=[1, 1, 10],
            metrics=[keras.metrics.BinaryAccuracy(),
                                             keras.metrics.TruePositives(),
                                             keras.metrics.TrueNegatives(),
                                             keras.metrics.FalsePositives(),
                                             keras.metrics.FalseNegatives()]
            
        )

        self.critic.trainable = False
        self.generator.trainable = True
 
        z_gen = Input(shape=(self.model.get("latent_dim"),))
        fake_sample = self.generator(z_gen)
        valid = self.critic(fake_sample)

        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(
            loss=wasserstein_loss, optimizer=optimizer,
            metrics=[keras.metrics.BinaryAccuracy(),
                                             keras.metrics.TruePositives(),
                                             keras.metrics.TrueNegatives(),
                                             keras.metrics.FalsePositives(),
                                             keras.metrics.FalseNegatives()] 

        )

    def build_generator(self):
        def build_block(model: Sequential , filters: int , kernel_size: int, alpha: float, padding: str):
            filters = int(filters)
            kernel_size = int(kernel_size)

            model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                             activation=LeakyReLU(alpha=alpha), padding=padding))
            model.add(BatchNormalization(momentum=self.generator_param.get("g_momentum")))

        model = Sequential(name='generator')
        model.add(Input(shape=(self.model.get("latent_dim"),)))
        model.add(Reshape((self.model.get("latent_dim"), 1)))

        build_block(model=model, filters=self.generator_param.get("g_filters"),
                    kernel_size=self.generator_param.get("g_kernel_size"),
                    alpha=self.generator_param.get("g_alpha"),
                    padding=self.generator_param.get("g_padding"))

        build_block(model=model, filters=self.generator_param.get("g_filters") // 2,
                    kernel_size=self.generator_param.get("g_kernel_size"),
                    alpha=self.generator_param.get("g_alpha"),
                    padding=self.generator_param.get("g_padding"))

        build_block(model=model, filters=self.generator_param.get("g_filters") // 4,
                    kernel_size=self.generator_param.get("g_kernel_size"),
                    alpha=self.generator_param.get("g_alpha"),
                    padding=self.generator_param.get("g_padding"))

        model.add(Flatten())
        model.add(Dense(np.product(self.sample_shape),
                        activation=self.generator_param.get("g_activation")))
        model.add(Reshape(self.sample_shape))

        noise = Input(shape=(self.model.get("latent_dim"),))
        output = model(noise)

        return Model(noise, output)

    def build_critic(self):
        model = Sequential(name='critic')
        model.add(Conv1D(filters=self.critic_param.get("c_filters"),
                         kernel_size=self.critic_param.get("c_kernel_size"),
                         activation=LeakyReLU(alpha=self.critic_param.get("c_alpha")),
                         input_shape=self.sample_shape,
                         padding=self.critic_param.get("c_padding")))

        model.add(Conv1D(filters=self.critic_param.get("c_filters") * 2,
                         kernel_size=self.critic_param.get("c_kernel_size"),
                         activation=LeakyReLU(alpha=self.critic_param.get("c_alpha")),
                         padding=self.critic_param.get("c_padding")))

        model.add(Conv1D(filters=self.critic_param.get("c_filters") * 4,
                         kernel_size=self.critic_param.get("c_kernel_size"),
                         activation=LeakyReLU(alpha=self.critic_param.get("c_alpha")),
                         padding=self.critic_param.get("c_padding")))

        model.add(Flatten())
        model.add(Dense(self.critic_param.get("c_last_output")))

        inp = Input(shape=self.sample_shape)
        validity = model(inp)

        return Model(inp, validity)

    # def save_model(self, save_path, file_name):
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     path = os.path.join(save_path, file_name)
    #     self.generator.save(path)
    
    def generate(self, batch_size) -> np.ndarray:
        random_normal_size = (batch_size,
                              self.model.get("latent_dim"))
        noise = np.random.normal(0, 1, size=random_normal_size)
        gen_samples = self.generator.predict(noise)
        new_shape = (batch_size,
                     self.model.get("num_neurons"))
        gen_samples = np.reshape(gen_samples, newshape=new_shape)
        return gen_samples
 



def main(yaml_model):
    with open(yaml_model, 'r') as f:
        params = yaml.full_load(f)

    params = params.get("modelmodule")
    model = WGANGPModel(model=params.get("model"),
                        discriminator_param=params.get("discriminator_param"),
                        critic_param=params.get("critic_param"),
                        generator_param=params.get("generator_param"))

    print("Success!") if model else print("Fail!")

if __name__ == "__main__":
    main("conf/model.yaml")





         