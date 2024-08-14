""" Create GAN gan_model and its components

Usage

>>> python -m src.model.gan_model

"""

import os
import yaml
import tensorflow as tf
import numpy as np
from omegaconf import DictConfig
from tensorflow.keras.layers import (BatchNormalization, Dense, Flatten, Input,
                                     LeakyReLU, Reshape)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class GANModel:
    def __init__(self, model: DictConfig,
                 discriminator_param: DictConfig,
                 critic_param: DictConfig,
                 generator_param: DictConfig) -> None:
        
        self.model = model
        self.discriminator_param = discriminator_param
        self.generator_param = generator_param
        self.sample_shape = (self.model.get("num_neurons"),)

        self.optimizerD =  tf.keras.optimizers.legacy.Adam(learning_rate=self.model.get("lr"),
                          beta_1=self.model.get("beta_1"))
        self.optimizerG =  tf.keras.optimizers.legacy.Adam(learning_rate=self.model.get("lr"),
                          beta_1=self.model.get("beta_1"))

        # === Build Discriminator === #
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.discriminator_param.get("d_loss"),
                                   optimizer=self.optimizerD,
                                   metrics= [keras.metrics.BinaryAccuracy(),
                                             keras.metrics.TruePositives(),
                                             keras.metrics.TrueNegatives(),
                                             keras.metrics.FalsePositives(),
                                             keras.metrics.FalseNegatives()] )     #self.discriminator_param.get("d_metrics"))

        # === Build Generator ===#
        self.generator = self.build_generator()

        # Create noise vectors as input for generator
        z = Input(shape=(self.model.get("latent_dim"),))
        random_input = self.generator(z)

        # We don't train Discriminator in the network   ???????????????????????? WHY FALSE ????????????????????????
        self.discriminator.trainable = False

        # Discriminator takes generated input as input and determines validity
        validity = self.discriminator(random_input)

        # === Network (aka combined G and D) === #
        self.network = Model(z, validity)

        self.network.compile(loss=generator_param.get("g_loss"),
                             optimizer=self.optimizerG,
                             metrics=[keras.metrics.BinaryAccuracy(),
                                             keras.metrics.TruePositives(),
                                             keras.metrics.TrueNegatives(),
                                             keras.metrics.FalsePositives(),
                                             keras.metrics.FalseNegatives()]  ) #generator_param.get("g_metrics"))

    def build_generator(self):
        def build_block(model: Sequential, output: int,
                        alpha: float, momentum: float,
                        input_dim: int = None) -> None:
            output = int(output)
            if input_dim:
                latent_dim = int( self.model.get("latent_dim") )
                model.add(Input( shape = (latent_dim,) )) # 100
                model.add(Dense(output))
                # model.add(Dense(output, input_dim=))
            else:
                model.add(Dense(output))

            # model.add(LeakyReLU(negative_slope=alpha))
            model.add(LeakyReLU(alpha=alpha))
            model.add(BatchNormalization(momentum=momentum))

        model = Sequential()
        build_block(model=model, output=self.generator_param.get("g_first_output"), # 200
                    alpha=self.generator_param.get("g_alpha"),
                    momentum=self.generator_param.get("g_momentum"),
                    input_dim=self.model.get("latent_dim"))

        build_block(model=model,
                    output=self.generator_param.get("g_first_output") * 2, # 400
                    alpha=self.generator_param.get("g_alpha"),
                    momentum=self.generator_param.get("g_momentum"))

        build_block(model=model,
                    output=self.generator_param.get("g_first_output") * 4, # 800
                    alpha=self.generator_param.get("g_alpha"),
                    momentum=self.generator_param.get("g_momentum")) 

        model.add(Dense(np.prod(self.sample_shape), # 100
                        activation=self.generator_param.get("g_activation")))
        model.add(Reshape(self.sample_shape)) # 100

        noise = Input(shape=(self.model.get("latent_dim"),))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        def build_block(model: Sequential, output: int, alpha: float) -> None:
            output = int(output)

            model.add(Dense(output))
            # model.add(LeakyReLU(negative_slope=alpha))
            model.add(LeakyReLU(alpha=alpha))

        model = Sequential()
        model.add(Input(self.sample_shape)) # 100
        model.add(Flatten())
        # model.add(Flatten(input_shape=self.sample_shape))

        build_block(model=model,
                    output=self.discriminator_param.get("d_first_output"), # 400
                    alpha=self.discriminator_param.get("d_alpha"))

        build_block(model=model,
                    output=self.discriminator_param.get("d_first_output") / 2, # 200
                    alpha=self.discriminator_param.get("d_alpha"))

        model.add(Dense(self.discriminator_param.get("d_last_output"), # 1
                        activation=self.discriminator_param.get("d_activation")))

        img = Input(shape=self.sample_shape)
        validity = model(img)

        return Model(img, validity)

    @staticmethod
    def smooth_positive_labels(labels):
        alpha = np.random.uniform(0, 1, labels.shape)
        labels = labels - 0.3 + (alpha * 0.5)
        return labels


    def save_model(self, save_path, file_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, file_name)
        self.generator.save(path)

    def generate(self, batch_size) -> np.ndarray:
        random_normal_size = (batch_size,
                              self.model.get("latent_dim"))
        noise = np.random.normal(0, 1, size=random_normal_size)
        gen_samples = self.generator.predict(noise)
        new_shape = (batch_size,
                     self.model.get("num_neurons"))
        gen_samples = np.reshape(gen_samples, newshape=new_shape)
        return gen_samples


# @hydra.main(config_path=str(PROJECT_ROOT / 'conf'), config_name='default')
def main(yaml_file): #cfg: omegaconf.DictConfig):
    with open(yaml_file, 'r') as f: 
        params = yaml.full_load(f)

    params = params.get("modelmodule")
    model = GANModel(model=params.get("model"),
                     discriminator_param=params.get("discriminator_param"),
                     critic_param=params.get("critic_param"),
                     generator_param=params.get("generator_param"))



    print("Success!") if model else print("Fail!")


if __name__ == "__main__":
    main("conf/model.yaml")