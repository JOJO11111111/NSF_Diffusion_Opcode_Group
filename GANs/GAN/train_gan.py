""" Train script to train GAN model

Usage

# >>> python -m src.training.train_gan

"""

import logging
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import yaml

# import coloredlogs
# import hydra
import matplotlib.pyplot as plt
# import mlflow
import numpy as np
# import omegaconf
import tensorflow as tf
# from omegaconf import DictConfig

from tensorboardX import SummaryWriter
import datetime
from tqdm import tqdm
from pathlib import Path
from numpy import savetxt

# from src.common.utils import PROJECT_ROOT, MyTimer
from data.data_module import MyDataModule
from GANs.GAN.gan_model import GANModel

# from src.visualization.visualization import plot_losses

# logging.getLogger('git').setLevel(logging.ERROR)
# logging.getLogger('urllib3').setLevel(logging.ERROR)
# logging.getLogger('matplotlib').setLevel(logging.ERROR)
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# logging.getLogger('h5py').setLevel(logging.ERROR)

# logger = logging.getLogger(__name__)
# coloredlogs.install(level=logging.DEBUG, logger=logger)


# def log_params(cfg: DictConfig) -> None:
#     # Log parameters
#     mlflow.log_params(cfg.data.datamodule.datasets.train)
#     mlflow.log_params(cfg.train.trainer)
#     for key, params in cfg.model.modelmodule.items():
#         if key == '_target_' or key == 'critic_param':
#             continue
#         mlflow.log_params(params)

#     for key, params in cfg.bert.bertmodule.items():
#         if key == '_target_':
#             continue
#         mlflow.log_params(params)

#     # Log hydra configs
#     mlflow.log_artifacts(os.path.join(PROJECT_ROOT, 'conf'))

#     # Log python scripts
#     mlflow.log_artifact(
#         os.path.join(PROJECT_ROOT, 'src/model/gan_model.py')
#     )
#     mlflow.log_artifact(
#         os.path.join(PROJECT_ROOT, 'src/training/train_gan.py')
#     )
#     mlflow.log_artifact(
#         os.path.join(PROJECT_ROOT, 'src/data/data_module.py')
#     )


def train(model_params, train_params, data_params, make_plot: bool = False) -> None:
    """Generic train loop"""
    # logger.info('Loading dataset and rescale to [-1, 1] ...')
    print("Loading dataset and rescale to [-1, 1] ...")

    # get dataset
    datamodule = MyDataModule(data_params) 
    X_train = datamodule.prepare_data()
    X_train = np.expand_dims(X_train, axis=2)
    
    print(f'Initializing {model_params.get("model").get("model_name")} '
                'model ...')
    
    # instantiate GAN Model
    gan_model = GANModel(model=model_params.get("model"),
                     discriminator_param=model_params.get("discriminator_param"),
                     critic_param=model_params.get("critic_param"),
                     generator_param=model_params.get("generator_param"))

    # Obtain parameters from hydra, prepare for training
    batch_size = train_params.get("batch_size")
    latent_dim = model_params.get("model").get("latent_dim")

    # Adversarial ground truths
    valid = gan_model.smooth_positive_labels( np.ones((batch_size, 1)) )
    fake = np.zeros((batch_size, 1))

    # valid = torch.from_numpy(valid)
    # fake = torch.from_numpy(fake)

    # Save losses for each epoch for graphs
    results = {}

    # setup tensorboard
    log_dir = "logs/gan_reproduce/" + data_params.get("train").get("family") + "/" + "size"+str(latent_dim)
    writer = SummaryWriter(log_dir)

    print(f'Beginning experiment {train_params.get("experiment_name")} ..')
    
    writer.add_hparams( model_params, {} )
    writer.add_hparams( train_params, {} )

    writer.add_text('Dataset_size: ', str(X_train.shape[0]))

    # @tf.function -- train_on_batch causes the warnings since it uses numpy array and an iterator
    def train_step(real_data, gen_samples, valid, fake):
        # === Train Discriminator on real and fake data, separately ===#
        d_loss_real, acc_real, tp_real, tn_real, fp_real, fn_real = gan_model.discriminator.train_on_batch(real_data, valid)
        d_loss_fake, acc_fake, tp_fake, tn_fake, fp_fake, fn_fake = gan_model.discriminator.train_on_batch(gen_samples, fake)
        
        # Compute Discriminator loss by averaging the 2 losses above
        d_loss = np.add(d_loss_real, d_loss_fake) / 2
        d_real_metrics = [acc_real, tp_real, tn_real, fp_real, fn_real]
        d_fake_metrics = [acc_fake, tp_fake, tn_fake, fp_fake, fn_fake]

 
        # === Train Generator === #
        # Create noise vector as input for Network (combined G and D)
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        g_loss = gan_model.network.train_on_batch(noise, valid)

        return d_loss, d_loss_real, d_loss_fake, g_loss, d_real_metrics, d_fake_metrics
    

    for epoch in tqdm( range(train_params.get("epochs")), "Epochs"):
        # Obtain `batch_size` real data randomly
        idx = np.random.randint(
            low=0, high=X_train.shape[0], size=batch_size
        )
        real_data = X_train[idx]
        # real_data = torch.from_numpy(real_data)

        # Create noise vector as input for Generator
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        # Generate `batch_size` of fake data
        gen_samples = gan_model.generator.predict(noise, verbose = 0)

        d_loss, d_loss_real, d_loss_fake, g_loss, d_real_metrics, d_fake_metrics = train_step(real_data, gen_samples, valid, fake)

        d_correct = d_real_metrics[1] + d_fake_metrics[1] + d_real_metrics[2] + d_fake_metrics[2]
        d_total = ( d_real_metrics[1] + d_real_metrics[2] + d_real_metrics[3] + d_real_metrics[4] ) * 2
        d_acc = d_correct / d_total

        # real_count = d_real_metrics[1] + d_real_metrics[2] + d_real_metrics[3] + d_real_metrics[4]
        # fake_count = d_fake_metrics[1] + d_fake_metrics[2] + d_fake_metrics[3] + d_fake_metrics[4]
        # fake_samples = gan_model.generate(1)
        # bad_predict = np.any((fake_samples < -1) | (fake_samples > 1))
        # if bad_predict:
        #     print("ERROR: Predictions outside valid range")
        #     raise

        writer.add_scalar('Disc_loss', d_loss, epoch)
        writer.add_scalar('Disc_real_loss', d_loss_real, epoch)
        writer.add_scalar('Disc_fake_loss', d_loss_fake, epoch)
        writer.add_scalar('Gen_loss', g_loss[0], epoch)

        writer.add_scalar('Disc_acc', d_acc, epoch)
        writer.add_scalar('Disc_real_acc', d_real_metrics[0], epoch)
        writer.add_scalar('Disc_fake_acc', d_fake_metrics[0], epoch)
        writer.add_scalar('Disc_real_tp', d_real_metrics[1], epoch)
        writer.add_scalar('Disc_fake_tp', d_fake_metrics[1], epoch)
        writer.add_scalar('Disc_real_tn', d_real_metrics[2], epoch)
        writer.add_scalar('Disc_fake_tn', d_fake_metrics[2], epoch)
        writer.add_scalar('Disc_real_fp', d_real_metrics[3], epoch)
        writer.add_scalar('Disc_fake_fp', d_fake_metrics[3], epoch)
        writer.add_scalar('Disc_real_fn', d_real_metrics[4], epoch)
        writer.add_scalar('Disc_fake_fn', d_fake_metrics[4], epoch)

        results[epoch] = [np.float64(d_loss), np.float64(g_loss[0])]

        # Save model every `save_interval`
        if epoch % train_params.get("save_interval") == 0:
            project_root = Path(os.getcwd())
    
            save_path = os.path.join(project_root,
                                     "models", "GAN",
                                     'save_interval',
                                     data_params.get("train").get("family")+"_"+str(latent_dim))
            
            file_name=("size"+str(latent_dim)+
                model_params.get("model").get("model_name") + '_' +
                data_params.get("train").get("family") + '_'
                + str(epoch) + '.keras')
    

            gan_model.save_model(save_path=save_path, file_name=file_name)
            # mlflow.log_artifact(local_path=save_path + '/' + file_name)
    ###########################
    # end indent for model train start_run
    ###########################

    # logger.info('Finished training! '
    #             f'Execution time: {my_timer.get_execution_time()}')

    # if make_plot:
    #     f_d, ax_d, f_g, ax_g = plot_losses(
    #         results, cfg.train.trainer.epochs,
    #         cfg.model.modelmodule.model.model_name
    #     )
    #     plt.legend(loc='upper left')
    #     f_d.savefig(os.path.join(save_path,
    #                              cfg.model.modelmodule.model.model_name +
    #                              '_discriminator.png'))
    #     f_g.savefig(os.path.join(save_path,
    #                              cfg.model.modelmodule.model.model_name +
    #                              '_generator.png'))
    # logger.info('Finished operation!!')

    # print('Generating Samples ...')
    # fake_samples = gan_model.generate(32)
    # project_root = Path(os.getcwd())
    # save_path = os.path.join(project_root,
    #                                  "GAN", "fake_samples")
    # save_path = str(save_path + data_params.get("train").get("family") + ".csv")
    # savetxt(save_path, fake_samples, delimiter=',')


    print('Finished operation!!')


# @hydra.main(config_path=str(PROJECT_ROOT / 'conf'), config_name='default')
def main(yaml_model, yaml_train, yaml_data): #cfg: omegaconf.DictConfig):
    # train(cfg, make_plot=True)
    with open(yaml_model, 'r') as f: 
        params = yaml.full_load(f)
    model = params.get("modelmodule")
    with open(yaml_train, 'r') as f: 
        params = yaml.full_load(f)
    train_p = params.get("trainer")
    with open(yaml_data, 'r') as f: 
        params = yaml.full_load(f)
    data = params.get("datamodule").get("datasets")



    families = [  'CeeInject','Beebone',
            'Hotbar', 'DelfInject', 'zeroaccess','Startpage']
    

    # families = ['Delf', 'Diplugem', 'Obfuscator', 'Vundo', 'VBInject',
    #         'OnLineGames', 'Renos','winwebsec', 'Vobfus',   'Enterak.A', 
    #         'Allaple.A', 'Injector', 'Systex.A', 'Expiro.BK', 'FakeRean', 
    #         'Small', 'Toga!rfn', 'Lamechi.B', 'CeeInject','Beebone',
    #         'Hotbar', 'DelfInject', 'zbot', 'zeroaccess','Startpage']

    # families = [  'zbot']

    for family in families:
        data['train']['family'] = family
        train(model, train_p, data, make_plot=False)



if __name__ == '__main__':
    # logger.propagate = False
    # logger.info("Num GPUs Available: "
    #             f"{len(tf.config.list_physical_devices('GPU'))}")
    # main("conf\\model.yaml", 
    #      "conf\\train.yaml",
    #      "conf\\data.yaml")
    main("conf/model.yaml", 
         "conf/train.yaml",
         "conf/data.yaml")