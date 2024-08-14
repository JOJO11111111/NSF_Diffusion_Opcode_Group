""" Train script to train WGAN-GP model """

import logging
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from data.data_module import MyDataModule
from GANs.WGAN_GP.wgangp_model import WGANGPModel
from numpy import savetxt
import numpy as np
import itertools



# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# logging.getLogger('git').setLevel(logging.ERROR)
# logging.getLogger('urllib3').setLevel(logging.ERROR)
# logging.getLogger('matplotlib').setLevel(logging.ERROR)
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# logging.getLogger('h5py').setLevel(logging.ERROR)

# logger = logging.getLogger(__name__)
# logger.propagate = False

def train(model_params, train_params, data_params, num_samples,make_plot=False):
    print("Loading dataset ...")
    datamodule = MyDataModule(data_params)     # get dataset    
    X_train = datamodule.prepare_data()
    X_train = np.expand_dims(X_train, axis=2)

    print(f'Initializing {model_params.get("model").get("model_name_wgangp")} model ...')
    
    # instantiate GAN Model
    wgangp_model = WGANGPModel(model=model_params.get("model"),
                               discriminator_param=model_params.get("discriminator_param"),
                               critic_param=model_params.get("critic_param"),
                               generator_param=model_params.get("generator_param"))

    batch_size = train_params.get("batch_size")
    latent_dim = model_params.get("model").get("latent_dim")
    
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = -np.ones((batch_size, 1)) # -1 
    dummy = np.zeros((batch_size, 1))
    
    # store the results
    results = dict()

    log_dir = "logs/wgangp/" + data_params.get("train").get("family") + "/" + ("size"+str(latent_dim)+"_critic"+str(model_params.get("critic_param")["n_critic_usual"])+
                            "_cf"+str(model_params.get("critic_param")["c_filters"])+
                            "_gf"+ str(model_params.get("generator_param")["g_filters"])+
                            "_lr"+ str(model_params.get("model")["lr"]))
    
    writer = SummaryWriter(log_dir)

    print(f'Beginning experiment {train_params.get("experiment_name")} ..')
    writer.add_hparams(model_params, {})
    writer.add_hparams(train_params, {})
    writer.add_text('Dataset_size: ', str(X_train.shape[0]))

    # trian_step to get critic and gen loss 
    def train_step(num_to_change, real_sample,noise, valid, fake, dummy):
        #  train critic alone with more times, without training the genarator, 
        for _ in range(num_to_change):
            c_loss_outputs = wgangp_model.critic_model.train_on_batch(x=[real_sample, noise], y=[valid, fake, dummy])
        
        c_loss= c_loss_outputs[0]
        c_loss_real= c_loss_outputs[1]
        c_loss_fake= c_loss_outputs[2]
        c_loss_gp = c_loss_outputs[3]

        g_loss = wgangp_model.generator_model.train_on_batch(noise, valid)
        # g_loss_outputs = wgangp_model.generator_model.train_on_batch(noise, valid)
        c_real_metrics = [c_loss_outputs[4], c_loss_outputs[5], c_loss_outputs[6], c_loss_outputs[7], c_loss_outputs[8]]
        c_fake_metrics = [c_loss_outputs[9], c_loss_outputs[10], c_loss_outputs[11], c_loss_outputs[12], c_loss_outputs[13]]

        return c_loss, c_loss_real, c_loss_fake, c_loss_gp, g_loss, c_real_metrics, c_fake_metrics
    


    
#
    for epoch in tqdm(range(train_params.get("epochs")), "Epochs"):

        # Obtain `batch_size` real data randomly
        idx = np.random.randint(low=0, high=X_train.shape[0], size=batch_size)
        real_sample = X_train[idx]
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))

        if epoch!= 0 and epoch % 500 == 0:
            num_to_change = model_params.get("critic_param").get("n_critic") # 100
        else:
            num_to_change = model_params.get("critic_param").get("n_critic_usual") # 7
        
        c_loss, c_loss_real, c_loss_fake, c_loss_gp, g_loss, c_real_metrics, c_fake_metrics = train_step(num_to_change,real_sample, noise, valid, fake, dummy)
            
        c_correct = c_real_metrics[1] + c_fake_metrics[1] + c_real_metrics[2] + c_fake_metrics[2]
        c_total = ( c_real_metrics[1] + c_real_metrics[2] + c_real_metrics[3] + c_real_metrics[4] ) * 2
        c_acc = c_correct / c_total

        writer.add_scalar('Critic_loss', -c_loss, epoch)
        writer.add_scalar('Critic_real_loss', c_loss_real, epoch)
        writer.add_scalar('Critic_fake_loss', c_loss_fake, epoch)
        writer.add_scalar('Gen_loss_wgangp', g_loss[0], epoch)

        writer.add_scalar('Critic_acc', c_acc, epoch)
        writer.add_scalar('Critic_real_acc', c_real_metrics[0], epoch)
        writer.add_scalar('Critic_fake_acc', c_fake_metrics[0], epoch)

        writer.add_scalar('Critic_real_tp', c_real_metrics[1], epoch)
        writer.add_scalar('Critic_fake_tp', c_fake_metrics[1], epoch)

        writer.add_scalar('Critic_real_tn', c_real_metrics[2], epoch)
        writer.add_scalar('Critic_fake_tn', c_fake_metrics[2], epoch)

        writer.add_scalar('Critic_real_fp', c_real_metrics[3], epoch)
        writer.add_scalar('Critic_fake_fp', c_fake_metrics[3], epoch)

        writer.add_scalar('Critic_real_fn', c_real_metrics[4], epoch)
        writer.add_scalar('Critic_fake_fn', c_fake_metrics[4], epoch)
        writer.add_scalar('Critic_loss_gp', c_loss_gp, epoch)


        results[epoch] = [np.float64(c_loss), np.float64(g_loss[0])]

        if epoch !=0 and (epoch% train_params.get("save_interval") == 0):
            project_root = Path(os.getcwd())
            save_path = os.path.join(project_root, "models", "WGAN-GP", 'save_interval',data_params.get("train").get("family")+"_"+str(latent_dim))
            
            file_name = ("size"+str(latent_dim)+
                            "_critic"+str(model_params.get("critic_param")["n_critic_usual"])+
                            "_cf"+str(model_params.get("critic_param")["c_filters"])+
                            "_gf"+ str(model_params.get("generator_param")["g_filters"])+
                            "_lr"+ str(model_params.get("model")["lr"])+
                            model_params.get("model").get("model_name_wgangp") + '_' +
                            data_params.get("train").get("family") + '_'
                            + str(epoch) + '.h5')

            wgangp_model.generator.save(save_path + '/' + file_name, save_format='h5')



import itertools

def main(yaml_model, yaml_train, yaml_data):
    with open(yaml_model, 'r') as f:
        model_params = yaml.full_load(f).get("modelmodule")
    with open(yaml_train, 'r') as f:
        train_params = yaml.full_load(f).get("trainer")
    with open(yaml_data, 'r') as f:
        data_params = yaml.full_load(f).get("datamodule").get("datasets")

    # families = [ 'zbot']
#

    # sample amounts in each family
    family_to_num = {
        'Delf': 1679,
        'Diplugem': 2269,
        'Obfuscator': 2102,
        'Vundo': 1877,
        'VBInject': 1688,
        'Vobfus': 4204,
        'Beebone': 1629,
        'Enterak.A': 1530,
        'OnLineGames': 1366,
        'Startpage': 1313,
        'Allaple.A': 1294,
        'Injector': 1161,
        'Systex.A': 1098,
        'Expiro.BK': 1095,
        'FakeRean': 1089,
        'Small': 1051,
        'Toga!rfn': 985,
        'Lamechi.B': 971,
        'CeeInject': 903,
        'Renos': 864,
        'Hotbar': 844,
        'DelfInject': 839,
        'winwebsec': 4355,
        'zbot': 2136,
        'zeroaccess': 1305
    }
    
    families = [ 'Vobfus', 'Diplugem', 'Obfuscator', 'Vundo', 'VBInject',
            'OnLineGames', 'Renos','winwebsec', 'Delf',  'Enterak.A', 
            'Allaple.A', 'Injector', 'Systex.A', 'Expiro.BK', 'FakeRean', 
            'Small', 'Toga!rfn', 'Lamechi.B', 'CeeInject','Beebone',
            'Hotbar', 'DelfInject', 'zbot', 'zeroaccess','Startpage']

    for family in families:
        data_params['train']['family']=family
        num_samples=family_to_num[family]
        train(model_params, train_params, data_params,num_samples,make_plot=False)

if __name__ == '__main__':
    main("conf/model.yaml", "conf/train.yaml", "conf/data.yaml")
