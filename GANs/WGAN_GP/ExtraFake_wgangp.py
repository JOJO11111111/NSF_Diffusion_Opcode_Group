import keras
import numpy as np
from numpy import savetxt
import os
from pathlib import Path
import yaml
from keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




families = [ 'Vobfus', 'Diplugem', 'Obfuscator', 'Vundo', 'VBInject',
            'OnLineGames', 'Renos','winwebsec', 'Delf',  'Enterak.A', 
            'Allaple.A', 'Injector', 'Systex.A', 'Expiro.BK', 'FakeRean', 
            'Small', 'Toga!rfn', 'Lamechi.B', 'CeeInject','Beebone',
            'Hotbar', 'DelfInject', 'zbot', 'zeroaccess','Startpage']
families = ['zbot']


# best full families


# family sample numbers
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


with open("conf/model.yaml", 'r') as f:
    model_params = yaml.full_load(f).get("modelmodule")

latent_dim = model_params.get("model").get("latent_dim")
num_neurons = model_params.get("model").get("num_neurons")

print (num_neurons) # 100
print(latent_dim)  # 100


# generate sammple function
def generate(model, batch_size) -> np.ndarray:
    random_normal_size = (batch_size, latent_dim)
    noise = np.random.normal(0, 1, size=random_normal_size)
    gen_samples = model.predict(noise) 
    # reshape
    new_shape = (batch_size, num_neurons)
    gen_samples = np.reshape(gen_samples, newshape=new_shape)
    return gen_samples



for family in families:
    num_samples=family_to_num[family]
    print("family is: "+family + "generating samples: " + str(num_samples))
    i = best_model[family]
    
    # generate malware using saved model per1k
    for i in range(1, 101):
        file_folder = "models/WGANGP/save_interval"+family+"_"+str(latent_dim)

        file_name = ("/size"+str(latent_dim)+
                "_critic"+str(model_params.get("critic_param")["n_critic_usual"])+
                "_cf"+str(model_params.get("critic_param")["c_filters"])+
                "_gf"+ str(model_params.get("generator_param")["g_filters"])+
                "_lr"+ str(model_params.get("model")["lr"])+
            "WGANGP_" +family+"_" + str(i) +"000.h5")

        file_path = file_folder+file_name
        model = load_model(file_path, compile=True)
        fake_samples = generate(model,num_samples)
        project_root = Path(os.getcwd())
        save_path_folder = os.path.join(project_root,"data","fake_samples", "WGAN-GP", family)

        if not os.path.exists(save_path_folder):
            os.makedirs(save_path_folder)

        save_path = ("size"+str(latent_dim)+
                    "_critic"+str(model_params.get("critic_param")["n_critic_usual"])+
                    "_cf"+str(model_params.get("critic_param")["c_filters"])+
                    "_gf"+ str(model_params.get("generator_param")["g_filters"])+
                    "_lr"+ str(model_params.get("model")["lr"]) +"_"+family+"_Fake_at" + str(i) + "k.csv")
        
        
  
        full_save_path = os.path.join(save_path_folder, save_path)
        savetxt(full_save_path, fake_samples, delimiter=',')

    print("Successfully printed extra fake data.")

