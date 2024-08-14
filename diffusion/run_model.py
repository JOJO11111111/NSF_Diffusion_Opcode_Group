from dataclasses import dataclass
import os

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import get_default_device
from model import train_model

@dataclass
class BaseConfig:
    def __init__(self, dataset):
        self.DEVICE = get_default_device()
        self.DATASET = dataset
        # For logging inferece images and saving checkpoints.
        self.root_log_dir = os.path.join("Logs_Checkpoints", "Inference")
        self.root_checkpoint_dir = os.path.join("Logs_Checkpoints", "checkpoints")
        # Current log and checkpoint directory.
        self.log_dir = "version_0"
        self.checkpoint_dir = "version_0"

@dataclass
class TrainingConfig:
    def __init__(self, tsteps, epochs, bsize, lr):
        self.TIMESTEPS = tsteps # Define number of diffusion timesteps
        self.IMG_SHAPE = (1, 104) #(1, 32, 32) if BaseConfig.DATASET == "MNIST" else (3, 32, 32)  # (Channels, Height, Width) -> for our implementation would be (1, 1, 256)
        self.NUM_EPOCHS = epochs #30
        self.BATCH_SIZE = bsize#128
        self.LR = lr #2e-4
        self.NUM_WORKERS = 2

@dataclass
class ModelConfig:
    def __init__(self, base_ch, base_ch_mult, att, drop, time_mult):
        self.BASE_CH = base_ch#64  # 64, 128, 256, 512
        self.BASE_CH_MULT = base_ch_mult # 32, 16, 8, 4
        self.APPLY_ATTENTION = att
        self.DROPOUT_RATE = drop #0.01 #0.1
        self.TIME_EMB_MULT = time_mult # 128

def create_base_config(dataset = 'zbot'):
    return BaseConfig(dataset=dataset)

def create_train_config(timesteps = 1000, num_epochs = 50, batch_size = 32, lr = 0.00005):
    return TrainingConfig(tsteps=timesteps, epochs=num_epochs, bsize=batch_size, lr=lr)

def create_model_config(base_ch = 32, base_ch_mult = (1,2,4,8), att = (False, False, False, False), drop = 0.01, time_mult = 2):
    return ModelConfig(base_ch=base_ch, base_ch_mult=base_ch_mult, att=att, drop=drop, time_mult=time_mult)

def evaluate_model(family, diff_path):
    # get real data
    real_data = pd.read_csv(str(f'data\embeddings\{family}.csv'), header=None)
    real_data = real_data[:100]
    real_data['label'] = 0
    # get fake data
    fake_data = pd.read_csv(diff_path, header=None)
    fake_data = fake_data[:100]
    fake_data['label'] = 1
    # Merge the datasets
    merged_data = pd.concat([real_data,fake_data])
    features = merged_data.drop(columns = ["label"])
    targets = merged_data["label"]
    # split data
    X_train,X_test,y_train,y_test = train_test_split( features, targets, test_size=0.2)
    # train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    # evaluate
    y_pred = clf.predict(X_test)
    metrics = {
        'Dataset': family + "_" + diff_path,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Confusion Matrix' : confusion_matrix(y_test, y_pred) ,
        'train_accuracy' : accuracy_score(y_train, clf.predict(X_train)),
        'test_accuracy' : accuracy_score(y_test, y_pred),
        'test_size': len(X_test),
        'train_size': len(X_train)
    }
    # add metrics to file
    file_path = os.path.join("Logs_Checkpoints", "Results")
    file_path = file_path + ".csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics])
    df.to_csv(file_path, index=False)



def main():
    lr = 5e-05
    b_ch = 32
    att = (True, True, True, True)
    t = 2
    families = ['zbot'] 

    for family in families:
        base = create_base_config(dataset=family)
        train = create_train_config(num_epochs=150, lr=lr)
        model = create_model_config(base_ch=b_ch, att=att, time_mult=t)

        sample_path = train_model(BaseConfig= base, TrainingConfig=train, ModelConfig=model)
        evaluate_model(family=base.DATASET, diff_path=sample_path)

def parse_config_file(filename):
    config = {}
    current_section = None

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('BaseConfig:') or line.startswith('ModelConfig:') or line.startswith('TrainingConfig:'):
                current_section = line.split(':')[0]
                config[current_section] = {}
            elif line and current_section:
                key, value = line.split(': ')
                config[current_section][key.strip()] = value.strip()

    return config



if __name__ == '__main__':
    print("Starting Diffusion Model")
    main()


    print("Complete")