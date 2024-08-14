""" Create Data Module

Usage

>>> python -m src.data.data_module

"""


import glob
import os

import yaml

# import coloredlogs
# import hydra
import numpy as np

import pandas as pd

from pathlib import Path

# from src.common.utils import PROJECT_ROOT, MyTimer

# logger = logging.getLogger(__name__)
# coloredlogs.install(level=logging.DEBUG, logger=logger)


class MyDataModule:
    """Build data module"""

    def __init__(self, datasets) -> None:
        """ Initialize data variables from hydra config

        Args:
            datasets (DictConfig): dataset contains train
        """
        self.datasets = datasets

    def prepare_data(self) -> list:
        project_root = Path(os.getcwd())
        path = os.path.join(
            project_root, self.datasets.get("train").get("raw_path_tiff"),
            # project_root, self.datasets.get("train").get("raw_path"),
            self.datasets.get("train").get("family")
        )

        path = path + ".csv"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} was not found. "
                                    f"Current dir: {os.getcwd()}")

        # path = os.path.join(path, '*.txt')
        

        files = glob.glob(path)
        # logger.info(f'Loading {self.datasets.train.family} --- '
        #             f'Number of files found: {len(files)}')

        data = []
        # for file in files:
        #     with open(file, 'r') as rf:
        #         content = rf.read().replace("\n", " ")
        #     data.append(content)

        # with open(path, 'r') as rf:
        #     reader = csv.reader(rf)
        #     for row in reader:
        #         content = " ".join(row)
        #         data.append(content)
                
        # nump = np.array(data)

        data = pd.read_csv(path, header=None)

        return np.array(data)

    def embed_data(self, data: list, distil_bert):
        embedded_data = distil_bert.embed(data)

        # Scale to [-1, 1]
        min_val = np.amin(embedded_data)
        max_val = np.amax(embedded_data)
        embedded_data = 2 * (embedded_data - min_val) / (max_val - min_val) - 1

        return embedded_data


# @hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg):
    # logger.propagate = False

    # Start timer
    # my_timer = MyTimer()

    datamodule = MyDataModule(cfg)
    # hydra.utils.instantiate(cfg.data.datamodule,
    #                                      _recursive_=False)
    data = datamodule.prepare_data()
    print(f'Data size: {len(data)}')
    # print(f"Execution Time: {my_timer.get_execution_time()}")
    print(data)


if __name__ == '__main__':
    yaml_file = ".\\conf\\data.yaml"
    with open(yaml_file, 'r') as f: 
        params = yaml.full_load(f)

    params = params.get("datamodule").get("datasets")
    main(params)
