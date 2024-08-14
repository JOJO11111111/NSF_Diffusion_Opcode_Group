import os
from PIL import Image
import torch
import torchvision
import csv

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get(element: torch.Tensor, t: torch.Tensor):
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, t)
    # return ele.reshape(-1, 1, 1, 1)
    return ele.reshape(-1, 1, 1)

def setup_log_directory(config):
    '''Log and Model checkpoint directory Setup'''
    v = 0
    if os.path.isdir(config.root_log_dir):
        # Get all folders numbers in the root_log_dir
        #folder_numbers = [int(folder.replace("version_", "")) for folder in os.listdir(config.root_log_dir)]
        folder_numbers = []
        for folder in os.listdir(config.root_log_dir):
          val = folder.replace("version_", "")
          if val != '.ipynb_checkpoints':
            val = int(val)
            folder_numbers.append(val)

        # Find the latest version number present in the log_dir
        last_version_number = max(folder_numbers)
        v = last_version_number
        # New version name
        version_name = f"version_{last_version_number + 1}"

    else:
        version_name = config.log_dir

    # Update the training config default directory
    log_dir        = os.path.join(config.root_log_dir,        version_name)
    checkpoint_dir = os.path.join(config.root_checkpoint_dir, version_name)

    # Create new directory for saving new experiment version
    os.makedirs(log_dir,        exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Logging at: {log_dir}")
    print(f"Model Checkpoint at: {checkpoint_dir}")

    return log_dir, checkpoint_dir, version_name

def write_config_to_file(filename: str, *configs):
    with open(filename, 'w') as f:
        for config in configs:
            f.write(f"{config.__class__.__name__}:\n")
            for field_name, value in config.__dict__.items():
                f.write(f"  {field_name}: {value}\n")
            f.write("\n")

def save_epoch_loss_to_csv(epoch, loss, file_path):
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Epoch', 'Loss'])
        writer.writerow([epoch, loss])

# def frames2vid(images, save_path):
#     WIDTH = images[0].shape[1]
#     HEIGHT = images[0].shape[0]

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video = cv2.VideoWriter(save_path, fourcc, 25, (WIDTH, HEIGHT))
#     # Appending the images to the video one by one
#     for image in images:
#         video.write(image)
#     video.release()
#     return

# def display_gif(gif_path):
    # b64 = base64.b64encode(open(gif_path,'rb').read()).decode('ascii')
    # display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))