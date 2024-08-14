# NSF_Diffusion_Opcode_Group

The contents of this repo are for the SJSU REU research project. It consists of three different generative models: GAN, WGAN - GP, and Diffusion.
The data is expected to be in the format of 1D vectors stored in a .csv file with each row being treated as a single sample. 


## GAN
keras==2.4.3

data scale: [-1,1]

To train GAN model, run python GANs/GAN/train_gan.py

After training, to generate fake samples, run python GANs/GAN/ExtraFake_gan.py

## WGAN-GP
keras==2.15.0

data scale: [-1,1]

To train WGANGP model, run python GANs/WGAN_GP/train_wgangp.py

After training, to generate fake samples, run python GANs/WGAN_GP/ExtraFake_wgangp.py


## Diffusion
