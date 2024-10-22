import torch
from classes.VQVAE import *
from parameters import codebook_params_cat, codebook_params_cont
from torch.distributions import Normal, OneHotCategorical
# from cm_ex import repo_data
from utilities import *
from bits_per_dim import *
# from train_cm import data_loaders
from torchvision.utils import save_image

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CONTINUOUS = True
    
    if CONTINUOUS:
        f="./logs/mnist/vqvae_cont/cm/version_0/checkpoints/best_epoch=0.ckpt" # 1.045
        vqvae = VQVAE.load_from_checkpoint(f).to(device)
        mixture = Normal(*codebook_params_cont(vqvae, device))
        print(mixture)
        
        sample = sampling_cont(mixture, n_components=vqvae.codebook_size, n_samples=64)
        save_image(sample, f'sampling_cont.png', nrow=8)

        _, _, test_loader = data_loaders()
        print(bpd_dataset(mixture, test_loader, vqvae.codebook_size, device))

        # hist=histogram(vqvae, test_loader, i=14, device=device)
        # print(hist.mean())
        # print(hist)
        # display_hist(hist)

    else:
        f="./logs/mnist/vqvae_cat/cm/version_8/checkpoints/best_epoch=17.ckpt" 
        vqvae = VQVAE.load_from_checkpoint(f).to(device)
        mixture = OneHotCategorical(logits=codebook_params_cat(vqvae, device))
        print(mixture)
        
        sample = sampling_cat(mixture, n_components=vqvae.codebook_size, device=device, n_samples=64)
        save_image(sample, f'sampling_cat.png', nrow=8)

        _, _, test_loader = repo_data()
        print(bpd_dataset_cat(mixture, test_loader, vqvae.codebook_size, device))

if __name__ == '__main__':
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.manual_seed(1)
    main()

# BEST SCORE BPD=1.258