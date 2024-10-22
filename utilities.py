import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.distributions import Normal
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from classes.UnsupervisedDataset import UnsupervisedDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from classes.architectures import *

class OneHotEncode:
    def __init__(self, num_classes=256):
        self.num_classes = num_classes
    
    def __call__(self, tensor):       
        one_hot = F.one_hot(tensor.to(torch.int64), num_classes=self.num_classes)
        return one_hot.squeeze(dim=0).permute(2, 0, 1).float() 

    def inverse(self, one_hot, dim=0):
        if one_hot.shape[dim] != self.num_classes:
            raise ValueError(f"The {dim}. dimension of the input tensor must be equal to num_classes={self.num_classes}, but the size dimension was {one_hot.shape[dim]}.")
        return torch.argmax(one_hot, dim=dim).unsqueeze(dim)
           

def histogram(vqvae, loader, i, device):
    hist = torch.zeros(2**i, device=device)
    for x in tqdm(loader):
        z_e_x = vqvae.encode(x.to(device))
        for index in vqvae.codebook(z_e_x):
            hist[index.item()] += 1
    return hist


def display_hist(
        hist,
        hist_file='histogram.png', 
        title='Histogram of Codebook Indices',
        xlabel='Codebook Index',
        ylabel='Frequency'
    ):
    hist_cpu = hist.cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(hist_cpu)), hist_cpu, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(hist_file, format='png')


def load_model(model, model_file, device):
    cp = torch.load(model_file, weights_only=True)
    return model.load_state_dict(cp).to(device)

def load_full_model(model_file, device):
    return torch.load(model_file).to(device)

def load_checkpoint(model, model_file, device):
    cp = torch.load(model_file, weights_only=False)
    return model.load_state_dict(cp['state_dict']).to(device)

def sampling_cont(mixture, n_components, n_samples, weights=None):
    if weights is None:
        weights = torch.tensor([1./n_components for _ in range(n_components)])
    assert weights.shape[0] == n_components
    assert sum(weights) == 1

    component = Categorical(weights).sample()
 
    sample = mixture.sample((n_samples,))[:,component,]
    sample = (sample + 1.) / 2.
    return sample

def sampling_cat(mixture, n_components, n_samples, device, weights=None):
    if weights is None:
        weights = torch.tensor([1./n_components for _ in range(n_components)])
    assert weights.shape[0] == n_components
    assert sum(weights) == 1
    samples = torch.tensor([]).to(device)
    for c in Categorical(weights).sample((n_samples,)):
        s = mixture.sample((1,))[:,c,].permute(0,3,1,2).argmax(dim=1) 
        s = s / 255.0
        samples = torch.cat((samples, s))
    return samples.unsqueeze(1)

def generate_recons_vae(model, dataloader, device, nsamples):
    model.eval()
    indices = torch.randperm(len(dataloader.dataset))[:nsamples]
    x = torch.stack([dataloader.dataset[i][0] for i in indices]).to(device) 
    x_tilde, _, _ = model(x)
    x_cat = torch.cat([x, x_tilde], 0).cpu()
    images = (x_cat + 1) / 2
    return images.to(device)


def generate_recons_vqvae(model, dataloader, device, nsamples):
    model.eval()
    indices = torch.randperm(len(dataloader.dataset))[:nsamples]
    x = torch.stack([dataloader.dataset[i][0] for i in indices]).to(device) 
    x_tilde, _, _, _ = model(x)
    x_cat = torch.cat([x, x_tilde], 0).cpu()
    images = (x_cat + 1) / 2
    return images.to(device)


def generate_samples(model, device, nsamples, dims=(128, 1, 1)):
    model.eval()
    z_e_x = Normal(0, 1).sample((nsamples, *dims)).to(device)
    x_tilde, _ = model.decode(z_e_x)
    images = (x_tilde.cpu() + 1) / 2
    return images.to(device)


def empty_folder(folder_path):
    for filename in os.listdir(folder_path): 
        file_path = os.path.join(folder_path, filename)  
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  
            elif os.path.isdir(file_path):  
                os.rmdir(file_path)  
        except Exception as e:  
            print(f"Error deleting {file_path}: {e}")
    print("Deletion done")

def data_loaders(dataset=MNIST, batch_size = 128):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = dataset(root='./mnist', train=True, download=True, transform=t)
    train = UnsupervisedDataset(train_data, labeling=False)
    train, valid = torch.utils.data.random_split(train, [50_000, 10_000])

    test_data = dataset(root='./mnist', train=False, download=True, transform=t)
    test = UnsupervisedDataset(test_data, labeling=False)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader


def repo_data_label(dataset = MNIST, batch_size = 128):
    #transf = transforms.Compose([transforms.PILToTensor()])
    t = transforms.Compose([transforms.PILToTensor(), OneHotEncode(num_classes=256)])
    train_data = dataset(root='./mnist', train=True, download=True, transform=t)
    train = UnsupervisedDataset(train_data, labeling=True)
    train, valid = torch.utils.data.random_split(train, [50_000, 10_000])
    
    test_data = dataset(root='./mnist', train=False, download=True, transform=t)
    test = UnsupervisedDataset(test_data, labeling=True)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    return train_loader, valid_loader, test_loader


def repo_data(dataset = MNIST, batch_size = 128):
    #transf = transforms.Compose([transforms.PILToTensor()])
    transf = transforms.Compose([transforms.PILToTensor(), OneHotEncode(num_classes=256)])
    
    train = UnsupervisedDataset(dataset(root='./mnist', train=True, download=True, transform=transf))
    train, valid = torch.utils.data.random_split(train, [50_000, 10_000])
    
    test = UnsupervisedDataset(dataset(root='./mnist', train=False, download=True, transform=transf))
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True