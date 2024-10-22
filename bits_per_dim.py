import torch
from tqdm import tqdm
from utilities import *

def bpd(mixture, x, n_components):
    ll_px = mixture.log_prob(x)
    ll_im = ll_px.sum(dim=[1,2,3]) - torch.tensor(n_components).log()
    mix_model = ll_im.logsumexp(dim=0) 
    return -mix_model.div(784).div(torch.tensor(2).log())

def bpd_batch(mixture, batch, n_components):
    return torch.tensor([bpd(mixture, x, n_components).item() for x in batch])

def bpd_dataset(mixture, loader, n_components, device):
    return torch.cat([bpd_batch(mixture, bx.to(device), n_components) for bx,_ in tqdm(loader)]).mean().item()


def bpd_cat(mixture, x, n_components):
    ll_px = mixture.log_prob(x)
    ll_im = ll_px.sum(dim=[1,2]) - torch.tensor(n_components).log()
    mix_model = ll_im.logsumexp(dim=0) 
    return -mix_model.div(784).div(torch.tensor(2).log())

def bpd_batch_cat(mixture, batch, n_components):
    return torch.tensor([bpd_cat(mixture, x, n_components).item() for x in batch.permute(0,2,3,1)])

def bpd_dataset_cat(mixture, loader, n_components, device):
    return torch.cat([bpd_batch_cat(mixture, bx.to(device), n_components) for bx in tqdm(loader)]).mean().item()


# def bpd(mixture, x, i):
#     log2 = torch.tensor(2).log()
#     ll_px = mixture.log_prob(x)
#     ll_im = ll_px.sum(dim=[1,2,3]) - log2.mul(i)
#     mix_model = ll_im.logsumexp(dim=0) 
#     return -mix_model.div(784).div(log2)

# def bpd_batch(mixture, batch, i):
#     return torch.tensor([bpd(mixture, x, i).item() for x in batch])

# def bpd_dataset(mixture, loader, i, device):
#     return torch.cat([bpd_batch(mixture, bx.to(device), i) for bx in tqdm(loader)]).mean().item()
