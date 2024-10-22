import torch
from tqdm import tqdm

@torch.no_grad()    
def codebook_params_cont(vqvae, device):
    mu, log_var = torch.tensor([]).to(device), torch.tensor([]).to(device)
    for w in tqdm(vqvae.codebook.embedding.weight):
        q_e_x = w.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        out = vqvae.decode(q_e_x)
        mu_, log_var_ = out.chunk(2, dim=1)
        mu = torch.cat((mu, mu_))
        log_var = torch.cat((log_var, log_var_))
    return mu, log_var.mul(0.5).exp()

@torch.no_grad()
def codebook_params_cat(vqvae, device):
    logits = torch.tensor([]).to(device)
    for w in tqdm(vqvae.codebook.embedding.weight):
        q_e_x = w.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        logits = torch.cat((logits, vqvae.decode(q_e_x)))
    return logits.permute(0, 2, 3, 1)

    