import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategorical
from torch.utils.data import DataLoader
from tqdm import tqdm
from classes.architectures import vqvae_cont, vqvae_cat

class VQVAE(pl.LightningModule):
    def __init__(
            self, 
            input_dim=1,
            output_dim=1,
            latent_dim=128,
            codebook_size=512, 
            dims=[32, 64, 96], 
            type="continuous", 
            beta=1.0
        ):
        super(VQVAE, self).__init__()
        if type == "continuous":
            e, c, d = vqvae_cont(
                input_dim=input_dim,
                output_dim=output_dim,
                latent_dim=latent_dim,
                codebook_size=codebook_size, 
                dims=dims
            )
        elif type == "categorical":
            e, c, d = vqvae_cat(
                input_dim=input_dim,
                output_dim=output_dim,
                latent_dim=latent_dim,
                codebook_size=codebook_size, 
                dims=dims
            )
        else:
            raise Exception(f"Type supplied was {self.type}, but it must be continuous or categorical.")
        self.encoder=e
        self.codebook=c
        self.decoder=d
        self.type=type
        self.beta=beta
        self.codebook_size=codebook_size
        self.save_hyperparameters()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z_e_x = self.encode(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)    
        return self.decode(z_q_x_st), z_e_x, z_q_x

    def loss_cont(self, batch:torch.Tensor):
        out, z_e_x, z_q_x = self(batch)   
        mu, log_var = out.chunk(2, dim=1)

        dist = Normal(mu, log_var.mul(0.5).exp())
        
        loss_recons = -dist.log_prob(batch).sum(dim=[2,3])
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
        return loss_recons + loss_vq + self.beta * loss_commit
    
    def loss_cat(self, batch:torch.Tensor):
        logits, z_e_x, z_q_x = self(batch)
        batch = batch.permute(0,2,3,1)

        dist = OneHotCategorical(logits=logits.permute(0, 2, 3, 1))

        loss_recons = -dist.log_prob(batch).sum(dim=[1,2])
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
        return loss_recons + loss_vq + self.beta * loss_commit

    def step(self, batch:torch.Tensor):
        if self.type == "continuous":
            loss = self.loss_cont(batch).mean()
        elif self.type == "categorical":
            loss = self.loss_cat(batch).mean()
        else:
            raise Exception(f"Type supplied was {self.type}, it must be continuous or categorical.")
        return loss

    def training_step(self, batch:torch.Tensor, batch_id:int):
        loss = self.step(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch:torch.Tensor, batch_id:int):
        loss = self.step(batch)
        self.log('valid_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def eval_loader(
        self, 
        loader:DataLoader, 
        progress_bar:bool = False, 
        device:str = 'cpu'
    ):
        self.eval()
        loader = tqdm(loader) if progress_bar else loader
        return torch.cat([self.forward(x.to(device)) for x in loader], dim=0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

