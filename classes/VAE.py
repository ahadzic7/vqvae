import torch
import pytorch_lightning as pl
from torch.distributions import Normal, OneHotCategorical, kl_divergence
from torch.utils.data import DataLoader
from tqdm import tqdm
from classes.architectures import vae_cat, vae_cont

class VAE(pl.LightningModule):
    def __init__(
            self, 
            input_dim=1,
            output_dim=1,
            latent_dim=128,
            dims=[32, 64, 96], 
            type="continuous", 
            beta=1.0
        ):
        super(VAE, self).__init__()
        if type == "continuous":
            e, d = vae_cont(
                input_dim=input_dim,
                output_dim=output_dim,
                latent_dim=latent_dim,
                dims=dims
            )
        elif type == "categorical":
            e, d = vae_cat(
                input_dim=input_dim,
                output_dim=output_dim,
                latent_dim=latent_dim,
                dims=dims
            )
        self.encoder=e
        self.decoder=d
        self.type=type
        self.beta=beta
        self.save_hyperparameters()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        out = self.encode(x)
        if self.type == "continuous":
            mu, log_var = out.chunk(2, dim=1)
            q_z_x = Normal(mu, log_var.mul(0.5).exp())
            p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
        elif self.type == "categorical":
            num_classes = out.shape[1]
            q_z_x = OneHotCategorical(logits=out)
            p_z = OneHotCategorical(torch.ones(num_classes).cuda() / num_classes)
        else:
            raise Exception(f"Type supplied was {self.type}, but it must be continuous or categorical.")    
        
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
        return self.decode(q_z_x.sample()), kl_div

    
    def loss_cont(self, batch:torch.Tensor):
        out, kl_div = self(batch)
        mu, log_var = out.chunk(2, dim=1)
        dist = Normal(mu, log_var.mul(0.5).exp())
        rec_loss = -dist.log_prob(batch).sum(dim=[2,3]).mean()
        return rec_loss + self.beta * kl_div
    
    def loss_cat(self, batch:torch.Tensor):
        logits, kl_div = self(batch)
        batch = batch.permute(0,2,3,1)
        dist = OneHotCategorical(logits=logits.permute(0, 2, 3, 1))
        rec_loss = -dist.log_prob(batch).sum(dim=[1,2]).mean()
        return rec_loss + self.beta * kl_div

    def step(self, batch):
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

