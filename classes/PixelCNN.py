import torch
import pytorch_lightning as pl
from classes.architectures import px_cnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class PixelCNN(pl.LightningModule):
    def __init__(
        self, 
        vqvae,
        input_dim=256, 
        dim=128, 
        n_layers=15, 
        n_classes=10
    ):
        super().__init__()
        e, ll, oc = px_cnn(
            input_dim=input_dim, 
            dim=dim, 
            n_layers=n_layers,
            n_classes=n_classes
        )
        self.dim = dim
        self.embedding = e
        self.layers = ll
        self.output_conv = oc
        self.vqvae = vqvae
        self.save_hyperparameters(ignore=['vqvae'])

    def forward(self, x, label):   
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp).permute(0, 3, 1, 2)  # (B, C, H, W)
        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    # prior loss
    def loss(self, batch_x, batch_y):
        k = self.vqvae.codebook_size
        with torch.no_grad():
            z_e_x = self.vqvae.encode(batch_x).contiguous()
            latents = self.vqvae.codebook(z_e_x)    
        logits = self.forward(latents, batch_y).permute(0, 2, 3, 1).contiguous()
        return F.cross_entropy(logits.view(-1, k), latents.view(-1))

    def training_step(self, batch:torch.Tensor, batch_id:int):
        batch_x, batch_y = batch
        loss = self.loss(batch_x, batch_y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch:torch.Tensor, batch_id:int):
        batch_x, batch_y = batch
        loss = self.loss(batch_x, batch_y)
        self.log('valid_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def eval_loader(
        self, 
        loader:DataLoader, 
        progress_bar:bool=False, 
        device:str='cpu'
    ):
        self.eval()
        loader = tqdm(loader) if progress_bar else loader
        return torch.cat([self.forward(x.to(device), y) for x,y in loader], dim=0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

    @torch.no_grad()
    def generate(self, label, shape, batch_size=128):
        device = next(self.parameters()).device
        x = torch.zeros((batch_size, *shape), dtype=torch.int64, device=device)

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(probs.multinomial(1).squeeze().data)
        return x

