from torchvision.datasets import MNIST
import pytorch_lightning as pl
from classes.VQVAE import VQVAE
from classes.VAE import VAE
from classes.PixelCNN import PixelCNN
from utilities import *

def train(kwargs, log_file, train_loader, valid_loader, max_epochs=100):
    cp_best_model_valid = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='valid_loss_epoch',
        mode='min',
        filename='best_{epoch}'
    )
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="valid_loss_epoch",
        min_delta=0.00,
        patience=15,
        verbose=False,
        mode='min'
    )
    callbacks = [cp_best_model_valid, early_stop_callback]

    logger = pl.loggers.TensorBoardLogger(log_file, name='cm')
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        deterministic=True
    )
    vqvae = VQVAE(**kwargs)
    trainer.fit(vqvae, train_loader, valid_loader)

def train_pixel_cnn(vqvae_file, kwargs, log_file, device, max_epochs=100):
    
    vqvae = VQVAE.load_from_checkpoint(vqvae_file).to(device)
    
    pixel_cnn = PixelCNN(vqvae, **kwargs)

    train_loader, valid_loader, _ = data_loaders(dataset=MNIST)
    cp_best_model_valid = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='valid_loss_epoch',
        mode='min',
        filename='best_{epoch}'
    )
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="valid_loss_epoch",
        min_delta=0.00,
        patience=15,
        verbose=False,
        mode='min'
    )
    callbacks = [cp_best_model_valid, early_stop_callback]
    
    logger = pl.loggers.TensorBoardLogger(log_file, name='cm')
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        logger=logger,
        deterministic=True
    )
    trainer.fit(pixel_cnn, train_loader, valid_loader)

def main():
    dataset_name='mnist'

    CONTINUOUS = True
    if CONTINUOUS:
        train_loader, valid_loader, _ = data_loaders(dataset=MNIST)
        log_file=f'./logs/{dataset_name}/vqvae_cont'
        kwargs = {
            'input_dim':1,
            'output_dim':2,
            'latent_dim':128,
            'codebook_size':512, 
            'dims':[32, 64, 96], 
            'type':"continuous", 
            'beta':1.0,
        }
    else:
        train_loader, valid_loader, _ = repo_data(dataset=MNIST)
        log_file=f'./logs/{dataset_name}/vqvae_cat'
        kwargs = {
            'input_dim':256,
            'output_dim':256,
            'latent_dim':512,
            'codebook_size':4096, 
            'dims':[320, 384, 448], 
            'type':"categorical", 
            'beta':1,
        }
    train(kwargs, log_file, train_loader, valid_loader, max_epochs=100)
    
if __name__ == '__main__':
    seed_everything(0)
    main()




# device = 'cpu' if torch.cuda.is_available() else 'cpu'
# f="./logs/mnist/vqvae_cont/cm/version_0/checkpoints/best_epoch=0.ckpt" 
# kwargs = {
#     'input_dim':256, 
#     'dim':128, 
#     'n_layers':15, 
#     'n_classes':10
# }
# log_file=f'./logs/{dataset_name}/pixel_cnn_cat'
# train_pixel_cnn(f, kwargs, log_file, device)
# exit()