import torch.nn as nn
from classes.Embedding import VQEmbedding, VQEmbedding_cat
from classes.GatedMaskedConv2d import GatedMaskedConv2d


def vqvae_cat(
        input_dim=256,
        output_dim=256,
        latent_dim=128,
        codebook_size=512, 
        dims=[32, 64, 96]
    ):
    
    encoder = nn.Sequential(
        nn.Conv2d(input_dim, dims[0], 4, 2, 1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.Conv2d(dims[0], dims[1], 4, 2, 1),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.Conv2d(dims[1], dims[2], 5, 1, 0),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.Conv2d(dims[2], latent_dim, 3, 1, 0)
    )

    # codebook = VQEmbedding(codebook_size, latent_dim)
    codebook = VQEmbedding_cat(codebook_size, latent_dim)

    decoder = nn.Sequential(
        nn.ConvTranspose2d(latent_dim, dims[2], 3, 1, 0),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[2], dims[1], 5, 1, 0),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[1], dims[0], 4, 2, 1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[0], output_dim, 4, 2, 1),

        # nn.Softmax(dim=1)
    )
    return encoder, codebook, decoder
 

def vqvae_cont(
        input_dim=1,
        output_dim=2,
        latent_dim=128,
        codebook_size=512, 
        dims=[32, 64, 96]
    ):

    encoder = nn.Sequential(
        nn.Conv2d(input_dim, dims[0], 4, 2, 1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.Conv2d(dims[0], dims[1], 4, 2, 1),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.Conv2d(dims[1], dims[2], 5, 1, 0),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.Conv2d(dims[2], latent_dim, 3, 1, 0),
    )

    codebook = VQEmbedding(codebook_size, latent_dim)

    decoder = nn.Sequential(
        nn.ConvTranspose2d(latent_dim, dims[2], 3, 1, 0),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[2], dims[1], 5, 1, 0),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[1], dims[0], 4, 2, 1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[0], output_dim, 4, 2, 1),
        nn.Tanh()
    )

    return encoder, codebook, decoder


def vae_cat(
        input_dim=256,
        output_dim=256,
        latent_dim=128,
        dims=[32, 64, 96]
    ):
    
    encoder = nn.Sequential(
        nn.Conv2d(input_dim, dims[0], 4, 2, 1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.Conv2d(dims[0], dims[1], 4, 2, 1),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.Conv2d(dims[1], dims[2], 5, 1, 0),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.Conv2d(dims[2], latent_dim, 3, 1, 0)
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(latent_dim, dims[2], 3, 1, 0),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[2], dims[1], 5, 1, 0),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[1], dims[0], 4, 2, 1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[0], output_dim, 4, 2, 1),

        # nn.Softmax(dim=1)
    )
    return encoder, decoder
 

def vae_cont(
        input_dim=1,
        output_dim=2,
        latent_dim=128,
        dims=[32, 64, 96]
    ):

    encoder = nn.Sequential(
        nn.Conv2d(input_dim, dims[0], 4, 2, 1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.Conv2d(dims[0], dims[1], 4, 2, 1),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.Conv2d(dims[1], dims[2], 5, 1, 0),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.Conv2d(dims[2], latent_dim * 2, 3, 1, 0),
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(latent_dim, dims[2], 3, 1, 0),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[2], dims[1], 5, 1, 0),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[1], dims[0], 4, 2, 1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[0], output_dim, 4, 2, 1),
        nn.Tanh()
    )

    return encoder, decoder

def px_cnn(
        input_dim=256, 
        dim=128, 
        n_layers=15, 
        n_classes=10
    ):
    embedding = nn.Embedding(input_dim, dim)
    layers = nn.ModuleList()

    # Initial block with Mask-A convolution. Rest with Mask-B convolutions
    for i in range(n_layers):
        mask_type = 'A' if i == 0 else 'B'
        kernel = 7 if i == 0 else 3
        residual = False if i == 0 else True

        layers.append(GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes))
    # Output layer
    output_conv = nn.Sequential(
        nn.Conv2d(dim, 512, 1),
        nn.ReLU(True),
        nn.Conv2d(512, input_dim, 1)
    )

    return embedding, layers, output_conv
