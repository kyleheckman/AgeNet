import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import os
import argparse

from ...models.resnet import ResNet
from ...utils.datasets import UTKFace

def train(
        weights_path: str,
        model_weights: str = None,
        optim_weights: str = None,
        batch_size: int = 128,
        epochs: int = 100,
        lr = 2e-5
):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Initialize dataset
    train_dataset = UTKFace('./data/UTKFace/utkface_aligned_cropped/UTKFace')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize model and optimizer
    model = ResNet().to(device)
    optim = Adam(model.parameters(), lr=lr)

    # Load checkpoints if present
    if model_weights:
        print(f'Loading model checkpoint {model_weights} ...')
        model.load_state_dict(torch.load(model_weights, weights_only=True))
    else:
        print(f'No checkpoint loaded (model)')
    
    if optim_weights:
        print(f'Loading optim checkpoint {optim_weights} ...')
        optim.load_state_dict(torch.load(optim_weights, weights_only=True))
    else:
        print(f'No checkpoint loaded (optim)')

    # Set loss function
    loss_func = nn.BCELoss(reduction='mean')

    # Training routine
    for i in range(epochs):
        total_loss = 0
        for _, (labels, x) in enumerate(tqdm(train_loader, desc=f'Epoch {i+1}/{epochs}')):
            # Zero gradients
            optim.zero_grad()

            x = x.to(device)
            labels = labels.to(device)
            pred = model(x)

            # Compute loss & backpropagate
            loss = loss_func(pred, labels)
            total_loss += loss.item()
            loss.backward()
            optim.step()
        
        # Print epoch result
        print(f'Epoch {i+1} | Loss: {total_loss/batch_size:.6f}')

        # Save checkpoint
        torch.save(model.state_dict(), f'{weights_path}/model_weights.pt')
        torch.save(optim.state_dict(), f'{weights_path}/optim_weights.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Path to weights', required=True)

    args = parser.parse_args()
    path = args.path

    if not os.path.exists(path):
        os.makedirs(path)
    
    model_weights = None
    optim_weights = None
    if os.path.exists(f'{path}/model_weights.pt'):
        model_weights = f'{path}/model_weights.pt'
    if os.path.exists(f'{path}/optim_weights.pt'):
        optim_weights = f'{path}/optim_weights.pt'
    
    train(weights_path=path, model_weights=model_weights, optim_weights=optim_weights, epochs=25)
