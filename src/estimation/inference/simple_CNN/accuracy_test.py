import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np

from ...models.simple_CNN import SimpleCNN
from ...utils.datasets import UTKFace

def inference(
        model_weights: str,
        test_length: int
):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Initialize dataset
    test_dataset = UTKFace('./data/UTKFace/utkface_aligned_cropped/UTKFace')
    test_loader = DataLoader(test_dataset, batch_size=test_length, shuffle=True)

    # Initialize model
    model = SimpleCNN().to(device).eval()

    # Load model weights
    print(f'Loading model checkpoint {model_weights} ...')
    model.load_state_dict(torch.load(model_weights, weights_only=True))

    labels, x = next(iter(test_loader))

    with torch.no_grad():
        x = x.to(device)
        pred = model(x).cpu()
    
    # Convert to numpy arrays
    labels = labels.numpy()
    pred = pred.numpy()

    # Get choices
    label_class = np.array([np.argmax(labels[i]) for i in range(labels.shape[0])])
    pred_class = np.array([np.argmax(pred[i]) for i in range(pred.shape[0])])

    # Caclulate accuracy
    accuracy = (lambda x: np.sum(x))([1 if label_class[i] == pred_class[i] else 0 for i in range(label_class.shape[0])])
    print(f'Accuracy: {accuracy/test_length:.6f} | Correct: {accuracy} / {test_length}')

    # Calculate MAE
    mae = (lambda x,y: np.mean(np.abs(x-y)) )(label_class, pred_class)
    mse = (lambda x,y: np.mean((x-y)**2))(label_class, pred_class)
    print(f'MAE: {mae:.6f} | MSE: {mse:.6f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Path to weights', required=True)

    args = parser.parse_args()
    path = args.path

    if not os.path.exists(path):
        print(f'Path does not exist')
        exit()
    
    if os.path.exists(f'{path}/model_weights.pt'):
        model_weights = f'{path}/model_weights.pt'
    else:
        print(f'Weights not found')
        exit()
    
    inference(model_weights=model_weights, test_length=200)