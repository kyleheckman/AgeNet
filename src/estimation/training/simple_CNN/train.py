import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import os
import argparse
import sys 

current_train_dir = os.path.dirname(os.path.abspath(__file__))
estimation_dir = os.path.dirname(os.path.dirname(current_train_dir)) # Up two levels to estimation
src_dir = os.path.dirname(estimation_dir) # Up one level to src
root_dir_from_train = os.path.dirname(src_dir) # Up one level to project root
if root_dir_from_train not in sys.path:
    sys.path.insert(0, root_dir_from_train)

try:
    from ...models.simple_CNN import SimpleCNN 
    from ...utils.datasets import UTKFace 
except ImportError:
    print("ImportError in train_simple_cnn.py. Check paths and structure.")
    print(f"Root path added: {root_dir_from_train}")
    print(f"Current sys.path: {sys.path}")
    exit(1)



def train(
        data_path: str, # Added data_path argument
        weights_path: str,
        model_weights: str = None,
        optim_weights: str = None,
        batch_size: int = 128,
        epochs: int = 100,
        lr = 2e-5
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    
    print(f"Initializing dataset from: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERROR: Data path not found: {data_path}")
        print("Please ensure the UTKFace data is available at the specified path.")
        exit(1)
   
    try:
       
        train_dataset = UTKFace(data_path) 
        print(f"Dataset size: {len(train_dataset)}")
        if len(train_dataset) == 0:
            print("ERROR: Dataset is empty. Check data path and dataset implementation.")
            exit(1)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        print("Ensure UTKFace class in utils/datasets.py is updated to handle the path and return integer labels.")
        exit(1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True) 

   
    print("Initializing model...")
    model = SimpleCNN().to(device) 
    optim = Adam(model.parameters(), lr=lr)

   
    start_epoch = 0
    if model_weights and os.path.exists(model_weights):
        print(f'Loading model checkpoint {model_weights} ...')
        try:
            model.load_state_dict(torch.load(model_weights, map_location=device))
        except Exception as e:
            print(f"Warning: Could not load model weights from {model_weights}. Error: {e}. Training from scratch.")
            model_weights = None
    else:
        print(f'No model checkpoint found or specified. Training from scratch.')

    if optim_weights and model_weights and os.path.exists(optim_weights):
        print(f'Loading optimizer checkpoint {optim_weights} ...')
        try:
            optim.load_state_dict(torch.load(optim_weights, map_location=device))
        except Exception as e:
             print(f"Warning: Could not load optimizer state from {optim_weights}. Error: {e}. Using fresh optimizer state.")
    else:
        print(f'No optimizer checkpoint found or specified, or model not loaded. Using fresh optimizer state.')


 
    loss_func = nn.CrossEntropyLoss() #CHANGED TO CROSS ENTROPY
    print("Using CrossEntropyLoss.")

   
    print(f"Starting training from epoch {start_epoch + 1}...")
    for i in range(start_epoch, epochs):
        model.train() 
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f'Epoch {i+1}/{epochs}')
        for batch_idx, (x, labels) in enumerate(pbar):
            
            x = x.to(device)
            labels = labels.to(device, dtype=torch.long) 

          
            optim.zero_grad()

            
            logits = model(x) 

            # Compute loss
            loss = loss_func(logits, labels)
            total_loss += loss.item()

            # Backpropagate and update weights
            loss.backward()
            optim.step()

            # Calculate accuracy
            _, predicted_labels = torch.max(logits, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct_predictions / total_samples:.2f}%'
             })

        # Print epoch summary
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct_predictions / total_samples
        print(f'Epoch {i+1} Summary | Average Loss: {avg_loss:.6f} | Accuracy: {accuracy:.2f}%')

        # Save checkpoint
        print(f"Saving checkpoint for epoch {i+1} to {weights_path}")
        if not os.path.exists(weights_path):
             os.makedirs(weights_path)
        torch.save(model.state_dict(), os.path.join(weights_path, 'model_weights.pt'))
        torch.save(optim.state_dict(), os.path.join(weights_path, 'optim_weights.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SimpleCNN model for Age Estimation")
    parser.add_argument('--data', type=str, required=True, help='Path to the root directory of the UTKFace dataset')
    parser.add_argument('--weights', type=str, required=True, help='Path to directory where weights will be saved')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for Adam optimizer')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoints in weights directory')


    args = parser.parse_args()

    weights_dir = args.weights
    data_dir = args.data

    if not os.path.exists(weights_dir):
        print(f"Creating weights directory: {weights_dir}")
        os.makedirs(weights_dir)

    model_ckpt = None
    optim_ckpt = None
    if args.resume:
        model_ckpt_path = os.path.join(weights_dir, 'model_weights.pt')
        optim_ckpt_path = os.path.join(weights_dir, 'optim_weights.pt')
        if os.path.exists(model_ckpt_path):
            model_ckpt = model_ckpt_path
        if os.path.exists(optim_ckpt_path):
            optim_ckpt = optim_ckpt_path
        print(f"Attempting to resume training. Model: {model_ckpt}, Optim: {optim_ckpt}")


    train(
        data_path=data_dir,
        weights_path=weights_dir,
        model_weights=model_ckpt,
        optim_weights=optim_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )