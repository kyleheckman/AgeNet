import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset # Added ConcatDataset
from torch.optim import Adam
from tqdm import tqdm
import os
import argparse
import sys # Added sys


current_script_path = os.path.abspath(__file__)
# Go up 5 levels to get to AgeNet-main: train.py -> resnet -> training -> estimation -> src -> AgeNet-main
root_dir_from_train = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
if root_dir_from_train not in sys.path:
    sys.path.insert(0, root_dir_from_train)


try:
    from src.estimation.models.resnet import ResNet
    from src.estimation.utils.datasets import UTKFace, CustomAgeDataset, train_transform 
except ImportError as e:
    print(f"ImportError in train_resnet.py: {e}. Check paths and structure.")
    print(f"Root path added: {root_dir_from_train}")
    print(f"Current sys.path: {sys.path}")
    # Add more specific checks if needed:
    print(f"Checking paths relative to added root: {root_dir_from_train}")
    print(f"Does src exist? {os.path.exists(os.path.join(root_dir_from_train, 'src'))}")
    print(f"Does src/__init__.py exist? {os.path.exists(os.path.join(root_dir_from_train, 'src', '__init__.py'))}")
    print(f"Does src/estimation exist? {os.path.exists(os.path.join(root_dir_from_train, 'src', 'estimation'))}")
    print(f"Does src/estimation/__init__.py exist? {os.path.exists(os.path.join(root_dir_from_train, 'src', 'estimation','__init__.py'))}")
    print(f"Does src/estimation/models exist? {os.path.exists(os.path.join(root_dir_from_train, 'src', 'estimation', 'models'))}")
    print(f"Does src/estimation/models/__init__.py exist? {os.path.exists(os.path.join(root_dir_from_train, 'src', 'estimation', 'models', '__init__.py'))}")
    print(f"Does src/estimation/utils exist? {os.path.exists(os.path.join(root_dir_from_train, 'src', 'estimation', 'utils'))}")
    print(f"Does src/estimation/utils/__init__.py exist? {os.path.exists(os.path.join(root_dir_from_train, 'src', 'estimation', 'utils', '__init__.py'))}")

    exit(1)


def train(
        utk_data_path: str, 
        custom_data_path: str, 
        weights_path: str,
        model_weights: str = None,
        optim_weights: str = None,
        batch_size: int = 128,
        epochs: int = 100,
        lr = 2e-5
):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    
    datasets_to_combine = []

    # UTKFace Dataset
    print(f"Initializing UTKFace dataset from: {utk_data_path}")
    if not os.path.exists(utk_data_path):
        print(f"ERROR: UTKFace Data path not found: {utk_data_path}")
        # Optionally exit or continue without this dataset
    else:
        try:
            utk_train_dataset = UTKFace(utk_data_path, transform=train_transform) # Use transform from datasets.py
            print(f"UTKFace dataset size: {len(utk_train_dataset)}")
            if len(utk_train_dataset) > 0:
                 datasets_to_combine.append(utk_train_dataset)
            else:
                print("WARNING: UTKFace dataset is empty or failed to load.")
        except Exception as e:
            print(f"Error initializing UTKFace dataset: {e}")

    # Custom Dataset
    if custom_data_path:
        print(f"Initializing Custom dataset from: {custom_data_path}")
        if not os.path.exists(custom_data_path):
             print(f"WARNING: Custom dataset path specified ({custom_data_path}), but not found. Skipping.")
        else:
            try:
                custom_train_dataset = CustomAgeDataset(custom_data_path, transform=train_transform) # Use transform
                print(f"Custom dataset size: {len(custom_train_dataset)}")
                if len(custom_train_dataset) > 0:
                    datasets_to_combine.append(custom_train_dataset)
                else:
                    print("WARNING: Custom dataset path provided, but no valid images found. Skipping.")
            except Exception as e:
                 print(f"Error initializing Custom dataset: {e}")
    else:
        print("No custom dataset path provided.")


 
    if not datasets_to_combine:
        print("ERROR: No valid datasets found to train on. Exiting.")
        exit(1)
    elif len(datasets_to_combine) > 1:
        print(f"Combining {len(datasets_to_combine)} datasets...")
        final_train_dataset = ConcatDataset(datasets_to_combine)
    else:
        final_train_dataset = datasets_to_combine[0] # Only one dataset loaded

    print(f"Total training samples: {len(final_train_dataset)}")

   
    train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True) 

    print("Initializing model...")
    model = ResNet().to(device) 
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


  
    loss_func = nn.CrossEntropyLoss()   #CROSS ENTROPY
    print("Using CrossEntropyLoss.")

    # Training routine
    print(f"Starting training from epoch {start_epoch + 1}...")
    for i in range(start_epoch, epochs):
        model.train() # Set model to training mode
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {i+1}/{epochs}')
        # Corrected loop variable order based on Dataset output (image, label)
        for batch_idx, (images, labels) in enumerate(pbar):
            # Ensure labels are integer type (long) and move data to device
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long) # Ensure labels are LongTensor for CrossEntropyLoss

            # Zero gradients
            optim.zero_grad()

            # Forward pass
            logits = model(images) # Get raw logits from the model

            # Compute loss
            loss = loss_func(logits, labels)
            total_loss += loss.item()

            # Backpropagate and update weights
            loss.backward()
            # Optional: Gradient clipping if gradients explode
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            # Calculate accuracy (optional but good for monitoring)
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
        print(f'\nEpoch {i+1} Summary | Average Loss: {avg_loss:.6f} | Accuracy: {accuracy:.2f}%')

        # Save checkpoint
        print(f"Saving checkpoint for epoch {i+1} to {weights_path}")
        if not os.path.exists(weights_path):
             os.makedirs(weights_path)
        torch.save(model.state_dict(), os.path.join(weights_path, 'model_weights.pt'))
        torch.save(optim.state_dict(), os.path.join(weights_path, 'optim_weights.pt'))
        # Optional: Save epoch-specific weights
        # torch.save(model.state_dict(), os.path.join(weights_path, f'model_epoch_{i+1}.pt'))

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResNet model for Age Estimation")
    parser.add_argument('--utk_data', type=str, required=True, help='Path to the root directory of the UTKFace dataset')
    parser.add_argument('--custom_data', type=str, default=None, help='Path to the root directory of the custom labeled dataset (optional)')
    parser.add_argument('--weights', type=str, required=True, help='Path to directory where weights will be saved')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for Adam optimizer')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoints in weights directory')

    args = parser.parse_args()

    weights_dir = args.weights

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
        utk_data_path=args.utk_data,
        custom_data_path=args.custom_data, # Pass the custom data path
        weights_path=weights_dir,
        model_weights=model_ckpt,
        optim_weights=optim_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )