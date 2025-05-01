import torch
import torch.nn as nn
# from torchvision.transforms.functional import pil_to_tensor # Keep if using torchvision transforms elsewhere
from torch.utils.data import Dataset, ConcatDataset # Import ConcatDataset
import os
from PIL import Image
import numpy as np # For image conversion if not using pil_to_tensor
from torchvision import transforms # Use torchvision transforms for flexibility

class UTKFace(Dataset):
    def __init__(
            self,
            path: str,
            min_age: int = 8, # Define min/max age for filtering and index calculation
            max_age: int = 90,
            transform=None # Add transform argument
    ):
        """
        Args:
            path (str): Path to the directory containing UTKFace images.
            min_age (int): Minimum age to include (inclusive).
            max_age (int): Maximum age to include (inclusive).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.path = path
        self.min_age = min_age
        self.max_age = max_age
        self.transform = transform
        self.ids = []
        self.labels = []

        print(f"Scanning dataset path: {path} for ages {min_age}-{max_age}...")
        all_files = os.listdir(path)
        valid_files_count = 0
        invalid_files_count = 0

        for filename in all_files:
            try:
                # Attempt to parse age from filename like 'age_gender_race_date.jpg.chip'
                parts = filename.split('_')
                if len(parts) < 1:
                    # print(f"Skipping invalid filename format: {filename}") # Optional: Verbose logging
                    invalid_files_count += 1
                    continue

                age = int(parts[0])

                # Filter based on age range
                if self.min_age <= age <= self.max_age:
                    self.ids.append(filename)
                    # Calculate the class index (age - min_age)
                    age_index = age - self.min_age
                    self.labels.append(age_index)
                    valid_files_count += 1
                else:
                    # print(f"Skipping file due to age out of range ({age}): {filename}") 
                    invalid_files_count += 1

            except (ValueError, IndexError) as e:
                # print(f"Skipping file due to parsing error ({e}): {filename}") 
                invalid_files_count += 1
                continue
            except Exception as e:
                # Catch any other unexpected errors during parsing
                # print(f"Skipping file due to unexpected error ({e}): {filename}") 
                invalid_files_count += 1
                continue

        print(f"Dataset scan complete. Found {valid_files_count} valid images, skipped {invalid_files_count} files.")
        if valid_files_count == 0:
            print(f"WARNING: No valid images found in the specified age range {min_age}-{max_age} at path {path}")


    def __getitem__(self, index):
        # Get filename and pre-calculated label index
        filename = self.ids[index]
        label = self.labels[index] # Integer label (0 to max_age - min_age)

        # Construct full image path
        img_path = os.path.join(self.path, filename)

        try:
            # Open image using PIL
            img = Image.open(img_path).convert('RGB') # Ensure image is RGB

            # Apply transformations if provided
            if self.transform:
                img_tensor = self.transform(img)
            else:
                # Default minimal transformation (if none provided)
                # Convert PIL Image to PyTorch tensor and normalize to [0, 1]
                # Ensure tensor is FloatTensor
                img_tensor = transforms.ToTensor()(img) # This scales to [0, 1] automatically

            # Return the integer label and the image tensor
            return img_tensor, label # Return image tensor FIRST, label SECOND 

        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            
            print("Returning first item as fallback...")
            return self.__getitem__(0)
        except Exception as e:
            print(f"Error loading or processing image {img_path}: {e}")
            print("Returning first item as fallback...")
            return self.__getitem__(0)


    def __len__(self):
        return len(self.ids)

train_transform = transforms.Compose([
    transforms.Resize((200, 200)), # Resize to expected model input size
    transforms.RandomHorizontalFlip(), # Data augmentation
    transforms.ToTensor(), # Converts PIL image to [0, 1] tensor (C, H, W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization 
])

# Define transformations for validation/testing 
eval_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class CustomAgeDataset(Dataset):
    def __init__(self, path, min_age=8, max_age=90, transform=None):
        self.path = path
        self.min_age = min_age
        self.max_age = max_age
        self.transform = transform
        self.ids = []
        self.labels = []

        print(f"Scanning CUSTOM dataset path: {path} for ages {min_age}-{max_age}...")
        all_files = os.listdir(path)
        valid_files_count = 0
        invalid_files_count = 0
      
        for filename in all_files:
             try:
                 parts = filename.split('_')
                 if len(parts) < 1:
                     invalid_files_count += 1
                     continue
                 age = int(parts[0])
                 if self.min_age <= age <= self.max_age:
                     self.ids.append(filename)
                     age_index = age - self.min_age
                     self.labels.append(age_index)
                     valid_files_count += 1
                 else:
                     invalid_files_count += 1
             except: 
                 invalid_files_count += 1
                 continue
        print(f"Custom Dataset scan complete. Found {valid_files_count} valid images, skipped {invalid_files_count} files.")
        if valid_files_count == 0:
            print(f"WARNING: No valid custom images found in range {min_age}-{max_age} at {path}")

    def __getitem__(self, index):
        filename = self.ids[index]
        label = self.labels[index]
        img_path = os.path.join(self.path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.ToTensor()(img)
            return img_tensor, label 
        except Exception as e:
             print(f"Error loading custom image {img_path}: {e}. Returning fallback.")
             empty_tensor = torch.zeros((3, 200, 200)) # Match expected size
             return empty_tensor, 0 # Return dummy data


    def __len__(self):
        return len(self.ids)

