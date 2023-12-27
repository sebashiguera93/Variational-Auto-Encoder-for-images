# Variational-Auto-Encoder-for-images
*By: Sebastian Higuera Pedraza*

This project will introduce the Variational Auto Encoder for processing images, using CelebA dataset in Python.


The aim of this project is to introduce the Variational Auto Encoder, both theoretically and practically, using Python.

For this purpose, I'll use CelebA data set, a very popular data set for training Deep Neural Networks. It is also commonly used for pedagogical purposes.

First, I'll introduce the traditional Autoencoder from the deep learning perspective, and afterwards I'll introduce the Variational Auto Encoder.

## Autoencoder 

As it is explained in Tenor Flow webpage, an autoencoder is simply a special type of deep neural network which is trained to replicate its input into its output. So essentially, this network learns how to represent the input data, like an image, into a lower dimension (latent) space, and then it learns how to reconstruct back the original input data, such that the reconstruction is as similar as possible to the original data.

The architecture of this Autoencoder is simple. First, there is an encoding process, where the input data (image) is transformed to a lower dimension space. So it takes the $x$ data points and use a neural network to generate an output $z$, where $z<n$. This output is living in a latent space of lower dimension and is merely a representation of the original data in a lower dimension space. The second part of the autoencoder is the decoder: it generates the process backwards. Therefore, the encoder compress the data, while the decoder reconstructs the data back to its original form.

An autoencoder consists of two main parts: the encoder and the decoder.

1. **Encoder:** This part of the network compresses the input into a latent-space representation. It can be represented as a function $f$ that maps an input $x$ to a hidden representation $z$: $z=f(x)$, where $f$ is just a neural network.
2. **Decoder:** This part of the network reconstructs the input data from the latent space representation. It can be represented as a function $g$ that maps the latent representation $z$ of $x$ to an approximation (reconstruction) $\hat{x} = g(z)$. The function $g$ is also a neural network that usually replicates the architecture of $f$, but backwards.

Therefore, the autoencoder is simply a composition of two functions: $\hat{x} = g(f(x))$. And its objective is to minimize the reconstruction error, which is the Mean Squared Error between $x$ and $\hat{x}$:

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2 $$

With $n$ the number of samples.

### A Python application

For illustrating how the Autoencoder works, I'll use the CelebA data set. It is a large-scale face attributes dataset with more than 200.000 celebrity images, each with 40 attribute annotations. The images in this dataset cover 10,177 number of identities, 202,599 number of face images, and 5 landmark locations, 40 binary attributes annotations per image.

This dataset is normally used as a benchmark for training models specialized in computer vision tasks: face attribute recognition, face recognition, face detection, landmark (or facial part) localization, and face editing & synthesis. 

The following code will implement a basic Autoencoder for replicate some celebrity faces.

First, I need to import the necessary libraries and packages to use:

```python
import os
import zipfile 
import gdown
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
## Setup
# Number of gpus available
ngpu = 1
device = torch.device('cuda:0' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')
```
Then, I'll upload the data set, which is located in a Google Drive folder. The following code is just for getting that data into the working space

```python
## Fetch data from Google Drive 
# Root directory for the dataset
data_root = 'data/celeba'
# Path to folder with the dataset
dataset_folder = f'{data_root}/img_align_celeba'
# URL for the CelebA dataset
url = 'https://drive.google.com/uc?id=1CeCv8rlc7OT8MHMTSr_uzfdbKop1a7pa'
# Path to download the dataset to
download_path = f'{data_root}/img_align_celeba.zip'
# Create required directories 
if not os.path.exists(data_root):
  os.makedirs(data_root)
  os.makedirs(dataset_folder)
# Download the dataset from google drive
gdown.download(url, download_path, quiet=False)
# Unzip the downloaded file 
with zipfile.ZipFile(download_path, 'r') as ziphandler:
  ziphandler.extractall(dataset_folder)
```
Then, I will create a Dataset class in Python, for reading the images and loading the data ready to be used.

```python
## Create a custom Dataset class
class CelebADataset(Dataset):
  def __init__(self, root_dir, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform 
    self.image_names = natsorted(image_names)

  def __len__(self): 
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image 
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)

    return img

## Load the dataset 
# Path to directory with all the images
img_folder = f'{dataset_folder}/img_align_celeba'
# Spatial size of training images, images are resized to this size.
image_size = 64
# Transformations to be applied to each individual image sample: resizing, cropping, converting to a tensor and normalizing
transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
])
# Load the dataset from file and apply transformations
celeba_dataset = CelebADataset(img_folder, transform)
```
The last step before training the model, is creating a data loader of the CelebA data set. This will import a number of images (batch size) for training the autoencoder, instead of training the model with the whole dataset.

```python
## Create a dataloader 
# Batch size during training
batch_size = 128
# Number of workers for the dataloader
num_workers = 0 if device.type == 'cuda' else 2
# Whether to put fetched data tensors to pinned memory
pin_memory = True if device.type == 'cuda' else False

celeba_dataloader = torch.utils.data.DataLoader(celeba_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                shuffle=True)
```

Now, I'll split the dataset into a training and a test set, using the data loader class previously coded:

```python
from torch.utils.data import random_split
# Define the proportion or size of the test set
test_size = 0.2  # for example, 20% of the dataset
total_size = len(celeba_dataset)
test_size = int(test_size * total_size)
train_size = total_size - test_size
# Split the dataset
train_dataset, test_dataset = random_split(celeba_dataset, [train_size, test_size])
# Create DataLoaders for train and test sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

```

And finally, here I code the Autoencoder structure: first, I initialize the model, then I define the encoder as a convolutional neural network (a neural network that applies convolutional layers, very useful when working with images), and then I define the decoder as another convolutional neural network, with the same structure as the encoder, but reversed. The autoencoder will be just the application of both the encoder and the decoder.

```python
#### Autoencoder structure #####
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [batch, 16, 32, 32]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [batch, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)                      # [batch, 64, 10, 10]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),           # [batch, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # [batch, 16, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),   # [batch, 3, 64, 64]
            nn.Sigmoid()  # Using Sigmoid to scale the output to [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```
For training the autoencoder, I define first the initialization, the loss function which is in this case the MSE, and the optimizer, which is Adam, with a learning rate of 0.001. The training will last 5 epochs and will iterate over each batch of the training set such that it minimizes the loss function.

```python
# Autoencoder initialization
autoencoder = Autoencoder().to(device)

# Define the Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
### Train the autoencoder
num_epochs = 5  
for epoch in range(num_epochs):
    for data in train_loader:  # Iterate over each batch in the DataLoader
        img = data.to(device)  # Move the data to the device
        # Forward pass
        output = autoencoder(img)
        loss = criterion(output, img)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

Finally, I'll evaluate the performance of the Autoencoder in the test set. For such purpose, I'll compare a sample of the original images with their respective reconstructed images, and compute the average reconstruction error over the test set:

```python
### Evaluate autoencoder
with torch.no_grad():  # No need to track gradients
    for data in test_loader:
        img = data.to(device)
        output = autoencoder(img)
        # Compare output images with original images
import matplotlib.pyplot as plt
import numpy as np

# Function to convert a tensor to a numpy image
def to_img(x):
    x = 0.5 * (x + 1)  # Unnormalize
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x.detach().cpu().numpy()

# Switch model to evaluation mode
autoencoder.eval()

# Calculate the loss - here we're using MSE
mse_loss = nn.MSELoss()
total_loss = 0.0

# We'll also visualize some of the outputs
num_images_to_show = 5
fig, axes = plt.subplots(nrows=2, ncols=num_images_to_show, figsize=(10, 4))

# Use the first `num_images_to_show` images from the test_loader
for i, data in enumerate(test_loader):
    img = data.to(device)
    output = autoencoder(img)
    loss = mse_loss(output, img)
    total_loss += loss.item()

    # Only plot the first batch of images
    if i == 0:
        # Convert the images to a displayable format
        original_images = to_img(img)
        reconstructed_images = to_img(output)

        # Plot original and reconstructed images
        for k in range(num_images_to_show):
            axes[0, k].imshow(np.transpose(original_images[k], (1, 2, 0)))
            axes[0, k].set_title('Original')
            axes[0, k].axis('off')

            axes[1, k].imshow(np.transpose(reconstructed_images[k], (1, 2, 0)))
            axes[1, k].set_title('Reconstructed')
            axes[1, k].axis('off')

        plt.show()

        break  # We only want to visualize one batch

# Calculate the average loss per image
average_loss = total_loss / len(test_loader.dataset)
print(f'Average reconstruction error: {average_loss}')
```

The results show that the average reconstruction error is quite small, 6.0968116286558865e-06, and the reconstructed images are quit similar to the original ones. So the autoencoder works well. The results can be found in the .ipynb file attached to this project.

## Variational Auto Encoder


