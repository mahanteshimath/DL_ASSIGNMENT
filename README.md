# Variational Autoencoder (VAE) Implementation on Frey Face Dataset

This project implements a Variational Autoencoder (VAE) to learn a latent representation of face images from the Frey Face dataset. The implementation demonstrates the power of VAEs in learning meaningful latent representations and generating new faces.

## Implementation Details

### Architecture

1. **Encoder Network**:
   - Input: 28x20 grayscale images
   - 3 Convolutional layers with ReLU activation
   - Max pooling layers for dimensionality reduction
   - Final dense layers for mean (μ) and log-variance (log σ²) of latent space
   - Architecture: Input(28x20) → Conv(32) → Conv(64) → Conv(128) → Dense(20)

2. **Latent Space**:
   - 20-dimensional latent space (as per requirement)
   - Uses reparameterization trick for backpropagation
   - Samples from N(μ, σ²) during training

3. **Decoder Network**:
   - Input: 20-dimensional latent vector
   - 3 Transposed Convolutional layers with ReLU activation
   - Final Sigmoid activation for pixel values
   - Additional upsampling to match original dimensions
   - Architecture: Input(20) → Dense(768) → ConvTranspose(64) → ConvTranspose(32) → ConvTranspose(1)

### Training Process

- **Loss Function**: Combines two terms:
  1. Reconstruction Loss (Binary Cross-Entropy)
  2. KL Divergence Loss
- **Optimizer**: Adam with learning rate 1e-3
- **Batch Size**: 32
- **Training/Test Split**: 80%/20%
- **Epochs**: 30

## Results and Visualizations

The training process generates several visualizations:

1. **Loss Curve** (`loss_curve.png`):
   - Shows convergence of both training and test loss
   - Demonstrates successful learning with decreasing reconstruction and KL divergence losses

2. **Reconstructions** (`reconstructions.png`):
   - Compares original images with their VAE reconstructions
   - Shows high-quality reconstruction capability

3. **Generated Samples** (`generated_samples.png`):
   - Random faces generated from the latent space
   - Demonstrates model's ability to generate new, realistic faces

4. **Latent Space Variations** (`latent_variations.png`):
   - Shows how varying each latent dimension affects the generated faces
   - Demonstrates learned meaningful features like expression, orientation, lighting

5. **2D Latent Space** (`latent_space_2d.png`):
   - 2D visualization of the learned latent space
   - Shows clustering and organization of face features

## Learning Outcomes

1. **Latent Space Understanding**:
   - The 20-dimensional latent space effectively captures facial features
   - Different dimensions control interpretable aspects of the faces
   - Smooth transitions in latent space produce realistic face variations

2. **Model Architecture**:
   - Importance of proper encoder-decoder balance
   - Role of convolutional layers in capturing spatial features
   - Impact of latent space dimensionality on reconstruction quality

3. **Training Insights**:
   - Balance between reconstruction and KL divergence losses
   - Importance of proper normalization and data preprocessing
   - Effect of batch size and learning rate on convergence

4. **Practical Considerations**:
   - Memory management for large datasets
   - Importance of proper data loading and batching
   - Trade-offs between model complexity and training time

## Requirements

- PyTorch >= 2.2.0
- torchvision >= 0.17.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- Pillow >= 8.3.0
- scipy >= 1.7.0

## Usage

1. Place `frey_rawface.mat` in the `data` folder
2. Run the training script:
   ```bash
   python Q2.py
   ```
3. The script will automatically:
   - Load and preprocess the Frey Face dataset
   - Train the VAE model
   - Generate visualizations
   - Save the trained model

## Model Checkpoints

The implementation saves model checkpoints:
- Every 10 epochs during training
- Final model saved as `vae_final_model.pth`
DL_ASSIGNMENT
