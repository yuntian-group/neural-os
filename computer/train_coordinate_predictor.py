import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from omegaconf import OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from latent_diffusion.ldm.util import instantiate_from_config
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

# Assuming your DataModule and dataset classes are accessible
# Adjust the import path as necessary
#from data.data_processing.datasets import DataModule, normalize_image, denormalize_image
def normalize_image(image_path): 

    """
    Takes in an image path or an image and returns the normalized image in a tensor.
    """
    
    if isinstance(image_path, str): image = Image.open(image_path)
    else: image = image_path

    if not image.mode == "RGB":
        image = image.convert("RGB")

    image = (np.array(image) / 127.5 - 1.0).astype(np.float32)

    return torch.tensor(image)


# --- New Image-based Coordinate Predictor (Revised Architecture) ---
class ImageCoordinatePredictor(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, num_out_coords=2, intermediate_conv_channels=2048, head_hidden_dim=512):
        """
        A CNN model to predict coordinates directly from images, revised to closer match
        the TensorFlow localizer example structure.

        Args:
            backbone_name (str): Name of the torchvision backbone (e.g., 'resnet18', 'resnet50'). Recommend 'resnet50'.
            pretrained (bool): Whether to use pretrained weights for the backbone.
            num_out_coords (int): Number of output coordinates (e.g., 2 for x, y).
            intermediate_conv_channels (int): Channels in the intermediate Conv2D layer.
            head_hidden_dim (int): Hidden dimension in the final regression head.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.num_out_coords = num_out_coords

        # --- Load Backbone Feature Extractor ---
        if hasattr(tv_models, backbone_name):
            model_func = getattr(tv_models, backbone_name)
            try:
                # Attempt new weights API
                if pretrained:
                    weights_enum_name = f"{backbone_name.capitalize()}_Weights"
                    if hasattr(tv_models, weights_enum_name):
                        weights_enum = getattr(tv_models, weights_enum_name)
                        if hasattr(weights_enum, 'DEFAULT'):
                            base_model = model_func(weights=weights_enum.DEFAULT)
                            print(f"--- Loaded {backbone_name} with pretrained weights (Weights enum API) ---")
                        else:
                             raise AttributeError(f"Weights enum {weights_enum_name} found but has no DEFAULT attribute.")
                    else:
                         raise AttributeError(f"Weights enum {weights_enum_name} not found.")
                else:
                    base_model = model_func(weights=None)
                    print(f"--- Loaded {backbone_name} without pretrained weights (using weights=None) ---")

            except (AttributeError, TypeError) as e:
                # Fallback to older `pretrained` argument
                print(f"--- New weights API failed ({type(e).__name__}: {e}), falling back to older 'pretrained={pretrained}' argument for {backbone_name} --- ")
                base_model = model_func(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone name: {backbone_name}")

        # ----- MODIFY LAYER 4 TO REDUCE DOWNSAMPLING -----
        # For ResNet, this is the last layer which would normally have stride=2
        # We modify it to have stride=1 and dilation=2 to maintain receptive field size
        # This reduces total downsampling from 32x to 16x
        
        # First extract the base model layers so we can modify them
        feature_extractor_list = list(base_model.children())[:-2]

        # ---- MODIFY FIRST CONV LAYER (conv1) ----
        if len(feature_extractor_list) > 0 and isinstance(feature_extractor_list[0], nn.Conv2d):
            print(f"--- Original conv1 stride: {feature_extractor_list[0].stride} ---")
            feature_extractor_list[0].stride = (1, 1)
            print(f"--- Modified conv1 stride to: {feature_extractor_list[0].stride} ---")
        else:
            print("--- Warning: Could not find expected Conv2d layer at index 0 to modify stride. ---")

        
        # The layer to modify is the last layer (typically layer4 in ResNet)
        layer4 = feature_extractor_list[-1]
        
        # Modify stride in the first bottleneck of layer4
        if hasattr(layer4[0], 'conv2'):
            # Change stride in 3x3 conv of first block
            layer4[0].conv2.stride = (1, 1)
            # Apply dilation to maintain receptive field
            layer4[0].conv2.dilation = (2, 2)
            # Adjust padding to compensate for dilation
            layer4[0].conv2.padding = (2, 2)
            
            # Also modify the downsample convolution if it exists
            if hasattr(layer4[0], 'downsample') and layer4[0].downsample is not None:
                if isinstance(layer4[0].downsample, nn.Sequential) and isinstance(layer4[0].downsample[0], nn.Conv2d):
                    layer4[0].downsample[0].stride = (1, 1)
            
            print(f"--- Modified Layer 4 stride from 2 to 1 and applied dilation=2 ---")
            print(f"--- This reduces total downsampling from 32x to 16x ---")
        
        # Use the modified backbone
        self.feature_extractor = nn.Sequential(*feature_extractor_list)

        print(f"--- Using {backbone_name} feature extractor (removed final pool/fc) ---")

        # Determine the number of output channels from the feature extractor
        # We need a dummy forward pass to determine this dynamically
        with torch.no_grad():
             # Create a dummy input matching expected image dims (e.g., B=1, C=3, H=224, W=224)
             # Use a standard size like 224x224 which ResNet expects, actual size handled later.
             dummy_input = torch.randn(1, 3, 224, 224)
             dummy_features = self.feature_extractor(dummy_input)
             backbone_out_channels = dummy_features.shape[1] # Get channels dim (B, C, H, W)
             feature_h, feature_w = dummy_features.shape[2], dummy_features.shape[3]
             print(f"--- Backbone feature extractor output shape: {dummy_features.shape} ---")
             print(f"--- Backbone feature extractor output channels: {backbone_out_channels} ---")

        # --- Intermediate Convolutional Layer ---
        # Mimics the Conv2D(1024, 3, 2, activation='relu') from TF example
        # padding=1 keeps feature map size roughly halved with stride=2
        self.intermediate_conv = nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=intermediate_conv_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.intermediate_relu = nn.ReLU()
        print(f"--- Added intermediate Conv2d layer (out_channels={intermediate_conv_channels}) ---")

        # --- NO POOLING --- Flattening only ---
        # self.pool = nn.AdaptiveMaxPool2d(pool_output_size) # Removed pooling
        self.flatten = nn.Flatten()
        print(f"--- Removed pooling layer. Using Flatten only. ---")

        # --- Regression Head (Large Input) ---
        # Input dimension MUST match the output of flatten
        # Determine spatial size after intermediate conv dynamically (safer)
        with torch.no_grad():
             # Need output shape from intermediate conv
             # First get output shape from feature extractor
             dummy_input = torch.randn(1, 3, 224, 224) # Use standard size again
             dummy_features = self.feature_extractor(dummy_input)
             # Then pass through intermediate conv
             dummy_intermediate_out = self.intermediate_conv(dummy_features)
             intermediate_H, intermediate_W = dummy_intermediate_out.shape[-2:]
             print(f"--- Intermediate conv output spatial size (HxW): {intermediate_H}x{intermediate_W} ---")

        # Flattened output size = C * H * W
        input_head_dim = 6291456 # 786432 * 4#196608  
        # Use head_hidden_dim from init args (default 512), multiplied by 4 as per user edit
        # Ensure head_hidden_dim is an integer
        simple_head_hidden_dim = input_head_dim 
        # print(input_head_dim)
        # print(196608)
        # Explicitly use calculated dimensions
        self.regression_head = nn.Sequential(
            # Layer 1: Correctly use large input_head_dim and calculated simple_head_hidden_dim
            # nn.Linear(in_features=input_head_dim, out_features=simple_head_hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # Layer 2: Hidden dim -> Output coordinates
            nn.Linear(in_features=simple_head_hidden_dim, out_features=num_out_coords)
            # Output raw coordinates (no final activation for regression)
        )


    def forward(self, image):
        """
        Processes an image batch [B, C, H, W] to predict coordinates [B, num_out_coords].
        Assumes input is already in BCHW format.
        NO POOLING is used.
        """
        # 1. Extract features using the backbone
        features = self.feature_extractor(image) # Output: [B, backbone_out_channels, H_feat, W_feat]
        # 2. Pass through intermediate conv layer
        x = self.intermediate_conv(features) # Output: [B, intermediate_conv_channels, H', W']
        x = self.intermediate_relu(x)
        # 3. NO POOLING
        # x = self.pool(x)
        # 4. Flatten the feature map
        x = self.flatten(x) # Output: [B, intermediate_conv_channels * H' * W']
        # 5. Predict coordinates using the regression head
        coords = self.regression_head(x) # Output: [B, num_out_coords]
        return coords


class CoordinateTrainer(pl.LightningModule):
    def __init__(self, image_predictor_config, learning_rate=1e-4):
        """
        LightningModule for training the ImageCoordinatePredictor using images.

        Args:
            image_predictor_config (OmegaConf): Configuration for ImageCoordinatePredictor.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
        self.expo = 0

        # Instantiate the image-based model from config
        # Ensure config specifies 'target', 'params' for ImageCoordinatePredictor
        # Example config section:
        # image_predictor_model:
        #   target: computer.train_coordinate_predictor.ImageCoordinatePredictor # Adjusted path
        #   params:
        #     backbone_name: resnet18
        #     pretrained: True
        #     num_out_coords: 2
        self.coord_predictor = instantiate_from_config(image_predictor_config)
        
        # Print number of trainable parameters
        num_params = sum(p.numel() for p in self.coord_predictor.parameters() if p.requires_grad)
        print(f"--- Image Coordinate Predictor Model Initialized ---")
        print(f"--- Trainable Parameters: {num_params:,} ---")

        print("USING LR", self.learning_rate)
        
        # Use L1 Loss
        self.loss_fn = nn.L1Loss()
        print("!!! Using L1 Loss (nn.L1Loss) !!!")

    def forward(self, batch):
        """
        Predicts coordinates from the IMAGE tensor in the batch.
        Assumes batch contains 'image' potentially in BHWC format and rearranges to BCHW.
        """
        # --- USE IMAGE KEY ---
        if 'image' in batch:
            images = batch['image']
            # Ensure image is float type (e.g., normalized to [0, 1] or [-1, 1])
            if images.dtype != torch.float32:
                 print(f"Warning: Input image tensor dtype is {images.dtype}, converting to float32.")
                 images = images.float()

            # print(f"Input image shape from batch: {images.shape}")

            # --- Rearrange to BCHW if needed ---
            if images.ndim == 4 and images.shape[1] != 3 and images.shape[3] == 3:
                #print(f"Detected channels-last format (shape: {images.shape}). Rearranging to channels-first (BCHW).")
                images = rearrange(images, 'b h w c -> b c h w')
                # print(f"Rearranged image shape for model: {images.shape}")
            elif images.ndim != 4 or images.shape[1] != 3:
                 # If it's not 4D BCHW or 4D BHWC that we can fix, raise an error.
                 raise ValueError(f"Unexpected image tensor shape: {images.shape}. "
                                  f"Expected 4D BCHW ([B, 3, H, W]) or BHWC ([B, H, W, 3]).")
            # --- End Rearrange ---

        else:
            raise ValueError("Batch missing 'image' key.")
        # --- END USE IMAGE KEY ---

        # Predict coordinates from the (now BCHW) images
        predicted_coords = self.coord_predictor(images)
        return predicted_coords

    def _get_targets(self, batch):
        """Helper function to extract raw target coordinates (x, y)."""
        # Assumes keys are 'x_1', 'y_1' and they are single values per sample
        if 'x_1' not in batch or 'y_1' not in batch:
             raise ValueError("Batch missing 'x_1' or 'y_1' keys for target coordinates.")

        x_target = batch['x_1']
        y_target = batch['y_1']

        # Ensure targets are tensors and have correct shape [B, 1]
        if not isinstance(x_target, torch.Tensor): x_target = torch.tensor(x_target, device=self.device)
        if not isinstance(y_target, torch.Tensor): y_target = torch.tensor(y_target, device=self.device)

        if x_target.ndim == 0: x_target = x_target.unsqueeze(0) # Handle single sample case
        if y_target.ndim == 0: y_target = y_target.unsqueeze(0)

        if x_target.ndim == 1: x_target = x_target.unsqueeze(1)
        if y_target.ndim == 1: y_target = y_target.unsqueeze(1)

        # Ensure shape is [B, 1]
        if x_target.shape[1] != 1 or y_target.shape[1] != 1:
            raise ValueError(f"Target coordinate tensors have unexpected shape. "
                             f"x_target: {x_target.shape}, y_target: {y_target.shape}. Expected [B, 1].")

        true_coords = torch.cat((x_target, y_target), dim=1).float()
        return true_coords # Return raw coordinates [B, 2]

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        true_coords = self._get_targets(batch) # Should be [B, 2]
        predicted_coords = self(batch) # Uses images via forward(), should be [B, 2]

        # Calculate L1 loss between raw predicted and raw true coords
        loss = self.loss_fn(predicted_coords, true_coords)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"!!! NaN or Inf detected in LOSS at step {batch_idx} !!!")
            self.log('nan_inf_loss', 1.0)
            print("!!! Returning zero loss to prevent crash !!!")
            return torch.zeros_like(loss, requires_grad=True)

        self.log('train_l1_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        self.expo = 0.99*self.expo + 0.01*loss
        self.log('train_l1_loss_exp', self.expo, on_step=True, on_epoch=True, prog_bar=True)

        # Log pixel error (L2 distance)
        with torch.no_grad():
            # Detach predictions before calculating metric if needed
            preds_detached = predicted_coords.detach()
            safe_preds = torch.nan_to_num(preds_detached, nan=-1.0, posinf=1e6, neginf=-1e6)
            mean_pixel_error = torch.sqrt(((safe_preds - true_coords)**2).sum(dim=1)).mean()
            if torch.isnan(mean_pixel_error) or torch.isinf(mean_pixel_error):
                 print(f"!!! NaN or Inf detected in pixel_error calculation !!! Preds: {safe_preds[0].tolist()}, True: {true_coords[0].tolist()}")
                 mean_pixel_error = torch.tensor(1000.0, device=self.device)
            self.log('pixel_error', mean_pixel_error, on_step=True, on_epoch=True, prog_bar=True)

        # Print raw coords
        if batch_idx > 0 and batch_idx % 100 == 0:
             with torch.no_grad():
                 for i in range(min(2, true_coords.size(0))):
                    # Calculate error directly here for printing
                    err = torch.sqrt(((safe_preds[i] - true_coords[i])**2).sum())
                    print(f"Step {batch_idx} Sample {i}: Pred {safe_preds[i].tolist()}, "
                          f"True {true_coords[i].tolist()}, L2 Error: {err.item():.4f}")

        return loss
        
    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.
        """
        true_coords = self._get_targets(batch)
        predicted_coords = self(batch)
        # Calculate L1 loss
        val_loss = self.loss_fn(predicted_coords, true_coords)
        self.log('val_l1_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log validation pixel error
        with torch.no_grad():
            preds_detached = predicted_coords.detach()
            safe_preds = torch.nan_to_num(preds_detached, nan=-1.0, posinf=1e6, neginf=-1e6)
            val_pixel_error = torch.sqrt(((safe_preds - true_coords)**2).sum(dim=1)).mean()
            if torch.isnan(val_pixel_error) or torch.isinf(val_pixel_error):
                val_pixel_error = torch.tensor(1000.0, device=self.device)
            self.log('val_pixel_error', val_pixel_error, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return val_loss

    def configure_optimizers(self):
        # Restore reasonable optimizer settings
        lr = self.hparams.learning_rate
        print(f"--- Using LR: {lr} ---")
        optimizer = torch.optim.Adam(
            # Filter parameters to ensure only trainable ones are optimized
            filter(lambda p: p.requires_grad, self.coord_predictor.parameters()),
            lr=lr,
            weight_decay=1e-5 # Keep weight decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10, # Consider increasing patience if LR reduces too quickly
            verbose=True
        )
        print("--- Using ReduceLROnPlateau LR Scheduler ---")
        print("--- Using Adam Optimizer with Weight Decay ---")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_l1_loss", # Monitor train loss
                "frequency": 1,
                "interval": "epoch"
            },
        }

    def predict(self, image_path, target_x, target_y, return_overlay=False, already_normalized=False):
        """
        Load a PNG image, predict coordinates, and calculate L1 loss against target.
        
        Args:
            image_path (str): Path to the PNG image
            target_x (float): Target x coordinate
            target_y (float): Target y coordinate
            return_overlay (bool): If True, returns the original image with predicted coordinate marked
            already_normalized (bool): If True, skip normalization (for when passing already normalized images)
            
        Returns:
            dict: Contains predicted coordinates and L1 loss, and optionally the overlay image
        """
        # Ensure model is in evaluation mode
        self.eval()
        
        # Load and preprocess image
        try:
            # Load the original image for overlay if requested
            if return_overlay:
                original_image = Image.open(image_path).convert('RGB')
            
            # Process the image
            if already_normalized:
                # Load the image without normalizing (it's a path to an already normalized image)
                image_tensor = Image.open(image_path)
                # Convert PIL to tensor
                to_tensor = transforms.ToTensor()
                image_tensor = to_tensor(image_tensor)
            else:
                image_tensor = normalize_image(image_path)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            # Move to the same device as the model
            image_tensor = image_tensor.to(self.device)
            
            # Create target tensor
            target_coords = torch.tensor([[target_x, target_y]], dtype=torch.float32).to(self.device)
            
            # Predict
            with torch.no_grad():
                predicted_coords = self({"image": image_tensor})
            
            # Calculate L1 loss
            loss = self.loss_fn(predicted_coords, target_coords)
            
            # Convert tensors to numpy for easier handling
            predicted_np = predicted_coords.cpu().numpy()[0]
            
            result = {
                "predicted_x": float(predicted_np[0]),
                "predicted_y": float(predicted_np[1]),
                "target_x": target_x,
                "target_y": target_y,
                "l1_loss": float(loss.item())
            }
            
            # Create and return overlay image if requested
            if return_overlay:
                # Create figure and axes
                fig = Figure(figsize=(10, 8))
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                
                # Display original image
                ax.imshow(np.array(original_image))
                
                # Plot predicted point (red) and target point (green)
                ax.scatter(float(predicted_np[0]), float(predicted_np[1]), 
                          c='red', s=100, marker='o', label='Prediction')
                ax.scatter(target_x, target_y, 
                          c='green', s=100, marker='x', label='Target')
                
                # Add legend and title
                ax.legend()
                ax.set_title(f"L1 Loss: {result['l1_loss']:.2f}")
                
                # Save to buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                
                # Create PIL image
                overlay_image = Image.open(buf)

                #overlay_image = denormalize_image(overlay_image)
                
                # Add to result
                result["overlay_image"] = overlay_image
            
            return result
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            # For debugging, print the full traceback
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "predicted_x": 0,
                "predicted_y": 0,
                "target_x": target_x,
                "target_y": target_y,
                "l1_loss": 9999.0
            }

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Image-based Coordinate Predictor Model')
    parser.add_argument('--config', type=str, default="DEBUG.yaml", help='Path to the configuration file.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training.')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # --- Setup ---
    save_path = config.save_path
    os.makedirs(save_path, exist_ok=True)
    
    # Instantiate DataModule
    # IMPORTANT: Ensure your dataset class specified in the config 
    # correctly loads the images and yields batches
    # containing dictionaries with 'image', 'x_1', and 'y_1' keys.
    # Ensure 'image' is a tensor [B, C, H, W] and normalized appropriately.
    # Ensure 'x_1', 'y_1' are the target coordinates.
    data_module: DataModule = instantiate_from_config(config.data)

    # Instantiate the Lightning Trainer Module
    # Make sure the config uses the correct key for the image predictor config,
    # e.g., 'image_predictor_model' instead of 'coordinate_predictor_model'
    if 'image_predictor_model' not in config:
        raise ValueError("Configuration file must contain 'image_predictor_model' section.")
    coord_trainer = CoordinateTrainer(
        image_predictor_config=config.image_predictor_model, # Use the new config section
        learning_rate=config.get("learning_rate", 1e-4)
    )

    # --- Trainer Configuration ---
    lightning_config = config.get("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # Handle potential legacy 'gpus' key vs 'devices'
    if 'gpus' in trainer_config and 'devices' not in trainer_config:
         trainer_config['devices'] = trainer_config.pop('gpus')
         if 'accelerator' not in trainer_config:
              trainer_config['accelerator'] = 'gpu' # Assume gpu if gpus was set

    trainer_opt = argparse.Namespace(**trainer_config)
    
    # Checkpoint Callback
    # Update filename format if desired
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=10000, # Save every 10k steps
        save_top_k=-1, # Save all checkpoints triggered by step count, ignore monitor
        save_last=config.lightning.checkpointing.get("save_last", True), # Optionally save the last checkpoint
        dirpath=save_path,
        filename=config.lightning.checkpointing.get("filename", 'image-coord-predictor-{epoch:02d}-{step}') # Added step to filename
    )

    trainer_kwargs = vars(trainer_opt)
    if 'gradient_clip_val' in config.lightning.trainer:
        print(f"!!! Enabling Gradient Clipping value: {config.lightning.trainer.gradient_clip_val} !!!")
        trainer_kwargs['gradient_clip_val'] = config.lightning.trainer.gradient_clip_val
        trainer_kwargs['gradient_clip_algorithm'] = config.lightning.trainer.get('gradient_clip_algorithm', 'norm')

    # Create Trainer
    trainer = pl.Trainer(**trainer_kwargs, callbacks=[checkpoint_callback])

    # --- Training ---
    print(f"Starting training with ImageCoordinatePredictor, saving checkpoints to: {save_path}")
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            trainer.fit(coord_trainer, datamodule=data_module, ckpt_path=args.resume)
        else:
            print(f"WARNING: Checkpoint path does not exist: {args.resume}")
            print("Starting from scratch instead.")
            trainer.fit(coord_trainer, datamodule=data_module)
    else:
        # Start training from scratch
        trainer.fit(coord_trainer, datamodule=data_module)

    # --- Save Final Model (Optional) ---
    final_save_path = os.path.join(save_path, "final_image_coordinate_predictor.ckpt")
    # Check if trainer object has save_checkpoint method before calling
    if hasattr(trainer, 'save_checkpoint'):
      trainer.save_checkpoint(final_save_path)
      print(f"Training finished. Final model saved to {final_save_path}")
    else:
      print("Trainer object does not have save_checkpoint method. Final model not saved automatically.") 
