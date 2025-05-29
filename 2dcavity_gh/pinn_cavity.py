import os
import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np

from utils_cavity_ngsolve import*

from contextlib import contextmanager
from diffusers import UNet2DModel
import h5py
import yaml
import argparse

torch.set_float32_matmul_precision('medium')

def parse_args():
    parser = argparse.ArgumentParser(description='Train PINN model for porous media using YAML config')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    return parser.parse_args()

def run_training(config_path):
    # Set multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    num_threads = os.cpu_count()
    print(f"You have access to {num_threads} CPU threads.")
    
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Print all contents of the YAML file
    print("\n=== YAML Configuration File Contents ===")
    with open(config_path, 'r') as file:
        yaml_contents = file.read()
        print(yaml_contents)
    print("======================================\n")
    
    # Extract configuration values
    train_num_samples = config['data']['train_num_samples']
    valid_num_samples = config['data']['valid_num_samples']
    
    ## Extract UNet model configuration if it exists
    model_config = config.get('model', {}).get('unet_config', None)

    train_file_path =  f'./data/train_data_{train_num_samples}.h5'
    valid_file_path =  f'./data/valid_data_{valid_num_samples}_May17.h5'
    #valid_file_path =  f'./data/train_data_{valid_num_samples}.h5'

    print(f'train_file_path = {train_file_path}')
    print(f'valid_file_path = {valid_file_path}')
    
    model_type = config['model']['type']
    learning_rate = config['model']['learning_rate']
    
    save_dir = config['training']['save_dir']
    read_ckpt = config['training']['read_ckpt']
    if read_ckpt == "None":
        read_ckpt = None
    
    fem_iterations = config['training']['fem_iterations']
    Tmax = config['training']['Tmax']
    lambda_ux = config['training']['lambda_ux']
    lambda_uy = config['training']['lambda_uy']
    lambda_p = config['training']['lambda_p']
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    
    # Enable progress bar if specified in config, default to disabled
    enable_progress_bar = config['training'].get('enable_progress_bar', False)
    enable_model_summary = config['training'].get('enable_model_summary', False)
    enable_validation = config['training'].get('enable_validation', True)
    
    # GPU configuration
    num_gpus = config['training'].get('num_gpus', 2)
    precision = config['training'].get('precision', '16-mixed')

    
    
    # Create PyTorch datasets
    train_dataset = CavityDataset(train_file_path)
    valid_dataset = CavityDataset(valid_file_path)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['dataloader'].get('num_workers', 1),
        pin_memory=config['dataloader'].get('pin_memory', True),
        persistent_workers=config['dataloader'].get('persistent_workers', True)
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['dataloader'].get('num_workers', 1),
        pin_memory=config['dataloader'].get('pin_memory', True),
        persistent_workers=config['dataloader'].get('persistent_workers', True)
    )
    
    # Initialize the model
    model = FEMPhysicsModule(
        model_type=model_type,
        learning_rate=learning_rate,
        fem_iterations=fem_iterations,
        Tmax = Tmax,
        lambda_ux = lambda_ux,
        lambda_uy = lambda_uy,
        lambda_p = lambda_p,
        model_config = model_config
    )

    # load in a previous checkpoint/weights if specified 
    if read_ckpt is not None:
        print(f"Loading weights from checkpoint: {read_ckpt}")
        checkpoint = torch.load(read_ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded successfully!")
    
    # Setup checkpointing with configurable parameters
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=config['checkpoint'].get('filename', 'model_{epoch}'),
        save_top_k=config['checkpoint'].get('save_top_k', -1),
        monitor=config['checkpoint'].get('monitor', 'train_loss'),
        mode=config['checkpoint'].get('mode', 'min'),
        save_last=config['checkpoint'].get('save_last', True),
        every_n_epochs=config['checkpoint'].get('every_n_epochs', 10),
    )
    
    # Setup custom output callback
    output_callback = SimplifiedOutputCallback(num_epochs)
    
    # Setup logger
    logger = TensorBoardLogger(save_dir=os.path.join(save_dir, 'logs'))
    
    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, output_callback],
        logger=logger,
        log_every_n_steps=config['training'].get('log_every_n_steps', 1),
        accelerator=config['training'].get('accelerator', 'auto'),
        devices=num_gpus,
        strategy=config['training'].get('strategy', 'ddp'),
        precision=precision,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
    )
    
    # Start training
    if enable_validation:
        trainer.fit(model, train_loader, valid_loader)
    else:
        trainer.fit(model, train_loader, None)
    
    print("Done!")

class CavityDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.num_samples = f['img1'].shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            img1 = torch.from_numpy(f['img1'][idx])
            if 'Ueq' in f:
                Ueq = torch.from_numpy(f['Ueq'][idx])
                Veq = torch.from_numpy(f['Veq'][idx])
                Peq = torch.from_numpy(f['Peq'][idx])
                return img1, Ueq, Veq, Peq
            else:
                return img1



class ImagePredictorUNet(nn.Module):
    def __init__(self, config=None):
        super(ImagePredictorUNet, self).__init__()
        
        # Use config if provided, otherwise use defaults
        if config is None:
            config = {}
        
        # Get UNet configuration parameters
        sample_size = config.get('sample_size', (32, 32))
        in_channels = config.get('in_channels', 1)
        out_channels = config.get('out_channels', 3)
        layers_per_block = config.get('layers_per_block', 1)
        block_out_channels = config.get('block_out_channels', (8, 16, 32))
        norm_num_groups = config.get('norm_num_groups', 2)
        down_block_types = config.get('down_block_types', ("DownBlock2D", "DownBlock2D", "DownBlock2D"))
        up_block_types = config.get('up_block_types', ("UpBlock2D", "UpBlock2D", "UpBlock2D"))
        attention_head_dim = config.get('attention_head_dim', 4)
        
        # Define the UNet2D model
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            norm_num_groups=norm_num_groups,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            attention_head_dim=attention_head_dim,
            act_fn = "silu"
        )
        
        # Parameters for output normalization
        self.pressure_min = config.get('pressure_min', 0.01)
        self.pressure_max = config.get('pressure_max', 3.0)

        self.ux_min = config.get('ux_min', -0.5)
        self.ux_max = config.get('ux_max', 0.5)

        self.uy_min = config.get('uy_min', -0.5)
        self.uy_max = config.get('uy_max', 0.5)

        # Initialize weights using the specified method
        init_method = config.get('init_method', 'kaiming')
        if init_method == "None":
            init_method = None
        gain = config.get('init_gain', 0.02)
        if init_method is not None:
            self.initialize_weights(init_method, gain)
        
    def initialize_weights(self, method='kaiming', gain=0.02):
        """Initialize the weights of the UNet model using the specified method.
        
        Args:
            method (str): Initialization method. Options: 'kaiming', 'xavier', 'orthogonal', 
                        'normal', 'zeros', 'near_zero'
            gain (float): The gain parameter used for some initialization methods
        """
        for m in self.unet.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif method == 'xavier':
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=gain)
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif method == 'zeros':
                    nn.init.zeros_(m.weight)
                elif method == 'near_zero':
                    # Initialize with very small values close to zero
                    nn.init.normal_(m.weight, mean=0, std=gain/10)  # Using a small fraction of gain
                
                if m.bias is not None:
                    if method == 'near_zero':
                        nn.init.normal_(m.bias, mean=0, std=gain/10)
                    else:
                        nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                if method == 'near_zero':
                    nn.init.normal_(m.weight, mean=1, std=gain/10)
                    nn.init.normal_(m.bias, mean=0, std=gain/10)
                else:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.Linear):
                if method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif method == 'xavier':
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=gain)
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif method == 'zeros':
                    nn.init.zeros_(m.weight)
                elif method == 'near_zero':
                    nn.init.normal_(m.weight, mean=0, std=gain/10)
                
                if method == 'near_zero':
                    nn.init.normal_(m.bias, mean=0, std=gain/10)
                else:
                    nn.init.constant_(m.bias, 0)
        
        print(f"UNet weights initialized using {method} initialization" + 
            (f" with gain={gain}" if method in ['xavier', 'orthogonal', 'normal', 'near_zero'] else ""))

    def forward(self, x1):
        device = x1.device
        batch_size = x1.shape[0]
        
        x1 = x1.unsqueeze(1)  # Add channel dimension
        
        x = torch.cat([x1], dim=1)
        
        # Create dummy timesteps and encoder hidden states for UNet3DConditionModel
        timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Forward pass through UNet
        output = self.unet(x, timesteps).sample

        # Normalize the density and velocities outputs
        #velx = self.ux_min + torch.sigmoid(output[:, 0:1]) * (self.ux_max - self.ux_min)
        #vely = self.uy_min + torch.sigmoid(output[:, 1:2]) * (self.uy_max - self.uy_min)
        #pressure = self.pressure_min + torch.sigmoid(output[:, 2:3]) * (self.pressure_max - self.pressure_min)
        
        #x = torch.cat([velx, vely, pressure], dim=1)
        x = output
        return x


class FEMPhysicsModule(pl.LightningModule):
    def __init__(self, 
                 model_type='unet', 
                 learning_rate=1e-5, 
                 fem_iterations=100,
                 Tmax = 50,
                 lambda_ux = 1.0,
                 lambda_uy = 1.0,
                 lambda_p = 1.0,
                 model_config=None):
        super().__init__()
        self.save_hyperparameters()
        
        if model_type == 'unet':
            self.model = ImagePredictorUNet(model_config)
        else:
            raise ValueError('No valid model!')
            
        self.criterion = nn.MSELoss()
        self.criterion_rel = self.relative_l2_loss
        self.learning_rate = learning_rate
        self.fem_iterations = fem_iterations
        self.Tmax = Tmax
        self.lambda_ux = lambda_ux
        self.lambda_uy = lambda_uy
        self.lambda_p = lambda_p
        
        # Keep track of epoch metrics for custom logging
        self.train_losses = []
        self.val_losses = []

        # Add tracking for problematic samples
        self.nan_inf_indices = []  # List to store global indices of problematic samples
        self.epoch_nan_inf_count = 0  # Count of NaN/Inf occurrences in current epoch
        self.total_nan_inf_count = 0  # Total count across all epochs

    def relative_l2_loss(self, pred, target):
        """
        Calculate relative L2 loss for each sample in the batch:
        ||pred - target||_2 / ||target||_2
        
        Args:
            pred: Predicted tensor of shape [batch_size, channels, ...]
            target: Target tensor of shape [batch_size, channels, ...]
            
        Returns:
            Mean of relative L2 losses across the batch
        """
        # Reshape tensors to [batch_size, -1] to calculate norm along all dimensions except batch
        batch_size = pred.size(0)
        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)
        
        # Calculate L2 norm of the difference for each sample (numerator)
        diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
        
        # Calculate L2 norm of the target for each sample (denominator)
        target_norm = torch.norm(target_flat, p=2, dim=1)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-8
        
        # Calculate relative L2 loss for each sample
        rel_l2_loss = diff_norm / (target_norm + epsilon)
        
        # Return mean loss across the batch
        return torch.mean(rel_l2_loss)
    
    def forward(self, x1):
        return self.model(x1)
    
    def _process_train_batch(self, batch, batch_idx):
        img1 = batch
        output = self(img1)
        
        batch_target1, batch_target2, batch_target3 = [], [], []
        valid_indices = []

        # Calculate global indices for this batch
        batch_size = img1.size(0)
        global_start_idx = batch_idx * batch_size

        for i in range(output.size(0)):
            single_img1 = img1[i].cpu().numpy().squeeze()
            single_output = output[i].detach().cpu().numpy()

            #Re_unnorm = (2600 - 2400)*single_img1[0,0] + 2400
            Re_unnorm = (3000 - 2000)*single_img1[0,0] + 2000
            #Re_unnorm = (1000 - 500)*single_img1[0,0] + 500
            
            t1, t2, t3, _ = run_ngsolve_custom(1.0/Re_unnorm,
                                            uin_max = 1.0,
                                            tau = 0.003,
                                            t_iter = self.fem_iterations,
                                            U_initial = single_output[0], 
                                            V_initial = single_output[1], 
                                            P_initial = single_output[2])

            # Check for NaN or Inf values
            if not (np.isnan(t1).any() or np.isnan(t2).any() or np.isnan(t3).any() or
                   np.isinf(t1).any() or np.isinf(t2).any() or np.isinf(t3).any()):
                t1 = torch.from_numpy(np.array(t1)).float().unsqueeze(0).unsqueeze(0).to(self.device)
                t2 = torch.from_numpy(np.array(t2)).float().unsqueeze(0).unsqueeze(0).to(self.device)
                t3 = torch.from_numpy(np.array(t3)).float().unsqueeze(0).unsqueeze(0).to(self.device)
                
                batch_target1.append(t1)
                batch_target2.append(t2)
                batch_target3.append(t3)
                valid_indices.append(i)
            else:
                # Track problematic samples
                global_idx = global_start_idx + i
                self.nan_inf_indices.append(global_idx)
                self.epoch_nan_inf_count += 1
                self.total_nan_inf_count += 1
                
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}], Error: NaN/Inf detected in sample {global_idx}")
            
        if len(valid_indices) > 0:
            target1 = torch.cat(batch_target1, dim=0)
            target2 = torch.cat(batch_target2, dim=0)
            target3 = torch.cat(batch_target3, dim=0)
            target = torch.cat([target1, target2, target3], dim=1)

            valid_output = output[valid_indices]
            
            # Calculate loss
            #loss = self.criterion(valid_output, target)
            #loss = self.criterion_rel(valid_output, target)

            # with regularizer terms
            # Split outputs and targets
            pred_ux, pred_uy, pred_p = valid_output[:, 0:1], valid_output[:, 1:2], valid_output[:, 2:3]
            true_ux, true_uy, true_p = target[:, 0:1], target[:, 1:2], target[:, 2:3]

            #pred_velmag = pred_ux**2 + pred_uy**2 #torch.sqrt(pred_ux**2 + pred_uy**2)
            #true_velmag = true_ux**2 + true_uy**2 #torch.sqrt(true_ux**2 + true_uy**2)

            # Individual losses
            #loss_ux = self.criterion_rel(pred_ux, true_ux)
            #loss_uy = self.criterion_rel(pred_uy, true_uy)
            #loss_p  = self.criterion_rel(pred_p, true_p)

            loss_ux = self.criterion(pred_ux, true_ux)
            loss_uy = self.criterion(pred_uy, true_uy)
            loss_p  = self.criterion(pred_p, true_p)
            #loss_velmag = self.criterion_rel(pred_velmag, true_velmag)

            # Total weighted loss
            loss = self.lambda_ux * loss_ux + self.lambda_uy * loss_uy + self.lambda_p * loss_p #+ loss_velmag

            
            return loss, len(valid_indices)
        
        return None, 0
    
    def _process_val_batch(self, batch, batch_idx):
        img1, Ueq, Veq, Peq = batch
        output = self(img1)

        # Ground truth
        target1 = Ueq.unsqueeze(1).to(self.device)
        target2 = Veq.unsqueeze(1).to(self.device)
        target3 = Peq.unsqueeze(1).to(self.device)
        target = torch.cat([target1, target2, target3], dim=1)

        # Predicted
        pred_ux, pred_uy, pred_p = output[:, 0:1], output[:, 1:2], output[:, 2:3]
        true_ux, true_uy, true_p = target[:, 0:1], target[:, 1:2], target[:, 2:3]

        # Loss
        loss_ux = self.criterion(pred_ux, true_ux)
        loss_uy = self.criterion(pred_uy, true_uy)
        loss_p = self.criterion(pred_p, true_p)
        loss = loss_ux + loss_uy + loss_p

        return loss, img1.size(0)

    def training_step(self, batch, batch_idx):
        loss, valid_count = self._process_train_batch(batch, batch_idx)
        
        if loss is not None:
            # Log metrics without progress bar
            self.log('train_loss', loss, prog_bar=False, sync_dist=True)
            self.log('train_valid_samples', valid_count, prog_bar=False, sync_dist=True)
            return loss
        
        # Return zero loss if no valid samples (will not contribute to gradients)
        return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def validation_step(self, batch, batch_idx):
        loss, valid_count = self._process_val_batch(batch, batch_idx)
        
        if loss is not None:
            # Log metrics without progress bar
            self.log('val_loss', loss, prog_bar=False, sync_dist=True)
            self.log('val_valid_samples', valid_count, prog_bar=False, sync_dist=True)
        
        # Force CUDA cache clearing
        torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        weight_decay = self.hparams.get("weight_decay", 1e-4)  # Default value
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)

        mycos = 0

        
        if mycos == 0:
            print("No cosine scheduler")
            return optimizer
        else:
            scheduler = CosineAnnealingLR(optimizer,T_max=self.Tmax, eta_min=1e-8)
            print("Using cosine scheducler")

            return {"optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1}
                    }

    
    def on_train_epoch_start(self):
        # Reset the epoch counter
        self.epoch_nan_inf_count = 0

    def on_train_epoch_end(self):
        # Access the trainer to get logged metrics
        train_loss = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0))
        self.train_losses.append(train_loss.item())
        
        # Log NaN/Inf statistics
        current_epoch = self.trainer.current_epoch
        print(f"Epoch {current_epoch}: Found {self.epoch_nan_inf_count} samples with NaN/Inf values")
        self.log('nan_inf_count', self.epoch_nan_inf_count, prog_bar=False, sync_dist=True)
        
        # Save problematic indices to a file at regular intervals
        if current_epoch % 5 == 0 or current_epoch == self.trainer.max_epochs - 1:
            self._save_problematic_indices()

    def _save_problematic_indices(self):
        """Save the indices of problematic samples to a file."""
        save_dir = self.trainer.checkpoint_callbacks[0].dirpath  # Get the checkpoint directory
        filename = os.path.join(save_dir, f"nan_inf_indices_epoch_{self.trainer.current_epoch}.txt")
        
        with open(filename, 'w') as f:
            f.write(f"Total NaN/Inf samples: {self.total_nan_inf_count}\n")
            f.write("Global indices of problematic samples:\n")
            for idx in self.nan_inf_indices:
                f.write(f"{idx}\n")
        
        print(f"Saved problematic indices to {filename}")
    
    def on_train_epoch_end(self):
        # Access the trainer to get the logged metrics
        train_loss = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0))
        self.train_losses.append(train_loss.item())
    
    def on_validation_epoch_end(self):
        # Access the trainer to get the logged metrics
        val_loss = self.trainer.callback_metrics.get('val_loss', torch.tensor(0.0))
        self.val_losses.append(val_loss.item())


class SimplifiedOutputCallback(Callback):
    def __init__(self, num_epochs):
        super().__init__()
        self.num_epochs = num_epochs
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item()
        val_loss = trainer.callback_metrics.get('val_loss', torch.tensor(0.0)).item()

        # Print GPU memory usage
        #for i in range(torch.cuda.device_count()):
        #    print(f"GPU {i} memory: {torch.cuda.memory_allocated(i)/1e9:.2f}GB / {torch.cuda.get_device_properties(i).total_memory/1e9:.2f}GB")
        
        # Print the customized output
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{current_time}] Epoch [{epoch+1}/{self.num_epochs}] Training Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}')


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Example usage:
if __name__ == "__main__":
    args = parse_args()
    run_training(args.config)
