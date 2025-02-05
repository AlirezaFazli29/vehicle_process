import torch
import torch.nn as nn
from model_handler import (
    BaseModel,
    UNetType
)
import segmentation_models_pytorch as smp
import time
from utils import train_custom_unet


class Plate_Unet(BaseModel):

    def __init__(
            self,
            model_type: UNetType = UNetType.Base, 
            encoder_name: str = "resnet34", 
            encoder_weights: str = "imagenet", 
            in_channels: int = 3,
            out_channels:int = 1,
            device:str = "cpu",
            verbose: bool = False
        ):
        """
        Initialize the Plate_Unet model.

        Parameters:
        encoder_name (str): Name of the encoder model (default is 'resnet34').
        encoder_weights (str): Weights to use for the encoder (default is 'imagenet').
        in_channels (int): Number of input channels (default is 3 for RGB images).
        out_channels (int): Number of output channels (default is 1 for binary segmentation).
        device (str): Device to run the model on (default is 'cpu').
        """
        self.model_type = model_type
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.verbose = verbose
        self.model = self.load_model()
        self.model_training_history = {
            'train_loss': [],
            'valid_loss': [],
            'train_iou': [],
            'valid_iou': [],
        }

    def load_model(self):
        """
        Load the U-Net model with specified parameters and change the last layer to Sigmoid.

        Returns:
        torch.nn.Module: The initialized U-Net model.
        """
        unet = smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,            
            classes=self.out_channels
        ).to(self.device)
        unet.segmentation_head[2] = nn.Sigmoid()
        if self.verbose: print(f"Selected model is {self.model_type}")
        if self.model_type != UNetType.Base:
            unet.load_state_dict(torch.load(self.model_type.value, 
                                            map_location=torch.device(self.device), 
                                            weights_only=True))
        if self.verbose: 
            print(f"Unet is successfully loaded on {self.device}")
        return unet
        
    def __call__(self, img_batch):
        """
        Run inference on a batch of images.

        Parameters:
        img_batch (torch.Tensor): Batch of input images.

        Returns:
        torch.Tensor: Model predictions for the input images.
        """
        if self.verbose:
            print("Running inference on Image Batch")
        if self.model.training:
            self.model.eval()
        img_batch = img_batch.to(self.device)
        with torch.no_grad():
            results = self.model(img_batch)
        if self.verbose:
            print("Outputs Generated")
        return results
    
    def freeze_unfreeze(self, freeze:bool=False):
        if self.verbose:
            print(f"Start {'freezing' if freeze==True else 'unfreezing'} encoder layers ...")
        for param in self.model.encoder.parameters():
            param.requires_grad = not(freeze)
        if self.verbose:
            print(f"Encoder layers {'freezed' if freeze==True else 'unfreezed'}")

    def fit(
        self,
        train_loader,
        valid_loader,
        optimizer=None,
        lr:float=0.01,
        weight_decay=0.001,
        epochs:int=10,
        loss_fn=torch.nn.BCELoss(),
        save_parameters=True,
        save_path:str="./weights"
    ):
        """
        Train the UNet model using the provided training and validation loaders.

        Parameters:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer, optional): Optimizer for model parameters. If None, a default optimizer is used.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
        weight_decay (float, optional): Weight decay (L2 regularization) parameter for the optimizer. Defaults to 0.001.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        loss_fn (torch.nn.Module, optional): Loss function used for training. Defaults to BCELoss.
        save_parameters (bool, optional): Flag to save model parameters. Defaults to True.
        save_path (str, optional): Path to save model weights. Defaults to './weights'.

        Returns:
        None
        """
        print(f"Starting training process for model {self.model_type} ...")
        print(f"Hyper Parameters:")
        print(f"- Epochs Number = {epochs}")
        print(f"- Learning Rate = {lr}")
        print(f"- Weight Decay = {weight_decay}")
        print(f"- Loss Function = {loss_fn}")
        print(f"- Optimizer = {optimizer if optimizer else torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)}\n")
        start_time = time.time()
        print(f"Start training on {self.device}: \n")
        training_history = train_custom_unet(
            model=self.model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            lr=lr,
            epochs=epochs,
            loss_fn=loss_fn,
            device=self.device,
            save_parameters=save_parameters,
            save_path=save_path
        )
        print("\nTraining completed")
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        print(f"Training time: {minutes} minutes and {seconds:.2f} seconds")
        self.model_training_history["train_loss"].extend(training_history["train_loss"])
        self.model_training_history["valid_loss"].extend(training_history["valid_loss"])
        self.model_training_history["train_iou"].extend(training_history["train_iou"])
        self.model_training_history["valid_iou"].extend(training_history["valid_iou"])

    def __repr__(self):
        return repr(self.model)
