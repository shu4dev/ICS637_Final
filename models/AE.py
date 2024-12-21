#Credit: strucutre of this code is taken from https://www.kaggle.com/code/voix97/jane-street-rmf-training-nn, but the model details is constructed by me.
from config import *
def r2_val(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2

class AE(LightningModule):
    def __init__(
        self,
        num_columns: int,
        num_labels: int,
        hidden_units: list,
        dropout_rates: list,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        noise_std: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()

        # Define a simple Gaussian noise layer inside this class
        class GaussianNoise(nn.Module):
            """Applies Gaussian noise during training."""
            def __init__(self, std=0.1):
                super().__init__()
                self.std = std
            
            def forward(self, x):
                if self.training:
                    return x + torch.randn_like(x) * self.std
                return x

        # -------------------
        # 1) ENCODER
        # -------------------
        self.encoder = nn.Sequential(
            GaussianNoise(std=noise_std),
            nn.Linear(num_columns, hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0]),
            nn.SiLU()
        )

        # -------------------
        # 2) DECODER
        # -------------------
        # Reconstruct back to num_columns
        self.decoder = nn.Sequential(
            nn.Dropout(dropout_rates[1]),
            nn.Linear(hidden_units[0], num_columns),
        )

        # -------------------
        # 3) AE Branch
        # -------------------
        self.ae_out = nn.Sequential(
            nn.Linear(num_columns, hidden_units[1]),
            nn.BatchNorm1d(hidden_units[1]),
            nn.SiLU(),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(hidden_units[1], num_labels),
        )

        # -------------------
        # 4) MAIN Branch
        # -------------------
        # We'll attach the original input + encoder output
        # hidden_units = [enc_dim, ae_dim, main_dim_1, main_dim_2, ..., main_dim_n]
        self.main_out = nn.Sequential(
            nn.BatchNorm1d(num_columns + hidden_units[0]),
            nn.Dropout(dropout_rates[3]),

            nn.Linear(num_columns + hidden_units[0], hidden_units[2]),
            nn.BatchNorm1d(hidden_units[2]),
            nn.SiLU(),
            nn.Dropout(dropout_rates[4]),

            nn.Linear(hidden_units[2], hidden_units[3]),
            nn.BatchNorm1d(hidden_units[3]),
            nn.SiLU(),
            nn.Dropout(dropout_rates[5]),

            nn.Linear(hidden_units[3], hidden_units[4]),
            nn.BatchNorm1d(hidden_units[4]),
            nn.SiLU(),
            nn.Dropout(dropout_rates[6]),

            nn.Linear(hidden_units[4], hidden_units[5]),
            nn.BatchNorm1d(hidden_units[5]),
            nn.SiLU(),
            nn.Dropout(dropout_rates[7]),

            nn.Linear(hidden_units[-1], num_labels),
        )

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay

        # For accumulating validation losses each epoch
        self.validation_step_outputs = []

    def forward(self, x):
        """
        Forward pass that returns:
          - decoder_out: reconstructed input
          - ae_out: output of the autoencoder branch
          - main_out: output of the main branch
        """
        encoder_out = self.encoder(x)              # [batch_size, enc_dim]
        decoder_out = self.decoder(encoder_out)    # [batch_size, num_columns]
        ae_out = self.ae_out(decoder_out)          # [batch_size, num_labels]
        main_out = self.main_out(torch.cat([x, encoder_out], dim=1))  # [batch_size, num_labels]
        
        return decoder_out, ae_out.squeeze(), main_out.squeeze()

    def training_step(self, batch):
        """
        Training step. Expects batch=(x, y). 
        If your dataset returns only x, remove the y part and adapt accordingly.
        """
        x, y, w = batch
        decoder_out, ae_out, main_out = self(x)
   
        ae_loss = F.mse_loss(ae_out, y, reduction='none') * w
        main_loss = F.mse_loss(main_out, y, reduction='none') * w
        decoder_loss = F.mse_loss(decoder_out, x, reduction='none')
     
        ae_loss = ae_loss.mean()
        main_loss = main_loss.mean()
        decoder_loss = decoder_loss.mean()

        main_loss = main_loss + ae_loss + decoder_loss
        # Logging
        self.log('train_total_loss', main_loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return main_loss
    

    def validation_step(self, batch):
        """
        Validation step. Similar to training_step but doesn't do backprop.
        """
        x, y, w = batch
        decoder_out, ae_out, main_out = self(x)

        ae_loss = F.mse_loss(ae_out, y,reduction='none') * w
        main_loss = F.mse_loss(main_out, y,reduction='none') * w
        decoder_loss = F.mse_loss(decoder_out, x,reduction='none')
 
        
        ae_loss = ae_loss.mean()
        main_loss = main_loss.mean()
        decoder_loss = decoder_loss.mean()

        main_loss = main_loss + ae_loss + decoder_loss
        # Log validation metrics
        self.log('val_loss', main_loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.validation_step_outputs.append((main_out, y, w))
        return main_loss
    
    def on_validation_epoch_end(self):
        """Calculate validation WRMSE at the end of the epoch."""
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
        if self.trainer.sanity_checking:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        else:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
            # r2_val
            val_r_square = r2_val(y, prob, weights)
            self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")