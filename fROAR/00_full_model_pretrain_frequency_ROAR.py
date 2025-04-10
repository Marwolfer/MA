#%%
import argparse
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
import yaml
import matplotlib
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from models.s4net import S4PatchedFinalNet, TrunkNet, HeadNet
from data.metrics import uceloss, sigma_scaling
#from data.data_loader import  get_pretrain_dataloader_transformer as get_pretrain_dataloader
from data.data_loader import  get_pretrain_dataloader
from torchinfo import summary
import accelerate
import lovely_tensors as lt
from scipy.stats import gaussian_kde
import numpy as np


# Initialize lovely_tensors and matplotlib
lt.monkey_patch()
matplotlib.rc_file("matplotlibrc")
# All subjects = [1,2,4,7,9,11,13,14,18,22,24,25,26,27,29,31,33,34,35,39,41,42,43,45,46,47,48,51,52,53,55,56,57,59,60,62,63,65,66,67,69,70,71,72,73,74,79,80,86,88,102]

# Configuration YAML
CFG_YAML = """
wandb:
 key: f0c92a0059bf12e2647f0a1c22fdcd12555fa6df
model:
dataset:
 data_directory: /home/marco/Documents/GitHub/tms_eeg_decoding/data/frequency_bandstop/bandpass
 file_name: subject_{:03d}_bandpass_{}_preprocessed_combined_py.fif
 test_subject_indices: 
 subject_index: 100
 #pretrain_subject_indices: [49, 28]
 pretrain_subject_indices: [4,7,9,11,14,18,22,25,26,31,33,35,39,41,45,48,51,53,55,59,63,65,66,70,71,74,79,86,88,102]
training:
 lr: 0.0003
 num_epochs: 1002
 nll_beta: 0.001
 num_warmup_epochs: 20
 batch_size: 50
 random_seed: 42
 precision: bf16
 kde_lambda: 0.5
exp_name: transformer_S4EEGNet_pretrain
"""

def load_config():
    cfg = OmegaConf.create(yaml.safe_load(CFG_YAML))
    cfg.exp_name = f"{cfg.exp_name}"
    return cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_conf", nargs="*", help="Updates to the configuration in the form of key=value pairs", default=[])
    parser.add_argument("-f", "--fff", help="A dummy argument to handle IPython's default argument", default="1")
    return parser.parse_args()

def update_config(cfg, cli_args):
    for update in cli_args.update_conf:
        key, value = update.split("=")
        try:
            value = eval(value)
        except:
            pass
        OmegaConf.update(cfg, key, value, force_add=True)
    cfg.exp_name = cfg.exp_name + "_" + "_".join(cli_args.update_conf)
    print(OmegaConf.to_yaml(cfg))
    return cfg

def save_config(cfg, subject_index):
    os.makedirs("conf/sweeps", exist_ok=True)
    os.makedirs("exp/tl", exist_ok=True)
    with open(f"conf/sweeps/tl_{cfg.exp_name}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def setup_wandb(cfg, run_name, subject_index, run_id=None):
    os.environ["WANDB_API_KEY"] = cfg.wandb.key
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="icmep_decoding", name=f"{run_name}", config=cfg_dict, mode="online", id=run_id, resume="allow")
    wandb.config.update(cfg_dict)

def build_model(cfg, input_shape_st, device):
    trunk_net = TrunkNet(n_chans=input_shape_st[0], n_times=input_shape_st[1])
    head_net = HeadNet(64, 1)
    model = S4PatchedFinalNet(64, trunk_net, head_net)
    model.to(device)

    print(summary(model, (1, input_shape_st[0], input_shape_st[1]), device=device))  # Changed batch size to 1
    return model


def initialize_optimizer_and_scheduler(cfg, model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    num_batches = len(train_loader)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_batches * cfg.training.num_warmup_epochs,
        num_training_steps=num_batches * cfg.training.num_epochs * 1.5,
    )
    return optimizer, lr_scheduler

def heteroscedastic_gaussian_nll(y_pred, y_true, log_vars):
    vars = torch.exp(log_vars)
    one_over_vars = 1 / (vars + 1e-4)
    nll = (log_vars + torch.clamp(one_over_vars * ((y_true - y_pred) ** 2), min=1e-4, max=1e3)).mean()
    return nll

def compute_kde_weights(y_true, lambda2=0.5):
    y_true_np = y_true.cpu().numpy()
    
    if y_true_np.ndim == 0 or len(y_true_np) == 1:
        # If we have a single data point, return a weight of 1
        return torch.ones_like(y_true)
    
    kde = gaussian_kde(y_true_np)
    densities = kde(y_true_np)
    weights = 1 / (densities ** lambda2)
    
    # Clip weights at 95th percentile
    percentile_95 = np.percentile(weights, 95)
    weights = np.clip(weights, a_min=None, a_max=percentile_95)
    
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, device=y_true.device)

def weighted_mse_loss(y_pred, y_true, kde_weights):
    return torch.mean(kde_weights * (y_pred - y_true) ** 2)

def total_loss_fn(y_pred, y_true, log_vars, beta=0.001, lambda2=0.5):
    kde_weights = compute_kde_weights(y_true, lambda2)
    
    # Weighted MSE loss
    mse_loss = weighted_mse_loss(y_pred, y_true, kde_weights)
    
    # Weighted heteroscedastic loss
    nll = beta * heteroscedastic_gaussian_nll(y_pred, y_true, log_vars)
    weighted_nll = (kde_weights * nll).mean()
    
    heteroscedastic_loss = beta * weighted_nll

    return heteroscedastic_loss + mse_loss


def calculate_regression_metrics(all_preds, all_targets):
    rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
    corr = torch.nn.functional.mse_loss(all_preds, all_targets).item()
    return {"RMSE": rmse, "Correlation": corr}

def calculate_calibration_metrics(y_pred, y_target, log_vars, num_bins=10, outlier=0.0):
    sigma2 = torch.exp(log_vars)
    #sigma2 = torch.clamp(sigma2, MIN_VAR, MAX_VAR)
    sq_error = (y_pred - y_target) ** 2
    sigma2_cal = sigma_scaling(sq_error, sigma2)
    uce, *_ = uceloss(sigma2_cal, sq_error, n_bins=num_bins, outlier=outlier)
    return {"sigma2": sigma2, "sigma2_cal": sigma2_cal, "sq_error": sq_error, "UCE": uce}

def train_one_epoch(epoch, model, train_loader, optimizer, lr_scheduler, accelerator, device, cfg):
    model.train()
    total_loss = 0
    all_preds = []
    all_logvars = []
    all_targets_smooth = []
    for data in train_loader:
        optimizer.zero_grad()
        outputs = model(data['epoch'].to(device))
        outputs_mean = outputs[:, 0] 
        log_vars = outputs[:, 1]
        loss = total_loss_fn(outputs_mean, data['label_raw'], log_vars, beta=cfg.training.nll_beta, lambda2=cfg.training.kde_lambda)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        all_preds.append(outputs_mean)
        all_logvars.append(log_vars)
        all_targets_smooth.append(data['label_raw'])
    all_preds = torch.cat(all_preds)
    all_logvars = torch.cat(all_logvars)
    all_targets_smooth = torch.cat(all_targets_smooth)
    metrics = calculate_regression_metrics(all_preds, all_targets_smooth)
    calibration_metrics = calculate_calibration_metrics(all_preds, all_targets_smooth, all_logvars)
    if (epoch + 1) % 20 == 0:
        plot_training_metrics(epoch, all_preds, all_targets_smooth, all_logvars, metrics, calibration_metrics)
    return total_loss / len(train_loader), metrics, calibration_metrics, None

def plot_training_metrics(epoch, all_preds, all_targets_smooth, all_logvars, metrics, calibration_metrics):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(all_targets_smooth.cpu().detach().numpy(), all_preds.cpu().detach().numpy())
    axs[0].set_xlabel('True')
    axs[0].set_ylabel('Predicted')
    axs[0].set_title(f'True vs Pred, RMSE: {metrics["RMSE"]:.3f}, Correlation: {metrics["Correlation"]:.3f}')
    axs[1].scatter(calibration_metrics["sigma2_cal"].detach().cpu(), calibration_metrics["sq_error"].detach().cpu())
    axs[1].set_xlabel('Sigma2_cal')
    axs[1].set_ylabel('MSE')
    axs[1].set_title(f'Calibration Metrics, UCE: {calibration_metrics["UCE"].detach().cpu().item():.3f}')
    fig.suptitle(f'Training Epoch {epoch}')
    wandb.log({f"metrics_train": wandb.Image(fig)})
    #plt.show()



def main(reps = 5):
    freq_bands = {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 45]}
    np.random_seed = 42
    torch.manual_seed(42)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    cli_args = parse_args()
    cfg = update_config(cfg, cli_args)
    save_config(cfg, cfg.dataset.subject_index)

    for freq_band, freq_range in freq_bands.items():
        #test = get_pretrain_dataloader(cfg, freq_band=freq_band)
        pretrain_loader, input_shape_st,_,_,_ = get_pretrain_dataloader(cfg, freq_band=freq_band)

        # Setup wandb for pretraining
        setup_wandb(cfg, f"{cfg.exp_name}_pretrain", cfg.dataset.subject_index)

        model = build_model(cfg, input_shape_st, device)

        optimizer, lr_scheduler = initialize_optimizer_and_scheduler(cfg, model, pretrain_loader)
    
        accelerator = accelerate.Accelerator(mixed_precision=cfg.training.precision, log_with="wandb")
        model, optimizer, lr_scheduler, pretrain_loader = accelerator.prepare(
            model, optimizer, lr_scheduler, pretrain_loader, 
        )
    
        save_path = os.path.join("exp", cfg.exp_name)
        os.makedirs(save_path, exist_ok=True)
        save_interval = 5  # Save checkpoint every 5 epochs
        with tqdm(range(cfg.training.num_epochs)) as pbar:
            for epoch in pbar:
                #print("Epoch: ", epoch)
                train_loss, train_metrics, train_calibration, train_classification = train_one_epoch(epoch, model, pretrain_loader, optimizer, lr_scheduler, accelerator, device, cfg)          
                current_lr = optimizer.param_groups[0]['lr']
            
                pbar.set_description(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train RMSE: {train_metrics['RMSE']:.4f}")
            
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_rmse": train_metrics["RMSE"],
                    "train_corr": train_metrics["Correlation"],
                    "train_uce": train_calibration["UCE"],
                    "learning_rate": current_lr,
                    "freq_band": freq_band
                })

                if epoch % save_interval == 0:
                    checkpoint_path = os.path.join(save_path, f"checkpoint_full_epoch_{epoch}_freq_band_{freq_band}.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'train_loss': train_loss,
                        'train_rmse': train_metrics["RMSE"],
                        'train_corr': train_metrics["Correlation"],
                        'train_uce': train_calibration["UCE"],
                    }, checkpoint_path)


                    # Save TrunkNet checkpoint
                    trunk_checkpoint_path = os.path.join(save_path, f"checkpoint_trunk_epoch_{epoch}_freq_band_{freq_band}.pt")
                    torch.save({
                        'epoch': epoch,
                        'trunk_state_dict': model.trunk_net.state_dict(),
                    }, trunk_checkpoint_path)

                    # Save HeadNet checkpoint
                    head_checkpoint_path = os.path.join(save_path, f"checkpoint_head_epoch_{epoch}_freq_band_{freq_band}.pt")
                    torch.save({
                        'epoch': epoch,
                        'head_state_dict': model.head_net.state_dict(),
                    }, head_checkpoint_path)
        
        wandb.finish()

#%%
if __name__ == "__main__":
    main()
# %%
