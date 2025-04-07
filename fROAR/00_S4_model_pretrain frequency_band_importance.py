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
from einops import rearrange
from sklearn.model_selection import train_test_split
import copy
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import FeatureAblation
from captum.attr import DeepLift
import pickle
import gc
import tracemalloc
#General ROAR pipeline
#Train model on train set and get importance estimate of each sample in train set after training(through selected XAI method)
#Get importance estimate on test set trials as well. Also evaluate performance(Accuracy) on test set.
#Remove top-k pixels from each trial of train set. Retrain model on modified train set.
#Remove top-k pixels from each trial of test-set. Evaluate performance on modified test set

# Initialize lovely_tensors and matplotlib
lt.monkey_patch()
matplotlib.rc_file("matplotlibrc")

# Configuration YAML
CFG_YAML_bandpass = """
wandb:
 key: f0c92a0059bf12e2647f0a1c22fdcd12555fa6df
model:
dataset:
 data_directory: /home/marco/Documents/GitHub/tms_eeg_decoding/data/frequency_bandstop/bandpass
 file_name: subject_{:03d}_bandpass_{}_preprocessed_combined_py.fif
 test_subject_indices: 
 subject_index: 10
 #pretrain_subject_indices: [2,4,7,9,11,13,14,18,22,24]
 pretrain_subject_indices: [2,4,7,9,11,13,14,18,22,24]
 #test_subject_indices: [2,4,7,9,11,13,14,18,22,24]
training:
 lr: 0.0003
 num_epochs: 500
 nll_beta: 0.001
 num_warmup_epochs: 20
 batch_size: 50
 random_seed: 42
 precision: bf16
 kde_lambda: 0.5
exp_name: S4_S4EEGNet_pretrain
"""

CFG_YAML= """
wandb:
 key: f0c92a0059bf12e2647f0a1c22fdcd12555fa6df
model:
dataset:
 data_directory: /home/marco/Documents/GitHub/tms_eeg_decoding/data
 file_name: subject_{:03d}_preprocessed_combined_py.fif
 test_subject_indices: 
 subject_index: 10
 #pretrain_subject_indices: [2,4,7,9,11,13,14,18,22,24]
 pretrain_subject_indices: [2,4,7,9,11,13,14,18,22,24]
 #test_subject_indices: [2,4,7,9,11,13,14,18,22,24]
training:
 lr: 0.0003
 num_epochs: 500
 nll_beta: 0.001
 num_warmup_epochs: 20
 batch_size: 50
 random_seed: 42
 precision: bf16
 kde_lambda: 0.5
exp_name: S4_S4EEGNet_pretrain
"""



def load_config(yaml_name=CFG_YAML):
    cfg = OmegaConf.create(yaml.safe_load(yaml_name))
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

    return total_loss / len(train_loader)

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

# Evaluate the model test set accuracy
def test_set_accuracy(model, data_loader, median, device):
    #Adapt tomorrow to turn raw_label output label into class label with fixed median
    with torch.no_grad():
        correct_total = 0

        for data in data_loader:
            x_batch, y_batch = data["epoch"].to(device), data["label_raw"].to(device)
            true_labels = data["label_binary"].to(device)
            outputs = model(x_batch)
            pred_mean, pred_uncertainty = outputs[:, 0], outputs[:, 1]
    
            #print(f"mean_batch_shape {pred_mean.shape}")
            #y_pred_max = torch.argmax(y_pred, dim=1)
            pred_label_fixed_batch = (pred_mean>median).to(torch.int64)
            pred_label_fixed_batch.to(device)

            #potentially cast boolean array to int array if required by torch

            correct_total += torch.sum(torch.eq(pred_label_fixed_batch, true_labels)).item()
        accuracy = correct_total / len(data_loader.dataset)
        del pred_label_fixed_batch, pred_mean, pred_uncertainty, outputs
        return accuracy

# start with removing 10% of pixels per input to see if it works

def main(reps = 5):
   
    freq_bands = {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 45]}
    np.random_seed = 42
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(yaml_name=CFG_YAML)
    cli_args = parse_args()
    cfg = update_config(cfg, cli_args)
    all_subjects = []
    for subject_index in cfg.dataset.pretrain_subject_indices:
        subject_dict = {k: [] for k in freq_bands.keys()}
        subject_dict["subj_index"] = subject_index
        subject_dict["original"] = []

        print(f"\nSubject index: {subject_index}\n")

        cfg_original = load_config(yaml_name=CFG_YAML)
        cfg_original.dataset.pretrain_subject_indices = [subject_index]
        cli_args = parse_args()
        cfg_original = update_config(cfg_original, cli_args)
        save_config(cfg_original, cfg_original.dataset.subject_index)

        set_mask_train = np.array([])
        pretrain_loader, input_shape_train,_,_,_ = get_pretrain_dataloader(cfg_original, set_mask_train)
        n_trials = len(pretrain_loader.dataset)

        for rep in range(reps):
            #split dataset into train and test set
            set_mask_train = np.random.choice([True,False], p=[0.8,0.2], size=n_trials)
            set_mask_test = ~set_mask_train
            print(f"n_trials: {n_trials}")
        

            pretrain_loader, input_shape_train, epochs_cal, labels_raw_cal, median_cal = get_pretrain_dataloader(cfg_original, set_mask_train)
            test_loader, input_shape_test,_,_,_ = get_pretrain_dataloader(cfg_original, set_mask_test, epochs_cal=epochs_cal, labels_raw_cal=labels_raw_cal, median_cal=median_cal)

            model = build_model(cfg_original, input_shape_train, device)
            optimizer, lr_scheduler = initialize_optimizer_and_scheduler(cfg_original, model, pretrain_loader)

            accelerator = accelerate.Accelerator(mixed_precision=cfg_original.training.precision, log_with="wandb")
            model, optimizer, lr_scheduler, pretrain_loader = accelerator.prepare(
            model, optimizer, lr_scheduler, pretrain_loader, 
            )
    
            save_path = os.path.join("exp", cfg_original.exp_name)
            os.makedirs(save_path, exist_ok=True)
            save_interval = 5  # Save checkpoint every 5 epochs
            with tqdm(range(cfg_original.training.num_epochs)) as pbar:
                for epoch in pbar:
                    train_loss= train_one_epoch(epoch, model, pretrain_loader, optimizer, lr_scheduler, accelerator, device, cfg_original)  
                    pbar.set_description(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")


            model.eval()

        
            accuracies_test = test_set_accuracy(model, test_loader, median_cal, device)
            subject_dict["original"].append(accuracies_test)

            for freq_band, freq_range in freq_bands.items():
                cfg_freqband = load_config(yaml_name=CFG_YAML_bandpass)
                cfg_freqband.dataset.pretrain_subject_indices = [subject_index]
                cli_args = parse_args()
                cfg_freqband = update_config(cfg_freqband, cli_args)

                pretrain_loader, input_shape_train, epochs_cal, labels_raw_cal, median_cal = get_pretrain_dataloader(cfg_freqband, set_mask_train, freq_band=freq_band)
                test_loader, input_shape_test,_,_,_ = get_pretrain_dataloader(cfg_freqband, set_mask_test, epochs_cal=epochs_cal, labels_raw_cal=labels_raw_cal, median_cal=median_cal, freq_band=freq_band)

                model = build_model(cfg_freqband, input_shape_train, device)
                optimizer, lr_scheduler = initialize_optimizer_and_scheduler(cfg_freqband, model, pretrain_loader)

                accelerator = accelerate.Accelerator(mixed_precision=cfg_freqband.training.precision, log_with="wandb")
                model, optimizer, lr_scheduler, pretrain_loader = accelerator.prepare(
                model, optimizer, lr_scheduler, pretrain_loader, 
                )
    
                save_path = os.path.join("exp", "freq_band", cfg_freqband.exp_name)
                os.makedirs(save_path, exist_ok=True)
                save_interval = 5  # Save checkpoint every 5 epochs
                with tqdm(range(cfg_freqband.training.num_epochs)) as pbar:
                    for epoch in pbar:
                        train_loss= train_one_epoch(epoch, model, pretrain_loader, optimizer, lr_scheduler, accelerator, device, cfg_freqband)  
                        pbar.set_description(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")


                model.eval()

        
                accuracies_test = test_set_accuracy(model, test_loader, median_cal, device)
                subject_dict[freq_band].append(accuracies_test)


            with open(f'freq_ROAR_subject_{subject_index}.pickle', 'wb') as handle:
                pickle.dump(subject_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            all_subjects.append(subject_dict)

    with open("freq_ROAR_all_subjects.pickle", "wb") as handle:
        pickle.dump(all_subjects, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
if __name__ == "__main__":
    main()
# %%


