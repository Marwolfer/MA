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
from models.s4net import S4PatchedFinalNet,TrunkNet, HeadNet
from data.metrics import uceloss, sigma_scaling
from data.data_loader import load_eeg_data, get_sliding_window_data, create_dataloader
from torchinfo import summary
import accelerate
import lovely_tensors as lt
import numpy as np
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from ema_pytorch import EMA
import time 
from sklearn.preprocessing import MinMaxScaler
import pickle

import timeit   

import seaborn
from sklearn.preprocessing import MinMaxScaler
import pickle

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import DeepLift
from captum.attr import visualization as viz
from data.data_loader import EEGCustomDataset_nt
from imputations import NoisyLinearImputer



# Initialize lovely_tensors and matplotlib
lt.monkey_patch()
matplotlib.rc_file("matplotlibrc")

# Configuration YAML
CFG_YAML = """
wandb:
 key: f0c92a0059bf12e2647f0a1c22fdcd12555fa6df
model:
dataset:
 data_directory: /home/marco/Documents/GitHub/tms_eeg_decoding/exp
 #file_name: subject_{:03d}_preprocessed_combined_py.fif
 file_name: subject_{:03d}_preprocessed_combined_py.fif
 exclude_timepoints: 100
 subject_index: 1
 #test_subject_indices: [2]
 test_subject_indices: [47,48,52,55,56,57,60,62,67,69,72,73,79,80,86,88,92,102]
training:
 training_start_len: 100
 pretrain_epochs: 100
 pretrain_lr: 0.0001
 val_window_len: 1
 epochs_per_window: 10
 num_warmup_epochs: 5
 num_epochs: 800
 slide_step: 1
 num_warmup_epochs_per_window: 0
 lr: 0.005 #maybe change back to 0.0001
 nll_beta: 0.001
 num_warmup_epochs: 0
 batch_size: 50 #better to use 50
 random_seed: 42
 precision: bf16
 kde_lambda: 0.5
 finetune_entire_model: true # Set to true to finetune the entire model, false for transformer only
exp_name: S4_S4EEGNet_ema
"""

def load_config():
    cfg = OmegaConf.create(yaml.safe_load(CFG_YAML))
    cfg.exp_name = f"{cfg.exp_name}_subject_{cfg.dataset.subject_index}"
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

def save_config(cfg):
    os.makedirs("conf/sweeps", exist_ok=True)
    os.makedirs("exp/withinsubs", exist_ok=True)
    with open(f"conf/sweeps/withinsubs_{cfg.exp_name}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

def setup_wandb(cfg):
    os.environ["WANDB_API_KEY"] = cfg.wandb.key
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="icmep_decoding", name=cfg.exp_name, config=cfg_dict, mode="online")

def build_model_for_finetuning(cfg, input_shape_st, device, save_path, epoch, explanation_func_name=None, k=None):
    # Initialize TrunkNet and HeadNet
    trunk_net = TrunkNet(n_chans=input_shape_st[0], n_times=input_shape_st[1])
    head_net = HeadNet(64, 1)  # Assuming these are the correct dimensions
    
    # Create new PatchedFinalNet
    model = S4PatchedFinalNet(64, trunk_net, head_net)


    if cfg.training.finetune_entire_model:
        # Load the full model checkpoint
        if explanation_func_name is None:
            full_checkpoint_path = os.path.join(save_path, f"checkpoint_full_epoch_{epoch}.pt")
            full_checkpoint = torch.load(full_checkpoint_path)
            model.load_state_dict(full_checkpoint['model_state_dict'])

            for param in model.parameters():
                param.requires_grad = True
        
        else:
            full_checkpoint_path = os.path.join(save_path, f"ROAD_checkpoint_full_epoch_1000_explanation_func_{explanation_func_name}_k_{k}.pt")
            full_checkpoint = torch.load(full_checkpoint_path)
            model.load_state_dict(full_checkpoint['model_state_dict'])
        
            # Make sure all parameters are trainable
            for param in model.parameters():
                param.requires_grad = True

    else:
        # Load the separate checkpoints for TrunkNet and HeadNet
        trunk_checkpoint_path = os.path.join(save_path, f"checkpoint_trunk_epoch_{epoch}.pt")
        head_checkpoint_path = os.path.join(save_path, f"checkpoint_head_epoch_{epoch}.pt")
        
        trunk_checkpoint = torch.load(trunk_checkpoint_path)
        head_checkpoint = torch.load(head_checkpoint_path)
        
        # Load the pretrained weights
        model.trunk_net.load_state_dict(trunk_checkpoint['trunk_state_dict'])
        model.head_net.load_state_dict(head_checkpoint['head_state_dict'])
        
        # Freeze TrunkNet and HeadNet
        for param in model.trunk_net.parameters():
            param.requires_grad = False
        for param in model.head_net.parameters():
            param.requires_grad = False
        
        # Ensure aggnet (transformer) is trainable
        for param in model.aggnet.parameters():
            param.requires_grad = True

    model.to(device)
    
    print(summary(model, (cfg.training.batch_size, input_shape_st[0], input_shape_st[1]), device=device))
    return model

def build_model(cfg, input_shape_st, device):
    trunk_net = TrunkNet(n_chans=input_shape_st[0], n_times= input_shape_st[1])
    head_net = HeadNet(64, 1)
    model = S4PatchedFinalNet(64, trunk_net, head_net)
    model.to(device)

    print(summary(model, (cfg.training.batch_size, input_shape_st[0], input_shape_st[1]), device=device))
    return model


def initialize_optimizer(cfg, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    return optimizer


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

def calculate_regression_metrics(all_preds, all_targets, labels_scaler, mean_mep):
    # Calculate RMSE in standardized space
    rmse_standardized = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()

    # Detach tensors and convert to NumPy arrays
    all_preds_np = all_preds.detach().cpu().numpy().reshape(-1, 1)
    all_targets_np = all_targets.detach().cpu().numpy().reshape(-1, 1)
    
    # Transform predictions and targets back to original space
    all_preds_original = labels_scaler.inverse_transform(all_preds_np).flatten()
    all_targets_original = labels_scaler.inverse_transform(all_targets_np).flatten()
    
    # Calculate RMSE in original space
    rmse_original = np.sqrt(np.mean((all_preds_original - all_targets_original) ** 2))
    
    # Calculate relative RMSE as percentage
    relative_rmse = (rmse_original / mean_mep) * 100
    
    return {
        "RMSE_standardized": rmse_standardized,
        "RMSE_original": rmse_original,
        "Relative_RMSE_percentage": relative_rmse
    }

def calculate_calibration_metrics(y_pred, y_target, log_vars, num_bins=10, outlier=0.0):
    sigma2 = torch.exp(log_vars)
    sq_error = (y_pred - y_target) ** 2
    sigma2_cal = sigma_scaling(sq_error, sigma2)
    uce, *_ = uceloss(sigma2_cal, sq_error, n_bins=num_bins, outlier=outlier)
    return {"sigma2": sigma2, "sigma2_cal": sigma2_cal, "sq_error": sq_error, "UCE": uce}


class ContinuousLearner:
    def __init__(self, window_size=50, fixed_threshold=None, labels_scaler=None, mean_mep=None):
        self.window_size = window_size
        self.fixed_threshold = fixed_threshold
        self.labels_scaler = labels_scaler
        self.mean_mep = mean_mep
        self.true_values = []
        self.pred_values = []
        self.log_vars = []
        self.binary_correct_rolling = []
        self.binary_correct_fixed = []
        self.three_class_correct = []

    def update(self, true_value, pred_value, log_var, binary_correct_rolling, binary_correct_fixed, three_class_correct):
        self.true_values.append(true_value)
        self.pred_values.append(pred_value)
        self.log_vars.append(log_var)
        self.binary_correct_rolling.append(binary_correct_rolling)
        self.binary_correct_fixed.append(binary_correct_fixed)
        self.three_class_correct.append(three_class_correct)

        if len(self.true_values) > self.window_size:
            self.true_values.pop(0)
            self.pred_values.pop(0)
            self.log_vars.pop(0)
            self.binary_correct_rolling.pop(0)
            self.binary_correct_fixed.pop(0)
            self.three_class_correct.pop(0)

    def get_metrics(self):
        if not self.true_values:
            return {
                "RMSE_standardized": 0, "RMSE_original": 0, "RMSE_relative": 0,
                "BinaryAccuracy_Rolling": 0, "BinaryAccuracy_Fixed": 0, "ThreeClassAccuracy": 0,
                "UCE": 0
            }

        true_array = np.array(self.true_values)
        pred_array = np.array(self.pred_values)
        log_vars_array = np.array(self.log_vars)

        # Standardized RMSE
        mse = np.mean((true_array - pred_array) ** 2)
        rmse_standardized = np.sqrt(mse)

        # Original RMSE
        if self.labels_scaler is not None:
            true_array_original = self.labels_scaler.inverse_transform(true_array.reshape(-1, 1)).flatten()
            pred_array_original = self.labels_scaler.inverse_transform(pred_array.reshape(-1, 1)).flatten()
            rmse_original = np.sqrt(np.mean((true_array_original - pred_array_original) ** 2))
        else:
            rmse_original = rmse_standardized

        # Relative RMSE
        if self.mean_mep is not None:
            rmse_relative = (rmse_original / self.mean_mep) * 100
        else:
            rmse_relative = 0

        binary_acc_rolling = np.mean(self.binary_correct_rolling)
        binary_acc_fixed = np.mean(self.binary_correct_fixed)
        three_class_acc = np.mean(self.three_class_correct)

        # Calculate UCE
        y_pred = torch.tensor(pred_array)
        y_target = torch.tensor(true_array)
        log_vars = torch.tensor(log_vars_array)
        cal_metrics = calculate_calibration_metrics(y_pred, y_target, log_vars)
        uce = cal_metrics['UCE']

        return {
            "RMSE_standardized": rmse_standardized,
            "RMSE_original": rmse_original,
            "RMSE_relative": rmse_relative,
            "BinaryAccuracy_Rolling": binary_acc_rolling,
            "BinaryAccuracy_Fixed": binary_acc_fixed,
            "ThreeClassAccuracy": three_class_acc,
            "UCE": uce.item()
        }

    @property
    def window(self):
        return list(zip(self.true_values, self.pred_values))
    

def train_one_epoch_adaptive(epoch, model, ema_model, train_loader, optimizer, lr_scheduler, accelerator, cfg, device,labels_scaler, mean_mep, window_idx=0,is_pretraining=False):
    model.train()
    total_loss = 0
    all_preds = []
    all_logvars = []
    all_targets_raw = []
    fixed_median = None # Initialize median label 

    for data in train_loader:
        optimizer.zero_grad()
        data = {k: v.to(device) for k, v in data.items()}
        outputs = model(data['epoch'])
        outputs_mean = outputs[:, 0] 
        log_vars = outputs[:, 1]
        if fixed_median is None:
            fixed_median = data['fixed_median'][0].item()
        
        loss = total_loss_fn(outputs_mean, data['label_raw'], log_vars, beta=cfg.training.nll_beta, lambda2=cfg.training.kde_lambda)
        

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += loss.item()

        # Update EMA model
        ema_model.update()
        
        if not is_pretraining:
            all_preds.append(outputs_mean)
            all_logvars.append(log_vars)
            all_targets_raw.append(data['label_raw'])


    avg_loss = total_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']

    metrics = {}
    calibration_metrics = {}
    
    if not is_pretraining:
        all_preds = torch.cat(all_preds)
        all_logvars = torch.cat(all_logvars)
        all_targets_raw = torch.cat(all_targets_raw)
        metrics = calculate_regression_metrics(all_preds, all_targets_raw, labels_scaler, mean_mep)
        calibration_metrics = calculate_calibration_metrics(all_preds, all_targets_raw, all_logvars)


    return avg_loss, metrics, calibration_metrics, current_lr


def predict_and_record(model, ema_model, data, device, fixed_median, fixed_q1, fixed_q3, continuous_learner, cfg, start_index, ch_names):
    model.eval()
    ema_model.eval()




    with torch.no_grad():
        data = {k: v.to(device) for k, v in data.items()}
        
        # Regular model prediction
        outputs_regular = model(data['epoch'])
        outputs_mean_regular = outputs_regular[:, 0]
        log_vars_regular = outputs_regular[:, 1]


        #fig = plt.figure()
        #print(data["epoch"].cpu().squeeze().numpy().shape)
        #plt.imshow(data["epoch"].cpu().squeeze().numpy())
        #plt.savefig(str(start_index)+".png")
        
        # EMA model prediction
        outputs_ema = ema_model(data['epoch'])
        outputs_mean_ema = outputs_ema[:, 0]
        log_vars_ema = outputs_ema[:, 1]
        
        true_label = data['label_raw']
        
        # Calculate loss for both models
        loss_regular = total_loss_fn(outputs_mean_regular, true_label, log_vars_regular, 
                                     beta=cfg.training.nll_beta, lambda2=cfg.training.kde_lambda)
        loss_ema = total_loss_fn(outputs_mean_ema, true_label, log_vars_ema, 
                                 beta=cfg.training.nll_beta, lambda2=cfg.training.kde_lambda)

        pred_label = outputs_mean_regular.item()
        log_var = log_vars_regular.item()
        true_label = true_label.item()

        """
        naive_baseline = data["epoch"]*0
        integrated_gradients = IntegratedGradients(model)
        attributions_ig = integrated_gradients.attribute(data['epoch'], target=0, n_steps=200)
        agg_dict["ig_agg"][start_index-100] = attributions_ig
        #hm = seaborn.heatmap(attributions_ig.squeeze().cpu().numpy(), yticklabels=ch_names)
        #hm.set_yticklabels(hm.get_yticklabels(), fontsize=3)
        #figure = hm.get_figure()
        #figure.savefig("IntegratedGradients"+str(start_index)+".png")
        #figure.clear()

        # this likely will not work too well, considering some activatoins are non-linear
        dl = DeepLift(model)
        dl_attr = dl.attribute(data['epoch'], target=0)
        agg_dict["dl_agg"][start_index-100] = dl_attr
        #hm_dl = seaborn.heatmap(dl_attr.squeeze().cpu().numpy(), yticklabels=ch_names)
        #hm_dl.set_yticklabels(hm_dl.get_yticklabels(), fontsize=3)
        #figure = hm_dl.get_figure()
        #figure.savefig("DeepLiftShap"+str(start_index)+".png")
        #figure.clear()


        gs = GradientShap(model)
        gs_attr = gs.attribute(data['epoch'], target=0, n_samples=50, baselines=naive_baseline)
        agg_dict["gs_agg"][start_index-100] = gs_attr
        #hm_gs = seaborn.heatmap(gs_attr.squeeze().cpu().numpy(), yticklabels=ch_names)
        #hm_gs.set_yticklabels(hm_gs.get_yticklabels(), fontsize=3)
        #figure = hm_gs.get_figure()
        #figure.savefig("GradientShap"+str(start_index)+".png")
        #figure.clear()

        # how important is each single sample in the epoch for the prediction
        oc_temporal = Occlusion(model)
        oc_temporal_attr = oc_temporal.attribute(data["epoch"],
                                       target=0,
                                       sliding_window_shapes=(60, 1),
                                       baselines=0)
        agg_dict["oc_temp_agg"][start_index-100] = oc_temporal_attr
        #hm_oc_temporal = seaborn.heatmap(oc_temporal_attr.squeeze().cpu().numpy(), yticklabels=ch_names)
        #hm_oc_temporal.set_yticklabels(hm_oc_temporal.get_yticklabels(), fontsize=3)
        #figure = hm_oc_temporal.get_figure()
        #figure.savefig("Occlusion_temporal"+str(start_index)+".png")
        #figure.clear()

        oc_channel = Occlusion(model)
        oc_channel_attr = oc_channel.attribute(data["epoch"],
                                       target=0,
                                       sliding_window_shapes=(1, 900),
                                       baselines=0)
        agg_dict["oc_ch_agg"][start_index-100] = oc_channel_attr
        #hm_oc_channel = seaborn.heatmap(oc_channel_attr.squeeze().cpu().numpy(), yticklabels=ch_names)
        #hm_oc_channel.set_yticklabels(hm_oc_channel.get_yticklabels(), fontsize=3)
        #figure = hm_oc_channel.get_figure()
        #figure.savefig("Occlusion_channel"+str(start_index)+".png")
        #figure.clear()
        
    
        fa_temporal = FeatureAblation(model)
        fa_temporal_attr = fa_temporal.attribute(data["epoch"],
                                       target=0,
                                       sliding_window_shapes=(60, 1),
                                       baselines=0)
        hm_fa_temporal = seaborn.heatmap(fa_temporal_attr.squeeze().cpu().numpy(), yticklabels=ch_names)
        hm_fa_temporal.set_yticklabels(hm_fa_temporal.get_yticklabels(), fontsize=3)
        figure = hm_fa_temporal.get_figure()
        figure.savefig("FeatureAblation_temporal"+str(start_index)+".png")
        figure.clear()

        
        fa_channel = FeatureAblation(model)
        fa_channel_attr = fa_channel.attribute(data["epoch"],
                                       target=0,
                                       sliding_window_shapes=(1, 900),
                                       baselines=0)
        hm_fa_channel = seaborn.heatmap(fa_channel_attr.squeeze().cpu().numpy(), yticklabels=ch_names)
        hm_fa_channel.set_yticklabels(hm_fa_channel.get_yticklabels(), fontsize=3)
        figure = hm_fa_channel.get_figure()
        figure.savefig("FeatureAblation_channel"+str(start_index)+".png")
        figure.clear()
        """



        # for now run with defaul setting. Ultimately think about whether attribution to layer input or attribution to layer output is more reasonable

        #define rules for custom layers
        #layer_LRP = LayerLRP(model, model.trunk_net.spatial_filter.layers.conv_temporal)
        #result_layer_LRP = layer_LRP.attribute(data["epoch"], target=0)




        #layer_DeepLift = LayerDeepLift(model, model.trunk_net.spatial_filter.layers.conv_temporal)
        #result_layer_DeepLift = layer_DeepLift.attribute(data["epoch"], target=0)
        #agg_dict["dl_layer_agg"][start_index-100] = result_layer_DeepLift

        #layer_IG = LayerIntegratedGradients(model, model.trunk_net.spatial_filter.layers.conv_temporal)
        #result_layer_IG = layer_IG.attribute(data["epoch"], target=0)
        #agg_dict["ig_layer_agg"][start_index-100] = result_layer_IG








        # Binary classification using fixed median
        true_binary_fixed = int(true_label > fixed_median)
        pred_binary_fixed = int(pred_label > fixed_median)
        binary_correct_fixed = int(true_binary_fixed == pred_binary_fixed)


        # Binary classification using rolling median
        rolling_median = np.median(continuous_learner.true_values[-50:] + [true_label])
        true_binary_rolling = int(true_label > rolling_median)
        pred_binary_rolling = int(pred_label > rolling_median)
        binary_correct_rolling = int(true_binary_rolling == pred_binary_rolling)

        # 3-class classification using fixed Q1 and Q3
        bins = [fixed_q1, fixed_q3]
        true_3class = np.digitize([true_label], bins)[0]
        pred_3class = np.digitize([pred_label], bins)[0]
        three_class_correct = int(true_3class == pred_3class)

        
        # Update continuous learner
        continuous_learner.update(true_label, pred_label, log_var, binary_correct_rolling, binary_correct_fixed, three_class_correct)
        
        result = {
            'true_label': true_label,
            'pred_label': pred_label,
            'binary_correct_rolling': binary_correct_rolling,
            'binary_correct_fixed': binary_correct_fixed,
            'three_class_correct': three_class_correct,
            'log_var': log_var,
            'fixed_median': fixed_median,
            'rolling_median': rolling_median,
            'fixed_q1': fixed_q1,
            'fixed_q3': fixed_q3,
            'loss_regular': loss_regular.item(),
            'loss_ema': loss_ema.item(),
        }
        
        return result

# Pretrain on the first 100 trials
def pretrain_model(model, ema_model, all_epochs, all_labels_raw, fixed_median, fixed_q1, fixed_q3, cfg, device, labels_scaler, mean_mep, accelerator):
    pretrain_epochs, pretrain_labels_raw = get_sliding_window_data(all_epochs, all_labels_raw, 0, 100, mode='pretrain')
    pretrain_labels = (pretrain_labels_raw >= fixed_median).astype(int)
    pretrain_loader = create_dataloader(pretrain_epochs, pretrain_labels, pretrain_labels_raw, fixed_median, fixed_q1, fixed_q3, cfg.training.batch_size, mode='train')
    
    if len(pretrain_loader) > 0:
        model, ema_model = pretrain_or_finetune_model(model, ema_model, pretrain_loader, cfg, device, labels_scaler, mean_mep, accelerator, window_idx=-1, is_pretraining=True)
    
    return model, ema_model


def pretrain_or_finetune_model(model, ema_model, train_loader, cfg, device, labels_scaler, mean_mep, accelerator, window_idx, is_pretraining=True, explanation_func_name=None, k=None):
    if not cfg.training.finetune_entire_model:
        for param in model.trunk_net.parameters():
            param.requires_grad = False

    
        for param in model.head_net.parameters():
            param.requires_grad = False
        for param in model.aggnet.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=cfg.training.pretrain_lr if is_pretraining else cfg.training.lr)
    optimizer = accelerator.prepare(optimizer)
    
    num_epochs = cfg.training.pretrain_epochs 
    
    for epoch in range(num_epochs):
        loss, metrics, calibration, current_lr = train_one_epoch_adaptive(
            epoch, model, ema_model, train_loader, optimizer, None, accelerator, cfg, device, labels_scaler, mean_mep,
            window_idx=-1 if is_pretraining else window_idx, is_pretraining=is_pretraining)
        
        prefix = 'pretrain' if is_pretraining else 'finetune'
        log_dict = {
            f'{prefix}_epoch': epoch,
            f'{prefix}_loss': loss,
            f'{prefix}_lr': current_lr,
            'window_idx': window_idx,
        }
        
        if not is_pretraining:
            log_dict.update({
                f'{prefix}_rmse': metrics['RMSE'],
                f'{prefix}_MSE': metrics['MSE'],
                f'{prefix}_uce': calibration['UCE'],
            })
        
    #if explanation_func_name is not None:
    #    path = cfg.dataset.data_directory
    #    path = os.path.join(path, "model_checkpoints", "calibration", f"subject_{subject_index}")
    #    if not os.path.exists(path):
    #        os.makedirs(path)
    #    if is_pretraining:
    #        PATH = os.path.join(path, f"ROAD_model_checkpoint_pretrain_subject_index_{subject_index}_window_idx_{window_idx}_explanation_func_name_{explanation_func_name}_k_{k}_ROAD.pt")
    #    else:
    #        PATH =  os.path.join(path, f"ROAD_model_checkpoint_pretrain_subject_index_{subject_index}_window_idx_{window_idx}_explanation_func_name_{explanation_func_name}_k_{k}_ROAD.pt")
    

    #    torch.save(model.state_dict(), PATH)
    
    return model, ema_model


def plot_metrics_over_time(all_trial_results, fixed_median, fixed_q1, fixed_q3, is_pretraining=False):
    trial_indices = range(len(all_trial_results))
    mep_sizes = [r['true_label'] for r in all_trial_results]
    binary_correct_fixed = [r['binary_correct_fixed'] for r in all_trial_results]
    binary_correct_rolling = [r['binary_correct_rolling'] for r in all_trial_results]
    three_class_correct = [r['three_class_correct'] for r in all_trial_results]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 32))

    # Fixed Binary classification plot
    ax1.axhline(y=fixed_median, color='black', linestyle='--', label='Median')
    for i, (mep, correct) in enumerate(zip(mep_sizes, binary_correct_fixed)):
        color = 'green' if correct else 'red'
        ax1.scatter(i, mep, c=color, alpha=0.7, s=70)
    ax1.set_title('Fixed Binary Classification: MEP Size vs Trial Index')
    ax1.set_xlabel('Trial Index')
    ax1.set_ylabel('MEP Size')
    ax1.legend()

    # Rolling Binary classification plot
    ax2.axhline(y=fixed_median, color='black', linestyle='--', label='Median')
    for i, (mep, correct) in enumerate(zip(mep_sizes, binary_correct_rolling)):
        color = 'green' if correct else 'red'
        ax2.scatter(i, mep, c=color, alpha=0.7, s=70)
    ax2.set_title('Rolling Binary Classification: MEP Size vs Trial Index')
    ax2.set_xlabel('Trial Index')
    ax2.set_ylabel('MEP Size')
    ax2.legend()

    # 3-class classification plot
    ax3.axhline(y=fixed_q1, color='blue', linestyle='--', label='Q1')
    ax3.axhline(y=fixed_median, color='black', linestyle='--', label='Median')
    ax3.axhline(y=fixed_q3, color='red', linestyle='--', label='Q3')
    for i, (mep, correct) in enumerate(zip(mep_sizes, three_class_correct)):
        color = 'green' if correct else 'red'
        ax3.scatter(i, mep, c=color, alpha=0.7, s=70)
    ax3.set_title('3-Class Classification: MEP Size vs Trial Index')
    ax3.set_xlabel('Trial Index')
    ax3.set_ylabel('MEP Size')
    ax3.legend()

    # Prediction vs True plot
    ax4.scatter(mep_sizes, [r['pred_label'] for r in all_trial_results], alpha=0.7)
    ax4.plot([min(mep_sizes), max(mep_sizes)], [min(mep_sizes), max(mep_sizes)], 'r--')
    ax4.set_title('Predicted vs True MEP Size')
    ax4.set_xlabel('True MEP Size')
    ax4.set_ylabel('Predicted MEP Size')

    plt.tight_layout()
    prefix = 'Pretraining' if is_pretraining else 'Finetuning'
    fig.suptitle(f'{prefix} Metrics Over Time')
    wandb.log({f"{prefix.lower()}_metrics_over_time": wandb.Image(fig)})
    plt.close()

def plot_next_trial_prediction(result, metrics, continuous_learner, all_results, window_idx, is_pretraining=False):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # True vs Predicted plot
    true_labels = continuous_learner.true_values
    pred_labels = continuous_learner.pred_values
    
    axs[0].scatter(true_labels, pred_labels, alpha=0.7)
    axs[0].scatter(result['true_label'], result['pred_label'], color='red', s=100, label='Current prediction')
    
    axs[0].set_xlabel('True Label')
    axs[0].set_ylabel('Predicted Label')
    
    # Add diagonal line
    lims = [
        min(axs[0].get_xlim()[0], axs[0].get_ylim()[0]),
        max(axs[0].get_xlim()[1], axs[0].get_ylim()[1]),
    ]
    axs[0].plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Diagonal')
    
    axs[0].legend()
    axs[0].set_title('True vs Predicted Label')
    
    # Metrics plot
    axs[1].bar(metrics.keys(), metrics.values())
    axs[1].set_title('Current Metrics')
    axs[1].set_ylabel('Value')
    axs[1].tick_params(axis='x', rotation=45)

    prefix = 'Pretraining' if is_pretraining else 'Finetuning'
    fig.suptitle(f'{prefix} Next Trial Prediction (Window {window_idx})')
    wandb.log({f"{prefix.lower()}_next_trial_prediction": wandb.Image(fig)})
    plt.close()

def plot_rolling_binary_accuracy(all_results, window_size=50):
    binary_accuracies_fixed = [result['binary_correct_fixed'] for result in all_results]
    binary_accuracies_rolling = [result['binary_correct_rolling'] for result in all_results]
    
    rolling_accuracies_fixed = np.convolve(binary_accuracies_fixed, np.ones(window_size)/window_size, mode='valid')
    rolling_accuracies_rolling = np.convolve(binary_accuracies_rolling, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_accuracies_fixed, label='Fixed Median')
    plt.plot(rolling_accuracies_rolling, label='Rolling Median')
    plt.xlabel('Trial')
    plt.ylabel('Rolling Binary Accuracy')
    plt.title(f'Rolling Binary Accuracy (window size: {window_size})')
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend()
    wandb.log({"rolling_binary_accuracy": wandb.Image(plt)})
    plt.close()


def plot_calibration_results(all_trial_results):
    # Extract data from all_trial_results
    true_values = np.array([r['true_label'] for r in all_trial_results])
    pred_values = np.array([r['pred_label'] for r in all_trial_results])
    log_vars = np.array([r['log_var'] for r in all_trial_results])
    sigma2 = np.exp(log_vars)
    squared_error = (pred_values - true_values) ** 2
    trial_indices = np.arange(len(all_trial_results))

    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    
    # 1. Predicted values over true values
    axs[0, 0].scatter(true_values, pred_values, alpha=0.5)
    axs[0, 0].plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    axs[0, 0].set_xlabel('True Values')
    axs[0, 0].set_ylabel('Predicted Values')
    axs[0, 0].set_title('Predicted vs True Values')

    # 2. Uncertainty (sigma2) over true values
    axs[0, 1].scatter(true_values, sigma2, alpha=0.5)
    axs[0, 1].set_xlabel('True Values')
    axs[0, 1].set_ylabel('Uncertainty (sigma^2)')
    axs[0, 1].set_title('Uncertainty vs True Values')

    # 3. Squared error over uncertainty (sigma2)
    axs[1, 0].scatter(sigma2, squared_error, alpha=0.5)
    axs[1, 0].plot([0, max(sigma2)], [0, max(sigma2)], 'r--')
    axs[1, 0].set_xlabel('Uncertainty (sigma^2)')
    axs[1, 0].set_ylabel('Squared Error')
    axs[1, 0].set_title('Squared Error vs Uncertainty')

    # 4. MEP amplitudes over time, colored by prediction uncertainty
    # Define uncertainty thresholds (you may need to adjust these)
    low_threshold = np.percentile(sigma2, 33)
    high_threshold = np.percentile(sigma2, 66)

    colors = np.where(sigma2 < low_threshold, 'green', 
                      np.where(sigma2 < high_threshold, 'yellow', 'red'))

    scatter = axs[1, 1].scatter(trial_indices, true_values, c=colors, alpha=0.7)
    axs[1, 1].set_xlabel('Trial Index')
    axs[1, 1].set_ylabel('MEP Amplitude')
    axs[1, 1].set_title('MEP Amplitudes over Time, Colored by Uncertainty')

    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Low Uncertainty',
                              markerfacecolor='g', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Medium Uncertainty',
                              markerfacecolor='y', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='High Uncertainty',
                              markerfacecolor='r', markersize=10)]
    axs[1, 1].legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig


def continuous_finetuning(model, ema_model, data_window, optimizer, scheduler, cfg, device, labels_scaler, mean_mep, accelerator, start_index, num_epochs=1, explanation_func_name=None, k=None, rep=None):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        all_preds = []
        all_logvars = []
        all_targets_raw = []

        # Create a TensorDataset
        dataset = torch.utils.data.TensorDataset(data_window['epoch'], data_window['label_raw'])
        
        # Create a DataLoader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)

        for batch_epochs, batch_labels_raw in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_epochs)
            outputs_mean = outputs[:, 0]
            log_vars = outputs[:, 1]

            loss = total_loss_fn(outputs_mean, batch_labels_raw, log_vars, beta=cfg.training.nll_beta, lambda2=cfg.training.kde_lambda)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            # Update EMA model
            ema_model.update()

            total_loss += loss.item() * batch_epochs.size(0)  # Multiply by batch size
            all_preds.append(outputs_mean)
            all_logvars.append(log_vars)
            all_targets_raw.append(batch_labels_raw)

        # Calculate metrics for this epoch
        all_preds = torch.cat(all_preds)
        all_logvars = torch.cat(all_logvars)
        all_targets_raw = torch.cat(all_targets_raw)
        metrics = calculate_regression_metrics(all_preds, all_targets_raw, labels_scaler, mean_mep)
        calibration_metrics = calculate_calibration_metrics(all_preds, all_targets_raw, all_logvars)
    
    #with torch.no_grad():
    #    if explanation_func_name is not None:
    #        path = cfg.dataset.data_directory
    #        path = os.path.join(path, "model_checkpoints","finetune",f"subject_idx_{str(subject_index)}")
    #        if not os.path.exists(path):
    #            os.makedirs(path)
    #        PATH = os.path.join(path, f"ROAD_model_checkpoint_finetune_subject_index_{subject_index}_start_idx_{start_index}_rep_{rep}_explanation_func_name_{explanation_func_name}_k_{k}.pth")
    #        torch.save(model.state_dict(), PATH)

    return total_loss / len(data_window['epoch']), metrics, calibration_metrics

# add mean(output of model 1), uncertainty(output of model 2), raw_label, pred_label, binary_true label and binary_pred label to the dict

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

def sort_expl_by_importance(explanations):
    """For each feature map in the explanations list, returns the indices
    that would sort the pixels according to the attribution.

    :param explanations: list of feature attribution maps (one per sample)
    :returns: list of explanation ranks
    """
    with torch.no_grad():
        all_indices = []
        if len(explanations) == 1:
            explanations = explanations[0]
        #one explanation per trial
        for explanation in explanations:
            sorted_indices = np.argsort(explanation, axis=None)[::-1]
            all_indices.append(sorted_indices)

    # for each trial, return ordered list of indices of  pixel-wise importances according to explanation method
        return all_indices

def remove_top_k(data, explanations, device, k=100):
    """Change given dataset in-place to replace top-k important indices per trial with 0.
    This assumes the replacement with 0 to the an accurate representation of information removal, which is debatable.

    :param model: Model to explain
    :param dataset: Image data
    :param explanations: list of explanations corresponding to the dataset
    :param k: number of features to remove at once, initialized at 100
    :returns: list of model accuracy at removal of k features.
    """
    # Note: For channel occlusion eval, k needs to be set to 900
    
    # the original function successively "removes" pixels from the test-set and test accuracy of the model on the deteriorated samples
    # e.g. test accuracy when top-k pixels(top-k according to explanation method) are removed from each sample in the test dataset
    # what we want to do instead is remove only the top k pixel from the dataset before training on the train set
    # before evaluation we also want to remove the top-k pixels(accordin to explanatin method) of each trial from the test set

    with torch.no_grad():
        sorted_attribution_indices = sort_expl_by_importance(explanations)
        
        


        channels, time_points = data.shape[1], data.shape[2]
        #data_iterable_list = dataset.epochs_list[0]
        #data_iterable_epochs = dataset.epochs
        #data_iterable_list = torch.from_numpy(data_iterable_list)
        #data_iterable_list.to(device)
        k_start = 0
        k_end = k
    # iterate over lists of top-k pixel importance per trial
        for i, indices in enumerate(sorted_attribution_indices):
            nlImputer = NoisyLinearImputer()
            #get indices of top-k important pixels of current trial
            indices = sorted(indices[k_start:k_end])
            #flatten data of current trial
            mask = np.ones((channels, time_points), dtype=bool)
            mask[np.unravel_index(indices, (channels, time_points))] = False
            mask = torch.from_numpy(mask)
            d = torch.from_numpy(data[i]).float()
            imputed_sample = nlImputer(d.unsqueeze(0), mask).numpy()
            data[i] = imputed_sample
            #flat_data_list = data[i].reshape(-1)

            # set top-k most importantant pixels to 0
            #flat_data_list[indices] = 0
            #reshape trial data to orginal data again. change dataset in-place to save on RAM(possibly bad idea to do this, but since we reload entire dataset in each
            # attempt I assume not a big deal)
            #data[i] = flat_data_list.reshape(channels, time_points)
        return data


    
def random_baseline(model, data_loader, device):
    """Randomly shuffle the pixels of the dataset and evaluate the model accuracy.

    :param model: Model to evaluate
    :param dataset: Image data
    :returns: Model accuracy
    """
    random_explanations = []
    dataset = data_loader.dataset
    # Note  
    with torch.no_grad():
        #print(type(dataset))
        if isinstance(dataset, EEGCustomDataset_nt):
            data_shape = dataset.epochs[0].shape
            channels, time_points = data_shape
            rng = np.random.default_rng(42)
        #data_iterable = dataset.epochs_list
        #for j in range(len(data_iterable)):
        #    for i in range(len(data_iterable[j])):
        #        # j is subject index, i is trial index
        #        flat_data = data_iterable[j,i].reshape(-1)
        #        indices = np.random.permutation(flat_data.size)
            #print(f" indices shape: {indices.shape}"
            for data in data_loader:
                flat_data = data["epoch"].reshape(data["epoch"].shape[0],-1).cpu().numpy()
                random_explanation = rng.uniform(0,100, size=(flat_data.shape[0], flat_data.shape[1]))
                random_explanations.extend(random_explanation.reshape(data["epoch"].shape[0],channels, time_points))

        elif isinstance(dataset, torch.utils.data.dataset.TensorDataset):
            data_shape = dataset.tensors[0][0].shape
            channels, time_points = data_shape
            rng = np.random.default_rng(42)
            for data,_ in data_loader:
                flat_data = data.reshape(data.shape[0],-1).cpu().numpy()
                random_explanation = rng.uniform(0,100, size=(flat_data.shape[0], flat_data.shape[1]))
                random_explanations.extend(random_explanation.reshape(data.shape[0],channels, time_points))     
        return random_explanations
        


def DeepLift_wrapper(model, data_loader, device):
    with torch.no_grad():
        #DeepLift_explanations = np.zeros((len(data_loader.dataset), data_loader.dataset["epoch"].shape[0], data_loader.dataset["epoch"].shape[1]))
        dl = DeepLift(model)
        DeepLift_explanations = []
        dataset = data_loader.dataset
        if isinstance(dataset, EEGCustomDataset_nt):
            for data in data_loader:
                #print(type(data["epoch"]))
                x_batch = data["epoch"].to(device)
                #x_batch = x_batch.unsqueeze(0)
                dl_attr = dl.attribute(x_batch, target=0)
                DeepLift_explanations.extend(dl_attr.cpu().numpy())
        elif isinstance(dataset, torch.utils.data.dataset.TensorDataset):
            for data,_ in data_loader:
                print(type(data))
                x_batch = data.to(device)
                dl_attr = dl.attribute(x_batch, target=0)
                DeepLift_explanations.extend(dl_attr.cpu().numpy())
        return np.abs(DeepLift_explanations)


def IntegratedGradient_wrapper(model, data_loader, device):
    ig = IntegratedGradients(model)
    with torch.no_grad():
        IG_explanations = []
        dataset = data_loader.dataset
        if isinstance(dataset, EEGCustomDataset_nt):
            for data in data_loader:
                x_batch = data["epoch"].to(device)
                ig_attr = ig.attribute(x_batch, target=0)
                IG_explanations.extend(ig_attr.cpu().numpy())
        elif isinstance(dataset, torch.utils.data.dataset.TensorDataset):
            for data,_ in data_loader:
                x_batch = data.to(device)
                ig_attr = ig.attribute(x_batch, target=0)
                IG_explanations.extend(ig_attr.cpu().numpy())
        return np.abs(IG_explanations)

"""
def DeepLiftShap_wrapper(model, data_loader, device):
    dlshap = DeepLiftShap(model)
    with torch.no_grad():
        DLShap_explanations = []
        for data in data_loader:
            x_batch = data["epoch"].to(device)
            dlshap_attr = dlshap.attribute(x_batch, target=0)
            DLShap_explanations.append(dlshap_attr)
        return DLShap_explanations
"""
        
def GradientShap_wrapper(model, data_loader, device):
    gs = GradientShap(model)
    with torch.no_grad():
        dataset = data_loader.dataset
        GS_explanations = []
        if isinstance(dataset, EEGCustomDataset_nt):
            for data in data_loader:
                naive_baseline = data["epoch"]*0
                naive_baseline = naive_baseline.to(device)
                x_batch = data["epoch"].to(device)
                gs_attr = gs.attribute(x_batch, target=0, baselines=naive_baseline)
                GS_explanations.extend(gs_attr.cpu().numpy())
        elif isinstance(dataset, torch.utils.data.dataset.TensorDataset):
            for data,_ in data_loader:
                naive_baseline = data*0
                naive_baseline = naive_baseline.to(device)
                x_batch = data.to(device)
                gs_attr = gs.attribute(x_batch, target=0, baselines=naive_baseline)
                GS_explanations.extend(gs_attr.cpu().numpy())

        return np.abs(GS_explanations)



Reps = 10
all_subject_results = []
explanation_functions = [random_baseline]
explanation_function_names = ["random_baseline"]
#explanation_functions = [IntegratedGradient_wrapper, GradientShap_wrapper]
#explanation_function_names = ["IntegratedGradient", "GradientShap"]

ks = (60*900)*np.arange(0.1, 1.1 ,0.1)
ks = ks.astype(int)

cfg = load_config()
for subject_index in cfg.dataset.test_subject_indices:
    cfg = load_config()
    cfg.dataset.subject_index = subject_index
    cfg.exp_name = f"S4_S4EEGNet_ema_100_cal_py_{cfg.dataset.subject_index}"

    cli_args = parse_args()
    cfg = update_config(cfg, cli_args)
    save_config(cfg)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape_st = (60, 900)

    save_path = "/home/marco/Documents/GitHub/tms_eeg_decoding/data"
    cfg.dataset.data_directory = "/home/marco/Documents/GitHub/tms_eeg_decoding/data"
    epoch = 1000
    model = build_model_for_finetuning(cfg, input_shape_st, device, save_path, epoch)
    ema_model = EMA(
        model,
        beta=0.999,
        update_after_step=100, #default 100

        update_every=10 #default 10
    )
    accelerator = accelerate.Accelerator(mixed_precision=cfg.training.precision, log_with="wandb")
    model, ema_model = accelerator.prepare(model, ema_model)
    # Load all data at once
    print(cfg)
    all_epochs, all_labels_raw, fixed_median, fixed_q1, fixed_q3, labels_scaler, mean_mep, ch_names = load_eeg_data(cfg)

    
    print(f"All epochs shape: {all_epochs.shape}")

    # Pretrain the model
    calibration_explanations = {key: [] for key in explanation_function_names}
    model, ema_model = pretrain_model(model, ema_model, all_epochs, all_labels_raw, fixed_median, fixed_q1, fixed_q3, cfg, device, labels_scaler, mean_mep, accelerator)
    for explanation_func, explanation_func_name in zip(explanation_functions, explanation_function_names):
        pretrain_epochs, pretrain_labels_raw = get_sliding_window_data(all_epochs, all_labels_raw, 0, 100, mode='pretrain')
        pretrain_labels = (pretrain_labels_raw >= fixed_median).astype(int)
        pretrain_loader = create_dataloader(pretrain_epochs, pretrain_labels, pretrain_labels_raw, fixed_median, fixed_q1, fixed_q3, cfg.training.batch_size, mode='train')
       
        explanations = explanation_func(model, pretrain_loader, device)
        calibration_explanations[explanation_func_name].append(explanations)
            

    # explanations for first 100 trials need to be generated here
    # calibration models gets stored here. It needs to be loaded later, s.t. the importance of each calibration trial can be evaluated for every explanation function
    # maybe it makes sense to directly do this here

    start_index = 100  # Start prediction after pretraining data
    all_trial_results = []
    trial_counter = 0
    window_size = cfg.training.training_start_len
    num_val_samples = 0

    # After pretraining
    fixed_threshold = np.median(all_labels_raw[:100])
    continuous_learner = ContinuousLearner(window_size=50, fixed_threshold=fixed_threshold, labels_scaler=labels_scaler, mean_mep=mean_mep)
 
    # Initialize optimizer for continuous fine-tuning
    # Create a dummy train_loader for initializing the scheduler
    dummy_train_loader = create_dataloader(all_epochs[:100], (all_labels_raw[:100] >= fixed_median).astype(int), all_labels_raw[:100], fixed_median, fixed_q1, fixed_q3, cfg.training.batch_size, mode='train')

    # Initialize optimizer and scheduler
    optimizer, scheduler = initialize_optimizer_and_scheduler(cfg, model, dummy_train_loader)
    optimizer = accelerator.prepare(optimizer)

    regular_model_count = 0
    ema_model_count = 0

    start_index = 100  # Start prediction after pretraining data
    all_trial_results = []
    all_trials = []

# for these 50 trials no explanations need to be generated
    # Initialize with the first 50 trials
    init_explanations = {key: [] for key in explanation_function_names}
    for i in range(50):
        trial_epochs, trial_labels_raw = get_sliding_window_data(all_epochs, all_labels_raw, start_index + i, 1, mode='finetune')
        trial_labels = (trial_labels_raw >= fixed_median).astype(int)
        current_trial_data = create_dataloader(trial_epochs, trial_labels, trial_labels_raw, fixed_median, fixed_q1, fixed_q3, batch_size=1, mode='valid')
        all_trials.extend(list(current_trial_data))
        for explanation_func, explanation_func_name in zip(explanation_functions, explanation_function_names):
            explanations = explanation_func(model, current_trial_data, device)
            init_explanations[explanation_func_name].append(explanations)

    # Initialize lists to store metrics for each trial
    trial_indices = []
    continuous_finetune_losses = []
    continuous_finetune_rmse_standardized = []
    continuous_finetune_rmse_original = []
    continuous_finetune_relative_rmse = []
    continuous_finetune_uce = []
    learning_rates = []
    rolling_rmse_standardized = []
    rolling_rmse_original = []
    rolling_rmse_relative = []
    rolling_binary_acc = []
    fixed_binary_acc = []
    rolling_3class_acc = []
    rolling_uce = []
    regular_losses = []
    ema_losses = []

    # Main loop
    
    finetune_explanations = {key: [] for key in explanation_function_names}
    while start_index + 50 < len(all_epochs):
        #print(f"start index: {start_index}")
        # Get data for the current trial
# for all of the following trials, explanations need to be generated

        trial_epochs, trial_labels_raw = get_sliding_window_data(all_epochs, all_labels_raw, start_index + 50, 1, mode='finetune')
        trial_labels = (trial_labels_raw >= fixed_median).astype(int)
        current_trial_data = create_dataloader(trial_epochs, trial_labels, trial_labels_raw, fixed_median, fixed_q1, fixed_q3, batch_size=1, mode='valid')


        # Predict for the current trial
        for data in current_trial_data:
            result = predict_and_record(model, ema_model, data, device, fixed_median, fixed_q1, fixed_q3, continuous_learner, cfg, start_index, ch_names)
            all_trial_results.append(result)

            # Store the new metrics
            regular_losses.append(result['loss_regular'])
            ema_losses.append(result['loss_ema'])

        # Add the new trial to all_trials
        all_trials.extend(list(current_trial_data))

        # Perform continuous fine-tuning on all trials
        data_window = {k: torch.cat([d[k] for d in all_trials]) for k in all_trials[0].keys()}
        data_window = {k: v.to(device) for k, v in data_window.items()}

        
        avg_loss, metrics, calibration_metrics = continuous_finetuning(model, ema_model, data_window, optimizer, scheduler, cfg, device, labels_scaler, mean_mep, accelerator, start_index, num_epochs=1)

        for explanation_func, explanation_func_name in zip(explanation_functions, explanation_function_names):
            #dataset = torch.utils.data.TensorDataset(data_window['epoch'], data_window['label_raw'])
            #dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)
            explanations = explanation_func(model, current_trial_data, device)
            finetune_explanations[explanation_func_name].append(explanations)
            
        #explanation of current trial needs to be evaluated here for every explanation function
        # Step the scheduler
        scheduler.step()


        start_index += 1



    for rep in range(Reps):
        print(f"\nRep: {rep}\n")
        for explanation_func, explanation_func_name in zip(explanation_functions, explanation_function_names):
              
            
            cfg = load_config()
            cfg.dataset.subject_index = subject_index
            cfg.exp_name = f"S4_S4EEGNet_ema_100_cal_py_{cfg.dataset.subject_index}"
            cli_args = parse_args()
            cfg = update_config(cfg, cli_args)

            #Load pre-trained model
            for k in ks:
                
                save_path_model_checkpoint = os.path.join("/home/marco/Documents/GitHub/tms_eeg_decoding/exp", explanation_func_name, str(k))
            
                epoch = 1000
                model = build_model_for_finetuning(cfg, input_shape_st, device, save_path_model_checkpoint, epoch, explanation_func_name=explanation_func_name, k=k)

                ema_model = EMA(
                    model,
                    beta=0.999,
                    update_after_step=100, #default 100
                    update_every=10 #default 10
                )

                start_index = 100 
                accelerator = accelerate.Accelerator(mixed_precision=cfg.training.precision, log_with="wandb")
                model, ema_model = accelerator.prepare(model, ema_model)

                # Load all data at once
                print(cfg)
                cfg.dataset.data_directory = "/home/marco/Documents/GitHub/tms_eeg_decoding/data"
                all_epochs, all_labels_raw, fixed_median, fixed_q1, fixed_q3, labels_scaler, mean_mep, ch_names = load_eeg_data(cfg)
                #remove top k pixels from every trial
                all_epochs[:start_index] = remove_top_k(all_epochs[:start_index], calibration_explanations[explanation_func_name], device, k=k)
                all_epochs[start_index:start_index+50] = remove_top_k(all_epochs[start_index:start_index+50], init_explanations[explanation_func_name], device, k=k)
                all_epochs[start_index+50:] = remove_top_k(all_epochs[start_index+50:], finetune_explanations[explanation_func_name], device, k=k)

                print(f"removed pixels of trial 10 in train set: {sum(all_epochs[9].flatten() == 0)}")
                print(f"removed pixels of trial 70 in test set: {sum(all_epochs[70].flatten() == 0)}")

                pretrain_epochs, pretrain_labels_raw = get_sliding_window_data(all_epochs, all_labels_raw, 0, 100, mode='pretrain')
                
            
                model, ema_model = pretrain_model(model, ema_model, all_epochs, all_labels_raw, fixed_median, fixed_q1, fixed_q3, cfg, device, labels_scaler, mean_mep, accelerator)

                start_index = 100  # Start prediction after pretraining data
                all_trial_results = []
                trial_counter = 0
                window_size = cfg.training.training_start_len
                num_val_samples = 0

                #After pretraining
                fixed_threshold = np.median(all_labels_raw[:100])
                continuous_learner = ContinuousLearner(window_size=50, fixed_threshold=fixed_threshold, labels_scaler=labels_scaler, mean_mep=mean_mep)
 
                # Initialize optimizer for continuous fine-tuning
                # Create a dummy train_loader for initializing the scheduler
                dummy_train_loader = create_dataloader(all_epochs[:100], (all_labels_raw[:100] >= fixed_median).astype(int), all_labels_raw[:100], fixed_median, fixed_q1, fixed_q3, cfg.training.batch_size, mode='train')

                # Initialize optimizer and scheduler
                optimizer, scheduler = initialize_optimizer_and_scheduler(cfg, model, dummy_train_loader)
                optimizer = accelerator.prepare(optimizer)

                regular_model_count = 0
                ema_model_count = 0

                start_index = 100  # Start prediction after pretraining data
                all_trial_results = []
                all_trials = []

                # Initialize with the first 50 trials
                for i in range(50):
                    trial_epochs, trial_labels_raw = get_sliding_window_data(all_epochs, all_labels_raw, start_index + i, 1, mode='finetune')
                    trial_labels = (trial_labels_raw >= fixed_median).astype(int)
                    current_trial_data = create_dataloader(trial_epochs, trial_labels, trial_labels_raw, fixed_median, fixed_q1, fixed_q3, batch_size=1, mode='valid')
                    all_trials.extend(list(current_trial_data))

                # Initialize lists to store metrics for each trial
                trial_indices = []
                continuous_finetune_losses = []
                continuous_finetune_rmse_standardized = []
                continuous_finetune_rmse_original = []
                continuous_finetune_relative_rmse = []
                continuous_finetune_uce = []
                learning_rates = []
                rolling_rmse_standardized = []
                rolling_rmse_original = []
                rolling_rmse_relative = []
                rolling_binary_acc = []
                fixed_binary_acc = []
                rolling_3class_acc = []
                rolling_uce = []
                regular_losses = []
                ema_losses = []

                # Main loop
    

                while start_index + 50 < len(all_epochs):
                    #print(f"Start index: {start_index}\n")

                    #print(f"start index: {start_index}")
                    # Get data for the current trial
                    trial_epochs, trial_labels_raw = get_sliding_window_data(all_epochs, all_labels_raw, start_index + 50, 1, mode='finetune')
                    trial_labels = (trial_labels_raw >= fixed_median).astype(int)
                    current_trial_data = create_dataloader(trial_epochs, trial_labels, trial_labels_raw, fixed_median, fixed_q1, fixed_q3, batch_size=1, mode='valid')


                    # Predict for the current trial
                    for data in current_trial_data:
                        result = predict_and_record(model, ema_model, data, device, fixed_median, fixed_q1, fixed_q3, continuous_learner, cfg, start_index, ch_names)
                        all_trial_results.append(result)

                    # Store the new metrics
                    regular_losses.append(result['loss_regular'])
                    ema_losses.append(result['loss_ema'])

                    # Add the new trial to all_trials
                    all_trials.extend(list(current_trial_data))

                    # Perform continuous fine-tuning on all trials
                    data_window = {k: torch.cat([d[k] for d in all_trials]) for k in all_trials[0].keys()}
                    data_window = {k: v.to(device) for k, v in data_window.items()}

                    avg_loss, metrics, calibration_metrics = continuous_finetuning(model, ema_model, data_window, optimizer, scheduler, cfg, device, labels_scaler, mean_mep, accelerator, start_index, num_epochs=1, explanation_func_name=explanation_func_name, k=k, rep=rep)

                    # Step the scheduler
                    scheduler.step()

                    #Store metrics for this trial
                    trial_indices.append(start_index)
                    continuous_finetune_losses.append(avg_loss)
                    continuous_finetune_rmse_standardized.append(metrics['RMSE_standardized'])
                    continuous_finetune_rmse_original.append(metrics['RMSE_original'])
                    continuous_finetune_relative_rmse.append(metrics['Relative_RMSE_percentage'])
                    continuous_finetune_uce.append(calibration_metrics['UCE'])
                    learning_rates.append(scheduler.get_last_lr()[0])

                    continuous_learner_metrics = continuous_learner.get_metrics()
                    rolling_rmse_standardized.append(continuous_learner_metrics['RMSE_standardized'])
                    rolling_rmse_original.append(continuous_learner_metrics['RMSE_original'])
                    rolling_rmse_relative.append(continuous_learner_metrics['RMSE_relative'])
                    rolling_binary_acc.append(continuous_learner_metrics['BinaryAccuracy_Rolling'])
                    fixed_binary_acc.append(continuous_learner_metrics['BinaryAccuracy_Fixed'])
                    rolling_3class_acc.append(continuous_learner_metrics['ThreeClassAccuracy'])
                    rolling_uce.append(continuous_learner_metrics['UCE'])
        
                    start_index += 1

                #Log all metrics at once
      

                def to_cpu_numpy(tensor):
                    if isinstance(tensor, torch.Tensor):
                        return tensor.detach().cpu().numpy()
                    elif isinstance(tensor, list) and isinstance(tensor[0], torch.Tensor):
                        return [t.detach().cpu().numpy() for t in tensor]
                    return tensor
     
                # Prepare all metrics
                all_metrics = {
                    'trial_indices': trial_indices,
                    'continuous_finetune_losses': continuous_finetune_losses,
                    'continuous_finetune_rmse_standardized': continuous_finetune_rmse_standardized,
                    'continuous_finetune_rmse_original': continuous_finetune_rmse_original,
                    'continuous_finetune_relative_rmse': continuous_finetune_relative_rmse,
                    'continuous_finetune_uce': continuous_finetune_uce,
                    'learning_rates': learning_rates,
                    'rolling_rmse_standardized': rolling_rmse_standardized,
                    'rolling_rmse_original': rolling_rmse_original,
                    'rolling_rmse_relative': rolling_rmse_relative,
                    'rolling_binary_acc': rolling_binary_acc,
                    'fixed_binary_acc': fixed_binary_acc,
                    'rolling_3class_acc': rolling_3class_acc,
                    'rolling_uce': rolling_uce,
                    'regular_losses': regular_losses,
                    'ema_losses': ema_losses,
                    'rep': rep,
                    'subject_index': subject_index,
                    'explanation_func_name': explanation_func_name,
                    'k': k,
                    }

                # Convert all tensors to CPU NumPy arrays
                all_metrics = {k: to_cpu_numpy(v) for k, v in all_metrics.items()}

                eval_save_path = os.path.join(save_path, "evaluation", f"{cfg.exp_name}_subject_{subject_index}")
                if not os.path.exists(eval_save_path):
                    os.makedirs(eval_save_path, exist_ok=True)

                # Save all metrics to a NPZ file
                metrics_filename = os.path.join(eval_save_path, f'ROAD_all_metrics_subject_{subject_index}_rep_{rep}_explantion_func_{explanation_func_name}_k_{k}.npz')
                np.savez(metrics_filename, **all_metrics)

               




  
        # %%
