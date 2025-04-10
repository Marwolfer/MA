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
from captum.attr import Saliency
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import FeatureAblation
from captum.attr import DeepLift
import pickle
import time

from imputations import NoisyLinearImputer
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
 data_directory: /home/marco/Documents/GitHub/tms_eeg_decoding/data
 file_name: subject_{:03d}_preprocessed_combined_pen.fif
 test_subject_indices: 
 subject_index: 100
 #pretrain_subject_indices: [49, 28]
 pretrain_subject_indices: [4,7,9,11,14,18,22,31,33,39,51,53,59,63,65,66,70,71,74,76,83,94,95,100,103,104,106,107,108,109,110,111]
training:
 lr: 0.0003
 num_epochs: 1002
 nll_beta: 0.001
 num_warmup_epochs: 20
 batch_size: 50
 random_seed: 42
 precision: bf16
 kde_lambda: 0.5
exp_name: channel_ROAR_S4EEGNet_pretrain
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
        loss = total_loss_fn(outputs_mean, data['label_raw'].to(device), log_vars, beta=cfg.training.nll_beta, lambda2=cfg.training.kde_lambda)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        all_preds.append(outputs_mean)
        all_logvars.append(log_vars)
        all_targets_smooth.append(data['label_raw'].to(device))
    all_preds = torch.cat(all_preds)
    all_logvars = torch.cat(all_logvars)
    all_targets_smooth = torch.cat(all_targets_smooth)
    metrics = calculate_regression_metrics(all_preds, all_targets_smooth)
    calibration_metrics = calculate_calibration_metrics(all_preds, all_targets_smooth, all_logvars)
    del all_preds, all_logvars, all_targets_smooth
    #if (epoch + 1) % 20 == 0:
    #    plot_training_metrics(epoch, all_preds, all_targets_smooth, all_logvars, metrics, calibration_metrics)
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
    
    
def remove_top_k(dataloader, explanations, device, k=100):
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
        
        dataset = dataloader.dataset
        data_shape = dataset[0]["epoch"].shape
        #dataset_test = dataset[0:50]["epoch"]

        channels, time_points = data_shape
        #data_iterable_list = dataset.epochs_list
        data_iterable_epochs = dataset.epochs
        #data_iterable_list = torch.from_numpy(data_iterable_list)
        #data_iterable_list.to(device)
        k_start = 0
        k_end = k
    # iterate over lists of top-k pixel importance per trial
        for i, indices in enumerate(sorted_attribution_indices):
            nlImputer = NoisyLinearImputer()
            #get indices of top-k important pixels of current trial
            indices = sorted(indices[k_start:k_end])
            # create boolean mask of top-k important pixels
            mask = np.ones((channels, time_points), dtype=bool)
            mask[np.unravel_index(indices, (channels, time_points))] = False
            mask = torch.from_numpy(mask)
            imputed_sample = nlImputer(data_iterable_epochs[i].unsqueeze(0), mask)
            data_iterable_epochs[i] = imputed_sample

    #Only interesting for test-data. Possibly include boolean option for this
    #performances.append(model_accuracy(model, dataset))

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
        data_shape = dataset[0]["epoch"].shape
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

            #random_explanation = rng.permutation(flat_data,axis=1)
            random_explanation = rng.uniform(0,100, size=(flat_data.shape[0], flat_data.shape[1]))
    
            random_explanations.extend(random_explanation.reshape(data["epoch"].shape[0],channels, time_points))
        return random_explanations
    
def DeepLift_wrapper(model, data_loader, device):
    with torch.no_grad():
        #DeepLift_explanations = np.zeros((len(data_loader.dataset), data_loader.dataset["epoch"].shape[0], data_loader.dataset["epoch"].shape[1]))
        dl = DeepLift(model)
        DeepLift_explanations = []
        for data in data_loader:
            x_batch = data["epoch"].to(device)
            dl_attr = dl.attribute(x_batch, target=0)
            DeepLift_explanations.extend(dl_attr.cpu().numpy())
        return np.abs(DeepLift_explanations)
    
    
def IntegratedGradient_wrapper(model, data_loader, device):
    ig = IntegratedGradients(model)
    with torch.no_grad():
        IG_explanations = []
        for data in data_loader:
            x_batch = data["epoch"].to(device)
            ig_attr = ig.attribute(x_batch, target=0)
            IG_explanations.extend(ig_attr.cpu().numpy())
        return np.abs(IG_explanations)
    

def GradientShap_wrapper(model, data_loader, device):
    gs = GradientShap(model)
    with torch.no_grad():
        
        GS_explanations = []
        for data in data_loader:
            naive_baseline = data["epoch"]*0
            naive_baseline = naive_baseline.to(device)
            x_batch = data["epoch"].to(device)
            gs_attr = gs.attribute(x_batch, target=0, baselines=naive_baseline)
            GS_explanations.extend(gs_attr.cpu().numpy())
        return np.abs(GS_explanations)


def Saliency_wrapper(model, data_loader, device):
    saliency = Saliency(model)
    with torch.no_grad():
        Saliency_explanations = []
        for data in data_loader:
            x_batch = data["epoch"].to(device)
            saliency_attr = saliency.attribute(x_batch, target=0)
            Saliency_explanations.extend(saliency_attr.cpu().numpy())
        return np.abs(Saliency_explanations)

    

def main(reps = 1):
    np.random_seed = 42
    torch.manual_seed(42)

    ks = (60*900)*np.arange(0.1, 1.1 ,0.1)
    ks = ks.astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    cli_args = parse_args()
    cfg = update_config(cfg, cli_args)
    save_config(cfg, cfg.dataset.subject_index)

    explanation_functions = [Saliency_wrapper]   
    explanation_function_names = ["Saliency"]

    
    #explanation_functions = [IntegratedGradient_wrapper, GradientShap_wrapper]
    #explanation_function_names = ["IntegratedGradient", "GradientShap"]

    pretrain_loader, input_shape_train,_,_,_ = get_pretrain_dataloader(cfg)
    pretrain_loader = pretrain_loader
    pretrain_loader_copy = copy.deepcopy(pretrain_loader)

    model = build_model(cfg, input_shape_train, device)
    optimizer, lr_scheduler = initialize_optimizer_and_scheduler(cfg, model, pretrain_loader_copy)
    accelerator = accelerate.Accelerator(mixed_precision=cfg.training.precision, log_with="wandb")
    model, optimizer, lr_scheduler, pretrain_loader = accelerator.prepare(
            model, optimizer, lr_scheduler, pretrain_loader, 
            )
    
    save_path = os.path.join("exp", cfg.exp_name)
    os.makedirs(save_path, exist_ok=True)
    save_interval = 5  # Save checkpoint every 5 epochs
    with tqdm(range(cfg.training.num_epochs)) as pbar:
        for epoch in pbar:
            train_loss,_,_,_= train_one_epoch(epoch, model, pretrain_loader_copy, optimizer, lr_scheduler, accelerator, device, cfg)  
            pbar.set_description(f"Epoch {epoch} - Train Loss: {train_loss:.2f}")

    for explanation_function, explanation_function_name in zip(explanation_functions, explanation_function_names):

        pretrain_loader_copy = copy.deepcopy(pretrain_loader)
        explanations_train = []
        explanations_train.append(explanation_function(model, pretrain_loader_copy, device))
        for k in ks:
            time1 = time.time()
            print(f"\nRunning explanation function {explanation_function_name} with k={k}\n")
            pretrain_loader_copy = copy.deepcopy(pretrain_loader)

            setup_wandb(cfg, f"{cfg.exp_name}_pretrain", cfg.dataset.subject_index)
            remove_top_k(pretrain_loader_copy, explanations_train, device, k=k)

            #print(f"removed pix#els of trial 0 in train set: {sum(pretrain_loader_copy.dataset.epochs.data[0].flatten() == 0)}")
            #print(f"removed pixels of trial 350 in test set: {sum(pretrain_loader_copy.dataset.epochs.data[350].flatten() == 0)}")
            #print(f"removed pixels of trial 10 in train set: {sum(pretrain_loader_copy.dataset.epochs.data[9].flatten() == 0)}")
            #print(f"removed pixels of trial 44 in test set: {sum(pretrain_loader_copy.dataset.epochs.data[43].flatten() == 0)}")

            model = build_model(cfg, input_shape_train, device)
            optimizer, lr_scheduler = initialize_optimizer_and_scheduler(cfg, model, pretrain_loader_copy)
            accelerator = accelerate.Accelerator(mixed_precision=cfg.training.precision, log_with="wandb")
            model, optimizer, lr_scheduler, pretrain_loader = accelerator.prepare(
                    model, optimizer, lr_scheduler, pretrain_loader_copy, 
            )
    
            save_path = os.path.join("exp", cfg.exp_name)
            os.makedirs(save_path, exist_ok=True)
            save_interval = 5  # Save checkpoint every 5 epochs
            with tqdm(range(cfg.training.num_epochs)) as pbar:
                for epoch in pbar:
                    train_loss, train_metrics, train_calibration, train_classification = train_one_epoch(epoch, model, pretrain_loader_copy, optimizer, lr_scheduler, accelerator, device, cfg)          
                    current_lr = optimizer.param_groups[0]['lr']
            
                    pbar.set_description(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train RMSE: {train_metrics['RMSE']:.4f}")
            
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_corr": train_metrics["Correlation"],
                        "train_uce": train_calibration["UCE"],
                        "learning_rate": current_lr,
                        'exp_name': explanation_function_name,
                        'k': k
                    })

                    if epoch==1000:
                        save_path = os.path.join("exp", explanation_function_name, str(k))
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        checkpoint_path = os.path.join(save_path, f"ROAD_checkpoint_full_epoch_{epoch}_explanation_func_{explanation_function_name}_k_{k}.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'train_loss': train_loss,
                            'train_rmse': train_metrics["RMSE"],
                            'train_corr': train_metrics["Correlation"],
                            'train_uce': train_calibration["UCE"],
                            'exp_name': explanation_function_name,
                            'k': k
                            }, checkpoint_path)
            time2 = time.time()
            print(f"Time taken for k={k}: {(time2-time1):.2f} seconds")

                        # Save TrunkNet checkpoint
                        #trunk_checkpoint_path = os.path.join(save_path, f"checkpoint_trunk_epoch_{epoch}_explanation_func_{explanation_function_name}_k_{k}.pt")
                        #torch.save({
                        #    'epoch': epoch,
                        #    'trunk_state_dict': model.trunk_net.state_dict(),
                        #}, trunk_checkpoint_path)

                        # Save HeadNet checkpoint
                        #head_checkpoint_path = os.path.join(save_path, f"checkpoint_head_epoch_{epoch}_explanation_func_{explanation_function_name}_k_{k}.pt")
                        #torch.save({
                        #    'epoch': epoch,
                        #    'head_state_dict': model.head_net.state_dict(),
                        #}, head_checkpoint_path)
        
            wandb.finish()

#%%
if __name__ == "__main__":
    main()
# %%
