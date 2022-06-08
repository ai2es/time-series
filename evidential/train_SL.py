from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch, sys, logging
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset, Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn import init

from evidential import LinearNormalGamma, evidential_regresssion_loss

import joblib
import yaml, tqdm
import pandas as pd
from collections import defaultdict
import random, os
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None
from scipy.stats import pearsonr

from torch.optim.lr_scheduler import ReduceLROnPlateau
from echo.src.base_objective import BaseObjective

from evidential import EvidentialMarginalLikelihood, EvidenceRegularizer, modified_mse


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
        
        
def load_model(input_size, middle_size, output_size, dropout):
    model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_size, middle_size)),
            #nn.BatchNorm1d(middle_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            #nn.Tanh(),
            LinearNormalGamma(middle_size, output_size)
    )
    return model


def load_mlp_model(input_size, middle_size, output_size, dropout):
    model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_size, middle_size)),
            #nn.BatchNorm1d(middle_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            #nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(middle_size, output_size))
    )
    return model
        
def init_weights(net, init_type='normal', init_gain=0.0, verbose=True):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    if verbose:
        logging.info('Initializing network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>        

    
class Objective(BaseObjective):

    def __init__(self, config, metric="val_loss", device="cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        return trainer(conf, trial = trial, evaluate = False, verbose = False)
        
        
def trainer(conf, trial = False, evaluate = False, verbose = True, device = None):
    
    if device == None:
        is_cuda = torch.cuda.is_available()
        device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
    
    # Set seeds for reproducibility
    save_loc = conf["save_loc"]
    data_seed = conf["data_seed"]
    model_seed = conf["model_seed"]
    input_cols = conf["input_cols"]
    output_cols = conf["output_cols"]
    
    if conf["use_idaho"]:
        data = pd.read_csv(conf["idaho"])
    elif conf["use_cabauw"]:
        data = pd.read_csv(conf["cabauw"])
    else:
        raise OSError("You must specify eithe idaho or cabauw")
        
    data["day"] = data["Time"].apply(lambda x: str(x).split(" ")[0])
    
    # Need the same test_data for all trained models (data and model ensembles)
    flat_seed = 1000
    gsp = GroupShuffleSplit(n_splits=1,  random_state = flat_seed, train_size=0.9)
    splits = list(gsp.split(data, groups = data["day"]))
    train_index, test_index = splits[0]
    train_data, test_data = data.iloc[train_index].copy(), data.iloc[test_index].copy() 

    
    # Make N train-valid splits using day as grouping variable
    gsp = GroupShuffleSplit(n_splits=100,  random_state = flat_seed, train_size=0.885)
    splits = list(gsp.split(train_data, groups = train_data["day"]))
    train_index, valid_index = splits[data_seed]
    train_data, valid_data = train_data.iloc[train_index].copy(), train_data.iloc[valid_index] .copy()  
    
    # Set the seed for the model here
    ## This can come before or after train_test_split, 
    ## as it had no effect on training when tested
    seed_everything(model_seed)
    
    # Load other config options
    input_size = len(input_cols)
    middle_size = conf["middle_size"]
    output_size = conf["output_size"]
    weight_init = conf["weight_init"]

    epochs = conf["epochs"]
    batch_size = conf["batch_size"]
    L1_penalty = conf["L1_penalty"]
    L2_penalty = conf["L2_penalty"]
    dropout = conf["dropout"]
    
    metric = conf["metric"]
    lr_patience = conf["lr_patience"]
    stopping_patience = conf["stopping_patience"]
    learning_rate = conf["learning_rate"]
    clip = conf["clip"]
    loss_coeff = conf["loss_coeff"]
    
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(train_data[input_cols])
    x_valid = x_scaler.transform(valid_data[input_cols])
    x_test = x_scaler.transform(test_data[input_cols])
    
    y_train = y_scaler.fit_transform(train_data[output_cols])
    y_valid = y_scaler.transform(valid_data[output_cols])
    y_test = y_scaler.transform(test_data[output_cols])
    
    train_split = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float()
    )
    train_loader = DataLoader(train_split, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=0)

    valid_split = TensorDataset(
        torch.from_numpy(x_valid).float(),
        torch.from_numpy(y_valid).float()
    )
    valid_loader = DataLoader(valid_split, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=0)
    
    model = load_model(input_size, middle_size, output_size, dropout).to(device)
    init_weights(model, init_type = weight_init, verbose = verbose)

    optimizer = torch.optim.Adam(model.parameters(),
                             lr = learning_rate,
                             weight_decay = L2_penalty,
                             eps = 1e-7,
                             betas = (0.9, 0.999),
                             amsgrad = False)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience = lr_patience, 
            verbose = verbose,
            min_lr = 1.0e-13
    )
    
    
    # load multi-task losses
    nll_loss = EvidentialMarginalLikelihood() ## original loss, NLL loss
    reg = EvidenceRegularizer() ## evidential regularizer
    mmse_loss = modified_mse ## lipschitz MSE loss
    
    
    results_dict = defaultdict(list)
    
    if verbose:
        my_iter = tqdm.tqdm(range(epochs),
                            total = epochs,
                            leave = True)
    else:
        my_iter = list(range(epochs))
    

    for epoch in my_iter:

        # Train in batch mode
        model.train()

        train_loss = []
        for k, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

#             loss = evidential_regresssion_loss(y.to(device), model(x.to(device)), loss_coeff)
#             l1_norm = sum(p.abs().sum() for p in model.parameters())
#             l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
#             loss += L1_penalty * l1_norm
#             loss += L2_penalty * l2_norm

            labels = y.to(device) 
            pred = model(x.to(device))
            gamma, nu, alpha, beta = pred
            #loss = evidential_regresssion_loss(labels, pred, loss_coeff)
            loss = nll_loss(gamma, nu, alpha, beta, labels)
            loss += reg(gamma, nu, alpha, beta, labels)
            loss += mmse_loss(gamma, nu, alpha, beta, labels)

            train_loss.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if not np.isfinite(loss.item()):
                print(nll_loss(gamma, nu, alpha, beta, labels), 
                      reg(gamma, nu, alpha, beta, labels), 
                      mmse_loss(gamma, nu, alpha, beta, labels))
                raise

        # Validate 
        model.eval()
        with torch.no_grad():

            # Validate in batch mode
            mae_loss = []
            valid_loss = []
            for k, (x, y) in enumerate(valid_loader):
                labels = y.to(device) 
                #gamma, nu, alpha, beta = model(x.to(device))
                pred = model(x.to(device))
                gamma, nu, alpha, beta = pred
                loss = nll_loss(gamma, nu, alpha, beta, labels)
                #loss = evidential_regresssion_loss(labels, pred, loss_coeff)
                loss += reg(gamma, nu, alpha, beta, labels)
                loss += mmse_loss(gamma, nu, alpha, beta, labels)
                #pred = model(x.to(device))
                #loss = evidential_regresssion_loss(y.to(device), pred, loss_coeff)
                #mae = torch.nn.L1Loss()(labels, pred[0])
                mae = torch.nn.L1Loss()(labels, gamma)
                valid_loss.append(loss.item())
                mae_loss.append(mae.item())

        if not np.isfinite(np.mean(valid_loss)):
            break

        results_dict["epoch"].append(epoch)
        results_dict["train_nll"].append(np.mean(train_loss))
        results_dict["val_nll"].append(np.mean(valid_loss))
        results_dict["val_mae"].append(np.mean(mae_loss))
        results_dict["lr"].append(optimizer.param_groups[0]['lr'])

        # Save the dataframe to disk
        df = pd.DataFrame.from_dict(results_dict).reset_index()
        df.to_csv(f"{save_loc}/training_log.csv", index = False)

        print_str = f'Epoch {epoch} train_nll {results_dict["train_nll"][-1]:4f}'
        print_str += f' val_nll {results_dict["val_nll"][-1]:4f}'
        print_str += f' val_mae {results_dict["val_mae"][-1]:4f}'
        print_str += f' lr {results_dict["lr"][-1]}'
        if verbose:
            my_iter.set_description(print_str)
            my_iter.refresh()

        # anneal the learning rate using just the box metric
        lr_scheduler.step(results_dict[metric][-1])

        if results_dict[metric][-1] == min(results_dict[metric]):
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': min(results_dict[metric])
            }
            torch.save(state_dict, f"{save_loc}/mlp.pt")

        # Stop training if we have not improved after X epochs
        best_epoch = [i for i,j in enumerate(results_dict[metric]) if j == min(results_dict[metric])][0]
        offset = epoch - best_epoch
        if offset >= stopping_patience:
            break

    best_epoch = [i for i,j in enumerate(results_dict[metric]) if j == min(results_dict[metric])][0]
    
    results = {
        "epoch": best_epoch,
        "train_nll": results_dict["train_nll"][best_epoch],
        "val_nll": results_dict["val_nll"][best_epoch],
        "val_mae": results_dict["val_mae"][best_epoch]
    }
    
    if evaluate:
        
        model = load_model(input_size, middle_size, output_size, dropout).to(device)
        
        checkpoint = torch.load(
            f"{save_loc}/mlp.pt",
            map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        x_splits = [x_train, x_valid, x_test]
        splits = [train_data, valid_data, test_data]
        
        sigma = train_data[output_cols[0]].std()
        
        for x,y in zip(x_splits, splits):
        
            with torch.no_grad():
                _input = torch.from_numpy(x).float()
                y_pred = model(_input.to(device))

            y["mu"] = y_scaler.inverse_transform(y_pred[0].cpu().numpy())
            y["v"] = y_pred[1].cpu().numpy()
            y["alpha"] = y_pred[2].cpu().numpy()
            y["beta"] = y_pred[3].cpu().numpy()
            
            inverse_evidence = 1. / ((y["alpha"].copy()-1) * y["v"])
            variance = y["beta"] * inverse_evidence
            
            y["var"] = (variance * sigma**2)
            y["conf"] = (inverse_evidence * sigma**2)
            y["error"] = y["mu"] - y[output_cols[0]]

            y.sort_values("var", ascending = True)
            y["dummy"] = 1
            y["cu_mae"] = y["error"].cumsum() / y["dummy"].cumsum()
            y["cu_var"] = y["var"].cumsum() / y["dummy"].cumsum()
            y["var_cov"] = 1 - y["dummy"].cumsum() / len(y)
        
        with open(f"{save_loc}/scalers.pkl", "wb") as fid:
            joblib.dump([x_scaler, y_scaler], fid)
            
        train_data.to_csv(f"{save_loc}/train.csv")
        valid_data.to_csv(f"{save_loc}/valid.csv") 
        test_data.to_csv(f"{save_loc}/test.csv")

    return results


def mlp_trainer(conf, trial = False, evaluate = False, verbose = True, device = None):
    
    if device == None:
        is_cuda = torch.cuda.is_available()
        device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
    
    # Set seeds for reproducibility
    save_loc = conf["save_loc"]
    data_seed = conf["data_seed"]
    model_seed = conf["model_seed"]
    
    input_cols = conf["input_cols"]
    output_cols = conf["output_cols"]
    
    if conf["use_idaho"]:
        data = pd.read_csv(conf["idaho"])
    elif conf["use_cabauw"]:
        data = pd.read_csv(conf["cabauw"])
    else:
        raise OSError("You must specify eithe idaho or cabauw")
        
    data["day"] = data["Time"].apply(lambda x: str(x).split(" ")[0])
    
    # Need the same test_data for all trained models (data and model ensembles)
    flat_seed = 1000
    gsp = GroupShuffleSplit(n_splits=1,  random_state = flat_seed, train_size=0.9)
    splits = list(gsp.split(data, groups = data["day"]))
    train_index, test_index = splits[0]
    train_data, test_data = data.iloc[train_index].copy(), data.iloc[test_index].copy() 

    # Make N train-valid splits using day as grouping variable
    gsp = GroupShuffleSplit(n_splits=100,  random_state = flat_seed, train_size=0.885)
    splits = list(gsp.split(train_data, groups = train_data["day"]))
    train_index, valid_index = splits[data_seed]
    train_data, valid_data = train_data.iloc[train_index].copy(), train_data.iloc[valid_index] .copy()  
    
    # Set the seed for the model here
    ## This can come before or after train_test_split, 
    ## as it had no effect on training when tested
    seed_everything(model_seed)
    
    # Load other config options
    model_type = conf["model_type"]
    input_size = len(input_cols)
    middle_size = conf["middle_size"]
    output_size = conf["output_size"]
    weight_init = conf["weight_init"]

    epochs = conf["epochs"]
    batch_size = conf["batch_size"]
    L1_penalty = conf["L1_penalty"]
    L2_penalty = conf["L2_penalty"]
    dropout = conf["dropout"]
    
    metric = conf["metric"]
    lr_patience = conf["lr_patience"]
    stopping_patience = conf["stopping_patience"]
    learning_rate = conf["learning_rate"]
    clip = conf["clip"]
    loss_coeff = conf["loss_coeff"]
    
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(train_data[input_cols])
    x_valid = x_scaler.transform(valid_data[input_cols])
    x_test = x_scaler.transform(test_data[input_cols])
    
    y_train = y_scaler.fit_transform(train_data[output_cols])
    y_valid = y_scaler.transform(valid_data[output_cols])
    y_test = y_scaler.transform(test_data[output_cols])
    
    train_split = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float()
    )
    train_loader = DataLoader(train_split, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=0)

    valid_split = TensorDataset(
        torch.from_numpy(x_valid).float(),
        torch.from_numpy(y_valid).float()
    )
    valid_loader = DataLoader(valid_split, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=0)
    

    model = load_mlp_model(input_size, middle_size, output_size, dropout).to(device)
    init_weights(model, init_type = weight_init, verbose = verbose)
    
    # Set the training loss
    training_loss = torch.nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(),
                             lr = learning_rate,
                             weight_decay = L2_penalty,
                             eps = 1e-7,
                             betas = (0.9, 0.999),
                             amsgrad = False)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience = lr_patience, 
            verbose = verbose,
            min_lr = 1.0e-13
    )
    
    results_dict = defaultdict(list)
    
    if verbose:
        my_iter = tqdm.tqdm(range(epochs),
                            total = epochs,
                            leave = True)
    else:
        my_iter = list(range(epochs))
    

    for epoch in my_iter:

        # Train in batch mode
        model.train()

        train_loss = []
        for k, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = training_loss(y.to(device), model(x.to(device)))
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss += L1_penalty * l1_norm
            loss += L2_penalty * l2_norm
            train_loss.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if not np.isfinite(loss.item()):
                print(x)
                raise

        # Validate 
        model.eval()
        with torch.no_grad():

            # Validate in batch mode
            valid_loss = []
            for k, (x, y) in enumerate(valid_loader):
                pred = model(x.to(device))
                loss = torch.nn.L1Loss()(y.to(device), pred)
                valid_loss.append(loss.item())

        if not np.isfinite(np.mean(valid_loss)):
            break

        results_dict["epoch"].append(epoch)
        results_dict["train_loss"].append(np.mean(train_loss))
        results_dict["val_mae"].append(np.mean(valid_loss))
        results_dict["lr"].append(optimizer.param_groups[0]['lr'])

        # Save the dataframe to disk
        df = pd.DataFrame.from_dict(results_dict).reset_index()
        df.to_csv(f"{save_loc}/training_log.csv", index = False)

        print_str = f'Epoch {epoch} train_loss {results_dict["train_loss"][-1]:4f}'
        print_str += f' val_mae {results_dict["val_mae"][-1]:4f}'
        print_str += f' lr {results_dict["lr"][-1]}'
        if verbose:
            my_iter.set_description(print_str)
            my_iter.refresh()

        # anneal the learning rate using just the box metric
        lr_scheduler.step(results_dict[metric][-1])

        if results_dict[metric][-1] == min(results_dict[metric]):
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': min(results_dict[metric])
            }
            torch.save(state_dict, f"{save_loc}/mlp.pt")

        # Stop training if we have not improved after X epochs
        best_epoch = [i for i,j in enumerate(results_dict[metric]) if j == min(results_dict[metric])][0]
        offset = epoch - best_epoch
        if offset >= stopping_patience:
            break

    best_epoch = [i for i,j in enumerate(results_dict[metric]) if j == min(results_dict[metric])][0]
    
    results = {
        "epoch": best_epoch,
        "train_loss": results_dict["train_loss"][best_epoch],
        "val_mae": results_dict["val_mae"][best_epoch]
    }
    
    if evaluate:
        
        model = load_mlp_model(input_size, middle_size, output_size, dropout).to(device)
        
        checkpoint = torch.load(
            f"{save_loc}/mlp.pt",
            map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        x_splits = [x_train, x_valid, x_test]
        splits = [train_data, valid_data, test_data]
        
        for x,y in zip(x_splits, splits):
        
            with torch.no_grad():
                _input = torch.from_numpy(x).float()
                y_pred = model(_input.to(device))
            y["y_pred"] = y_scaler.inverse_transform(y_pred.cpu().numpy())
        
        with open(f"{save_loc}/scalers.pkl", "wb") as fid:
            joblib.dump([x_scaler, y_scaler], fid)
            
        train_data.to_csv(f"{save_loc}/train.csv")
        valid_data.to_csv(f"{save_loc}/valid.csv") 
        test_data.to_csv(f"{save_loc}/test.csv")

    return results

    
if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python train_SL.py model.yml [data seed] [model seed]")
        sys.exit()
        
    # ### Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    # ### Load the configuration and get the relevant variables
    config = sys.argv[1]
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
        
    data_seed, model_seed = None, None
    if len(sys.argv) == 4:
        data_seed = int(sys.argv[2])
        model_seed = int(sys.argv[3])
        save_loc = os.path.join(conf["save_loc"], f"{data_seed}_{model_seed}")
        conf["data_seed"] = data_seed
        conf["model_seed"] = model_seed
        conf["save_loc"] = save_loc
        
    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok = True)
    
    if data_seed == None and model_seed == None:
        if not os.path.join(save_loc, "model.yml"):
            shutil.copyfile(config, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)
        
    result = trainer(conf, evaluate = True)
    
    logging.info(f'Result of training:')
    logging.info(f'{result}')