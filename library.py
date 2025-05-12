import math
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import shap
import numpy as np
from numpy import log as ln
import random
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter


# Define the Dataframe object
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, dtype=torch.float32):
        self.data = data
        self.target = target
        
        self.dtype = dtype

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        values = self.data.iloc[idx]
        target = self.target.iloc[idx]
        
        values = torch.tensor(values, dtype = self.dtype)
        target = torch.tensor(target, dtype = self.dtype)
        
        return values, target

    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, restore_best_params=False, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_parameters = None
        self.restore_best_params = restore_best_params
        self.epoch = 1
        self.best_epoch = 1
        self.verbose = verbose

    def early_stop(self, validation_loss, model):
        if self.restore_best_params and self.epoch==1:
            self.best_parameters = model.state_dict()
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_parameters = copy.deepcopy(model.state_dict())
            self.best_epoch = self.epoch
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_params:
                    if self.verbose:
                        print(f"Restoring best weights at epoch {self.best_epoch} (loss:{self.min_validation_loss:>8f}) ...")
                    model.load_state_dict(self.best_parameters)
                return True
        self.epoch += 1
        return False
    

class Constraint:
    def __init__(self, index:int, effect_size:float, weight:float, step_size:float, variable_name:str=None):
        self.name = variable_name
        self.index = index
        self.effect_size = effect_size
        self.step_size = step_size
        self.weight = weight
    
    def __str__(self):
        return f"Constraint(Name: {self.name}, Index: {self.index}, Effect Size: {round(self.effect_size, 2)}, Weight: {round(self.weight, 2)}, Step: {round(self.step, 2)})"
 

class ConstraintsEnsemble:
    def __init__(self):
        self.constraints = []

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)

    def remove_constraint(self, index: int):
        # Removes a constraint by its index in the ensemble
        self.constraints = [c for i, c in enumerate(self.constraints) if i != index]

    def find_constraint_by_name(self, name: str):
        # Finds a constraint by its name
        for constraint in self.constraints:
            if constraint.name == name:
                return constraint
        return None
    
    def find_constraint_by_index(self, index: int):
        for constraint in self.constraints:
            if constraint.index == index:
                return constraint
        return None

    def get_constraints(self):
        return self.constraints

    def __str__(self):
        constraint_lines = []
        for c in self.constraints:
            line = f"Constraint Name: {c.name:<25} | Index: {c.index:<5} | Effect Size: {round(c.effect_size, 4):<10} | Step size: {c.step_size:<10} | Weight: {c.weight:<10}"
            constraint_lines.append(line)
        return "\n".join(constraint_lines)

# MLP for logistic output (binary classification problems)     
class LogMLP(nn.Module):
    def __init__(self, n_features, initial_size_multiplier, n_hidden_layers, size_decrease_multiplier):
        super().__init__()
        self.dim = n_features
        
        # input layer
        hidden_size = int(n_features * initial_size_multiplier)
        self.input_layer = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.LeakyReLU(),
            # nn.Dropout(dropout_rate),
        )
        # hidden layers
        self.hidden_layers = nn.Sequential()
        for i in range(n_hidden_layers):
            self.hidden_layers.add_module(f"linear_{i+1}", nn.Linear(hidden_size, max(2, int(hidden_size*size_decrease_multiplier))))
            hidden_size = max(2, int(hidden_size * size_decrease_multiplier))
            # print(hidden_size)
            self.hidden_layers.add_module(f"LK_{i+1}", nn.LeakyReLU())
            # self.hidden_layers.add_module(f"dropout_{i+1}", nn.Dropout(dropout_rate))
        # output layer
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x)
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        o = self.sigmoid(x)
        assert(x.size()[0] > 0 and x.size()[1] > 0)
        return o.reshape(x.size()[0])
    

class LogSEANN(LogMLP):
    def __init__(self, n_features: int,
                 initial_size_multiplier: float,
                 n_hidden_layers: int,
                 size_decrease_multiplier: float,
                 constraints: ConstraintsEnsemble,
                 rescale_loss_weight: float):
        super().__init__(n_features, initial_size_multiplier, n_hidden_layers, size_decrease_multiplier)
        self.constraints = constraints
        self.rescale_loss_weight = rescale_loss_weight
        self.n_constraints = len(constraints.get_constraints())
 

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        # Ensure output is at least two-dimensional
        return output.unsqueeze(-1)

# puts seeds to pytorch env. to ensure determinism.
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    
# puts seed for pytorch dataloader worker    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def make_dataloader(features:pd.DataFrame, target:pd.DataFrame, dtype=torch.float32):
    g = torch.Generator()
    g.manual_seed(0)
    customdf = CustomDataset(features, target, dtype=dtype)
    dataloader = DataLoader(customdf, batch_size=64, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g)    # suffle = True is non deterministic
    return dataloader

def split_and_standardize(features, target, train_ratio, validation_ratio, test_ratio, categorical_cols_list):
    assert(train_ratio + validation_ratio + test_ratio == 1)
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 1 - train_ratio, random_state=42, shuffle=True)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=0, shuffle=True)
    
    # Standardize the data
    X_train_cpy, X_valid_cpy, X_test_cpy = X_train.copy(deep=True), X_valid.copy(deep=True), X_test.copy(deep=True)
    columns_to_standardize = [col for col in list(features.columns) if col not in categorical_cols_list]
    train_mean = X_train[columns_to_standardize].mean()
    train_std = X_train[columns_to_standardize].std()
    X_train_cpy[columns_to_standardize] = (X_train[columns_to_standardize] - train_mean) / train_std
    X_valid_cpy[columns_to_standardize] = (X_valid[columns_to_standardize] - train_mean) / train_std
    X_test_cpy[columns_to_standardize] = (X_test[columns_to_standardize] - train_mean) / train_std
    
    return (X_train_cpy, y_train), (X_valid_cpy, y_valid), (X_test_cpy, y_test)

# perform training
# def train_model(model, loss_function, optimizer, scheduler, train_data_loader, valid_data_loader, device, epochs=200, patience=10, min_delta=0.001, verbose=True, 
def train_model(model, loss_function, optimizer, train_data_loader, valid_data_loader, device, epochs=200, patience=10, min_delta=0.001, verbose=True, 
                beta:torch.Tensor=None, h:float=None, plot:bool=True):
    # handle expeptions
    if beta is not None:
        if h is None:
            raise ValueError("Non null beta must be paired with non null h")
    if h is not None:
        if beta is None:
            raise ValueError("Non null h must be paired with non null beta")
    
    # training loop
    # torch.autograd.set_detect_anomaly(True)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, restore_best_params=True, verbose=verbose)
    array_tloss = []
    array_vloss = []
    for t in range(epochs):
        if verbose:
            print(f"Epoch {t+1}\n-------------------------------")
        if beta is not None and h is not None: # tracking custom loss in agnostic model
            train_one_epoch_tracking()
        else: # basic model (no custom loss tracking)
            # t_loss = train_one_epoch(train_data_loader, model, loss_function, optimizer, scheduler, device, verbose)
            t_loss = train_one_epoch(train_data_loader, model, loss_function, optimizer, device, verbose)
            array_tloss.append(t_loss)
            v_loss = validate_one_epoch(valid_data_loader, model, loss_function, device, verbose)
            array_vloss.append(v_loss)
            if early_stopper.early_stop(v_loss, model):
                break
            # scheduler.step(v_loss)
    if plot:
        plot_training_losses(array_tloss, array_vloss, "Total loss")

        
# define traning and testing functions for the base DNN.
def train_one_epoch(dataloader, model, loss_fn, optimizer, device, verbose=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_training_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        main_pred = model(X)
        optimizer.zero_grad()
        # print(main_pred)
        # print(y)
        loss = loss_fn(main_pred, y.reshape(main_pred.size()[0]))

        # Backpropagation
        loss.backward()
        optimizer.step()
        # scheduler.step()
        # plot_grad_flow(model.named_parameters())
        
        total_training_loss += loss

    total_training_loss /= num_batches
    loss = total_training_loss.item()
    if verbose:
        print(f"Epoch Mean Training Loss: {total_training_loss:>7f}")
    return total_training_loss.detach().numpy().copy()

def validate_one_epoch(dataloader, model, loss_fn, device, verbose=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    test_loss = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y.reshape(pred.size()[0]))
            
            test_loss += loss
            
    test_loss /= num_batches
    if verbose:
        print(f"Test Error: \n Total Error: {(test_loss):>8f}% \n")
    return test_loss


def predict(dataloader, model, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    pred_list = []
    
    with torch.no_grad():
        for X, y in dataloader:            
            X = X.to(device)
            pred = model(X)
            pred_list.append(pred)
            
    final_pred = torch.cat(pred_list, 0)
    return final_pred.numpy()


# define traning and testing functions for the custom model
def train_one_epoch_custom(dataloader, model, loss_fn, optimizer, device, step:int, verbose=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_aggregated_loss, total_predictive_task_loss, total_individual_tasks = 0, 0, [0 for i in range(model.n_constraints)]
    
    # writer = SummaryWriter()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        main_task_loss, total_loss, tasks_losses = loss_fn(model, X, y, mode="train")
        
        # Track gradient flow on tensorboard
        # for name, param in model.named_parameters():
        #     # print(name, param, step)
        #     writer.add_histogram(name + '/grad', param.grad, global_step=step)

        # Backpropagation
        # total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        # plot_grad_flow(model.named_parameters())
        
        total_aggregated_loss += total_loss
        total_predictive_task_loss += main_task_loss
        for i in range(model.n_constraints):
            total_individual_tasks[i] += tasks_losses[i]
    
    total_aggregated_loss /= num_batches
    total_predictive_task_loss /= num_batches
    total_individual_tasks = [x/num_batches for x in total_individual_tasks]
    if verbose:
        print(f"Epoch Mean Training Loss: {total_aggregated_loss:>7f}, Predictive Loss: {total_predictive_task_loss:>7f}")
        
    return total_aggregated_loss.detach().numpy().copy(), total_predictive_task_loss.detach().numpy().copy(), [task.detach().numpy().copy() for task in total_individual_tasks]

    
def validate_one_epoch_custom(dataloader, model, loss_fn, device, verbose=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    total_aggregated_loss, total_predictive_task_loss, total_individual_tasks = 0, 0, [0 for i in range(model.n_constraints)]
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            main_task_loss, total_loss, tasks_losses = loss_fn(model, X, y, mode="eval")
            
            total_predictive_task_loss += main_task_loss
            total_aggregated_loss += total_loss
            for i in range(model.n_constraints):
                total_individual_tasks[i] += tasks_losses[i]
            
    total_aggregated_loss /= num_batches
    total_predictive_task_loss /= num_batches
    total_individual_tasks = [x/num_batches for x in total_individual_tasks]
    # print(total_individual_tasks)
    if verbose:
        print(f"Test Error: \n Total Error: {total_aggregated_loss:>8f}, Predictive Error: {total_predictive_task_loss:>8f} \n")
    return total_aggregated_loss, total_predictive_task_loss, total_individual_tasks

def plot_training_losses(train_loss: list, val_loss: list, title: str):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle(title)
    line1, = ax.plot(np.arange(len(train_loss)), train_loss, c='b', label='Training loss')
    line2, = ax.plot(np.arange(len(val_loss)), val_loss, c='g', label='Validation loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean error")
    ax.legend(handles=[line1, line2])
    plt.show()
    
    
def columns_with_low_value_counts(df, threshold):
    low_count_columns = []
    for col in df.columns:
        value_counts = df[col].value_counts()
        if len(value_counts) <= threshold:
            low_count_columns.append(col)
    return low_count_columns


def train_custom_model(model,
                       loss_functions, 
                       optimizer,
                       train_data_loader,
                       valid_data_loader,
                       device, epochs=200,
                       early_stopping=True,
                       patience=10,
                       min_delta=0.001,
                       verbose=True,
                       plot:bool=True):
    
    # torch.autograd.set_detect_anomaly(True)
    if early_stopping:
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, restore_best_params=True, verbose=verbose)
    array_tloss_tot, array_tloss_pred, array_tloss_task = [], [], []
    array_vloss_tot, array_vloss_pred, array_vloss_task = [], [], []
        
    for t in range(epochs):
        if verbose and t%100==0:
            print(f"Training: Epoch {t+1}\n-------------------------------")
        
        # print(alphas)
        train_loss_tot, train_loss_pred, train_task_losses = train_one_epoch_custom(train_data_loader, model, loss_functions, optimizer, device, t, verbose)
        array_tloss_tot.append(train_loss_tot)
        array_tloss_pred.append(train_loss_pred)
        array_tloss_task.append(train_task_losses)
        
        
        val_loss_tot, val_loss_pred, val_task_losses = validate_one_epoch_custom(valid_data_loader, model, loss_functions, device, verbose)
        array_vloss_tot.append(val_loss_tot)
        array_vloss_pred.append(val_loss_pred)
        array_vloss_task.append(val_task_losses)
        
        if early_stopping:
            if early_stopper.early_stop(val_loss_tot, model):
                break
    
    transpose_tloss_task = [list(i) for i in zip(*array_tloss_task)]
    transpose_vloss_task = [list(i) for i in zip(*array_vloss_task)]
    if plot:
        plot_training_losses(array_tloss_tot, array_vloss_tot, "Total loss")
        plot_training_losses(array_tloss_pred, array_vloss_pred, "Predictive loss")
        print("individual tasks:")
        for i in range(model.n_constraints):
            name = model.constraints.get_constraints()[i].name
            plot_training_losses(transpose_tloss_task[i], transpose_vloss_task[i], f"Task loss {name}")

            
def alternative_custom_loss_or(model, inputs, target, mode:list="train"):
    assert(mode=="train" or mode=="eval")
    
    binary_cross_entropy_lossfn = torch.nn.BCELoss()
        
    main_pred = model(inputs)
    bceloss = binary_cross_entropy_lossfn(main_pred, target)
    main_loss = model.rescale_loss_weight * bceloss

    aggregated_loss = main_loss.clone()
    task_losses = []
    for c in model.constraints.get_constraints():
        step_tensor = torch.tensor([[c.step_size if i==c.index else 0 for i in range(inputs.shape[1])] for j in range(inputs.shape[0])], dtype=torch.float32, requires_grad=False)
        x_prime = torch.add(inputs.clone(), step_tensor) # add h to input
        dpred = model(x_prime)
        
        determination_coeff = c.effect_size * (torch.sigmoid(torch.mul(inputs.clone()[:, c.index], c.effect_size)) * (1 - torch.sigmoid(torch.mul(inputs[:, c.index], c.effect_size)))) 
        constraint_term = determination_coeff * c.step_size
                      
        reconstructed_targ = torch.sub(dpred, constraint_term)
        
        added_loss = torch.mean((reconstructed_targ - main_pred)**2)
        scaled_added_loss = torch.mul(added_loss, c.weight)
        aggregated_loss = torch.add(aggregated_loss, scaled_added_loss)
        task_losses.append(scaled_added_loss)
    
    if mode == "train":
        aggregated_loss.backward(create_graph=False)
    
    return main_loss, aggregated_loss, task_losses


def sigmoid(x, min=0, max=1):
    x = np.array(x)  # Ensure x is a NumPy array to handle both single values and lists
    result = np.where(x < 0, 
                      min + (max-min) * (np.exp(x) / (1+np.exp(x))),
                      min + (max-min) * (1 / (1 + np.exp(-1 * x))))
    return result

def compute_scaling_weights_binary_classif(inputs, target, indexes, effect_sizes, step_sizes, naive_pred=0.5):
    assert(len(indexes) == len(effect_sizes) == len(step_sizes))
    
    initial_loss_amplitudes = []
    predictive_initial_loss_amplitude = np.mean(-1 * (target * np.log(naive_pred) + (1-target) * np.log(1 - naive_pred)))
    initial_loss_amplitudes.append(predictive_initial_loss_amplitude)
    
    for i in range(len(indexes)):
        local_derivative = effect_sizes[i] * (sigmoid(-inputs.copy().iloc[:, indexes[i]]*effect_sizes[i]) * (1 - sigmoid(-inputs.copy().iloc[:, indexes[i]]*effect_sizes[i])))
        task_initial_loss_amplitude = (np.mean(local_derivative)*step_sizes[i])**2
        initial_loss_amplitudes.append(task_initial_loss_amplitude)
        
    scaling_weights = [1/x for x in initial_loss_amplitudes]
    return scaling_weights


def create_constraints(indexes:list, effect_sizes:list, step_sizes:list, weights:list, name_list:list, verbose:bool=True):
    assert(len(indexes) == len(effect_sizes) == len(step_sizes) == len(weights) == len(name_list))
    ce = ConstraintsEnsemble()
    for i in range(len(indexes)):
        c = Constraint(index=indexes[i], effect_size=effect_sizes[i], weight=weights[i], step_size=step_sizes[i], variable_name=name_list[i])
        ce.add_constraint(constraint=c)
    if verbose:
        print(ce)
    return ce

def ln_normalize(X:list):
    for x in X: # each elements in X must be positive integers
        assert(isinstance(x, int) and x > 0)
    # return [coeff*np.emath.logn(base, x)/np.sum(coeff*np.emath.logn(base, X)) for x in X]
    return [np.log(x)/np.sum(np.log(X)) for x in X]

def linear_normalize(X:list):
    for x in X: # each elements in X must be positive integers
        assert(isinstance(x, int) and x > 0)
    return [x/np.sum(X) for x in X]


def compute_log_scaling_weights(literature_sample_sizes:dict, data_sample_size:int, data_n_variables:int, data_importance_mult:int=1):
    data_w = data_importance_mult * int(data_sample_size*data_n_variables)
    weights = ln_normalize([data_w] + list(literature_sample_sizes.values()))
    res = {"data": weights[0]}
    for i, key in enumerate(literature_sample_sizes.keys()):
        res[key] = weights[i+1]
    # print(res)
    return res


def compute_linear_scaling_weights(literature_sample_sizes:dict, data_sample_size:int, data_n_variables:int, data_importance_mult:int=1):
    data_w = data_importance_mult * int(data_sample_size*data_n_variables)
    weights = linear_normalize([data_w] + list(literature_sample_sizes.values()))
    res = {"data": weights[0]}
    for i, key in enumerate(literature_sample_sizes.keys()):
        res[key] = weights[i+1]
    # print(res)
    return res


def plot_deepshap_response_function(shapley_values:np.ndarray, inputs: pd.DataFrame, variable_name: str, jitter:float=0.2, figsize=(8, 2)):
    actual_values = inputs[variable_name]
    jittered_values = actual_values + np.random.normal(0, jitter, size=actual_values.shape)
    
    # Initialize a figure for plotting
    plt.figure(figsize=figsize)
    shapley_values = pd.DataFrame(shapley_values, columns=inputs.columns)
    shap_values_for_var = shapley_values[variable_name]    
    
    plt.scatter(jittered_values, shap_values_for_var, alpha=0.5)
    plt.title(f"SHAP Values vs. {variable_name}")
    plt.xlabel(variable_name)
    plt.ylabel("SHAP Value")
    plt.grid(True)
    plt.show()
    
def compare_deepshap_response_function(shapley_values_1:np.ndarray,
                                       shapley_values_2:np.ndarray,
                                       true_shapley_values,
                                       label_1:str,
                                       label_2:str,
                                       inputs: pd.DataFrame, variable_name: str, jitter:float=0.2):
    actual_values = inputs[variable_name]
    jittered_values = actual_values + np.random.normal(0, jitter, size=actual_values.shape)
    
    # Initialize a figure for plotting
    plt.figure(figsize=(10, 3))
    # true_shapley_values = pd.DataFrame(true_shapley_values.values, columns=true_shapley_values.feature_names)
    shapley_values_1 = pd.DataFrame(shapley_values_1, columns=inputs.columns)
    shapley_values_2 = pd.DataFrame(shapley_values_2, columns=inputs.columns)
    # print(true_shapley_values.shape, shapley_values_1.shape)
    shap_values_for_truth = true_shapley_values[variable_name]
    shap_values_for_var1 = shapley_values_1[variable_name]
    shap_values_for_var2 = shapley_values_2[variable_name]  
    
    plt.scatter(jittered_values, shap_values_for_truth, alpha=0.5, c='g', label="Truth")
    plt.scatter(jittered_values, shap_values_for_var1, alpha=0.5, c='r', label=label_1)
    plt.scatter(jittered_values, shap_values_for_var2, alpha=0.5, c='b', label=label_2)
    plt.legend(title='Model', loc='best')
    plt.title(f"SHAP Values vs. {variable_name}")
    plt.xlabel(variable_name)
    plt.ylabel("SHAP Value")
    plt.grid(True)
    plt.show()
    
def compare_deepshap_dashboard(shapley_values_1: np.ndarray,
                               shapley_values_2: np.ndarray,
                               label_1: str,
                               label_2: str,
                               inputs: pd.DataFrame,
                               feature_names: list,
                               true_shapley_values: dict=None,
                               jitter: float = 0.2,
                               layout_n_cols: int=3,
                               save_file_name:str=None):
    num_features = len(feature_names)
    if num_features % layout_n_cols != 0:
        rows = ((num_features + 1) + layout_n_cols - 1) // layout_n_cols
    else:
        rows = (num_features + layout_n_cols - 1) // layout_n_cols
    
    plt.figure(figsize=(5 * layout_n_cols, 4 * rows))
    for i, variable_name in enumerate(feature_names, 1):
        plt.subplot(rows, layout_n_cols, i)
        actual_values = inputs[variable_name]
        jittered_values = actual_values + np.random.normal(0, jitter, size=actual_values.shape)
        
        shapley_values_1_df = pd.DataFrame(shapley_values_1, columns=inputs.columns)
        shapley_values_2_df = pd.DataFrame(shapley_values_2, columns=inputs.columns)
        
        shap_values_for_var1 = shapley_values_1_df[variable_name]
        shap_values_for_var2 = shapley_values_2_df[variable_name]
        plt.scatter(jittered_values, shap_values_for_var1, alpha=0.4, c='r', label=label_1, s=15)
        plt.scatter(jittered_values, shap_values_for_var2, alpha=0.4, c='b', label=label_2, s=15)
        
        if true_shapley_values is not None and variable_name in true_shapley_values.keys():
            shap_values_for_truth = true_shapley_values[variable_name]
            plt.scatter(jittered_values, shap_values_for_truth, alpha=0.4, c='g', label="Literature", s=10)
        
        if num_features % layout_n_cols == 0 and i % layout_n_cols == 1:  # Add legend to the first subplot for clarity
            plt.legend(title='Source', loc='best', markerscale=2)
        
        if variable_name == 'smoking_intensity:_exsmoker': # better name without redoing the full analysis
            variable_name = 'smoking_intensity:_exsmoker_unknown_dur'
        plt.title(f"{str(variable_name).replace('_', ' ')}")
        plt.xlabel(str(variable_name).replace('_', ' '))
        plt.ylabel("SHAP Value")
        
    # Add an extra subplot for the legend at the end.
    if num_features % layout_n_cols != 0:
        plt.subplot(rows, layout_n_cols, num_features + 1)
        plt.axis('off')
        plt.scatter([], [], alpha=0.4, c='r', label=label_1, s=50)
        plt.scatter([], [], alpha=0.4, c='b', label=label_2, s=50)
        if true_shapley_values is not None:
            plt.scatter([], [], alpha=0.4, c='g', label="Literature", s=50)
        plt.legend(title='Source', loc='center', ncol=3, fontsize='large', markerscale=1.8, title_fontsize='x-large')
        
    plt.tight_layout()
    if save_file_name is not None:
        plt.savefig(f"../../../results/GCAT/{save_file_name}.png")
    plt.show()