'''
Project: PolyGNN
                    GHGNN with extra global feature                    
-------------------------------------------------------------------------------
'''

# Scientific computing
import pandas as pd

# RDKiT
from rdkit import Chem

# Internal utilities
from GNNGH_T_architecture import GNNGH_T, count_parameters
from utilities.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features, n_bond_features
from utilities.Train_eval import train, eval, MAE
from utilities.save_info import save_train_traj

# External utilities
from tqdm import tqdm
#tqdm.pandas()
from collections import OrderedDict
import copy
import time
import os

# Pytorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
from torch.cuda.amp import GradScaler

   
def train_GNNGH_T(df, model_name, hyperparameters, spec, split):
    
    path = os.getcwd()
    path = path + '/' + split + '/' + spec + '/' + model_name
    
    if not os.path.exists(path):
        os.makedirs(path)

    # Open report file
    report = open(path+'/Report_training_' + model_name + '.txt', 'w')
    def print_report(string, file=report):
        print(string)
        file.write('\n' + string)

    print_report(' Report for ' + model_name)
    print_report('-'*50)
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    train_index = df.index.tolist()
    
    target = 'log_omega'
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, 
                                                 target, extra='log_'+spec)
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']
    lr          = hyperparameters['lr']
    n_epochs    = hyperparameters['n_epochs']
    batch_size  = hyperparameters['batch_size']
    
    start       = time.time()
    
    # Data loaders
    train_loader = get_dataloader_pairs_T(df, 
                                          train_index, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size, 
                                          shuffle=True, 
                                          drop_last=True)
    
    # Model
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3 # ap, bp, topopsa
    model    = GNNGH_T(v_in, e_in, u_in, hidden_dim)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    
    print('    Number of model parameters: ', count_parameters(model))
    
    # Optimizer                                                           
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)     
    task_type = 'regression'
    scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
    
    # Mixed precision training with autocast
    if torch.cuda.is_available():
        pbar = tqdm(range(n_epochs))
        scaler = GradScaler()
    else:
        pbar = tqdm(range(n_epochs))
        scaler=None
    
    # To save trajectory
    mae_train = []; train_loss = []
    
    for epoch in pbar:
        stats = OrderedDict()
        
        # Train
        stats.update(train(model, device, train_loader, optimizer, task_type, stats, scaler))
        # Evaluation
        stats.update(eval(model, device, train_loader, MAE, stats, 'Train', task_type))
        # Scheduler
        scheduler.step(stats['MAE_Train'])
        # Save info
        train_loss.append(stats['Train_loss'])
        mae_train.append(stats['MAE_Train'])
        pbar.set_postfix(stats) # include stats in the progress bar
    
    print_report('-'*30)
    print_report('Training MAE   : '+ str(mae_train[-1]))
    
    # Save training trajectory
    df_model_training = pd.DataFrame(train_loss, columns=['Train_loss'])
    df_model_training['MAE_Train']  = mae_train
    save_train_traj(path, df_model_training, valid=False)
    
    # Save best model
    final_model = copy.deepcopy(model.state_dict())
    torch.save(final_model, path + '/' + model_name + '.pth')
    
    end       = time.time()
    
    print_report('\nTraining time (min): ' + str((end-start)/60))
    report.close()
   

n_epochs = 50

hyperparameters_dict = {'hidden_dim'  : 113,
                        'lr'          : 0.0002532501358651798,
                        'n_epochs'    : n_epochs,
                        'batch_size'  : 32
                        }

for split in ['Random_split', 'Extrapolation_solute', 'Extrapolation_solvent']:
    print('='*70)
    print(split)
    print('='*70)
    if split == 'Random_split':
        n_folds = 10
    else:
        n_folds = 5
    for spec in ['MN', 'MW', 'PDI']:
        print('='*70)
        print(spec)
        print('='*70)
        for f in range(n_folds):    
            df = pd.read_csv('../../data/'+split+'/'+spec+'/fold_'+str(f)+'_train.csv')
            train_GNNGH_T(df, 'GHGNN_epochs_'+str(n_epochs)+'_fold_'+str(f), 
                          hyperparameters_dict,
                          spec, split)
