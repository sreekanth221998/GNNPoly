'''
Project: PolyGNN
-------------------------------------------------------------------------------
'''

import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error
                             )
import numpy as np

n_epochs = 50

def save_latextable(df, filename):
    """dataframe to latex"""
    LATEX_TABLE = r'''\documentclass{{standalone}}
                        \usepackage{{booktabs}}
                        \usepackage{{multirow}}
                        \usepackage{{graphicx}}
                        \usepackage{{xcolor,colortbl}}
                        \begin{{document}}
                        {}
                        \end{{document}}
                        '''
    a_str = df.style.to_latex()
    with open(filename, 'w') as f:
        f.write(LATEX_TABLE.format(a_str))
    
for split in ['Random_split', 'Extrapolation_solute', 'Extrapolation_solvent']:
    print('='*70)
    print(split)
    print('='*70)
    if split == 'Random_split':
        n_folds = 10
    else:
        n_folds = 5
    for spec in ['MN', 'MW', 'PDI']:
        for mode in ['train', 'test']:
            print('='*50)
            maes, r2s, rmses, mapes = [], [], [], []
            for f in range(n_folds):  
                model_name = 'GHGNN_epochs_'+str(n_epochs)+'_fold_'+str(f)
            
                df = pd.read_csv(split + '/' + spec+ '/'+ model_name+'/'+mode+'_pred.csv')
                y_true = df['log_omega'].to_numpy()
                y_pred = df[model_name].to_numpy()
                
                maes.append(mean_absolute_error(y_true, y_pred))
                r2s.append(r2_score(y_true, y_pred))
                rmses.append(mean_squared_error(y_true, y_pred)**0.5)
                mapes.append(mean_absolute_percentage_error(y_true, y_pred)*100)
                
            df_res = pd.DataFrame({
                'MAE': maes,
                'R2': r2s,
                'RMSE': rmses,
                'MAPE': mapes
                })
            
            print('MAE : ', np.mean(maes))
            print('R2  : ', np.mean(r2s))
            print('RMSE: ', np.mean(rmses))
            print('MAPE: ', np.mean(mapes))
            
            df_res.to_csv(split + '/' + spec+'/performance.csv', index=False)
            
            save_latextable(df_res, split + '/' + spec+ '/report_performance_'+mode+'.txt')
        

