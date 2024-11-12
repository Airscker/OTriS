'''
Author: airscker
Date: 2024-10-29 01:06:10
LastEditors: airscker
LastEditTime: 2024-11-12 18:56:42
Description: NULL

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''
import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '0'

import argparse
import matplotlib.pyplot as plt
import netket as nk
from netket.driver import AbstractVariationalDriver
from netket.vqs import VariationalState
from utils import (Config,
                   load_state, save_state, load_log)
from model import *
from system import *
from netket.optimizer import *

def main(exp_config:Config):
    global_env = globals()
    system=exp_config._build_system(global_env)
    print(system)
    workdir=exp_config.workdir
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    exp_config.move_config(save_path=os.path.join(workdir, 'config.py'))
    
    model=exp_config._build_model(global_env)
    print(model)
    sampler = nk.sampler.MetropolisLocal(system.Hilbert_space)
    vstate:VariationalState=getattr(nk.vqs, exp_config.vstate_backbone)(sampler, model, **exp_config.vstate_params)
    optimizer = exp_config._build_optimizer(global_env)
    driver_modules=[nk, nk.driver]
    for module in driver_modules:
        if hasattr(module, exp_config.driver_backbone):
            vmc_dirver:AbstractVariationalDriver=getattr(module, exp_config.driver_backbone)
            break
    if exp_config.SR_enabled:
        vmc_dirver=vmc_dirver(system.Hamiltonian, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=exp_config.SR_diag_shift))
    else:
        vmc_dirver=vmc_dirver(system.Hamiltonian, optimizer, variational_state=vstate)
    
    checkpoint_folder=exp_config.load_from
    if checkpoint_folder!='' and os.path.exists(checkpoint_folder):
        vstate.parameters=load_state(os.path.join(checkpoint_folder, "log.mpack"))['params']
    
    vmc_dirver.run(n_iter=exp_config.epochs,
                   save_params_every=exp_config.save_inter,
                   out=os.path.join(workdir, "log"),
                   obs=system.Observable)
    save_state(vstate, os.path.join(workdir, "log.mpack"))
    

    print('Calculating exact Ground State(GS) energy...')
    try:
        E_eigen=system.eigen_energies().min()
        print(f'Exact GS energy: {E_eigen}')
    except:
        print('Failed to calculate exact GS energy')
        E_eigen=None

    try:
        plt.figure(figsize=(10, 6))
        log_data = load_log(os.path.join(workdir, "log.log"))
        iters=log_data['Energy']['iters']
        E_mean=log_data['Energy']['Mean']
        plt.plot(iters,E_mean)
        plt.hlines([E_eigen], xmin=min(iters), xmax=max(iters), color='black', label=f"Exact GS energy: {E_eigen}")
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(workdir, 'Energy.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
        print('Energy convergence plot saved in', os.path.join(workdir, 'Energy.png'))
    except:
        print('Failed to plot energy convergence')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        '-c',
                        default='E:\OneDrive\StonyBrook\QML\OTriS\config\sample_lindblad.py',
                        type=str,
                        help='the path of config file')
    args = parser.parse_args()
    exp_config=Config(args.config)
    print(f"Experiment config:\n{exp_config}")
    main(exp_config)
