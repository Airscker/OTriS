'''
Author: airscker
Date: 2024-10-29 01:06:10
LastEditors: airscker
LastEditTime: 2024-11-22 00:06:19
Description: NULL

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''
import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '0'

import json
import argparse
import matplotlib.pyplot as plt
import netket as nk
from netket.driver import AbstractVariationalDriver
from netket.vqs import VariationalState
from utils import (Config,
                   load_state, save_state, load_log)
from model import *
from system import _Base_System
from netket.optimizer import *

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(exp_config:Config,system:_Base_System=None, idx:int=0):
    global_env = globals()
    # system=exp_config._build_system(global_env)
    # systems=exp_config.systems
    print(system)
    workdir=exp_config.workdir
    _sys_subdir=os.path.join(workdir, "systems", f"system_{idx:03d}")
    create_dir(_sys_subdir)
    system.save_params(os.path.join(_sys_subdir, 'system_params.json'))

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
                   out=os.path.join(_sys_subdir, "log"),
                   obs=system.Observable)
    save_state(vstate, os.path.join(_sys_subdir, "log.mpack"))
    

    print('Calculating exact Ground State(GS) energy...')
    try:
        E_eigen=system.eigen_energies().min()
        print(f'Exact GS energy: {E_eigen}')
    except:
        print('Failed to calculate exact GS energy')
        E_eigen=None

    # log_data = load_log(os.path.join(workdir, "log.log"))
    # system.plot(log_data, workdir)

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
    _coupling_list=[]
    for idx,(_coupling,_sys) in enumerate(exp_config.systems):
        _coupling_list.append(_coupling)
        main(exp_config, _sys, idx)
    _lattice_length=exp_config.system_params['Lattice_length']
    sx_means = []
    sy_means = []
    sz_means = []
    for i in range(len(_coupling_list)):
        log_file=os.path.join(exp_config.workdir, f"systems/system_{i:03d}/log.log")
        with open(log_file, "r") as f:
            data = json.load(f)
            sx_means.append(data["Sx"]["Mean"]["real"][-1] / _lattice_length)
            sy_means.append(data["Sy"]["Mean"]["real"][-1] / _lattice_length)
            sz_means.append(data["Sz"]["Mean"]["real"][-1] / _lattice_length)
        
    # Plot Sx
    plt.figure()
    plt.plot(_coupling_list, sx_means, label="Sx Mean", marker='o')
    plt.xlabel("gp")
    plt.ylabel("Sx Mean (Last Iteration)")
    plt.title("Sx Mean vs gp")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(exp_config.workdir,"Sx_vs_gp.png"))
    plt.show()

    # Plot Sy
    plt.figure()
    plt.plot(_coupling_list, sy_means, label="Sy Mean", marker='o')
    plt.xlabel("gp")
    plt.ylabel("Sy Mean (Last Iteration)")
    plt.title("Sy Mean vs gp")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(exp_config.workdir,"Sy_vs_gp.png"))
    plt.show()

    # Plot Sz
    plt.figure()
    plt.plot(_coupling_list, sz_means, label="Sz Mean", marker='o')
    plt.xlabel("gp")
    plt.ylabel("Sz Mean (Last Iteration)")
    plt.title("Sz Mean vs gp")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(exp_config.workdir,"Sz_vs_gp.png"))
    plt.show()
