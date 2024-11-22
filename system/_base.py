'''
Author: airscker
Date: 2024-11-12 16:51:47
LastEditors: airscker
LastEditTime: 2024-11-21 22:42:01
Description: NULL

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''
import os
import json
import matplotlib.pyplot as plt
from netket.exact import lanczos_ed
from abc import ABCMeta, abstractmethod

class _Base_System(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.Lattice_length=None
        self.Lattice_dim=None
        self.PBC=None
        self.Spin=None
        self.Coupling=None
        self.Field_tranverse=None
        self.Lattice_graph=None
        self.Hilbert_space=None
        self.Hamiltonian=None
        self.Observable=None
    def eigen_energies(self, n_eigen:int=4):
        return lanczos_ed(self.Hamiltonian, k=n_eigen)
    def plot(self,log_data,workdir):
        try:
            iters=log_data['Energy']['iters']
            E_mean=log_data['Energy']['Mean']
            E_eigen=self.eigen_energies().min()
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
    def save_params(self,path):
        _params={}
        for _name in self.__dict__:
            if _name not in ['Lattice_graph','Hilbert_space','Hamiltonian','Observable','str_repr']:
                _params[_name]=self.__dict__[_name]
        with open(path,'w+') as f:
            json.dump(_params,f)