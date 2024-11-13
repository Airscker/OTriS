'''
Author: airscker
Date: 2024-11-12 16:24:52
LastEditors: airscker
LastEditTime: 2024-11-13 00:59:51
Description: NULL

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import matplotlib.pyplot as plt
import netket as nk
from netket.exact import lanczos_ed
from ._base import _Base_System

class Liouvillian_System(_Base_System):
    def __init__(self,
                Lattice_length:int=4,
                Lattice_dim:int=2,
                PBC:bool=False,
                Spin:float=1/2,
                Coupling:float=1,
                Field_tranverse:float=2):
        super().__init__()
        self.Lattice_length=Lattice_length
        self.Lattice_dim=Lattice_dim
        self.PBC=PBC
        self.Spin=Spin
        self.Coupling=Coupling
        self.Field_tranverse=Field_tranverse
        self.init_system()
    def _hamiltonian(self,hilbert_space):
        ha = nk.operator.LocalOperator(hilbert_space)
        j_ops = []
        obs_sx = nk.operator.LocalOperator(hilbert_space)
        obs_sy = nk.operator.LocalOperator(hilbert_space, dtype=complex)
        obs_sz = nk.operator.LocalOperator(hilbert_space)
        for i in range(self.Lattice_length):
            ha += (self.Coupling / 2.0) * nk.operator.spin.sigmax(hilbert_space, i)
            ha += (
                (self.Field_tranverse / 4.0)
                * nk.operator.spin.sigmaz(hilbert_space, i)
                * nk.operator.spin.sigmaz(hilbert_space, (i + 1) % self.Lattice_length)
            )
            j_ops.append(nk.operator.spin.sigmam(hilbert_space, i))
            obs_sx += nk.operator.spin.sigmax(hilbert_space, i)
            obs_sy += nk.operator.spin.sigmay(hilbert_space, i)
            obs_sz += nk.operator.spin.sigmaz(hilbert_space, i)
        self.Observable = {"Sx": obs_sx, "Sy": obs_sy, "Sz": obs_sz}
        self.Hamiltonian=nk.operator.LocalLiouvillian(ha, j_ops)
        self.Hilbert_space=self.Hamiltonian.hilbert

    def init_system(self):
        self.Lattice_graph=nk.graph.Hypercube(length=self.Lattice_length, n_dim=self.Lattice_dim, pbc=self.PBC)
        self.Hilbert_space=nk.hilbert.Spin(s=self.Spin, N=self.Lattice_graph.n_nodes)
        self._hamiltonian(self.Hilbert_space)

        self.str_repr=f'{self.__class__.__name__}\n\tLattice_length={self.Lattice_length}\n\tLattice_dim={self.Lattice_dim}\n\tPBC={self.PBC}\n\tSpin={self.Spin}\n\tCoupling={self.Coupling}\n\tHilbert_space={self.Hilbert_space}\n\tHamiltonian={self.Hamiltonian}\nLattice_graph={self.Lattice_graph}'
    def eigen_energies(self, n_eigen:int=4):
        return None

    def plot(self,log_data,workdir):
        try:
            obs_list=['Sx', 'Sy', 'Sz']
            print(f'Plotting observables: {obs_list}')
            for obs_name, obs_data in log_data.items():
                if obs_name not in obs_list:
                    continue
                iters = obs_data['iters']
                mean_real = obs_data['Mean']['real']
                mean_imag = obs_data['Mean'].get('imag', [0] * len(mean_real))  # 如果没有虚部，默认为0

                # Plot Real Part
                plt.figure(figsize=(10, 6))
                plt.plot(iters, mean_real, label=f'{obs_name} real')
                plt.xlabel('iter')
                plt.ylabel('mean')
                plt.title(f'{obs_name} real change with iter')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(workdir,f'{obs_name}_real.png'))
                plt.close()
                print(f'{obs_name}_real.png saved in {workdir}')
                # If Imaginary Part exists, plot it
                
                plt.figure(figsize=(10, 6))
                plt.plot(iters, mean_imag, label=f'{obs_name} imag', color='orange')
                plt.xlabel('iter')
                plt.ylabel('mean')
                plt.title(f'{obs_name} imag change with iter')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(workdir,f'{obs_name}_imag.png'))
                plt.close()
                print(f'{obs_name}_imag.png saved in {workdir}')
        except:
            print('Failed to obseravbles, please check the log data')

    def __str__(self):
        return self.str_repr
    def __repr__(self):
        return self.str_repr