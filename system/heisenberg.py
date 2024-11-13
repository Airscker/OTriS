'''
Author: airscker
Date: 2024-10-30 21:54:01
LastEditors: airscker
LastEditTime: 2024-11-12 17:14:26
Description: NULL

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''

import netket as nk
from netket.exact import lanczos_ed
from ._base import _Base_System

class Heisenberg_System(_Base_System):
    def __init__(self,
                Lattice_length:int=4,
                Lattice_dim:int=2,
                PBC:bool=True,
                Spin:float=1/2,
                Coupling:float=1
                ) -> None:
        super().__init__()
        self.Lattice_length=Lattice_length
        self.Lattice_dim=Lattice_dim
        self.PBC=PBC
        self.Spin=Spin
        self.Coupling=Coupling
        self.init_system()

    def init_system(self):
        self.Lattice_graph=nk.graph.Hypercube(length=self.Lattice_length, n_dim=self.Lattice_dim, pbc=self.PBC)
        self.Hilbert_space=nk.hilbert.Spin(s=self.Spin, N=self.Lattice_graph.n_nodes)
        self.Hamiltonian=nk.operator.Heisenberg(hilbert=self.Hilbert_space, graph=self.Lattice_graph, J=self.Coupling)

        self.str_repr=f'{self.__class__.__name__}\n\tLattice_length={self.Lattice_length}\n\tLattice_dim={self.Lattice_dim}\n\tPBC={self.PBC}\n\tSpin={self.Spin}\n\tCoupling={self.Coupling}\n\tHilbert_space={self.Hilbert_space}\n\tHamiltonian={self.Hamiltonian}\nLattice_graph={self.Lattice_graph}'
    
    def eigen_energies(self, n_eigen:int=4):
        return super().eigen_energies(n_eigen)

    def __str__(self):
        return self.str_repr
    def __repr__(self):
        return self.str_repr