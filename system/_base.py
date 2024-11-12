'''
Author: airscker
Date: 2024-11-12 16:51:47
LastEditors: airscker
LastEditTime: 2024-11-12 18:15:38
Description: NULL

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''

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