'''
Author: airscker
Date: 2024-10-29 01:28:28
LastEditors: airscker
LastEditTime: 2024-10-29 15:01:31
Description: NULL

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''
import flax.linen as nn
from abc import ABCMeta, abstractmethod

class BaseModel(nn.Module,metaclass=ABCMeta):
    @abstractmethod
    def setup(self):
        pass
    @abstractmethod
    def __call__(self, x):
        pass
    def __str__(self):
        _str=f"{self.__class__.__name__}:\n"
        for para in self.__dict__:
            _str+=f"\t{para}: {self.__dict__[para]}\n"
        return _str
    def __repr__(self):
        return self.__str__()