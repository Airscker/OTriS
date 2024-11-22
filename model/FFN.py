'''
Author: airscker
Date: 2024-10-28 17:09:42
LastEditors: airscker
LastEditTime: 2024-10-29 15:01:35
Description: NULL

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''
import flax.linen as nn
import jax.numpy as jnp

class FFNModel(nn.Module):
    alpha : int = 1
    @nn.compact
    def __call__(self, x):
        dense = nn.Dense(features=self.alpha * x.shape[-1])
        # we apply the dense layer to the input
        y = dense(x)
        # the non-linearity is a simple ReLu
        y = nn.relu(y)
        # sum the output
        return jnp.sum(y, axis=-1)