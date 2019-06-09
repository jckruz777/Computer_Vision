#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 23:43:53 2019

@author: luisalonsomurillorojas
"""

import json

#meanshift parameters
data =	{
  "scale": 100,
  "sigma":0.5,
  "min_size":50
}

# Serializing
filename = 'felzenszwalb.json'
with open(filename, 'w') as f:
    json.dump(data, f, sort_keys=True)