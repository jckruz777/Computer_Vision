#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 23:43:53 2019

@author: luisalonsomurillorojas
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

#meanshift parameters
data =	{
  "bandwidth": 17
}

# Serializing
filename = 'meanshift.json'
with open(filename, 'w') as f:
    json.dump(data, f, sort_keys=True)