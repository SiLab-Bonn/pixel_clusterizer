# Pixel Clusterizer [![Build Status](https://travis-ci.org/SiLab-Bonn/pixel_clusterizer.svg?branch=master)](https://travis-ci.org/SiLab-Bonn/pixel_clusterizer) [![Build Status](https://ci.appveyor.com/api/projects/status/github/SiLab-Bonn/pixel_clusterizer)](https://ci.appveyor.com/project/SiLab-Bonn/pixel_clusterizer)

pixel_clusterizer is an easy to use pixel hit-clusterizer for Python. It clusters hits on an event basis in space and time. 

The hits have to be a numpy recarray with data types as defined in data_struct.HitInfo. The fields are:
- event_number
- frame
- column
- row
- charge

Cluster information (data_struct.ClusterInfo) is created with the following fields:
- event_number
- ID
- size
- charge
- seed_column
- seed_row
- mean_column
- mean_row

Also cluster hit information (data_struct.ClusterHitInfo) is created. The cluster hit information is the hit information extended by the following fields:
- cluster_ID
- is_seed
- cluster_size
- n_cluster

# Installation

The stable code is hosted on PyPI and can be installed by typing:

pip install pixel_clusterizer

# Usage

import numpy as np

from pixel_clusterizer.clusterizer import HitClusterizer

from pixel_clusterizer import data_struct

hits = np.ones(shape=(3, ), dtype=data_struct.HitInfo)  # create some data

clusterizer = HitClusterizer()  # initialize clusterizer

clusterizer.add_hits(hits)  # cluster hits  # add hits to clusterizer

print (clusterizer.get_cluster())  # print cluster

print (clusterizer.get_hit_cluster())  # print hits + cluster info

Also take a look at the example folder!

