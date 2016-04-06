# Pixel Clusterizer [![Build Status](https://travis-ci.org/SiLab-Bonn/pixel_clusterizer.svg?branch=master)](https://travis-ci.org/SiLab-Bonn/pixel_clusterizer) [![Build Status](https://ci.appveyor.com/api/projects/status/github/SiLab-Bonn/pixel_clusterizer)](https://ci.appveyor.com/project/SiLab-Bonn/pixel_clusterizer) [![Coverage Status](https://coveralls.io/repos/github/SiLab-Bonn/pixel_clusterizer/badge.svg?branch=master)](https://coveralls.io/github/SiLab-Bonn/pixel_clusterizer?branch=master)

Pixel_clusterizer is an easy to use pixel hit-clusterizer for Python. It clusters hits on an event basis in space and time.
 
The hits have to be defined as a numpy recarray. The array has to have the following fields:
- event_number
- frame
- column
- row
- charge

or a mapping of the names has to be provided. The data type does not matter.

The result of the clustering is the hit array extended by the following fields:
- cluster_ID
- is_seed
- cluster_size
- n_cluster

A new array with cluster information is also created created and has the following fields:
- event_number
- ID
- size
- charge
- seed_column
- seed_row
- mean_column
- mean_row



# Installation

The stable code is hosted on PyPI and can be installed by typing:

pip install pixel_clusterizer

# Usage

```
import numpy as np

from pixel_clusterizer import clusterizer

hits = np.ones(shape=(3, ), dtype=clusterizer.hit_data_type)  # Create some data with std. hit data type

cr = clusterizer.HitClusterizer()  # Initialize clusterizer

hits_clustered, cluster = cr.cluster_hits(hits)  # Cluster hits

```
Also take a look at the example folder!

# Test installation
```
nosetests pixel_clusterizer
```
