# Pixel Clusterizer [![Build Status](https://travis-ci.org/SiLab-Bonn/pixel_clusterizer.svg?branch=master)](https://travis-ci.org/SiLab-Bonn/pixel_clusterizer) [![Build status](https://ci.appveyor.com/api/projects/status/c8jqu9ow696opevf?svg=true)](https://ci.appveyor.com/project/laborleben/pixel-clusterizer) [![Coverage Status](https://coveralls.io/repos/github/SiLab-Bonn/pixel_clusterizer/badge.svg?branch=master)](https://coveralls.io/github/SiLab-Bonn/pixel_clusterizer?branch=master)
[![tests](https://github.com/SiLab-Bonn/pixel_clusterizer/actions/workflows/tests.yml/badge.svg)](https://github.com/SiLab-Bonn/pixel_clusterizer/actions/workflows/tests.yml)

## Intended Use

Pixel_clusterizer is an easy to use pixel hit clusterizer for Python. It clusters hits connected to unique event numbers in space and time.

The hits must be provided in a numpy recarray. The array must contain the following columns ("fields"):
- ```event_number```
- ```frame```
- ```column```
- ```row```
- ```charge```

If the column names are different, a mapping of the names to the default names can be provided. The data type of each column can vary and is not fixed. The ```column```/```row``` values can be either indices (integer, default) or positions (float). ```Charge``` can be either integer or float (default).

After clustering, two new arrays are returned:
1. The cluster hits array is the hits array extended by the following columns:
    - ```cluster_ID```
    - ```is_seed```
    - ```cluster_size```
    - ```n_cluster```
2. The cluster array contains in each row the information about a single cluster. It has the following columns:
    - ```event_number```
    - ```ID```
    - ```n_hits```
    - ```charge```
    - ```seed_column```
    - ```seed_row```
    - ```mean_column```
    - ```mean_row```

## Installation

Python 2.7 or Python 3 or higher must be used. There are many ways to install Python, though we recommend using [Anaconda Python](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Prerequisites

The following packages are required:
```
numpy numba>=0.24.0
```

### Installation of pixel_clusterizer

The stable code is hosted on PyPI and can be installed by typing:
```
pip install pixel_clusterizer
```

For developer, clone the pixel_clusterizer git repository and use the following command to install pixel_clusterizer:
```
pip install -e .
```

For testing the basic functionality of pixel_clusterizer, execute the following command:
```
nosetests pixel_clusterizer
```

## Usage

```
import numpy as np

from pixel_clusterizer import clusterizer

hits = np.ones(shape=(3, ), dtype=clusterizer.default_hits_dtype)  # Create some data with std. hit data type

cr = clusterizer.HitClusterizer()  # Initialize clusterizer

cluster_hits, clusters = cr.cluster_hits(hits)  # Cluster hits

```
Also please have a look at the ```examples``` folder!

## Support

Please use GitHub's [issue tracker](https://github.com/SiLab-Bonn/pixel_clusterizer/issues) for bug reports/feature requests/questions.
