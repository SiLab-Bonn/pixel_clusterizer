# http://stackoverflow.com/questions/17583443/what-is-the-correct-way-to-share-package-version-with-setup-py-and-the-package
from pkg_resources import get_distribution
from pixel_clusterizer.clusterizer import HitClusterizer, default_hits_descr, default_hits_dtype, default_cluster_hits_descr, default_cluster_hits_dtype, default_clusters_descr, default_clusters_dtype


__version__ = get_distribution('pixel_clusterizer').version
_all_ = ["HitClusterizer", "default_hits_dtype", "default_cluster_hits_descr", "default_cluster_hits_dtype", "default_clusters_descr", "default_clusters_dtype"]
