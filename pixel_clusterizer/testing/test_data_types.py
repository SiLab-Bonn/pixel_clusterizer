''' Script to check the correctness of the clustering for different data types.
'''

import unittest
import os

import numpy as np

from pixel_clusterizer.clusterizer import HitClusterizer, default_hits_descr, default_hits_dtype, default_clusters_dtype, default_clusters_descr, default_cluster_hits_descr, default_cluster_hits_dtype


def create_hits(n_hits, max_column, max_row, max_frame, max_charge, hit_dtype=default_hits_dtype, hit_fields=None):
    hits = np.zeros(shape=(n_hits, ), dtype=hit_dtype)
    if not hit_fields:
        for i in range(n_hits):
            hits[i]['event_number'], hits[i]['frame'], hits[i]['column'], hits[i]['row'], hits[i]['charge'] = i / 3, i % max_frame, i % max_column + 1, 2 * i % max_row + 1, i % max_charge
    else:
        hit_fields_inverse = dict((v, k) for k, v in hit_fields.items())
        for i in range(n_hits):
            hits[i][hit_fields_inverse['event_number']], hits[i][hit_fields_inverse['frame']], hits[i][hit_fields_inverse['column']], hits[i][hit_fields_inverse['row']], hits[i][hit_fields_inverse['charge']] = i / 3, i % max_frame, i % max_column + 1, 2 * i % max_row + 1, i % max_charge
    return hits


class TestClusterizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pure_python = os.getenv('PURE_PYTHON', False)

    def test_different_hit_data_types(self):
        # Define a different hit data structure with standard names but
        # different data types and number of fields. Numba automatically
        # recompiles and the result should not change
        hit_data_types = []
        hit_data_types.append([
            ('event_number', '<i8'),
            ('frame', '<u2'),
            ('column', '<u4'),
            ('row', '<u4'),
            ('charge', '<f4'),
            ('parameter', '<i4')])
        hit_data_types.append([
            ('event_number', '<i4'),
            ('frame', '<u8'),
            ('column', '<u2'),
            ('row', '<i2'),
            ('charge', '<u1'),
            ('parameter', '<u1'),
            ('parameter_1', '<i4'),
            ('parameter_2', 'f4')])

        # Initialize clusterizer
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, charge_correction=1, charge_weighted_clustering=True, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)

        for hit_data_type in hit_data_types:
            clusterizer.set_hit_dtype(np.dtype(hit_data_type))
            # Create fake data with actual hit data structure
            hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=np.dtype(hit_data_type))
            hits['parameter'] = 1  # check for number different from zero
            cluster_hits, clusters = clusterizer.cluster_hits(hits)
            array_size_before = clusterizer._clusters.shape[0]

            # Define expected output
            expected_clusters = np.zeros(shape=(4, ), dtype=default_clusters_dtype)
            expected_clusters['event_number'] = [0, 1, 2, 3]
            expected_clusters['n_hits'] = [3, 3, 3, 1]
            expected_clusters['charge'] = [1, 2, 1, 1]
            expected_clusters['seed_column'] = [2, 4, 8, 10]
            expected_clusters['seed_row'] = [3, 7, 15, 19]
            expected_clusters['mean_column'] = [2.0, 5.0, 8.0, 10.0]
            expected_clusters['mean_row'] = [3.0, 9.0, 15.0, 19.0]

            # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
            hit_data_type.extend([
                ('cluster_ID', '<i2'),
                ('is_seed', '<u1'),
                ('cluster_size', '<u2'),
                ('n_cluster', '<u2')])
            expected_cluster_hits = np.zeros(shape=(10, ), dtype=hit_data_type)
            expected_cluster_hits['event_number'] = hits['event_number']
            expected_cluster_hits['frame'] = hits['frame']
            expected_cluster_hits['column'] = hits['column']
            expected_cluster_hits['row'] = hits['row']
            expected_cluster_hits['charge'] = hits['charge']
            expected_cluster_hits['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
            expected_cluster_hits['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
            expected_cluster_hits['n_cluster'] = 1
            expected_cluster_hits['parameter'] = 1  # was set to 1 before and copied to the cluster hits array

            # Test results
            self.assertTrue(np.array_equal(clusters, expected_clusters))
            self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

            # Test same size array
            hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=np.dtype(hit_data_type))
            cluster_hits, clusters = clusterizer.cluster_hits(hits)
            array_size_after = clusterizer._clusters.shape[0]

            # Test results
            self.assertTrue(array_size_before == array_size_after)
            self.assertTrue(np.array_equal(clusters, expected_clusters))
            expected_cluster_hits['parameter'] = 0  # created new hits, this is zero again
            self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

            # Test increasing size array
            hits = create_hits(n_hits=20, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=np.dtype(hit_data_type))
            cluster_hits, clusters = clusterizer.cluster_hits(hits)
            array_size_after = clusterizer._clusters.shape[0]

            # Define expected output
            expected_clusters = np.zeros(shape=(7, ), dtype=default_clusters_dtype)
            expected_clusters['event_number'] = [0, 1, 2, 3, 4, 5, 6]
            expected_clusters['n_hits'] = [3, 3, 3, 3, 3, 3, 2]
            expected_clusters['charge'] = [1, 2, 1, 2, 1, 2, 1]
            expected_clusters['seed_column'] = [2, 4, 8, 10, 14, 16, 20]
            expected_clusters['seed_row'] = [3, 7, 15, 19, 27, 31, 39]
            expected_clusters['mean_column'] = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, (1 * 19 + 2 * 20) / 3.0]
            expected_clusters['mean_row'] = [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, (1 * 37 + 2 * 39) / 3.0]

            # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
            expected_cluster_hits = np.zeros(shape=(20, ), dtype=hit_data_type)
            expected_cluster_hits['event_number'] = hits['event_number']
            expected_cluster_hits['frame'] = hits['frame']
            expected_cluster_hits['column'] = hits['column']
            expected_cluster_hits['row'] = hits['row']
            expected_cluster_hits['charge'] = hits['charge']
            expected_cluster_hits['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
            expected_cluster_hits['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]
            expected_cluster_hits['n_cluster'] = 1

            # Test results
            self.assertTrue(array_size_before < array_size_after)
            self.assertTrue(np.array_equal(clusters, expected_clusters))
            self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

        # Initialize Clusterizer and test charge weighted clustering (charge is float)
        clusterizer = HitClusterizer(pure_python=self.pure_python, charge_weighted_clustering=True)

        # Create some fake data
        hits = np.ones(shape=(4, ), dtype=default_hits_dtype)
        clusterizer.set_hit_dtype(hits.dtype)
        hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 17, 36, 0.0, 19
        hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 37, 10.5, 19
        hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'] = 17, 36, 1.0, 20
        hits[3]['column'], hits[3]['row'], hits[3]['charge'], hits[3]['event_number'] = 18, 37, 10.5, 20

        cluster_hits, clusters = clusterizer.cluster_hits(hits)  # cluster hits

        # Define expected output
        expected_clusters = np.zeros(shape=(2, ), dtype=default_clusters_dtype)
        expected_clusters['event_number'] = [19, 20]
        expected_clusters['n_hits'] = [2, 2]
        expected_clusters['charge'] = [10.5, 11.5]
        expected_clusters['seed_column'] = [18, 18]
        expected_clusters['seed_row'] = [37, 37]
        expected_clusters['mean_column'] = [18.0, (1.0 * 17 + 10.5 * 18) / 11.5]
        expected_clusters['mean_row'] = [37.0, (1.0 * 36 + 10.5 * 37) / 11.5]
        # Define expected output
        expected_cluster_hits = np.zeros(shape=(4, ), dtype=default_cluster_hits_dtype)
        expected_cluster_hits['event_number'] = hits['event_number']
        expected_cluster_hits['frame'] = hits['frame']
        expected_cluster_hits['column'] = hits['column']
        expected_cluster_hits['row'] = hits['row']
        expected_cluster_hits['charge'] = hits['charge']
        expected_cluster_hits['is_seed'] = [0, 1, 0, 1]
        expected_cluster_hits['cluster_size'] = [2, 2, 2, 2]
        expected_cluster_hits['n_cluster'] = [1, 1, 1, 1]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_clusters))
        self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

    def test_different_cluster_data_types(self):
        # Define a different hit data structure with standard names but
        # different data types and number of fields. Numba automatically
        # recompiles and the result should not change
        cluster_data_types = []
        cluster_data_types.append([
            ('event_number', '<f8'),
            ('ID', '<u2'),
            ('n_hits', '<u2'),
            ('charge', 'f4'),
            ('seed_column', '<i2'),
            ('seed_row', '<i2'),
            ('mean_column', 'f4'),
            ('mean_row', 'f4')])
        cluster_data_types.append([
            ('event_number', '<u8'),
            ('ID', '<u2'),
            ('n_hits', '<u2'),
            ('charge', 'u4'),
            ('seed_column', '<u2'),
            ('seed_row', '<u2'),
            ('mean_column', 'f4'),
            ('mean_row', 'f4')])

        # Initialize clusterizer
        clusterizer = HitClusterizer(
            pure_python=self.pure_python,
            min_hit_charge=0,
            max_hit_charge=13,
            charge_correction=1,
            charge_weighted_clustering=True,
            column_cluster_distance=2,
            row_cluster_distance=2,
            frame_cluster_distance=4,
            ignore_same_hits=True)

        for cluster_data_type in cluster_data_types:
            clusterizer.set_cluster_dtype(np.dtype(cluster_data_type))
            # Create fake data with actual hit data structure
            hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
            cluster_hits, clusters = clusterizer.cluster_hits(hits)  # Cluster hits
            array_size_before = clusterizer._clusters.shape[0]

            # Define expected output
            expected_clusters = np.zeros(shape=(4, ), dtype=np.dtype(cluster_data_type))
            expected_clusters['event_number'] = [0, 1, 2, 3]
            expected_clusters['n_hits'] = [3, 3, 3, 1]
            expected_clusters['charge'] = [1, 2, 1, 1]
            expected_clusters['seed_column'] = [2, 4, 8, 10]
            expected_clusters['seed_row'] = [3, 7, 15, 19]
            expected_clusters['mean_column'] = [2.0, 5.0, 8.0, 10.0]
            expected_clusters['mean_row'] = [3.0, 9.0, 15.0, 19.0]

            expected_cluster_hits = np.zeros(shape=(10, ), dtype=default_cluster_hits_dtype)
            expected_cluster_hits['event_number'] = hits['event_number']
            expected_cluster_hits['frame'] = hits['frame']
            expected_cluster_hits['column'] = hits['column']
            expected_cluster_hits['row'] = hits['row']
            expected_cluster_hits['charge'] = hits['charge']
            expected_cluster_hits['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
            expected_cluster_hits['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
            expected_cluster_hits['n_cluster'] = 1

            # Test results
            self.assertTrue(np.array_equal(clusters, expected_clusters))
            self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

            # Test same size array
            hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
            cluster_hits, clusters = clusterizer.cluster_hits(hits)
            array_size_after = clusterizer._clusters.shape[0]

            # Test results
            self.assertTrue(array_size_before == array_size_after)
            self.assertTrue(np.array_equal(clusters, expected_clusters))
            self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

            # Test increasing size array
            hits = create_hits(n_hits=20, max_column=100, max_row=100, max_frame=1, max_charge=2)
            cluster_hits, clusters = clusterizer.cluster_hits(hits)
            array_size_after = clusterizer._clusters.shape[0]

            # Define expected output
            expected_clusters = np.zeros(shape=(7, ), dtype=np.dtype(cluster_data_type))
            expected_clusters['event_number'] = [0, 1, 2, 3, 4, 5, 6]
            expected_clusters['n_hits'] = [3, 3, 3, 3, 3, 3, 2]
            expected_clusters['charge'] = [1, 2, 1, 2, 1, 2, 1]
            expected_clusters['seed_column'] = [2, 4, 8, 10, 14, 16, 20]
            expected_clusters['seed_row'] = [3, 7, 15, 19, 27, 31, 39]
            expected_clusters['mean_column'] = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, (1 * 19 + 2 * 20) / 3.0]
            expected_clusters['mean_row'] = [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, (1 * 37 + 2 * 39) / 3.0]

            # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
            expected_cluster_hits = np.zeros(shape=(20, ), dtype=default_cluster_hits_dtype)
            expected_cluster_hits['event_number'] = hits['event_number']
            expected_cluster_hits['frame'] = hits['frame']
            expected_cluster_hits['column'] = hits['column']
            expected_cluster_hits['row'] = hits['row']
            expected_cluster_hits['charge'] = hits['charge']
            expected_cluster_hits['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
            expected_cluster_hits['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]
            expected_cluster_hits['n_cluster'] = 1

            # Test results
            self.assertTrue(array_size_before < array_size_after)
            self.assertTrue(np.array_equal(clusters, expected_clusters))
            self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

    def test_custom_hit_fields(self):
        # Define a different hit data structure with different names but standard data types.
        hit_dtype = np.dtype([
            ('eventNumber', '<i8'),
            ('relBCID', '<u1'),
            ('column', '<u2'),
            ('row', '<u2'),
            ('tot', '<u2')])

        hit_clustered_dtype = np.dtype([
            ('eventNumber', '<i8'),
            ('relBCID', '<u1'),
            ('column', '<u2'),
            ('row', '<u2'),
            ('tot', '<u2'),
            ('cluster_ID', '<i2'),
            ('is_seed', '<u1'),
            ('cluster_size', '<u2'),
            ('n_cluster', '<u2')])

        hit_fields = {
            'eventNumber': 'event_number',
            'column': 'column',
            'row': 'row',
            'tot': 'charge',
            'relBCID': 'frame'}

        # Initialize clusterizer and cluster test hits with self defined data type names
        clusterizer = HitClusterizer(hit_fields=hit_fields, hit_dtype=hit_dtype, pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, charge_correction=1, charge_weighted_clustering=True, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=hit_dtype, hit_fields=hit_fields)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_before = clusterizer._clusters.shape[0]

        # Define expected output
        expected_clusters = np.zeros(shape=(4, ), dtype=default_clusters_dtype)
        expected_clusters['event_number'] = [0, 1, 2, 3]
        expected_clusters['n_hits'] = [3, 3, 3, 1]
        expected_clusters['charge'] = [1, 2, 1, 1]
        expected_clusters['seed_column'] = [2, 4, 8, 10]
        expected_clusters['seed_row'] = [3, 7, 15, 19]
        expected_clusters['mean_column'] = [2.0, 5.0, 8.0, 10.0]
        expected_clusters['mean_row'] = [3.0, 9.0, 15.0, 19.0]

        # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
        expected_cluster_hits = np.zeros(shape=(10, ), dtype=hit_clustered_dtype)
        expected_cluster_hits['eventNumber'] = hits['eventNumber']
        expected_cluster_hits['relBCID'] = hits['relBCID']
        expected_cluster_hits['column'] = hits['column']
        expected_cluster_hits['row'] = hits['row']
        expected_cluster_hits['tot'] = hits['tot']
        expected_cluster_hits['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
        expected_cluster_hits['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
        expected_cluster_hits['n_cluster'] = 1

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_clusters))
        self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

        # Test same size array
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=hit_dtype, hit_fields=hit_fields)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Test results
        self.assertTrue(array_size_before == array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_clusters))
        self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

        # Test increasing size array
        hits = create_hits(n_hits=20, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=hit_dtype, hit_fields=hit_fields)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Define expected output
        expected_clusters = np.zeros(shape=(7, ), dtype=default_clusters_dtype)
        expected_clusters['event_number'] = [0, 1, 2, 3, 4, 5, 6]
        expected_clusters['n_hits'] = [3, 3, 3, 3, 3, 3, 2]
        expected_clusters['charge'] = [1, 2, 1, 2, 1, 2, 1]
        expected_clusters['seed_column'] = [2, 4, 8, 10, 14, 16, 20]
        expected_clusters['seed_row'] = [3, 7, 15, 19, 27, 31, 39]
        expected_clusters['mean_column'] = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, (1 * 19 + 2 * 20) / 3.0]
        expected_clusters['mean_row'] = [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, (1 * 37 + 2 * 39) / 3.0]

        # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
        expected_cluster_hits = np.zeros(shape=(20, ), dtype=hit_clustered_dtype)
        expected_cluster_hits['eventNumber'] = hits['eventNumber']
        expected_cluster_hits['relBCID'] = hits['relBCID']
        expected_cluster_hits['column'] = hits['column']
        expected_cluster_hits['row'] = hits['row']
        expected_cluster_hits['tot'] = hits['tot']
        expected_cluster_hits['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        expected_cluster_hits['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]
        expected_cluster_hits['n_cluster'] = 1

        # Test results
        self.assertTrue(array_size_before < array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_clusters))
        self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

    def test_custom_cluster_fields(self):
        # Define a different cluster data structure with different names but standard data types.
        clusters_dtype = np.dtype([
            ('eventNumber', '<i8'),
            ('ID', '<u2'),
            ('size', '<u2'),
            ('tot', 'f4'),
            ('seed_column', '<u2'),
            ('seed_row', '<u2'),
            ('mean_column', 'f4'),
            ('mean_row', 'f4')])

        clusters_fields = {
            'eventNumber': 'event_number',
            'ID': 'ID',
            'size': 'n_hits',
            'tot': 'charge',
            'seed_column': 'seed_column',
            'seed_row': 'seed_row',
            'mean_column': 'mean_column',
            'mean_row': 'mean_row'}

        # Initialize clusterizer and cluster test hits with self defined data type names
        clusterizer = HitClusterizer(cluster_fields=clusters_fields, cluster_dtype=clusters_dtype, pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, charge_correction=1, charge_weighted_clustering=True, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_before = clusterizer._clusters.shape[0]

        # Define expected output
        expected_clusters = np.zeros(shape=(4, ), dtype=clusters_dtype)
        expected_clusters['eventNumber'] = [0, 1, 2, 3]
        expected_clusters['size'] = [3, 3, 3, 1]
        expected_clusters['tot'] = [1, 2, 1, 1]
        expected_clusters['seed_column'] = [2, 4, 8, 10]
        expected_clusters['seed_row'] = [3, 7, 15, 19]
        expected_clusters['mean_column'] = [2.0, 5.0, 8.0, 10.0]
        expected_clusters['mean_row'] = [3.0, 9.0, 15.0, 19.0]

        # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
        expected_cluster_hits = np.zeros(shape=(10, ), dtype=default_cluster_hits_dtype)
        expected_cluster_hits['event_number'] = hits['event_number']
        expected_cluster_hits['frame'] = hits['frame']
        expected_cluster_hits['column'] = hits['column']
        expected_cluster_hits['row'] = hits['row']
        expected_cluster_hits['charge'] = hits['charge']
        expected_cluster_hits['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
        expected_cluster_hits['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
        expected_cluster_hits['n_cluster'] = 1

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_clusters))
        self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

        # Test same size array
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Test results
        self.assertTrue(array_size_before == array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_clusters))
        self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

        # Test increasing size array
        hits = create_hits(n_hits=20, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Define expected output
        expected_clusters = np.zeros(shape=(7, ), dtype=clusters_dtype)
        expected_clusters['eventNumber'] = [0, 1, 2, 3, 4, 5, 6]
        expected_clusters['size'] = [3, 3, 3, 3, 3, 3, 2]
        expected_clusters['tot'] = [1, 2, 1, 2, 1, 2, 1]
        expected_clusters['seed_column'] = [2, 4, 8, 10, 14, 16, 20]
        expected_clusters['seed_row'] = [3, 7, 15, 19, 27, 31, 39]
        expected_clusters['mean_column'] = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, (1 * 19 + 2 * 20) / 3.0]
        expected_clusters['mean_row'] = [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, (1 * 37 + 2 * 39) / 3.0]

        # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
        expected_cluster_hits = np.zeros(shape=(20, ), dtype=default_cluster_hits_dtype)
        expected_cluster_hits['event_number'] = hits['event_number']
        expected_cluster_hits['frame'] = hits['frame']
        expected_cluster_hits['column'] = hits['column']
        expected_cluster_hits['row'] = hits['row']
        expected_cluster_hits['charge'] = hits['charge']
        expected_cluster_hits['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        expected_cluster_hits['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]
        expected_cluster_hits['n_cluster'] = 1

        # Test results
        self.assertTrue(array_size_before < array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_clusters))
        self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

    def test_adding_hit_field(self):
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, charge_correction=1, charge_weighted_clustering=True, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)
        with self.assertRaises(TypeError):
            clusterizer.add_hit_field(description=['extra_field', 'f4'])  # also test list of 2 items
        clusterizer.add_hit_field(description=[('extra_field', 'f4')])  # also test list of 2-tuples
        modified_hits_descr = default_hits_descr[:]
        modified_hits_descr.append(('extra_field', 'f4'))
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=np.dtype(modified_hits_descr))
        hits['extra_field'][1:] = range(hits.shape[0] - 1)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)

        # Define expected cluster output with extra field
        expected_clusters = np.zeros(shape=(4, ), dtype=default_clusters_dtype)
        expected_clusters['event_number'] = [0, 1, 2, 3]
        expected_clusters['n_hits'] = [3, 3, 3, 1]
        expected_clusters['charge'] = [1, 2, 1, 1]
        expected_clusters['seed_column'] = [2, 4, 8, 10]
        expected_clusters['seed_row'] = [3, 7, 15, 19]
        expected_clusters['mean_column'] = [2.0, 5.0, 8.0, 10.0]
        expected_clusters['mean_row'] = [3.0, 9.0, 15.0, 19.0]

        # Define expected hit clustered output
        modified_cluster_hits_descr = default_cluster_hits_descr[:]
        modified_cluster_hits_descr.append(('extra_field', 'f4'))
        expected_cluster_hits = np.zeros(shape=(10, ), dtype=np.dtype(modified_cluster_hits_descr))
        expected_cluster_hits['event_number'] = hits['event_number']
        expected_cluster_hits['frame'] = hits['frame']
        expected_cluster_hits['column'] = hits['column']
        expected_cluster_hits['row'] = hits['row']
        expected_cluster_hits['charge'] = hits['charge']
        expected_cluster_hits['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
        expected_cluster_hits['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
        expected_cluster_hits['n_cluster'] = 1
        expected_cluster_hits['extra_field'] = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        # Test results
        print("\n")
        print(clusters)
        print(expected_clusters)
        self.assertTrue(np.array_equal(clusters, expected_clusters))
        self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

    def test_adding_cluster_field(self):
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, charge_correction=1, charge_weighted_clustering=True, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)
        with self.assertRaises(TypeError):
            clusterizer.add_cluster_field(description=['extra_field', 'f4'])  # also test list of 2 items
        clusterizer.add_cluster_field(description=[('extra_field', 'f4')])  # also test list of 2-tuples

        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_before = clusterizer._clusters.shape[0]

        # Define expected cluster output with extra field
        modified_clusters_descr = default_clusters_descr[:]
        modified_clusters_descr.append(('extra_field', 'f4'))
        expected_clusters = np.zeros(shape=(4, ), dtype=np.dtype(modified_clusters_descr))
        expected_clusters['event_number'] = [0, 1, 2, 3]
        expected_clusters['n_hits'] = [3, 3, 3, 1]
        expected_clusters['charge'] = [1, 2, 1, 1]
        expected_clusters['seed_column'] = [2, 4, 8, 10]
        expected_clusters['seed_row'] = [3, 7, 15, 19]
        expected_clusters['mean_column'] = [2.0, 5.0, 8.0, 10.0]
        expected_clusters['mean_row'] = [3.0, 9.0, 15.0, 19.0]
        expected_clusters['extra_field'] = [0.0, 0.0, 0.0, 0.0]

        # Define expected hit clustered output
        expected_cluster_hits = np.zeros(shape=(10, ), dtype=default_cluster_hits_dtype)
        expected_cluster_hits['event_number'] = hits['event_number']
        expected_cluster_hits['frame'] = hits['frame']
        expected_cluster_hits['column'] = hits['column']
        expected_cluster_hits['row'] = hits['row']
        expected_cluster_hits['charge'] = hits['charge']
        expected_cluster_hits['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
        expected_cluster_hits['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
        expected_cluster_hits['n_cluster'] = 1

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_clusters))
        self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

        # Test same size array
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Test results
        self.assertTrue(array_size_before == array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_clusters))
        self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))

        # Test increasing size array
        hits = create_hits(n_hits=20, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Define expected cluster output with extra field
        modified_clusters_descr = default_clusters_descr[:]
        modified_clusters_descr.append(('extra_field', 'f4'))
        expected_clusters = np.zeros(shape=(7, ), dtype=np.dtype(modified_clusters_descr))
        expected_clusters['event_number'] = [0, 1, 2, 3, 4, 5, 6]
        expected_clusters['n_hits'] = [3, 3, 3, 3, 3, 3, 2]
        expected_clusters['charge'] = [1, 2, 1, 2, 1, 2, 1]
        expected_clusters['seed_column'] = [2, 4, 8, 10, 14, 16, 20]
        expected_clusters['seed_row'] = [3, 7, 15, 19, 27, 31, 39]
        expected_clusters['mean_column'] = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, (1 * 19 + 2 * 20) / 3.0]
        expected_clusters['mean_row'] = [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, (1 * 37 + 2 * 39) / 3.0]
        expected_clusters['extra_field'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Define expected hit clustered output
        expected_cluster_hits = np.zeros(shape=(20, ), dtype=default_cluster_hits_dtype)
        expected_cluster_hits['event_number'] = hits['event_number']
        expected_cluster_hits['frame'] = hits['frame']
        expected_cluster_hits['column'] = hits['column']
        expected_cluster_hits['row'] = hits['row']
        expected_cluster_hits['charge'] = hits['charge']
        expected_cluster_hits['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        expected_cluster_hits['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]
        expected_cluster_hits['n_cluster'] = 1

        # Test results
        self.assertTrue(array_size_before < array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_clusters))
        self.assertTrue(np.array_equal(cluster_hits, expected_cluster_hits))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClusterizer)
    unittest.TextTestRunner(verbosity=2).run(suite)
