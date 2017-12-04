''' Script to check the correctness of the clustering for different data types.
'''

import unittest
import os

import numpy as np

from pixel_clusterizer.clusterizer import HitClusterizer


def create_hits(n_hits, max_column, max_row, max_frame, max_charge, hit_dtype=np.dtype([('event_number', '<i8'),
                                                                                        ('frame', '<u1'),
                                                                                        ('column', '<u2'),
                                                                                        ('row', '<u2'),
                                                                                        ('charge', '<u2')]), hit_fields=None):
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
        hit_data_types.append([('event_number', '<i8'),
                               ('frame', '<u1'),
                               ('column', '<u4'),
                               ('row', '<u4'),
                               ('charge', '<u1'),
                               ('parameter', '<i4')])
        hit_data_types.append([('event_number', '<i4'),
                               ('frame', '<u8'),
                               ('column', '<u2'),
                               ('row', '<i2'),
                               ('charge', '<u1'),
                               ('parameter', '<u1'),
                               ('parameter_1', '<i4'),
                               ('parameter_2', 'f4')])

        # Initialize clusterizer
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)

        for hit_data_type in hit_data_types:
            clusterizer.set_hit_dtype(np.dtype(hit_data_type))
            # Create fake data with actual hit data structure
            hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=np.dtype(hit_data_type))
            hits['parameter'] = 1  # check for number different from zero
            cluster_hits, clusters = clusterizer.cluster_hits(hits)
            array_size_before = clusterizer._clusters.shape[0]

            # Define expected output
            expected_cluster_result = np.zeros(shape=(4, ), dtype=np.dtype([('event_number', '<i8'),
                                                                            ('ID', '<u2'),
                                                                            ('n_hits', '<u2'),
                                                                            ('charge', 'f4'),
                                                                            ('seed_column', '<u2'),
                                                                            ('seed_row', '<u2'),
                                                                            ('mean_column', 'f4'),
                                                                            ('mean_row', 'f4')]))
            expected_cluster_result['event_number'] = [0, 1, 2, 3]
            expected_cluster_result['n_hits'] = [3, 3, 3, 1]
            expected_cluster_result['charge'] = [1, 2, 1, 1]
            expected_cluster_result['seed_column'] = [2, 4, 8, 10]
            expected_cluster_result['seed_row'] = [3, 7, 15, 19]
            expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 10.0]
            expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 19.0]

            # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
            hit_data_type.extend([('cluster_ID', '<i2'),
                                  ('is_seed', '<u1'),
                                  ('cluster_size', '<u2'),
                                  ('n_cluster', '<u2')])
            expected_hit_result = np.zeros(shape=(10, ), dtype=hit_data_type)
            expected_hit_result['event_number'] = hits['event_number']
            expected_hit_result['frame'] = hits['frame']
            expected_hit_result['column'] = hits['column']
            expected_hit_result['row'] = hits['row']
            expected_hit_result['charge'] = hits['charge']
            expected_hit_result['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
            expected_hit_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
            expected_hit_result['n_cluster'] = 1
            expected_hit_result['parameter'] = 1  # was set to 1 before and copied to the cluster hits array

            # Test results
            self.assertTrue(np.array_equal(clusters, expected_cluster_result))
            self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

            # Test same size array
            hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=np.dtype(hit_data_type))
            cluster_hits, clusters = clusterizer.cluster_hits(hits)
            array_size_after = clusterizer._clusters.shape[0]

            # Test results
            self.assertTrue(array_size_before == array_size_after)
            self.assertTrue(np.array_equal(clusters, expected_cluster_result))
            expected_hit_result['parameter'] = 0  # created new hits, this is zero again
            self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

            # Test increasing size array
            hits = create_hits(n_hits=20, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=np.dtype(hit_data_type))
            cluster_hits, clusters = clusterizer.cluster_hits(hits)
            array_size_after = clusterizer._clusters.shape[0]

            # Define expected output
            expected_cluster_result = np.zeros(shape=(7, ), dtype=np.dtype([('event_number', '<i8'),
                                                                            ('ID', '<u2'),
                                                                            ('n_hits', '<u2'),
                                                                            ('charge', 'f4'),
                                                                            ('seed_column', '<u2'),
                                                                            ('seed_row', '<u2'),
                                                                            ('mean_column', 'f4'),
                                                                            ('mean_row', 'f4')]))
            expected_cluster_result['event_number'] = [0, 1, 2, 3, 4, 5, 6]
            expected_cluster_result['n_hits'] = [3, 3, 3, 3, 3, 3, 2]
            expected_cluster_result['charge'] = [1, 2, 1, 2, 1, 2, 1]
            expected_cluster_result['seed_column'] = [2, 4, 8, 10, 14, 16, 20]
            expected_cluster_result['seed_row'] = [3, 7, 15, 19, 27, 31, 39]
            expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, (1 * 19 + 2 * 20) / 3.0]
            expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, (1 * 37 + 2 * 39) / 3.0]

            # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
            expected_hit_result = np.zeros(shape=(20, ), dtype=hit_data_type)
            expected_hit_result['event_number'] = hits['event_number']
            expected_hit_result['frame'] = hits['frame']
            expected_hit_result['column'] = hits['column']
            expected_hit_result['row'] = hits['row']
            expected_hit_result['charge'] = hits['charge']
            expected_hit_result['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
            expected_hit_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]
            expected_hit_result['n_cluster'] = 1

            # Test results
            self.assertTrue(array_size_before < array_size_after)
            self.assertTrue(np.array_equal(clusters, expected_cluster_result))
            self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

    def test_different_cluster_data_types(self):
        # Define a different hit data structure with standard names but
        # different data types and number of fields. Numba automatically
        # recompiles and the result should not change
        cluster_data_types = []
        cluster_data_types.append([('event_number', '<f8'),
                                   ('ID', '<u2'),
                                   ('n_hits', '<u2'),
                                   ('charge', 'f4'),
                                   ('seed_column', '<i2'),
                                   ('seed_row', '<i2'),
                                   ('mean_column', 'f4'),
                                   ('mean_row', 'f4')])
        cluster_data_types.append([('event_number', '<u8'),
                                   ('ID', '<u2'),
                                   ('n_hits', '<u2'),
                                   ('charge', 'u4'),
                                   ('seed_column', '<u2'),
                                   ('seed_row', '<u2'),
                                   ('mean_column', 'f4'),
                                   ('mean_row', 'f4')])

        # Initialize clusterizer
        clusterizer = HitClusterizer(pure_python=self.pure_python,
                                     min_hit_charge=0, max_hit_charge=13,
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
            expected_cluster_result = np.zeros(shape=(4, ), dtype=np.dtype(cluster_data_type))
            expected_cluster_result['event_number'] = [0, 1, 2, 3]
            expected_cluster_result['n_hits'] = [3, 3, 3, 1]
            expected_cluster_result['charge'] = [1, 2, 1, 1]
            expected_cluster_result['seed_column'] = [2, 4, 8, 10]
            expected_cluster_result['seed_row'] = [3, 7, 15, 19]
            expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 10.0]
            expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 19.0]

            expected_hit_result = np.zeros(shape=(10, ), dtype=np.dtype([('event_number', '<i8'),
                                                                         ('frame', '<u1'),
                                                                         ('column', '<u2'),
                                                                         ('row', '<u2'),
                                                                         ('charge', '<u2'),
                                                                         ('cluster_ID', '<i2'),
                                                                         ('is_seed', '<u1'),
                                                                         ('cluster_size', '<u2'),
                                                                         ('n_cluster', '<u2')]))
            expected_hit_result['event_number'] = hits['event_number']
            expected_hit_result['frame'] = hits['frame']
            expected_hit_result['column'] = hits['column']
            expected_hit_result['row'] = hits['row']
            expected_hit_result['charge'] = hits['charge']
            expected_hit_result['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
            expected_hit_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
            expected_hit_result['n_cluster'] = 1

            # Test results
            self.assertTrue(np.array_equal(clusters, expected_cluster_result))
            self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

            # Test same size array
            hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
            cluster_hits, clusters = clusterizer.cluster_hits(hits)
            array_size_after = clusterizer._clusters.shape[0]

            # Test results
            self.assertTrue(array_size_before == array_size_after)
            self.assertTrue(np.array_equal(clusters, expected_cluster_result))
            self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

            # Test increasing size array
            hits = create_hits(n_hits=20, max_column=100, max_row=100, max_frame=1, max_charge=2)
            cluster_hits, clusters = clusterizer.cluster_hits(hits)
            array_size_after = clusterizer._clusters.shape[0]

            # Define expected output
            expected_cluster_result = np.zeros(shape=(7, ), dtype=np.dtype(cluster_data_type))
            expected_cluster_result['event_number'] = [0, 1, 2, 3, 4, 5, 6]
            expected_cluster_result['n_hits'] = [3, 3, 3, 3, 3, 3, 2]
            expected_cluster_result['charge'] = [1, 2, 1, 2, 1, 2, 1]
            expected_cluster_result['seed_column'] = [2, 4, 8, 10, 14, 16, 20]
            expected_cluster_result['seed_row'] = [3, 7, 15, 19, 27, 31, 39]
            expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, (1 * 19 + 2 * 20) / 3.0]
            expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, (1 * 37 + 2 * 39) / 3.0]

            # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
            expected_hit_result = np.zeros(shape=(20, ), dtype=np.dtype([('event_number', '<i8'),
                                                                         ('frame', '<u1'),
                                                                         ('column', '<u2'),
                                                                         ('row', '<u2'),
                                                                         ('charge', '<u2'),
                                                                         ('cluster_ID', '<i2'),
                                                                         ('is_seed', '<u1'),
                                                                         ('cluster_size', '<u2'),
                                                                         ('n_cluster', '<u2')]))
            expected_hit_result['event_number'] = hits['event_number']
            expected_hit_result['frame'] = hits['frame']
            expected_hit_result['column'] = hits['column']
            expected_hit_result['row'] = hits['row']
            expected_hit_result['charge'] = hits['charge']
            expected_hit_result['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
            expected_hit_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]
            expected_hit_result['n_cluster'] = 1

            # Test results
            self.assertTrue(array_size_before < array_size_after)
            self.assertTrue(np.array_equal(clusters, expected_cluster_result))
            self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

    def test_custom_hit_fields(self):
        # Define a different hit data structure with different names but standard data types.
        hit_dtype = np.dtype([('eventNumber', '<i8'),
                              ('relBCID', '<u1'),
                              ('column', '<u2'),
                              ('row', '<u2'),
                              ('tot', '<u2')])

        hit_clustered_dtype = np.dtype([('eventNumber', '<i8'),
                                        ('relBCID', '<u1'),
                                        ('column', '<u2'),
                                        ('row', '<u2'),
                                        ('tot', '<u2'),
                                        ('cluster_ID', '<i2'),
                                        ('is_seed', '<u1'),
                                        ('cluster_size', '<u2'),
                                        ('n_cluster', '<u2')])

        hit_fields = {'eventNumber': 'event_number',
                      'column': 'column',
                      'row': 'row',
                      'tot': 'charge',
                      'relBCID': 'frame'
                      }

        # Initialize clusterizer and cluster test hits with self defined data type names
        clusterizer = HitClusterizer(hit_fields=hit_fields, hit_dtype=hit_dtype, pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=hit_dtype, hit_fields=hit_fields)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_before = clusterizer._clusters.shape[0]

        # Define expected output
        expected_cluster_result = np.zeros(shape=(4, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0, 1, 2, 3]
        expected_cluster_result['n_hits'] = [3, 3, 3, 1]
        expected_cluster_result['charge'] = [1, 2, 1, 1]
        expected_cluster_result['seed_column'] = [2, 4, 8, 10]
        expected_cluster_result['seed_row'] = [3, 7, 15, 19]
        expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 10.0]
        expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 19.0]

        # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
        expected_hit_result = np.zeros(shape=(10, ), dtype=hit_clustered_dtype)
        expected_hit_result['eventNumber'] = hits['eventNumber']
        expected_hit_result['relBCID'] = hits['relBCID']
        expected_hit_result['column'] = hits['column']
        expected_hit_result['row'] = hits['row']
        expected_hit_result['tot'] = hits['tot']
        expected_hit_result['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
        expected_hit_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
        expected_hit_result['n_cluster'] = 1

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Test same size array
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=hit_dtype, hit_fields=hit_fields)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Test results
        self.assertTrue(array_size_before == array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Test increasing size array
        hits = create_hits(n_hits=20, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=hit_dtype, hit_fields=hit_fields)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Define expected output
        expected_cluster_result = np.zeros(shape=(7, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0, 1, 2, 3, 4, 5, 6]
        expected_cluster_result['n_hits'] = [3, 3, 3, 3, 3, 3, 2]
        expected_cluster_result['charge'] = [1, 2, 1, 2, 1, 2, 1]
        expected_cluster_result['seed_column'] = [2, 4, 8, 10, 14, 16, 20]
        expected_cluster_result['seed_row'] = [3, 7, 15, 19, 27, 31, 39]
        expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, (1 * 19 + 2 * 20) / 3.0]
        expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, (1 * 37 + 2 * 39) / 3.0]

        # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
        expected_hit_result = np.zeros(shape=(20, ), dtype=hit_clustered_dtype)
        expected_hit_result['eventNumber'] = hits['eventNumber']
        expected_hit_result['relBCID'] = hits['relBCID']
        expected_hit_result['column'] = hits['column']
        expected_hit_result['row'] = hits['row']
        expected_hit_result['tot'] = hits['tot']
        expected_hit_result['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        expected_hit_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]
        expected_hit_result['n_cluster'] = 1

        # Test results
        self.assertTrue(array_size_before < array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

    def test_custom_cluster_fields(self):
        # Define a different cluster data structure with different names but standard data types.
        cluster_dtype = np.dtype([('eventNumber', '<i8'),
                                  ('ID', '<u2'),
                                  ('size', '<u2'),
                                  ('tot', 'f4'),
                                  ('seed_column', '<u2'),
                                  ('seed_row', '<u2'),
                                  ('mean_column', 'f4'),
                                  ('mean_row', 'f4')])

        cluster_fields = {'eventNumber': 'event_number',
                          'ID': 'ID',
                          'size': 'n_hits',
                          'tot': 'charge',
                          'seed_column': 'seed_column',
                          'seed_row': 'seed_row',
                          'mean_column': 'mean_column',
                          'mean_row': 'mean_row'
                          }

        # Initialize clusterizer and cluster test hits with self defined data type names
        clusterizer = HitClusterizer(cluster_fields=cluster_fields, cluster_dtype=cluster_dtype, pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_before = clusterizer._clusters.shape[0]

        # Define expected output
        expected_cluster_result = np.zeros(shape=(4, ), dtype=cluster_dtype)
        expected_cluster_result['eventNumber'] = [0, 1, 2, 3]
        expected_cluster_result['size'] = [3, 3, 3, 1]
        expected_cluster_result['tot'] = [1, 2, 1, 1]
        expected_cluster_result['seed_column'] = [2, 4, 8, 10]
        expected_cluster_result['seed_row'] = [3, 7, 15, 19]
        expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 10.0]
        expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 19.0]

        # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
        expected_hit_result = np.zeros(shape=(10, ), dtype=np.dtype([('event_number', '<i8'),
                                                                     ('frame', '<u1'),
                                                                     ('column', '<u2'),
                                                                     ('row', '<u2'),
                                                                     ('charge', '<u2'),
                                                                     ('cluster_ID', '<i2'),
                                                                     ('is_seed', '<u1'),
                                                                     ('cluster_size', '<u2'),
                                                                     ('n_cluster', '<u2')]))
        expected_hit_result['event_number'] = hits['event_number']
        expected_hit_result['frame'] = hits['frame']
        expected_hit_result['column'] = hits['column']
        expected_hit_result['row'] = hits['row']
        expected_hit_result['charge'] = hits['charge']
        expected_hit_result['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
        expected_hit_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
        expected_hit_result['n_cluster'] = 1

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Test same size array
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Test results
        self.assertTrue(array_size_before == array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Test increasing size array
        hits = create_hits(n_hits=20, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Define expected output
        expected_cluster_result = np.zeros(shape=(7, ), dtype=cluster_dtype)
        expected_cluster_result['eventNumber'] = [0, 1, 2, 3, 4, 5, 6]
        expected_cluster_result['size'] = [3, 3, 3, 3, 3, 3, 2]
        expected_cluster_result['tot'] = [1, 2, 1, 2, 1, 2, 1]
        expected_cluster_result['seed_column'] = [2, 4, 8, 10, 14, 16, 20]
        expected_cluster_result['seed_row'] = [3, 7, 15, 19, 27, 31, 39]
        expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, (1 * 19 + 2 * 20) / 3.0]
        expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, (1 * 37 + 2 * 39) / 3.0]

        # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
        expected_hit_result = np.zeros(shape=(20, ), dtype=np.dtype([('event_number', '<i8'),
                                                                     ('frame', '<u1'),
                                                                     ('column', '<u2'),
                                                                     ('row', '<u2'),
                                                                     ('charge', '<u2'),
                                                                     ('cluster_ID', '<i2'),
                                                                     ('is_seed', '<u1'),
                                                                     ('cluster_size', '<u2'),
                                                                     ('n_cluster', '<u2')]))
        expected_hit_result['event_number'] = hits['event_number']
        expected_hit_result['frame'] = hits['frame']
        expected_hit_result['column'] = hits['column']
        expected_hit_result['row'] = hits['row']
        expected_hit_result['charge'] = hits['charge']
        expected_hit_result['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        expected_hit_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]
        expected_hit_result['n_cluster'] = 1

        # Test results
        self.assertTrue(array_size_before < array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

    def test_adding_cluster_field(self):
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)
        with self.assertRaises(TypeError):
            clusterizer.add_cluster_field(description=['extra_field', 'f4'])  # also test list of 2 items
        clusterizer.add_cluster_field(description=[('extra_field', 'f4')])  # also test list of 2-tuples

        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_before = clusterizer._clusters.shape[0]

        # Define expected cluster output with extra field
        expected_cluster_result = np.zeros(shape=(4, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4'),
                                                                        ('extra_field', 'f4')]))
        expected_cluster_result['event_number'] = [0, 1, 2, 3]
        expected_cluster_result['n_hits'] = [3, 3, 3, 1]
        expected_cluster_result['charge'] = [1, 2, 1, 1]
        expected_cluster_result['seed_column'] = [2, 4, 8, 10]
        expected_cluster_result['seed_row'] = [3, 7, 15, 19]
        expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 10.0]
        expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 19.0]
        expected_cluster_result['extra_field'] = [0.0, 0.0, 0.0, 0.0]

        # Define expected hit clustered output
        expected_hit_result = np.zeros(shape=(10, ), dtype=np.dtype([('event_number', '<i8'),
                                                                     ('frame', '<u1'),
                                                                     ('column', '<u2'),
                                                                     ('row', '<u2'),
                                                                     ('charge', '<u2'),
                                                                     ('cluster_ID', '<i2'),
                                                                     ('is_seed', '<u1'),
                                                                     ('cluster_size', '<u2'),
                                                                     ('n_cluster', '<u2')]))
        expected_hit_result['event_number'] = hits['event_number']
        expected_hit_result['frame'] = hits['frame']
        expected_hit_result['column'] = hits['column']
        expected_hit_result['row'] = hits['row']
        expected_hit_result['charge'] = hits['charge']
        expected_hit_result['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
        expected_hit_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
        expected_hit_result['n_cluster'] = 1

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Test same size array
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Test results
        self.assertTrue(array_size_before == array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Test increasing size array
        hits = create_hits(n_hits=20, max_column=100, max_row=100, max_frame=1, max_charge=2)
        cluster_hits, clusters = clusterizer.cluster_hits(hits)
        array_size_after = clusterizer._clusters.shape[0]

        # Define expected cluster output with extra field
        expected_cluster_result = np.zeros(shape=(7, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4'),
                                                                        ('extra_field', 'f4')]))
        expected_cluster_result['event_number'] = [0, 1, 2, 3, 4, 5, 6]
        expected_cluster_result['n_hits'] = [3, 3, 3, 3, 3, 3, 2]
        expected_cluster_result['charge'] = [1, 2, 1, 2, 1, 2, 1]
        expected_cluster_result['seed_column'] = [2, 4, 8, 10, 14, 16, 20]
        expected_cluster_result['seed_row'] = [3, 7, 15, 19, 27, 31, 39]
        expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, (1 * 19 + 2 * 20) / 3.0]
        expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, (1 * 37 + 2 * 39) / 3.0]
        expected_cluster_result['extra_field'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Define expected hit clustered output
        expected_hit_result = np.zeros(shape=(20, ), dtype=np.dtype([('event_number', '<i8'),
                                                                     ('frame', '<u1'),
                                                                     ('column', '<u2'),
                                                                     ('row', '<u2'),
                                                                     ('charge', '<u2'),
                                                                     ('cluster_ID', '<i2'),
                                                                     ('is_seed', '<u1'),
                                                                     ('cluster_size', '<u2'),
                                                                     ('n_cluster', '<u2')]))
        expected_hit_result['event_number'] = hits['event_number']
        expected_hit_result['frame'] = hits['frame']
        expected_hit_result['column'] = hits['column']
        expected_hit_result['row'] = hits['row']
        expected_hit_result['charge'] = hits['charge']
        expected_hit_result['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        expected_hit_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]
        expected_hit_result['n_cluster'] = 1

        # Test results
        self.assertTrue(array_size_before < array_size_after)
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClusterizer)
    unittest.TextTestRunner(verbosity=2).run(suite)
