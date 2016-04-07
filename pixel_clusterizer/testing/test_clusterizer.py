''' Script to check the correctness of the clustering.
'''

import unittest
import os
import numpy as np

from docutils.utils.roman import OutOfRangeError  # For numba exception

from pixel_clusterizer.clusterizer import HitClusterizer


def create_hits(n_hits, max_column, max_row, max_frame, max_charge, hit_dtype=np.dtype([('event_number', '<i8'),
                                                                                        ('frame', '<u1'),
                                                                                        ('column', '<u2'),
                                                                                        ('row', '<u2'),
                                                                                        ('charge', '<u2')]), hit_fields=None):
    hits = np.ones(shape=(n_hits, ), dtype=hit_dtype)
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

    def test_exceptions(self):
        # TEST 1: Check to add more hits than supported
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

        clusterizer = HitClusterizer(pure_python=self.pure_python)
        clusterizer.set_max_cluster_hits(1)
        with self.assertRaises(OutOfRangeError):
            clusterizer.cluster_hits(hits)
        # TEST 2: Set Custom mapping that is correct and should not throw an exception
        hit_mapping = {'event_number': 'event_number',
                       'column': 'column',
                       'row': 'row',
                       'charge': 'charge',
                       'frame': 'frame'
                       }
        hit_dtype = np.dtype([('event_number', '<i8'),
                              ('frame', '<u1'),
                              ('column', '<u2'),
                              ('row', '<u2'),
                              ('charge', '<u2')])
        clusterizer = HitClusterizer(hit_fields=hit_mapping, hit_dtype=hit_dtype, pure_python=self.pure_python)
        # TEST 3: Set custom clustered hit struct that is incorrect and should throw an exception
        hit_dtype_new = np.dtype([('not_defined', '<i8'),
                                  ('frame', '<u1'),
                                  ('column', '<u2'),
                                  ('row', '<u2'),
                                  ('charge', '<u2')])
        with self.assertRaises(ValueError):
            clusterizer = HitClusterizer(hit_fields=hit_mapping, hit_dtype=hit_dtype_new, pure_python=self.pure_python)
        # TEST 4 Set custom and correct hit mapping, no eception expected
        hit_mapping = {'not_defined': 'event_number',
                       'column': 'column',
                       'row': 'row',
                       'charge': 'charge',
                       'frame': 'frame'
                       }
        clusterizer = HitClusterizer(hit_fields=hit_mapping, hit_dtype=hit_dtype_new, pure_python=self.pure_python)

    def test_cluster_algorithm(self):  # Check with multiple jumps data
        # Inititalize Clusterizer
        clusterizer = HitClusterizer(pure_python=self.pure_python)

        # TEST 1
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

        clusterizer.cluster_hits(hits)  # cluster hits
        _, clusters = clusterizer.get_hit_cluster(), clusterizer.get_cluster()

        # Define expected output
        expected_result = np.zeros(shape=(4, ), dtype=np.dtype([('event_number', '<i8'),
                                                                ('ID', '<u2'),
                                                                ('n_hits', '<u2'),
                                                                ('charge', 'f4'),
                                                                ('seed_column', '<u2'),
                                                                ('seed_row', '<u2'),
                                                                ('mean_column', 'f4'),
                                                                ('mean_row', 'f4')]))
        expected_result['event_number'] = [0, 1, 2, 3]
        expected_result['n_hits'] = [3, 3, 3, 1]
        expected_result['charge'] = [1, 2, 1, 1]
        expected_result['seed_column'] = [2, 4, 8, 10]
        expected_result['seed_row'] = [3, 7, 15, 19]
        expected_result['mean_column'] = [2.0, 5.0, 8.0, 10.0]
        expected_result['mean_row'] = [3.0, 9.0, 15.0, 19.0]

        # Test results
        self.assertTrue((clusters == expected_result).all())

        # TEST 2
        clusterizer.create_cluster_hit_info_array(True)
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

        clusterizer.cluster_hits(hits)  # cluster hits
        cluster_hits, clusters = clusterizer.get_hit_cluster(), clusterizer.get_cluster()

        # Define expected output
        expected_result = np.zeros(shape=(10, ), dtype=np.dtype([('event_number', '<i8'),
                                                                 ('frame', '<u1'),
                                                                 ('column', '<u2'),
                                                                 ('row', '<u2'),
                                                                 ('charge', '<u2'),
                                                                 ('cluster_ID', '<i2'),
                                                                 ('is_seed', '<u1'),
                                                                 ('cluster_size', '<u2'),
                                                                 ('n_cluster', '<u2')]))
        expected_result['event_number'] = hits['event_number']
        expected_result['frame'] = hits['frame']
        expected_result['column'] = hits['column']
        expected_result['row'] = hits['row']
        expected_result['charge'] = hits['charge']
        expected_result['is_seed'] = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
        expected_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
        expected_result['n_cluster'] = 1

        # Test results
        self.assertEqual(cluster_hits.shape[0], 10)  # hit clustering activated, thus this array have 10 entries
        self.assertTrue(np.array_equal(cluster_hits, expected_result))

    def test_cluster_cuts(self):
        # Create some fake data
        hits = np.ones(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
                                                    ('frame', '<u1'),
                                                    ('column', '<u2'),
                                                    ('row', '<u2'),
                                                    ('charge', '<u2')]))
        hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 17, 36, 30, 19
        hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 36, 6, 19

        # Create clusterizer object
        clusterizer = HitClusterizer(pure_python=self.pure_python)
        clusterizer.create_cluster_hit_info_array(True)

        # Case 1: Test max hit charge cut, accept all hits
        clusterizer.set_max_hit_charge(30)  # only add hits with charge <= 30
        clusterizer.cluster_hits(hits)  # cluster hits

        # Check cluster
        cluster = clusterizer.get_cluster()
        expected_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                ('ID', '<u2'),
                                                                ('n_hits', '<u2'),
                                                                ('charge', 'f4'),
                                                                ('seed_column', '<u2'),
                                                                ('seed_row', '<u2'),
                                                                ('mean_column', 'f4'),
                                                                ('mean_row', 'f4')]))
        expected_result['event_number'] = [19]
        expected_result['n_hits'] = [2]
        expected_result['charge'] = [36]
        expected_result['seed_column'] = [17]
        expected_result['seed_row'] = [36]
        expected_result['mean_column'] = [17.18420982]
        expected_result['mean_row'] = [36.0]

        self.assertTrue(np.array_equal(cluster, expected_result))

        # Check cluster hit info
        cluster_hits = clusterizer.get_hit_cluster()
        expected_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
                                                                ('frame', '<u1'),
                                                                ('column', '<u2'),
                                                                ('row', '<u2'),
                                                                ('charge', '<u2'),
                                                                ('cluster_ID', '<i2'),
                                                                ('is_seed', '<u1'),
                                                                ('cluster_size', '<u2'),
                                                                ('n_cluster', '<u2')]))
        expected_result['event_number'] = hits['event_number']
        expected_result['frame'] = hits['frame']
        expected_result['column'] = hits['column']
        expected_result['row'] = hits['row']
        expected_result['charge'] = hits['charge']
        expected_result['is_seed'] = [1, 0]
        expected_result['cluster_size'] = [2, 2]
        expected_result['n_cluster'] = 1

        self.assertTrue(np.array_equal(cluster_hits, expected_result))

        # Case 2: Test max hit charge cut, omit charge > 29 hits
        hits['event_number'] = 20
        clusterizer.set_max_hit_charge(29)  # only add hits with charge <= 30
        clusterizer.cluster_hits(hits)  # cluster hits
        # Check cluster
        cluster = clusterizer.get_cluster()
        expected_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                ('ID', '<u2'),
                                                                ('n_hits', '<u2'),
                                                                ('charge', 'f4'),
                                                                ('seed_column', '<u2'),
                                                                ('seed_row', '<u2'),
                                                                ('mean_column', 'f4'),
                                                                ('mean_row', 'f4')]))
        expected_result['event_number'] = [20]
        expected_result['n_hits'] = [1]
        expected_result['charge'] = [6]
        expected_result['seed_column'] = [18]
        expected_result['seed_row'] = [36]
        expected_result['mean_column'] = [18.0]
        expected_result['mean_row'] = [36.0]
        self.assertTrue(np.array_equal(cluster, expected_result))

        # Check cluster hit info
        cluster_hits = clusterizer.get_hit_cluster()
        expected_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
                                                                ('frame', '<u1'),
                                                                ('column', '<u2'),
                                                                ('row', '<u2'),
                                                                ('charge', '<u2'),
                                                                ('cluster_ID', '<i2'),
                                                                ('is_seed', '<u1'),
                                                                ('cluster_size', '<u2'),
                                                                ('n_cluster', '<u2')]))
        expected_result['event_number'] = hits['event_number']
        expected_result['frame'] = hits['frame']
        expected_result['column'] = hits['column']
        expected_result['row'] = hits['row']
        expected_result['charge'] = hits['charge']
        expected_result['cluster_ID'] = [-1, 0]
        expected_result['is_seed'] = [0, 1]
        expected_result['cluster_size'] = [0, 1]
        expected_result['n_cluster'] = [1, 1]

        self.assertTrue(np.array_equal(cluster_hits, expected_result))

        # Case 3: Add the same hit within an event
        # Create some fake data
        hits = np.ones(shape=(3, ), dtype=np.dtype([('event_number', '<i8'),
                                                    ('frame', '<u1'),
                                                    ('column', '<u2'),
                                                    ('row', '<u2'),
                                                    ('charge', '<u2')]))
        hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 18, 36, 6, 19
        hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 36, 6, 19
        hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'] = 18, 38, 6, 19

        expected_hit_result = np.zeros(shape=(3, ), dtype=np.dtype([('event_number', '<i8'),
                                                                    ('frame', '<u1'),
                                                                    ('column', '<u2'),
                                                                    ('row', '<u2'),
                                                                    ('charge', '<u2'),
                                                                    ('cluster_ID', '<i2'),
                                                                    ('is_seed', '<u1'),
                                                                    ('cluster_size', '<u2'),
                                                                    ('n_cluster', '<u2')]))
        expected_cluster_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_hit_result['event_number'] = hits['event_number']
        expected_hit_result['frame'] = hits['frame']
        expected_hit_result['column'] = hits['column']
        expected_hit_result['row'] = hits['row']
        expected_hit_result['charge'] = hits['charge']
        expected_hit_result['cluster_ID'] = [0, -2, 0]
        expected_hit_result['is_seed'] = [1, 0, 0]
        expected_hit_result['cluster_size'] = [2, 0, 2]
        expected_hit_result['n_cluster'] = [1, 1, 1]
        expected_cluster_result['event_number'] = [19]
        expected_cluster_result['n_hits'] = [2]
        expected_cluster_result['charge'] = [12]
        expected_cluster_result['seed_column'] = [18]
        expected_cluster_result['seed_row'] = [36]
        expected_cluster_result['mean_column'] = [18.0]
        expected_cluster_result['mean_row'] = [37.0]

        clusterizer.ignore_same_hits(True)  # If a hit occured 2 times in an event it is ignored and gets the cluster index -2
        cluster_hits, cluster = clusterizer.cluster_hits(hits)  # Cluster hits

        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))
        self.assertTrue(np.array_equal(cluster, expected_cluster_result))

        clusterizer.ignore_same_hits(False)  # If a hit occured 2 times in an event it is used as a normal hit
        cluster_hits, cluster = clusterizer.cluster_hits(hits)  # Cluster hits

        expected_hit_result['cluster_ID'] = [0, 0, 0]
        expected_hit_result['is_seed'] = [1, 0, 0]
        expected_hit_result['cluster_size'] = [3, 3, 3]
        expected_hit_result['n_cluster'] = [1, 1, 1]

        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

    def test_different_hit_data_types(self):
        # Define a different hit data structure with standard names but different data types and number of fields. Numba automatically recompiles and the result should not change
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
                               ('parameter', '<i4'),
                               ('parameter2', 'f4')])

        # Initialize clusterizer
        clusterizer = HitClusterizer(pure_python=self.pure_python)

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

        for hit_data_type in hit_data_types:
            clusterizer.set_hit_dtype(np.dtype(hit_data_type))
            # Create fake data with actual hit data structure
            hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=np.dtype(hit_data_type))
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

            hits_clustered, clusters = clusterizer.cluster_hits(hits)  # Cluster hits
            # Test results
            self.assertTrue((clusters == expected_cluster_result).all())
            self.assertTrue((hits_clustered == expected_hit_result).all())

    def test_different_cluster_data_types(self):
        # Define a different hit data structure with standard names but different data types and number of fields. Numba automatically recompiles and the result should not change
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
        clusterizer = HitClusterizer(pure_python=self.pure_python)

        # Create fake data with actual hit data structure
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

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

        for cluster_data_type in cluster_data_types:
            clusterizer.set_cluster_dtype(np.dtype(cluster_data_type))

            # Define expected output
            expected_cluster_result = np.zeros(shape=(4, ), dtype=np.dtype(cluster_data_type))
            expected_cluster_result['event_number'] = [0, 1, 2, 3]
            expected_cluster_result['n_hits'] = [3, 3, 3, 1]
            expected_cluster_result['charge'] = [1, 2, 1, 1]
            expected_cluster_result['seed_column'] = [2, 4, 8, 10]
            expected_cluster_result['seed_row'] = [3, 7, 15, 19]
            expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 10.0]
            expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 19.0]

            hits_clustered, clusters = clusterizer.cluster_hits(hits)  # Cluster hits
            # Test results
            self.assertTrue((clusters == expected_cluster_result).all())
            self.assertTrue((hits_clustered == expected_hit_result).all())

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
        clusterizer = HitClusterizer(hit_fields=hit_fields, hit_dtype=hit_dtype, pure_python=self.pure_python)
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=hit_dtype, hit_fields=hit_fields)
        hits_clustered, clusters = clusterizer.cluster_hits(hits)

        # Define expected output
        expected_cluster_result = np.zeros(shape=(4, ), dtype=np.dtype([('eventNumber', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('tot', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['eventNumber'] = [0, 1, 2, 3]
        expected_cluster_result['n_hits'] = [3, 3, 3, 1]
        expected_cluster_result['tot'] = [1, 2, 1, 1]
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
        clusterizer = HitClusterizer(cluster_fields=cluster_fields, cluster_dtype=cluster_dtype, pure_python=self.pure_python)
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
        hits_clustered, clusters = clusterizer.cluster_hits(hits)

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

        self.assertTrue((clusters == expected_cluster_result).all())
        self.assertTrue((hits_clustered == expected_hit_result).all())

    def test_adding_cluster_field(self):
        clusterizer = HitClusterizer(pure_python=self.pure_python)

        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

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
        expected_cluster_result['extra_field'] = [0., 0., 0., 0.]

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

        clusterizer.add_cluster_field(description=('extra_field', 'f4'))
        hits_clustered, clusters = clusterizer.cluster_hits(hits)

        self.assertTrue((clusters == expected_cluster_result).all())
        self.assertTrue((hits_clustered == expected_hit_result).all())

    def test_set_end_of_cluster_function(self):
        # Initialize clusterizer object
        clusterizer = HitClusterizer(pure_python=self.pure_python)

        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

        # Define expected output
        expected_cluster_result = np.zeros(shape=(4, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4'),
                                                                        ('seed_charge', 'f4')]))
        expected_cluster_result['event_number'] = [0, 1, 2, 3]
        expected_cluster_result['n_hits'] = [3, 3, 3, 1]
        expected_cluster_result['charge'] = [1, 2, 1, 1]
        expected_cluster_result['seed_column'] = [2, 4, 8, 10]
        expected_cluster_result['seed_row'] = [3, 7, 15, 19]
        expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 10.0]
        expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 19.0]
        expected_cluster_result['seed_charge'] = [1., 1., 1., 1.]

        #
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

        clusterizer.add_cluster_field(description=('seed_charge', 'f4'))  # Add an additional field to hold the result of the end_of_cluster_function calculation (here: seed charge)

        # The end of loop function has to define all of the following arguments, even when they are not used
        # It has to be compile able by numba in non python mode
        # This end_of_cluster_function sets the additional seed_charge field
        def end_of_cluster_function(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_cluster_index, actual_event_hit_index, actual_cluster_hit_indices, seed_index):
            cluster[actual_cluster_index]['seed_charge'] = hits[seed_index]['charge']

        clusterizer.set_end_of_cluster_function(end_of_cluster_function)  # Set the new end_of_cluster_function

        # Main function
        hits_clustered, cluster = clusterizer.cluster_hits(hits)  # cluster hits

        self.assertTrue((cluster == expected_cluster_result).all())
        self.assertTrue((hits_clustered == expected_hit_result).all())

    def test_set_end_of_event_function(self):
        # Initialize clusterizer object
        clusterizer = HitClusterizer(pure_python=self.pure_python)

        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

        # Define expected output
        expected_cluster_result = np.zeros(shape=(4, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4'),
                                                                        ('n_cluster', '<u1')]))
        expected_cluster_result['event_number'] = [0, 1, 2, 3]
        expected_cluster_result['n_hits'] = [3, 3, 3, 1]
        expected_cluster_result['charge'] = [1, 2, 1, 1]
        expected_cluster_result['seed_column'] = [2, 4, 8, 10]
        expected_cluster_result['seed_row'] = [3, 7, 15, 19]
        expected_cluster_result['mean_column'] = [2.0, 5.0, 8.0, 10.0]
        expected_cluster_result['mean_row'] = [3.0, 9.0, 15.0, 19.0]
        expected_cluster_result['n_cluster'] = [1, 1, 1, 1]

        #
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

        clusterizer.add_cluster_field(description=('n_cluster', '<u1'))  # Add an additional field to hold the result of the end_of_cluster_function calculation (here: seed charge)

        # The end of loop function has to define all of the following arguments, even when they are not used
        # It has to be compile able by numba in non python mode
        # This end_of_event_function sets the additional n_cluster field
        def end_of_event_function(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_event_hit_index, new_actual_event_hit_index, next_cluster_id, actual_event_cluster_index):
            # Set the number of clusters info (n_cluster)for clusters of the event
            for i in range(actual_event_cluster_index, actual_event_cluster_index + next_cluster_id):
                cluster[i]['n_cluster'] = n_cluster[actual_event_hit_index]

        clusterizer.set_end_of_event_function(end_of_event_function)  # Set the new end_of_cluster_function

        # Main function
        hits_clustered, cluster = clusterizer.cluster_hits(hits)  # cluster hits

        self.assertTrue((cluster == expected_cluster_result).all())
        self.assertTrue((hits_clustered == expected_hit_result).all())

    def test_chunked_clustering(self):  # Big tables have to be chunked and analyzed with clusterizer.cluster_hits(hits_chunk) calls
        clusterizer = HitClusterizer(pure_python=self.pure_python)

        hits = create_hits(n_hits=100, max_column=100, max_row=100, max_frame=1, max_charge=2)

        hits_clustered, cluster = clusterizer.cluster_hits(hits)  # Cluster all at once
        hits_clustered, cluster = hits_clustered.copy(), cluster.copy()  # Be aware that the returned array are references to be stored! An additional call of clusterizer.cluster_hits will overwrite the data

        hits_clustered_chunked, cluster_chunked = None, None
        chunk_size = 6  # Chunk size has to be chosen to not split events between chunks!
        for i in range(int(100 / chunk_size + 1)):  # Cluster in chunks
            hits_chunk = hits[i * chunk_size:i * chunk_size + chunk_size]
            hits_clustered_chunk, cluster_chunk = clusterizer.cluster_hits(hits_chunk)
            if hits_clustered_chunked is None:
                hits_clustered_chunked = hits_clustered_chunk.copy()
            else:
                hits_clustered_chunked = np.append(hits_clustered_chunked, hits_clustered_chunk)
            if cluster_chunked is None:
                cluster_chunked = cluster_chunk.copy()
            else:
                cluster_chunked = np.append(cluster_chunked, cluster_chunk)

        self.assertTrue((hits_clustered == hits_clustered_chunked).all())
        self.assertTrue((cluster == cluster_chunked).all())

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClusterizer)
    unittest.TextTestRunner(verbosity=2).run(suite)
