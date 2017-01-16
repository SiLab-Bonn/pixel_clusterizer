''' Script to check the correctness of the clustering.
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
        # TEST 1: Set Custom mapping that is correct and should not throw an exception
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
        _ = HitClusterizer(hit_fields=hit_mapping, hit_dtype=hit_dtype, pure_python=self.pure_python)
        # TEST 2: Set custom clustered hit struct that is incorrect and should throw an exception
        hit_dtype_new = np.dtype([('not_defined', '<i8'),
                                  ('frame', '<u1'),
                                  ('column', '<u2'),
                                  ('row', '<u2'),
                                  ('charge', '<u2')])
        with self.assertRaises(ValueError):
            _ = HitClusterizer(hit_fields=hit_mapping, hit_dtype=hit_dtype_new, pure_python=self.pure_python)
        # TEST 3 Set custom and correct hit mapping, no eception expected
        hit_mapping = {'not_defined': 'event_number',
                       'column': 'column',
                       'row': 'row',
                       'charge': 'charge',
                       'frame': 'frame'
                       }
        _ = HitClusterizer(hit_fields=hit_mapping, hit_dtype=hit_dtype_new, pure_python=self.pure_python)

    def test_cluster_algorithm(self):  # Check with multiple jumps data
        # Inititalize Clusterizer
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)

        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

        cluster_hits, clusters = clusterizer.cluster_hits(hits)  # cluster hits

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

        # Define expected output
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

    def test_disabled_pixels(self):
        # Create some fake data
        hits = np.ones(shape=(7, ), dtype=np.dtype([('event_number', '<i8'),
                                                    ('frame', '<u1'),
                                                    ('column', '<u2'),
                                                    ('row', '<u2'),
                                                    ('charge', '<u2')]))
        hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'], hits[0]['frame'] = 1, 2, 4, 0, 0
        hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'], hits[1]['frame'] = 2, 2, 4, 0, 0
        hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'], hits[2]['frame'] = 2, 2, 5, 1, 10
        hits[3]['column'], hits[3]['row'], hits[3]['charge'], hits[3]['event_number'], hits[3]['frame'] = 2, 2, 6, 2, 0
        hits[4]['column'], hits[4]['row'], hits[4]['charge'], hits[4]['event_number'], hits[4]['frame'] = 2, 3, 6, 2, 0
        hits[5]['column'], hits[5]['row'], hits[5]['charge'], hits[5]['event_number'], hits[5]['frame'] = 3, 3, 6, 2, 0
        hits[6]['column'], hits[6]['row'], hits[6]['charge'], hits[6]['event_number'], hits[6]['frame'] = 3, 3, 7, 3, 11

        # Create clusterizer object
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)

        # Case 1: Test max hit charge cut, accept all hits
        cluster_hits, clusters = clusterizer.cluster_hits(hits, disabled_pixels=[[2, 2], [3, 3]])  # cluster hits

        # Check cluster
        expected_cluster_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0, 2]
        expected_cluster_result['ID'] = [0, 0]
        expected_cluster_result['n_hits'] = [1, 1]
        expected_cluster_result['charge'] = [4, 6]
        expected_cluster_result['seed_column'] = [1, 2]
        expected_cluster_result['seed_row'] = [2, 3]
        expected_cluster_result['mean_column'] = [1.0, 2.0]
        expected_cluster_result['mean_row'] = [2.0, 3.0]

        expected_hit_result = np.zeros(shape=(7, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, -1, -1, -1, 0, -1, -1]
        expected_hit_result['is_seed'] = [1, 0, 0, 0, 1, 0, 0]
        expected_hit_result['cluster_size'] = [1, 0, 0, 0, 1, 0, 0]
        expected_hit_result['n_cluster'] = [1, 1, 0, 1, 1, 1, 0]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

    def test_noisy_pixels(self):
        # Create some fake data
        hits = np.ones(shape=(9, ), dtype=np.dtype([('event_number', '<i8'),
                                                    ('frame', '<u1'),
                                                    ('column', '<u2'),
                                                    ('row', '<u2'),
                                                    ('charge', '<u2')]))
        hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'], hits[0]['frame'] = 1, 2, 8, 0, 0
        hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'], hits[1]['frame'] = 2, 2, 4, 0, 0
        hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'], hits[2]['frame'] = 2, 2, 5, 1, 10
        hits[3]['column'], hits[3]['row'], hits[3]['charge'], hits[3]['event_number'], hits[3]['frame'] = 2, 2, 12, 2, 0
        hits[4]['column'], hits[4]['row'], hits[4]['charge'], hits[4]['event_number'], hits[4]['frame'] = 2, 3, 6, 2, 0
        hits[5]['column'], hits[5]['row'], hits[5]['charge'], hits[5]['event_number'], hits[5]['frame'] = 3, 3, 3, 2, 0
        hits[6]['column'], hits[6]['row'], hits[6]['charge'], hits[6]['event_number'], hits[6]['frame'] = 3, 3, 7, 3, 11
        hits[7]['column'], hits[7]['row'], hits[7]['charge'], hits[7]['event_number'], hits[7]['frame'] = 3, 15, 1, 4, 0
        hits[8]['column'], hits[8]['row'], hits[8]['charge'], hits[8]['event_number'], hits[8]['frame'] = 20, 15, 1, 5, 0

        # Create clusterizer object
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)

        # Case 1: Test max hit charge cut, accept all hits
        cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=[[2, 2], [3, 3], [3, 15]])  # cluster hits

        # Check cluster
        expected_cluster_result = np.zeros(shape=(3, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0, 2, 5]
        expected_cluster_result['ID'] = [0, 0, 0]
        expected_cluster_result['n_hits'] = [2, 3, 1]
        expected_cluster_result['charge'] = [8 + 4, 12 + 6 + 3, 1]
        expected_cluster_result['seed_column'] = [1, 2, 20]
        expected_cluster_result['seed_row'] = [2, 2, 15]
        expected_cluster_result['mean_column'] = [(9 * 1 + 5 * 2) / float(9 + 5), (13 * 2 + 7 * 2 + 4 * 3) / float(13 + 7 + 4), 20]
        expected_cluster_result['mean_row'] = [(9 * 2 + 5 * 2) / float(9 + 5), (13 * 2 + 7 * 3 + 4 * 3) / float(13 + 7 + 4), 15]

        expected_hit_result = np.zeros(shape=(9, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0, -1, 0, 0, 0, -1, -1, 0]
        expected_hit_result['is_seed'] = [1, 0, 0, 1, 0, 0, 0, 0, 1]
        expected_hit_result['cluster_size'] = [2, 2, 0, 3, 3, 3, 0, 0, 1]
        expected_hit_result['n_cluster'] = [1, 1, 0, 1, 1, 1, 0, 0, 1]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

    def test_noisy_and_disabled_pixels(self):
        # Create single hit data
        hits = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                    ('frame', '<u1'),
                                                    ('column', '<u2'),
                                                    ('row', '<u2'),
                                                    ('charge', '<u2')]))
        hits[0]['column'], hits[0]['row'] = 1, 1

        # Case 1: Test single noisy pixel
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=[[1, 1]])

        expected_cluster_result = np.zeros(shape=(0, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))

        expected_hit_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [-1]
        expected_hit_result['is_seed'] = [0]
        expected_hit_result['cluster_size'] = [0]
        expected_hit_result['n_cluster'] = [0]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Case 2: Test single disabled pixel
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, disabled_pixels=[[1, 1]])

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Create double hit data
        hits = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
                                                    ('frame', '<u1'),
                                                    ('column', '<u2'),
                                                    ('row', '<u2'),
                                                    ('charge', '<u2')]))
        hits[0]['column'], hits[0]['row'] = 1, 1
        hits[1]['column'], hits[1]['row'] = 1, 2

        # Case 3: Test double noisy pixel
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=[[1, 1], [1, 2]])

        expected_cluster_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0]
        expected_cluster_result['ID'] = [0]
        expected_cluster_result['n_hits'] = [2]
        expected_cluster_result['charge'] = [0]
        expected_cluster_result['seed_column'] = [1]
        expected_cluster_result['seed_row'] = [1]
        expected_cluster_result['mean_column'] = [1.0]
        expected_cluster_result['mean_row'] = [1.5]

        expected_hit_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0]
        expected_hit_result['is_seed'] = [1, 0]
        expected_hit_result['cluster_size'] = [2, 2]
        expected_hit_result['n_cluster'] = [1, 1]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Case 4: Test double noisy pixel
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, disabled_pixels=[[1, 1], [1, 2]])

        expected_cluster_result = np.zeros(shape=(0, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))

        expected_hit_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [-1, -1]
        expected_hit_result['is_seed'] = [0, 0]
        expected_hit_result['cluster_size'] = [0, 0]
        expected_hit_result['n_cluster'] = [0, 0]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Case 5: Test noisy and disabled pixel
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=[[1, 1]], disabled_pixels=[[1, 2]])

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Create triple hit data
        hits = np.zeros(shape=(3, ), dtype=np.dtype([('event_number', '<i8'),
                                                    ('frame', '<u1'),
                                                    ('column', '<u2'),
                                                    ('row', '<u2'),
                                                    ('charge', '<u2')]))
        hits[0]['column'], hits[0]['row'] = 1, 1
        hits[1]['column'], hits[1]['row'] = 1, 2
        hits[2]['column'], hits[2]['row'] = 1, 3

        # Case 6: Test triple pixel
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, disabled_pixels=[[1, 1]], noisy_pixels=[[1, 2], [1, 3]])

        expected_cluster_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0]
        expected_cluster_result['ID'] = [0]
        expected_cluster_result['n_hits'] = [2]
        expected_cluster_result['charge'] = [0]
        expected_cluster_result['seed_column'] = [1]
        expected_cluster_result['seed_row'] = [2]
        expected_cluster_result['mean_column'] = [1.0]
        expected_cluster_result['mean_row'] = [2.5]

        expected_hit_result = np.zeros(shape=(3, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [-1, 0, 0]
        expected_hit_result['is_seed'] = [0, 1, 0]
        expected_hit_result['cluster_size'] = [0, 2, 2]
        expected_hit_result['n_cluster'] = [1, 1, 1]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Create quadruple hit data
        hits = np.zeros(shape=(4, ), dtype=np.dtype([('event_number', '<i8'),
                                                    ('frame', '<u1'),
                                                    ('column', '<u2'),
                                                    ('row', '<u2'),
                                                    ('charge', '<u2')]))
        hits[0]['column'], hits[0]['row'] = 1, 1
        hits[1]['column'], hits[1]['row'] = 1, 2
        hits[2]['column'], hits[2]['row'] = 1, 3
        hits[3]['column'], hits[3]['row'] = 1, 4

        # Case 7: Test quadruple pixel with single disabled pixel
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, disabled_pixels=[[1, 3]])

        expected_cluster_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0, 0]
        expected_cluster_result['ID'] = [0, 1]
        expected_cluster_result['n_hits'] = [2, 1]
        expected_cluster_result['charge'] = [0, 0]
        expected_cluster_result['seed_column'] = [1, 1]
        expected_cluster_result['seed_row'] = [1, 4]
        expected_cluster_result['mean_column'] = [1.0, 1.0]
        expected_cluster_result['mean_row'] = [1.5, 4.0]

        expected_hit_result = np.zeros(shape=(4, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0, -1, 1]
        expected_hit_result['is_seed'] = [1, 0, 0, 1]
        expected_hit_result['cluster_size'] = [2, 2, 0, 1]
        expected_hit_result['n_cluster'] = [2, 2, 2, 2]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Case 8: Test quadruple pixel with single noisy pixel
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=[[1, 3]])

        expected_cluster_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0]
        expected_cluster_result['ID'] = [0]
        expected_cluster_result['n_hits'] = [4]
        expected_cluster_result['charge'] = [0]
        expected_cluster_result['seed_column'] = [1]
        expected_cluster_result['seed_row'] = [1]
        expected_cluster_result['mean_column'] = [1.0]
        expected_cluster_result['mean_row'] = [2.5]

        expected_hit_result = np.zeros(shape=(4, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0, 0, 0]
        expected_hit_result['is_seed'] = [1, 0, 0, 0]
        expected_hit_result['cluster_size'] = [4, 4, 4, 4]
        expected_hit_result['n_cluster'] = [1, 1, 1, 1]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Create quintuple hit data
        hits = np.zeros(shape=(5, ), dtype=np.dtype([('event_number', '<i8'),
                                                    ('frame', '<u1'),
                                                    ('column', '<u2'),
                                                    ('row', '<u2'),
                                                    ('charge', '<u2')]))
        hits[0]['column'], hits[0]['row'] = 1, 1
        hits[1]['column'], hits[1]['row'] = 1, 2
        hits[2]['column'], hits[2]['row'] = 1, 3
        hits[3]['column'], hits[3]['row'] = 1, 4
        hits[4]['column'], hits[4]['row'] = 1, 5

        # Case 9: Test quintuple pixel with 2 disabled pixels
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, disabled_pixels=[[1, 3], [1, 4]])

        expected_cluster_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0, 0]
        expected_cluster_result['ID'] = [0, 1]
        expected_cluster_result['n_hits'] = [2, 1]
        expected_cluster_result['charge'] = [0, 0]
        expected_cluster_result['seed_column'] = [1, 1]
        expected_cluster_result['seed_row'] = [1, 5]
        expected_cluster_result['mean_column'] = [1.0, 1.0]
        expected_cluster_result['mean_row'] = [1.5, 5.0]

        expected_hit_result = np.zeros(shape=(5, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0, -1, -1, 1]
        expected_hit_result['is_seed'] = [1, 0, 0, 0, 1]
        expected_hit_result['cluster_size'] = [2, 2, 0, 0, 1]
        expected_hit_result['n_cluster'] = [2, 2, 2, 2, 2]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Case 10: Test quintuple pixel with 2 noisy pixel
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=[[1, 3]])

        expected_cluster_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0]
        expected_cluster_result['ID'] = [0]
        expected_cluster_result['n_hits'] = [5]
        expected_cluster_result['charge'] = [0]
        expected_cluster_result['seed_column'] = [1]
        expected_cluster_result['seed_row'] = [1]
        expected_cluster_result['mean_column'] = [1.0]
        expected_cluster_result['mean_row'] = [3.0]

        expected_hit_result = np.zeros(shape=(5, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0, 0, 0, 0]
        expected_hit_result['is_seed'] = [1, 0, 0, 0, 0]
        expected_hit_result['cluster_size'] = [5, 5, 5, 5, 5]
        expected_hit_result['n_cluster'] = [1, 1, 1, 1, 1]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Case 11: Test quintuple pixel with single noisy and disabled pixels
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=[[1, 3]], disabled_pixels=[[1, 4]])

        expected_cluster_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0, 0]
        expected_cluster_result['ID'] = [0, 1]
        expected_cluster_result['n_hits'] = [3, 1]
        expected_cluster_result['charge'] = [0, 0]
        expected_cluster_result['seed_column'] = [1, 1]
        expected_cluster_result['seed_row'] = [1, 5]
        expected_cluster_result['mean_column'] = [1.0, 1.0]
        expected_cluster_result['mean_row'] = [2.0, 5.0]

        expected_hit_result = np.zeros(shape=(5, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0, 0, -1, 1]
        expected_hit_result['is_seed'] = [1, 0, 0, 0, 1]
        expected_hit_result['cluster_size'] = [3, 3, 3, 0, 1]
        expected_hit_result['n_cluster'] = [2, 2, 2, 2, 2]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Case 12: Test quintuple pixel with single noisy and disabled pixels
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=2, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=[[1, 3]], disabled_pixels=[[1, 4]])

        expected_cluster_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0]
        expected_cluster_result['ID'] = [0]
        expected_cluster_result['n_hits'] = [4]
        expected_cluster_result['charge'] = [0]
        expected_cluster_result['seed_column'] = [1]
        expected_cluster_result['seed_row'] = [1]
        expected_cluster_result['mean_column'] = [1.0]
        expected_cluster_result['mean_row'] = [2.75]

        expected_hit_result = np.zeros(shape=(5, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0, 0, -1, 0]
        expected_hit_result['is_seed'] = [1, 0, 0, 0, 0]
        expected_hit_result['cluster_size'] = [4, 4, 4, 0, 4]
        expected_hit_result['n_cluster'] = [1, 1, 1, 1, 1]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Create sextuple hit data
        hits = np.zeros(shape=(6, ), dtype=np.dtype([('event_number', '<i8'),
                                                    ('frame', '<u1'),
                                                    ('column', '<u2'),
                                                    ('row', '<u2'),
                                                    ('charge', '<u2')]))
        hits[0]['column'], hits[0]['row'] = 1, 1
        hits[1]['column'], hits[1]['row'] = 1, 2
        hits[2]['column'], hits[2]['row'] = 1, 3
        hits[3]['column'], hits[3]['row'] = 1, 4
        hits[4]['column'], hits[4]['row'] = 1, 5
        hits[5]['column'], hits[5]['row'] = 1, 6

        # Case 13: Test sextuple pixel with noisy and disabled pixels
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=[[1, 3], [1, 5]], disabled_pixels=[[1, 4]])

        expected_cluster_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0, 0]
        expected_cluster_result['ID'] = [0, 1]
        expected_cluster_result['n_hits'] = [3, 2]
        expected_cluster_result['charge'] = [0, 0]
        expected_cluster_result['seed_column'] = [1, 1]
        expected_cluster_result['seed_row'] = [1, 5]
        expected_cluster_result['mean_column'] = [1.0, 1.0]
        expected_cluster_result['mean_row'] = [2.0, 5.5]

        expected_hit_result = np.zeros(shape=(6, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0, 0, -1, 1, 1]
        expected_hit_result['is_seed'] = [1, 0, 0, 0, 1, 0]
        expected_hit_result['cluster_size'] = [3, 3, 3, 0, 2, 2]
        expected_hit_result['n_cluster'] = [2, 2, 2, 2, 2, 2]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Case 14: Test sextuple pixel with noisy and disabled pixels
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=2, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=[[1, 3], [1, 5]], disabled_pixels=[[1, 4]])

        expected_cluster_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0]
        expected_cluster_result['ID'] = [0]
        expected_cluster_result['n_hits'] = [5]
        expected_cluster_result['charge'] = [0]
        expected_cluster_result['seed_column'] = [1]
        expected_cluster_result['seed_row'] = [1]
        expected_cluster_result['mean_column'] = [1.0]
        expected_cluster_result['mean_row'] = [3.4]

        expected_hit_result = np.zeros(shape=(6, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0, 0, -1, 0, 0]
        expected_hit_result['is_seed'] = [1, 0, 0, 0, 0, 0]
        expected_hit_result['cluster_size'] = [5, 5, 5, 0, 5, 5]
        expected_hit_result['n_cluster'] = [1, 1, 1, 1, 1, 1]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Case 15: Test sextuple pixel with noisy and disabled pixels
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, disabled_pixels=[[1, 3], [1, 5]], noisy_pixels=[[1, 4]])

        expected_cluster_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0, 0]
        expected_cluster_result['ID'] = [0, 1]
        expected_cluster_result['n_hits'] = [2, 1]
        expected_cluster_result['charge'] = [0, 0]
        expected_cluster_result['seed_column'] = [1, 1]
        expected_cluster_result['seed_row'] = [1, 6]
        expected_cluster_result['mean_column'] = [1.0, 1.0]
        expected_cluster_result['mean_row'] = [1.5, 6.0]

        expected_hit_result = np.zeros(shape=(6, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0, -1, -1, -1, 1]
        expected_hit_result['is_seed'] = [1, 0, 0, 0, 0, 1]
        expected_hit_result['cluster_size'] = [2, 2, 0, 0, 0, 1]
        expected_hit_result['n_cluster'] = [2, 2, 2, 2, 2, 2]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Case 16: Test sextuple pixel with noisy and disabled pixels
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=1, row_cluster_distance=2, frame_cluster_distance=1, ignore_same_hits=True)
        cluster_hits, clusters = clusterizer.cluster_hits(hits, disabled_pixels=[[1, 3], [1, 5]], noisy_pixels=[[1, 4]])

        expected_cluster_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [0]
        expected_cluster_result['ID'] = [0]
        expected_cluster_result['n_hits'] = [4]
        expected_cluster_result['charge'] = [0]
        expected_cluster_result['seed_column'] = [1]
        expected_cluster_result['seed_row'] = [1]
        expected_cluster_result['mean_column'] = [1.0]
        expected_cluster_result['mean_row'] = [3.25]

        expected_hit_result = np.zeros(shape=(6, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [0, 0, -1, 0, -1, 0]
        expected_hit_result['is_seed'] = [1, 0, 0, 0, 0, 0]
        expected_hit_result['cluster_size'] = [4, 4, 0, 4, 0, 4]
        expected_hit_result['n_cluster'] = [1, 1, 1, 1, 1, 1]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

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
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)

        # Case 1: Test max hit charge cut, accept all hits
        clusterizer.set_max_hit_charge(30)  # only add hits with charge <= 30
        cluster_hits, clusters = clusterizer.cluster_hits(hits)  # cluster hits

        # Check cluster
        expected_cluster_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [19]
        expected_cluster_result['n_hits'] = [2]
        expected_cluster_result['charge'] = [36]
        expected_cluster_result['seed_column'] = [17]
        expected_cluster_result['seed_row'] = [36]
        expected_cluster_result['mean_column'] = [17.18420982]
        expected_cluster_result['mean_row'] = [36.0]

        # Check cluster hit info
        expected_hit_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['is_seed'] = [1, 0]
        expected_hit_result['cluster_size'] = [2, 2]
        expected_hit_result['n_cluster'] = 1

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        # Case 2: Test max hit charge cut, omit charge > 29 hits
        hits['event_number'] = 20
        clusterizer.set_max_hit_charge(29)  # only add hits with charge <= 30
        cluster_hits, clusters = clusterizer.cluster_hits(hits)  # cluster hits
        # Check cluster
        expected_cluster_result = np.zeros(shape=(1, ), dtype=np.dtype([('event_number', '<i8'),
                                                                        ('ID', '<u2'),
                                                                        ('n_hits', '<u2'),
                                                                        ('charge', 'f4'),
                                                                        ('seed_column', '<u2'),
                                                                        ('seed_row', '<u2'),
                                                                        ('mean_column', 'f4'),
                                                                        ('mean_row', 'f4')]))
        expected_cluster_result['event_number'] = [20]
        expected_cluster_result['n_hits'] = [1]
        expected_cluster_result['charge'] = [6]
        expected_cluster_result['seed_column'] = [18]
        expected_cluster_result['seed_row'] = [36]
        expected_cluster_result['mean_column'] = [18.0]
        expected_cluster_result['mean_row'] = [36.0]

        # Check cluster hit info
        expected_hit_result = np.zeros(shape=(2, ), dtype=np.dtype([('event_number', '<i8'),
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
        expected_hit_result['cluster_ID'] = [-1, 0]
        expected_hit_result['is_seed'] = [0, 1]
        expected_hit_result['cluster_size'] = [0, 1]
        expected_hit_result['n_cluster'] = [1, 1]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

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
        cluster_hits, clusters = clusterizer.cluster_hits(hits)  # Cluster hits

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

        clusterizer.ignore_same_hits(False)  # If a hit occured 2 times in an event it is used as a normal hit
        cluster_hits, clusters = clusterizer.cluster_hits(hits)  # Cluster hits

        expected_hit_result['cluster_ID'] = [0, 0, 0]
        expected_hit_result['is_seed'] = [1, 0, 0]
        expected_hit_result['cluster_size'] = [3, 3, 3]
        expected_hit_result['n_cluster'] = [1, 1, 1]
        expected_cluster_result['n_hits'] = [3]
        expected_cluster_result['charge'] = [18]
        expected_cluster_result['mean_row'] = [(2 * 36 + 38) / 3.0]

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
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
                               ('parameter_2', 'f4')])

        # Initialize clusterizer
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)

        for hit_data_type in hit_data_types:
            clusterizer.set_hit_dtype(np.dtype(hit_data_type))
            # Create fake data with actual hit data structure
            hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_dtype=np.dtype(hit_data_type))
            cluster_hits, clusters = clusterizer.cluster_hits(hits)  # Cluster hits
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
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)

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
        clusterizer.add_cluster_field(description=('extra_field', 'f4'))

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

    def test_set_end_of_cluster_function(self):
        # Initialize clusterizer object
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)

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
        def end_of_cluster_function(hits, clusters, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels, disabled_pixels, seed_hit_index):
            clusters[cluster_index]['seed_charge'] = hits[seed_hit_index]['charge']

        clusterizer.set_end_of_cluster_function(end_of_cluster_function)  # Set the new end_of_cluster_function

        # Main function
        cluster_hits, clusters = clusterizer.cluster_hits(hits)  # cluster hits

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

    def test_set_end_of_event_function(self):
        # Initialize clusterizer object
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)

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
        def end_of_event_function(hits, clusters, start_event_hit_index, stop_event_hit_index, start_event_cluster_index, stop_event_cluster_index):
            # Set the number of clusters info (n_cluster)for clusters of the event
            for i in range(start_event_cluster_index, stop_event_cluster_index):
                clusters[i]['n_cluster'] = hits["n_cluster"][start_event_hit_index]

        clusterizer.set_end_of_event_function(end_of_event_function)  # Set the new end_of_cluster_function

        # Main function
        cluster_hits, clusters = clusterizer.cluster_hits(hits)  # cluster hits

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

    def test_chunked_clustering(self):  # Big tables have to be chunked and analyzed with clusterizer.cluster_hits(hits_chunk) calls
        clusterizer = HitClusterizer(pure_python=self.pure_python, min_hit_charge=0, max_hit_charge=13, column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=4, ignore_same_hits=True)

        n_hits = 100
        hits = create_hits(n_hits=n_hits, max_column=100, max_row=100, max_frame=1, max_charge=2)

        cluster_hits, clusters = clusterizer.cluster_hits(hits)  # Cluster all at once
        cluster_hits, clusters = cluster_hits.copy(), clusters.copy()  # Be aware that the returned array are references to be stored! An additional call of clusterizer.cluster_hits will overwrite the data

        cluster_hits_chunked, clusters_chunked = None, None
        chunk_size = 6  # Chunk size has to be chosen to not split events between chunks!
        for i in range(int(n_hits / chunk_size + 1)):  # Cluster in chunks
            hits_chunk = hits[i * chunk_size:i * chunk_size + chunk_size]
            cluster_hits_chunk, clusters_chunk = clusterizer.cluster_hits(hits_chunk)
            if cluster_hits_chunked is None:
                cluster_hits_chunked = cluster_hits_chunk.copy()
            else:
                cluster_hits_chunked = np.append(cluster_hits_chunked, cluster_hits_chunk)
            if clusters_chunked is None:
                clusters_chunked = clusters_chunk.copy()
            else:
                clusters_chunked = np.append(clusters_chunked, clusters_chunk)

        # Test results
        self.assertTrue(np.array_equal(clusters, clusters_chunked))
        self.assertTrue(np.array_equal(cluster_hits, cluster_hits_chunked))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClusterizer)
    unittest.TextTestRunner(verbosity=2).run(suite)
