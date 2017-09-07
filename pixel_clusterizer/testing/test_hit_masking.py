''' Script to check the correctness of clustering for masked pixels.

    Masked pixels are pixels that are disabled or noisy.
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


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClusterizer)
    unittest.TextTestRunner(verbosity=2).run(suite)
