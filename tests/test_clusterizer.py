''' Script to check the correctness of the clustering.
'''

import unittest
import numpy as np

from pixel_clusterizer.clusterizer import HitClusterizer
from pixel_clusterizer import data_struct


def create_hits(n_hits, max_column, max_row, max_frame, max_charge):
    hits = np.ones(shape=(n_hits, ), dtype=data_struct.HitInfo)
    for i in range(n_hits):
        hits[i]['event_number'], hits[i]['frame'], hits[i]['column'], hits[i]['row'], hits[i]['charge'] = i / 3, i % max_frame, i % max_column + 1, 2 * i % max_row + 1, i % max_charge
    return hits


class TestClusterizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):  # remove created files
        pass

    def test_hit_definition(self):  # colum/row has to start at 1 and , otherwise IndexError exception
        clusterizer = HitClusterizer(n_columns=10, n_rows=10, n_frames=2, n_charges=2)
        clusterizer.set_warning_output(False)  # Supress eent alignment warning
        hits = np.zeros(shape=(1, ), dtype=data_struct.HitInfo)
        with self.assertRaises(IndexError):
            clusterizer.add_hits(hits)  # cluster hits with illigal column/row index = 0/0
        hits = np.ones(shape=(1, ), dtype=data_struct.HitInfo)
        hits['column'] = 11
        with self.assertRaises(IndexError):
            clusterizer.add_hits(hits)  # column = 11 is too large for n_columns=10
        hits['frame'] = 2
        with self.assertRaises(IndexError):
            clusterizer.add_hits(hits)  # frame = 2 is too large for n_frames=2
        hits['charge'] = 2
        with self.assertRaises(IndexError):
            clusterizer.add_hits(hits)  # charge = 2 is too large for n_charges=2
        clusterizer.reset()
        hits['column'] = 10
        hits['frame'] = 1
        hits['charge'] = 1
        clusterizer.add_hits(hits)  # column = 10 has too fit for n_columns=10

    def test_clustering(self):  # check with multiple jumps data
        # Create hits and cluster them
        clusterizer = HitClusterizer(n_charges=2)
        # TEST 1
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

        clusterizer.add_hits(hits)  # cluster hits
        cluster_hits, clusters = clusterizer.get_hit_cluster(), clusterizer.get_cluster()
        # Define expected output
        expected_result = np.zeros(shape=(4, ), dtype=data_struct.ClusterInfo)
        expected_result['event_number'] = [0, 1, 2, 3]
        expected_result['size'] = [3, 3, 3, 1]
        expected_result['charge'] = [1, 2, 1, 1]
        expected_result['seed_column'] = [2, 6, 8, 10]
        expected_result['seed_row'] = [3, 11, 15, 19]
        expected_result['mean_column'] = [2.5, 5.5, 8.5, 10.5]
        expected_result['mean_row'] = [3.5, 9.5, 15.5, 19.5]
        # Test results
        self.assertEqual(cluster_hits.shape[0], 0)  # hit clustering not activated, thus this array has to be empty
        self.assertTrue((clusters == expected_result).all())

        # TEST 2
        clusterizer.create_cluster_hit_info_array(True)
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
        clusterizer.add_hits(hits)  # cluster hits
        cluster_hits, clusters = clusterizer.get_hit_cluster(), clusterizer.get_cluster()
        # Define expected output
        expected_result = np.zeros(shape=(10, ), dtype=data_struct.ClusterHitInfo)
        expected_result['event_number'] = hits['event_number']
        expected_result['frame'] = hits['frame']
        expected_result['column'] = hits['column']
        expected_result['row'] = hits['row']
        expected_result['charge'] = hits['charge']
        expected_result['is_seed'] = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1]
        expected_result['cluster_size'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
        expected_result['n_cluster'] = 1
        # Test results
        self.assertEqual(cluster_hits.shape[0], 10)  # hit clustering not activated, thus this array has to be empty
        self.assertTrue(np.array_equal(cluster_hits, expected_result))

    def test_cluster_cuts(self):
        # create some fake data
        hits = np.ones(shape=(2, ), dtype=data_struct.HitInfo)
        hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 17, 36, 30, 19
        hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 36, 6, 19

        # create clusterizer object
        clusterizer = HitClusterizer(n_columns=100, n_rows=100, n_frames=2, n_charges=31)
        clusterizer.create_cluster_hit_info_array(True)

        # Case 1: Test max hit charge cut, accept all hits
        clusterizer.set_max_hit_charge(30)  # only add hits with charge <= 30
        clusterizer.add_hits(hits)  # cluster hits
        # Check cluster
        cluster = clusterizer.get_cluster()
        expected_result = np.zeros(shape=(1, ), dtype=data_struct.ClusterInfo)
        expected_result['event_number'] = [19]
        expected_result['size'] = [2]
        expected_result['charge'] = [36]
        expected_result['seed_column'] = [17]
        expected_result['seed_row'] = [36]
        expected_result['mean_column'] = [17.68420982]
        expected_result['mean_row'] = [36.5]
        self.assertTrue(np.array_equal(cluster, expected_result))
        # Check cluster hit info
        cluster_hits = clusterizer.get_hit_cluster()
        expected_result = np.zeros(shape=(2, ), dtype=data_struct.ClusterHitInfo)
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
        clusterizer.add_hits(hits)  # cluster hits
        # Check cluster
        cluster = clusterizer.get_cluster()
        expected_result = np.zeros(shape=(1, ), dtype=data_struct.ClusterInfo)
        expected_result['event_number'] = [20]
        expected_result['size'] = [1]
        expected_result['charge'] = [6]
        expected_result['seed_column'] = [18]
        expected_result['seed_row'] = [36]
        expected_result['mean_column'] = [18.5]
        expected_result['mean_row'] = [36.5]
        self.assertTrue(np.array_equal(cluster, expected_result))
        # Check cluster hit info
        cluster_hits = clusterizer.get_hit_cluster()
        expected_result = np.zeros(shape=(2, ), dtype=data_struct.ClusterHitInfo)
        expected_result['event_number'] = hits['event_number']
        expected_result['frame'] = hits['frame']
        expected_result['column'] = hits['column']
        expected_result['row'] = hits['row']
        expected_result['charge'] = hits['charge']
        expected_result['cluster_ID'] = [-1, 0]
        expected_result['is_seed'] = [0, 1]
        expected_result['cluster_size'] = [0, 1]
        expected_result['n_cluster'] = [0, 1]
        self.assertTrue(np.array_equal(cluster_hits, expected_result))

    def test_not_implemented(self):
        clusterizer = HitClusterizer(n_columns=1, n_rows=1)
        with self.assertRaises(NotImplementedError):
            clusterizer.set_max_cluster_charge(0)
        with self.assertRaises(NotImplementedError):
            clusterizer.set_max_cluster_hit_charge(0)

    def test_exceptions(self):
        with self.assertRaises(MemoryError):
            HitClusterizer(n_columns=1000000, n_rows=1000000)  # Memory exception

        # create some fake data
        hits = np.ones(shape=(2, ), dtype=data_struct.HitInfo)
        hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 17, 36, 3, 19
        hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 36, 6, 19

        clusterizer = HitClusterizer(n_columns=10, n_rows=10, n_frames=2, n_charges=4)

        with self.assertRaises(IndexError):
            clusterizer.add_hits(hits)  # hits col/row too large

        clusterizer = HitClusterizer(n_columns=100, n_rows=100, n_frames=2, n_charges=7)
        clusterizer.set_cluster_info_array_size(0)  # set cluster array size to 0
        hits['event_number'] = 20

        with self.assertRaises(IndexError):
            clusterizer.add_hits(hits)  # cluster array size too small

        clusterizer.set_cluster_info_array_size(1)  # set cluster array size to 1
        clusterizer.create_cluster_hit_info_array(True)
        clusterizer.set_cluster_hit_info_array_size(0)  # set cluster array size to 0
        hits['event_number'] = 21

        with self.assertRaises(IndexError):
            clusterizer.add_hits(hits)  # cluster hit array size too small

        hits['event_number'] = 22
        clusterizer.set_cluster_hit_info_array_size(2)  # set cluster array size to 0

        with self.assertRaises(IndexError):
            clusterizer.add_hits(hits)  # although the size fits now, there are old hit still stored, thus clusterizer.reset has to be called

        clusterizer.reset()
        clusterizer.add_hits(hits)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClusterizer)
    unittest.TextTestRunner(verbosity=2).run(suite)
