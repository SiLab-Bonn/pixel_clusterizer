''' Script to check the correctness of the clustering.
'''

import unittest
import numpy as np

from pixel_clusterizer.clusterizer2 import HitClusterizer
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

    #@unittest.SkipTest
    def test_clustering(self):  # check with multiple jumps data
        # Create hits and cluster them
        clusterizer = HitClusterizer()
        # TEST 1
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
 
        clusterizer.add_hits(hits)  # cluster hits
        _, clusters = clusterizer.get_hit_cluster(), clusterizer.get_cluster()
        # Define expected output
        expected_result = np.zeros(shape=(4, ), dtype=data_struct.ClusterInfo)
        expected_result['event_number'] = [0, 1, 2, 3]
        expected_result['n_hits'] = [3, 3, 3, 1]
        expected_result['charge'] = [1, 2, 1, 1]
        expected_result['seed_column'] = [2, 6, 8, 10]
        expected_result['seed_row'] = [3, 11, 15, 19]
        expected_result['mean_column'] = [2.5, 5.5, 8.5, 10.5]
        expected_result['mean_row'] = [3.5, 9.5, 15.5, 19.5]
        # Test results
        # self.assertEqual(cluster_hits.shape[0], 0)  # hit clustering not activated, thus this array has to be empty
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
        self.assertEqual(cluster_hits.shape[0], 10)  # hit clustering activated, thus this array have 10 entries
        self.assertTrue(np.array_equal(cluster_hits, expected_result))

    #@unittest.SkipTest
    def test_cluster_cuts(self):
        # create some fake data
        hits = np.ones(shape=(2, ), dtype=data_struct.HitInfo)
        hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 17, 36, 30, 19
        hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 36, 6, 19

        # create clusterizer object
        clusterizer = HitClusterizer()
        clusterizer.create_cluster_hit_info_array(True)

        # Case 1: Test max hit charge cut, accept all hits
        clusterizer.set_max_hit_charge(30)  # only add hits with charge <= 30
        clusterizer.add_hits(hits)  # cluster hits
        # Check cluster
        cluster = clusterizer.get_cluster()
        expected_result = np.zeros(shape=(1, ), dtype=data_struct.ClusterInfo)
        expected_result['event_number'] = [19]
        expected_result['n_hits'] = [2]
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
        expected_result['n_hits'] = [1]
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


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClusterizer)
    unittest.TextTestRunner(verbosity=2).run(suite)
