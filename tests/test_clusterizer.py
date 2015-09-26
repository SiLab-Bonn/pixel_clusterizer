''' Script to check the correctness of the clustering.
'''

import unittest
import numpy as np

from pixel_clusterizer.clusterizer import HitClusterizer
from pixel_clusterizer import data_struct


def create_hits(n_hits, max_column, max_row):
    hits = np.ones(shape=(n_hits, ), dtype=data_struct.HitInfo)
    for i in range(n_hits):
        hits[i]['event_number'], hits[i]['frame'], hits[i]['column'], hits[i]['row'], hits[i]['charge'] = i / 3, i % 129, i % max_column + 1, 2 * i % max_row + 1, i % 2
    return hits


class TestClusterizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):  # remove created files
        pass

    def test_hit_definition(self):  # colum/row has to start at 1 and , otherwise IndexError exception
        clusterizer = HitClusterizer(n_columns=10, n_rows=10)
        hits = np.zeros(shape=(1, ), dtype=data_struct.HitInfo)
        with self.assertRaises(IndexError):
            clusterizer.add_hits(hits)  # cluster hits with illigal column/row index = 0/0
        hits = np.ones(shape=(1, ), dtype=data_struct.HitInfo)
        hits['column'] = 11
        with self.assertRaises(IndexError):
            clusterizer.add_hits(hits)  # column = 11 is too large for n_columns=10
        clusterizer.reset()
        hits['column'] = 10
        clusterizer.add_hits(hits)  # column = 10 has too fit for n_columns=10

    def test_clustering(self):  # check with multiple jumps data
        # Create hits and cluster them
        clusterizer = HitClusterizer()

        # TEST 1
        hits = create_hits(10, 100, 100)
        clusterizer.add_hits(hits)  # cluster hits
        cluster_hits, cluster = clusterizer.get_hit_cluster(), clusterizer.get_cluster()
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
        self.assertTrue((cluster == expected_result).all())

        # TEST 2
        clusterizer.create_cluster_hit_info_array(True)
        hits = create_hits(10, 100, 100)
        clusterizer.add_hits(hits)  # cluster hits
        cluster_hits, cluster = clusterizer.get_hit_cluster(), clusterizer.get_cluster()
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
        self.assertTrue((cluster_hits == expected_result).all())


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClusterizer)
    unittest.TextTestRunner(verbosity=2).run(suite)
