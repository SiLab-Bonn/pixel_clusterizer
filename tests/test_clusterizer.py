''' Script to check the correctness of the clustering.
'''

import unittest
import numpy as np

from pixel_clusterizer.clusterizer import HitClusterizer
from pixel_clusterizer import data_struct
from docutils.utils.roman import OutOfRangeError  # For numba exception


def create_hits(n_hits, max_column, max_row, max_frame, max_charge, hit_struct=data_struct.HitInfo):
    hits = np.ones(shape=(n_hits, ), dtype=hit_struct)
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

    def test_exceptions(self):
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)
        clusterizer = HitClusterizer()
        clusterizer.set_max_cluster_hits(1)

        with self.assertRaises(OutOfRangeError):
            clusterizer.cluster_hits(hits)

    def test_cluster_algorithm(self):  # Check with multiple jumps data
        # Inititalize Clusterizer
        clusterizer = HitClusterizer()

        # TEST 1
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

        clusterizer.cluster_hits(hits)  # cluster hits
        _, clusters = clusterizer.get_hit_cluster(), clusterizer.get_cluster()

        # Define expected output
        expected_result = np.zeros(shape=(4, ), dtype=data_struct.ClusterInfo)
        expected_result['event_number'] = [0, 1, 2, 3]
        expected_result['n_hits'] = [3, 3, 3, 1]
        expected_result['charge'] = [1, 2, 1, 1]
        expected_result['seed_column'] = [2, 4, 8, 10]
        expected_result['seed_row'] = [3, 7, 15, 19]
        expected_result['mean_column'] = [2.5, 5.5, 8.5, 10.5]
        expected_result['mean_row'] = [3.5, 9.5, 15.5, 19.5]

        # Test results
        self.assertTrue((clusters == expected_result).all())

        # TEST 2
        clusterizer.create_cluster_hit_info_array(True)
        hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

        clusterizer.cluster_hits(hits)  # cluster hits
        cluster_hits, clusters = clusterizer.get_hit_cluster(), clusterizer.get_cluster()

        # Define expected output
        expected_result = np.zeros(shape=(10, ), dtype=data_struct.ClusterHitInfo)
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
        hits = np.ones(shape=(2, ), dtype=data_struct.HitInfo)
        hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 17, 36, 30, 19
        hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 36, 6, 19

        # Create clusterizer object
        clusterizer = HitClusterizer()
        clusterizer.create_cluster_hit_info_array(True)

        # Case 1: Test max hit charge cut, accept all hits
        clusterizer.set_max_hit_charge(30)  # only add hits with charge <= 30
        clusterizer.cluster_hits(hits)  # cluster hits

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
        clusterizer.cluster_hits(hits)  # cluster hits
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
        expected_result['n_cluster'] = [1, 1]

        self.assertTrue(np.array_equal(cluster_hits, expected_result))

        # Case 3: Add the same hit within an event
        # Create some fake data
        hits = np.ones(shape=(3, ), dtype=data_struct.HitInfo)
        hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 18, 36, 6, 19
        hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 18, 36, 6, 19
        hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'] = 18, 38, 6, 19

        expected_hit_result = np.zeros(shape=(3, ), dtype=data_struct.ClusterHitInfo)
        expected_cluster_result = np.zeros(shape=(1, ), dtype=data_struct.ClusterInfo)
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
        expected_cluster_result['mean_column'] = [18.5]
        expected_cluster_result['mean_row'] = [37.5]

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

    def test_different_data_type(self):
        # Define a different hit data structure to cluster. Numba automatically recompiles and the result should not change
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
        clusterizer = HitClusterizer()

        # Define expected output
        expected_cluster_result = np.zeros(shape=(4, ), dtype=data_struct.ClusterInfo)
        expected_cluster_result['event_number'] = [0, 1, 2, 3]
        expected_cluster_result['n_hits'] = [3, 3, 3, 1]
        expected_cluster_result['charge'] = [1, 2, 1, 1]
        expected_cluster_result['seed_column'] = [2, 4, 8, 10]
        expected_cluster_result['seed_row'] = [3, 7, 15, 19]
        expected_cluster_result['mean_column'] = [2.5, 5.5, 8.5, 10.5]
        expected_cluster_result['mean_row'] = [3.5, 9.5, 15.5, 19.5]

        for hit_data_type in hit_data_types:
            # Create fake data with actual hit data structure
            hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2, hit_struct=np.dtype(hit_data_type))
            # Define expected output. Cluster hit data types are different and thus the expected results have to have different data types
            hit_data_type.extend([('cluster_ID', '<i2'),
                                  ('is_seed', '<u1'),
                                  ('cluster_size', '<u2'),
                                  ('n_cluster', '<u2')])
            expected_hit_result = np.zeros(shape=(10, ), dtype=data_struct.ClusterHitInfo)
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

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClusterizer)
    unittest.TextTestRunner(verbosity=2).run(suite)
