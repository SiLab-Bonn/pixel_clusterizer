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
        clusterizer = HitClusterizer(hit_fields=hit_mapping, hit_dtype=hit_dtype_new, pure_python=self.pure_python)
        with self.assertRaises(TypeError):
            _, _ = clusterizer.cluster_hits(np.array([], dtype=hit_dtype))  # missing "not_defined"
        with self.assertRaises(TypeError):
            _, _ = clusterizer.cluster_hits(np.array([], dtype=hit_dtype_new))  # missing "event_number"
        # TEST 3 Set custom and correct hit mapping, no eception expected
        hit_mapping = {'not_defined': 'event_number',
                       'column': 'column',
                       'row': 'row',
                       'charge': 'charge',
                       'frame': 'frame'
                       }
        clusterizer = HitClusterizer(hit_fields=hit_mapping, hit_dtype=hit_dtype_new, pure_python=self.pure_python)
        _, _ = clusterizer.cluster_hits(np.array([], dtype=hit_dtype_new))

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

        end_of_cluster_function_jitted = clusterizer._jitted(end_of_cluster_function)
        clusterizer.set_end_of_cluster_function(end_of_cluster_function_jitted)  # Set jitted end_of_cluster_function

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

        end_of_event_function_jitted = clusterizer._jitted(end_of_event_function)
        clusterizer.set_end_of_event_function(end_of_event_function_jitted)  # Set jitted end_of_cluster_function

        # Main function
        cluster_hits, clusters = clusterizer.cluster_hits(hits)  # cluster hits

        # Test results
        self.assertTrue(np.array_equal(clusters, expected_cluster_result))
        self.assertTrue(np.array_equal(cluster_hits, expected_hit_result))

    def test_chunked_clustering(self):  # Big tables have to be chunked and analyzed with clusterizer.cluster_hits(hits_chunk) calls
        clusterizer = HitClusterizer(pure_python=self.pure_python,
                                     min_hit_charge=0, max_hit_charge=13,
                                     column_cluster_distance=2,
                                     row_cluster_distance=2,
                                     frame_cluster_distance=4,
                                     ignore_same_hits=True)

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
