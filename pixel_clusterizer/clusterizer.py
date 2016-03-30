import os
import logging
import numpy as np

hit_data_type = np.dtype([('event_number', '<i8'),
                          ('frame', '<u1'),
                          ('column', '<u2'),
                          ('row', '<u2'),
                          ('charge', '<u2')])


class HitClusterizer(object):

    ''' Clusterizer class providing an interface for the jitted functions and stores settings.'''

    def __init__(self, hit_fields=None, hit_dtype=None, cluster_fields=None, cluster_dtype=None, pure_python=False):
        # Activate pute python mode by setting the evnironment variable NUMBA_DISABLE_JIT
        self.pure_python = pure_python
        if self.pure_python:
            logging.warning('PURE PYTHON MODE: USE FOR TESTING ONLY! YOU CANNOT SWITCH THE MODE WITHIN ONE PYTHON INTERPRETER INSTANCE!')
            os.environ['NUMBA_DISABLE_JIT'] = '1'
        else:
            os.environ['NUMBA_DISABLE_JIT'] = '0'

        # Delayed import of numba.njit, since the environment 'NUMBA_DISABLE_JIT' is evaluated on import.
        # To allow pure_python mode this dirty hack is needed; issues occur when within the same python instance the mode is switched, since python does
        # NOT provide a proper method to reload modules.
        self.cluster_functions = __import__('pixel_clusterizer.cluster_functions').cluster_functions
        self.njit = __import__('numba').njit

        # Std. settings
        self._create_cluster_hit_info_array = False
        self._max_cluster_hit_charge = 13
        self._max_hits = 10000
        self._x_cluster_distance = 2
        self._y_cluster_distance = 2
        self._frame_cluster_distance = 4
        self._max_cluster_hits = 300
        self._ignore_same_hits = True

        # Set the translation dictionary for the important hit value names
        if hit_fields:
            self.set_hit_fields(hit_fields)
        else:
            self.set_hit_fields({'event_number': 'event_number',
                                 'column': 'column',
                                 'row': 'row',
                                 'charge': 'charge',
                                 'frame': 'frame'
                                 })

        # Set the translation dictionary for the important hit value names
        if cluster_fields:
            self.set_cluster_fields(cluster_fields)
        else:
            self.set_cluster_fields({'event_number': 'event_number',
                                     'ID': 'ID',
                                     'n_hits': 'n_hits',
                                     'charge': 'charge',
                                     'seed_column': 'seed_column',
                                     'seed_row': 'seed_row',
                                     'mean_column': 'mean_column',
                                     'mean_row': 'mean_row'
                                     })

        # Set the result data struct for clustered hits
        if hit_dtype:
            self.set_hit_dtype(hit_dtype)
        else:
            self._hit_clustered_descr = [('event_number', '<i8'),
                                         ('frame', '<u1'),
                                         ('column', '<u2'),
                                         ('row', '<u2'),
                                         ('charge', '<u2'),
                                         ('cluster_ID', '<i2'),
                                         ('is_seed', '<u1'),
                                         ('cluster_size', '<u2'),
                                         ('n_cluster', '<u2')]

        # Set the result data struct for clustered hits
        if cluster_dtype:
            self.set_cluster_dtype(cluster_dtype)
        else:
            self._cluster_descr = [('event_number', '<i8'),
                                   ('ID', '<u2'),
                                   ('n_hits', '<u2'),
                                   ('charge', 'f4'),
                                   ('seed_column', '<u2'),
                                   ('seed_row', '<u2'),
                                   ('mean_column', 'f4'),
                                   ('mean_row', 'f4')]

        self.hits_clustered = np.zeros(shape=(self._max_hits, ), dtype=self._hit_clustered_descr)
        self.cluster = np.zeros(shape=(self._max_hits, ), dtype=np.dtype(self._cluster_descr))

        self.n_cluster = 0
        self.n_hits = 0

        self.reset()

    def reset(self):  # Resets the maybe overwritten function hooks, otherwise they are stored as a module global and not reset on clusterizer initialization
        def end_of_cluster_function(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_cluster_index, actual_event_hit_index, actual_cluster_hit_indices, seed_index):
            return

        def end_of_event_function(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_cluster_index, actual_event_hit_index, actual_cluster_hit_indices, seed_index):
            return
        self.set_end_of_cluster_function(end_of_cluster_function)
        self.set_end_of_event_function(end_of_event_function)

    def set_hit_fields(self, hit_fields):
        ''' Tell the clusterizer the meaning of the field names (e.g.: the field name x means column). Field that are not mentioned here are NOT copied into the result array.'''
        self._hit_fields_mapping = dict((v, k) for k, v in hit_fields.items())  # Create also the inverse dictionary for faster lookup
        try:
            self._hit_fields_mapping['event_number'], self._hit_fields_mapping['column'], self._hit_fields_mapping['row'], self._hit_fields_mapping['charge'], self._hit_fields_mapping['frame']
        except KeyError:
            raise ValueError('The hit fields event_number, column, row, charge and frame have to be defined!')
        self._hit_fields_mapping_inverse = hit_fields

    def set_cluster_fields(self, cluster_fields):
        ''' Tell the clusterizer the meaning of the field names (e.g.: the field name seed_x means seed_column). '''
        self._cluster_fields_mapping = dict((v, k) for k, v in cluster_fields.items())  # Create also the inverse dictionary for faster lookup

        try:
            self._cluster_fields_mapping['event_number'], self._cluster_fields_mapping['ID'], self._cluster_fields_mapping['n_hits'], self._cluster_fields_mapping['charge'], self._cluster_fields_mapping['seed_column'], self._cluster_fields_mapping['seed_row'], self._cluster_fields_mapping['mean_column'], self._cluster_fields_mapping['mean_row']
        except KeyError:
            raise ValueError('The cluster fields event_number, ID, n_hits, charge, seed_column, seed_row, mean_column and mean_row have to be defined!')
        self._cluster_fields_mapping_inverse = cluster_fields

    def set_hit_dtype(self, hit_dtype):
        ''' Set the data type of the hits. Clusterizer has to know the data type to produce the clustered hit result with the same data types.'''
        hit_clustered_descr = []
        for dtype_name, dtype in hit_dtype.descr:
            try:
                hit_clustered_descr.append((self._hit_fields_mapping_inverse[dtype_name], dtype))
            except KeyError:  # The hit has an unknown field, thus also add it to the hit_clustered
                hit_clustered_descr.append((dtype_name, dtype))
        hit_clustered_descr.extend([('cluster_ID', '<i2'), ('is_seed', '<u1'), ('cluster_size', '<u2'), ('n_cluster', '<u2')])
        self._hit_clustered_descr = hit_clustered_descr
        hit_clustered_dtype = np.dtype(hit_clustered_descr)  # Convert to numpy dtype for following sanity check
        # Check if the minimum required fields are there
        try:
            hit_clustered_dtype['event_number'], hit_clustered_dtype['column'], hit_clustered_dtype['row'], hit_clustered_dtype['charge'], hit_clustered_dtype['frame']
        except KeyError:
            raise ValueError('The clustered hit struct has to have a valid mapping to the fields: event_number, column, row, charge. Consider to set the mapping with set_hit_fields method first!')
        self.hits_clustered = np.zeros(shape=(self._max_hits, ), dtype=np.dtype(self._hit_clustered_descr))  # Hit clustered result array has to be reinitialized with new data types

    def set_cluster_dtype(self, cluster_dtype):
        ''' Set the data type of the cluster.'''
        cluster_descr = []
        for dtype_name, dtype in cluster_dtype.descr:
            try:
                cluster_descr.append((self._cluster_fields_mapping_inverse[dtype_name], dtype))
            except KeyError:  # The hit has an unknown field, thus also add it to the cluster
                cluster_descr.append((dtype_name, dtype))
        self._cluster_descr = cluster_descr
        cluster_dtype = np.dtype(cluster_descr)  # Convert to numpy dtype for following sanity check
        # Check if the minimum required fields are there
        try:
            cluster_dtype['event_number'], cluster_dtype['ID'], cluster_dtype['n_hits'], cluster_dtype['charge'], cluster_dtype['seed_column'], cluster_dtype['seed_row'], cluster_dtype['mean_column'], cluster_dtype['mean_row']
        except KeyError:
            raise ValueError('The cluster struct has to have a valid mapping to the fields: event_number, ID, n_hits, charge, seed_column, seed_row, mean_column and mean_row. Consider to set the mapping with set_hit_fields method first!')
        self.cluster = np.zeros(shape=(self._max_hits, ), dtype=np.dtype(self._cluster_descr))  # Hit clustered result array has to be reinitialized with new data types

    def add_cluster_field(self, description):
        ''' Adds a field or a list of fields to the cluster result array. Has to be defined as a numpy dtype entry, e.g.: ('parameter', '<i4') '''
        if isinstance(description, list):
            for one_parameter in description:
                self._cluster_descr.append(one_parameter)
        else:
            self._cluster_descr.append(description)
        self.cluster = np.zeros(shape=(self._max_hits, ), dtype=np.dtype(self._cluster_descr))  # Cluster result array has to be reinitialized with new data types

    def set_end_of_cluster_function(self, function):
        if not self.pure_python:
            self.cluster_functions._end_of_cluster_function = self.njit()(function)  # Overwrite end of cluster function by new provided function
        else:
            self.cluster_functions._end_of_cluster_function = function

    def set_end_of_event_function(self, function):
        if not self.pure_python:
            self.cluster_functions._end_of_event_function = self.njit()(function)  # Overwrite end of cluster function by new provided function
        else:
            self.cluster_functions._end_of_event_function = function

    def set_max_hits(self, value):
        self._max_hits = value
        self.hits_clustered = np.zeros(shape=(self._max_hits, ), dtype=np.dtype(self._hit_clustered_descr))
        self.cluster = np.zeros(shape=(self._max_hits, ), dtype=np.dtype(self._cluster_descr))

    def set_max_hit_charge(self, value):
        self._max_cluster_hit_charge = value

    def set_x_cluster_distance(self, value):
        self._x_cluster_distance = value

    def set_y_cluster_distance(self, value):
        self._y_cluster_distance = value

    def set_frame_cluster_distance(self, value):
        self._frame_cluster_distance = value

    def set_max_cluster_hits(self, value):
        self._max_cluster_hits = value

    def ignore_same_hits(self, value):  # Ignore same hit in an event for clustering
        self._ignore_same_hits = value

    def create_cluster_hit_info_array(self, value=True):  # TODO: do not create cluster hit info of false to save time
        self._create_cluster_hit_info_array = value

    def cluster_hits(self, hits):
        self.n_hits = 0  # Effectively deletes the already clustered hits
        self._delete_cluster()  # Delete the already created cluster
        self.hits_clustered.dtype.names = self._unmap_hit_field_names(self.hits_clustered.dtype.names)  # Reset the data fields from previous renaming
        self._check_struct_compatibility(hits)

        # The hit info is extended by the cluster info; this is only possible by creating a new hit info array and copy data
        for internal_name, external_name in self._hit_fields_mapping.items():
            self.hits_clustered[internal_name][self.n_hits:hits.shape[0]] = hits[external_name]

        self.hits_clustered['cluster_ID'][self.n_hits:hits.shape[0]], self.hits_clustered['is_seed'][self.n_hits:hits.shape[0]], self.hits_clustered['cluster_size'][self.n_hits:hits.shape[0]], self.hits_clustered['n_cluster'][self.n_hits:hits.shape[0]], self.n_cluster = self.cluster_functions._cluster_hits(self.hits_clustered[self.n_hits:hits.shape[0]],
                                                                                                                                                                                                                                                                                                                    self.cluster,
                                                                                                                                                                                                                                                                                                                    n_hits=hits.shape[0],
                                                                                                                                                                                                                                                                                                                    x_cluster_distance=self._x_cluster_distance,
                                                                                                                                                                                                                                                                                                                    y_cluster_distance=self._y_cluster_distance,
                                                                                                                                                                                                                                                                                                                    frame_cluster_distance=self._frame_cluster_distance,
                                                                                                                                                                                                                                                                                                                    max_n_cluster_hits=self._max_cluster_hits,
                                                                                                                                                                                                                                                                                                                    max_cluster_hit_charge=self._max_cluster_hit_charge,
                                                                                                                                                                                                                                                                                                                    ignore_same_hits=self._ignore_same_hits)

        self.n_hits += hits.shape[0]
        self.hits_clustered.dtype.names = self._map_hit_field_names(self.hits_clustered.dtype.names)  # Rename the data fields for the result
        self.cluster.dtype.names = self._map_cluster_field_names(self.cluster.dtype.names)  # Rename the data fields for the result

        return self.hits_clustered[:self.n_hits], self.cluster[:self.n_cluster]

    def get_hit_cluster(self):
        hits_clustered = self.hits_clustered[:self.n_hits]
        self.n_hits = 0
        return hits_clustered

    def get_cluster(self):
        cluster = self.cluster[:self.n_cluster]
        self._delete_cluster()
        return cluster

    def _delete_cluster(self):
        self.cluster = np.zeros(shape=(self._max_hits, ), dtype=np.dtype(self._cluster_descr))
        self.n_cluster = 0

    def _map_hit_field_names(self, dtype_names):  # Maps the hit field names from the internal convention to the external defined one
        unpatched_field_names = list(dtype_names)
        for index, unpatched_field_name in enumerate(unpatched_field_names):
            if unpatched_field_name in self._hit_fields_mapping.keys():
                unpatched_field_names[index] = self._hit_fields_mapping[unpatched_field_name]
        return tuple(unpatched_field_names)

    def _unmap_hit_field_names(self, dtype_names):  # Maps the hit field names from the external convention to the internal defined one
        unpatched_field_names = list(dtype_names)
        for index, unpatched_field_name in enumerate(unpatched_field_names):
            if unpatched_field_name in self._hit_fields_mapping_inverse.keys():
                unpatched_field_names[index] = self._hit_fields_mapping_inverse[unpatched_field_name]
        return tuple(unpatched_field_names)

    def _map_cluster_field_names(self, dtype_names):  # Maps the cluster field names from the internal convention to the external defined one
        unpatched_field_names = list(dtype_names)
        for index, unpatched_field_name in enumerate(unpatched_field_names):
            if unpatched_field_name in self._cluster_fields_mapping.keys():
                unpatched_field_names[index] = self._cluster_fields_mapping[unpatched_field_name]
        return tuple(unpatched_field_names)

    def _check_struct_compatibility(self, hits):
        ''' Takes the hit array and checks if the important data fields have the same data type than the hit clustered array and that the field names are correct'''
        try:
            if self.hits_clustered['frame'].dtype != hits[self._hit_fields_mapping['frame']].dtype or self.hits_clustered['column'].dtype != hits[self._hit_fields_mapping['column']].dtype or self.hits_clustered['row'].dtype != hits[self._hit_fields_mapping['row']].dtype or self.hits_clustered['charge'].dtype != hits[self._hit_fields_mapping['charge']].dtype or self.hits_clustered['event_number'].dtype != hits[self._hit_fields_mapping['event_number']].dtype:
                raise TypeError('The hit data type(s) do not match. Consider calling the method set_hit_dtype first! Got/Expected:', hits.dtype, self.hits_clustered.dtype)
        except ValueError:
            raise TypeError('The hit field names are unexpected. Consider calling the method set_hit_fields! Got:', hits.dtype.names)
