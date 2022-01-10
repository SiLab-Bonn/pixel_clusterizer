import logging
import os
from operator import itemgetter

import numpy as np

default_hits_descr = [
    ('event_number', '<i8'),
    ('frame', '<u2'),
    ('column', '<u2'),
    ('row', '<u2'),
    ('charge', '<f4')]

default_hits_dtype = np.dtype(default_hits_descr)

default_cluster_hits_descr = [
    ('event_number', '<i8'),
    ('frame', '<u2'),
    ('column', '<u2'),
    ('row', '<u2'),
    ('charge', '<f4'),
    ('cluster_ID', '<i2'),
    ('is_seed', '<u1'),
    ('cluster_size', '<u2'),
    ('n_cluster', '<u2')]

default_cluster_hits_dtype = np.dtype(default_cluster_hits_descr)

default_clusters_descr = [
    ('event_number', '<i8'),
    ('ID', '<u2'),
    ('n_hits', '<u2'),
    ('charge', '<f4'),
    ('seed_column', '<u2'),
    ('seed_row', '<u2'),
    ('mean_column', '<f4'),
    ('mean_row', '<f4')]

default_clusters_dtype = np.dtype(default_clusters_descr)


def np_uint_type_chooser(number):
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if number <= np.iinfo(dtype).max:
            return dtype
    raise ValueError('{} is too big!'.format(number))


def np_int_type_chooser(number):
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        if number <= np.iinfo(dtype).max:
            return dtype
    raise ValueError('{} is too big!'.format(number))


class HitClusterizer(object):

    ''' Clusterizer class providing an interface for the jitted functions and stores settings.'''

    def __init__(self, hit_fields=None, hit_dtype=None, cluster_fields=None, cluster_dtype=None, pure_python=False, min_hit_charge=None, max_hit_charge=None, charge_correction=None, charge_weighted_clustering=False, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=0, ignore_same_hits=True):
        '''
        Parameters
        ----------
        pure_python : bool
            If True, disable JIT compiler. The (n)jit decorator act as if it performs no operation.
        '''
        # Activate pure python mode by setting the environment variable NUMBA_DISABLE_JIT
        self.pure_python = pure_python
        if self.pure_python:
            logging.info('PURE PYTHON MODE: INTENDED FOR TESTING ONLY! YOU CANNOT SWITCH THE MODE WITHIN THE PYTHON INTERPRETER INSTANCE!')
            os.environ['NUMBA_DISABLE_JIT'] = '1'
        else:
            os.environ['NUMBA_DISABLE_JIT'] = '0'

        # Delayed import of numba.njit, since the environment 'NUMBA_DISABLE_JIT' is evaluated on import.
        # To allow pure_python mode this dirty hack is needed; issues occur when within the same python instance the mode is switched, since python does
        # NOT provide a proper method to reload modules.
        self.cluster_functions = __import__('pixel_clusterizer.cluster_functions').cluster_functions

        # Set the translation dictionary for the important hit value names
        self._default_hit_fields_mapping = {item: item for item in map(itemgetter(0), default_cluster_hits_descr)}
        if hit_fields:
            self.set_hit_fields(hit_fields)
        else:
            self.set_hit_fields(None)

        # Set the translation dictionary for the important hit value names
        self._default_cluster_fields_mapping = {item: item for item in map(itemgetter(0), default_clusters_descr)}
        if cluster_fields:
            self.set_cluster_fields(cluster_fields)
        else:
            self.set_cluster_fields(None)

        # Set hit data structure for clustered hits
        self._default_cluster_hits_descr = default_cluster_hits_descr
        if hit_dtype:
            self.set_hit_dtype(hit_dtype)
        else:
            self.set_hit_dtype(None)

        # Set cluster data struct for clustered hits
        self._default_clusters_descr = default_clusters_descr
        if cluster_dtype:
            self.set_cluster_dtype(cluster_dtype)
        else:
            self.set_cluster_dtype(None)

        self._initialized = True

        # Std. settings
        self.set_min_hit_charge(min_hit_charge)
        self.set_max_hit_charge(max_hit_charge)
        self.set_charge_correction(charge_correction)
        self.set_charge_weighted_clustering(charge_weighted_clustering)
        self.set_column_cluster_distance(column_cluster_distance)
        self.set_row_cluster_distance(row_cluster_distance)
        self.set_frame_cluster_distance(frame_cluster_distance)
        self.ignore_same_hits(ignore_same_hits)
        self.reset()

    @property
    def initialized(self):
        return hasattr(self, "_initialized") and self._initialized

    def _init_arrays(self, size=0):
        if self.initialized:
            self._cluster_hits = np.zeros(shape=(size, ), dtype=np.dtype(self._cluster_hits_descr))
            self._clusters = np.zeros(shape=(size, ), dtype=np.dtype(self._clusters_descr))
            self._assigned_hit_array = np.zeros(shape=(size, ), dtype=bool)
            self._cluster_hit_indices = np.full(shape=(size, ), fill_value=-1, dtype=np_int_type_chooser(size))

    def reset(self):  # Resets the overwritten function hooks, otherwise they are stored as a module global and not reset on clusterizer initialization
        self._init_arrays(size=0)
        self._last_event_number = None

        def end_of_cluster_function(hits, clusters, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels, disabled_pixels, seed_hit_index):
            pass

        def end_of_event_function(hits, clusters, start_event_hit_index, stop_event_hit_index, start_event_cluster_index, stop_event_cluster_index):
            pass

        self.set_end_of_cluster_function(end_of_cluster_function)
        self.set_end_of_event_function(end_of_event_function)

    def set_hit_fields(self, hit_fields):
        ''' Tell the clusterizer the meaning of the field names of the cluster hits array.

        The hit_fields parameter is a dict, e.g., {"new field name": "standard field name"}.

        If None default mapping is set.

        Example:
        --------
        Internally, the clusterizer uses the hit fields names "column"/"row". If the name of the hits fields are "x"/"y", call:
        set_hit_fields(hit_fields={'x': 'column',
                                   'y': 'row'})
        '''
        if not hit_fields:
            hit_fields_mapping_inverse = {}
            hit_fields_mapping = {}
        else:
            # Create also the inverse dictionary for faster lookup
            hit_fields_mapping_inverse = dict((k, v) for k, v in hit_fields.items())
            hit_fields_mapping = dict((v, k) for k, v in hit_fields.items())

        for old_name, new_name in self._default_hit_fields_mapping.items():
            if old_name not in hit_fields_mapping:
                hit_fields_mapping[old_name] = new_name
                hit_fields_mapping_inverse[new_name] = old_name

        self._hit_fields_mapping = hit_fields_mapping
        self._hit_fields_mapping_inverse = hit_fields_mapping_inverse

    def set_cluster_fields(self, cluster_fields):
        ''' Tell the clusterizer the meaning of the field names of the clusters array.

        The cluster_fields parameter is a dict, e.g., {"new filed name": "standard field name"}.
        '''
        if not cluster_fields:
            cluster_fields_mapping_inverse = {}
            cluster_fields_mapping = {}
        else:
            # Create also the inverse dictionary for faster lookup
            cluster_fields_mapping_inverse = dict((k, v) for k, v in cluster_fields.items())
            cluster_fields_mapping = dict((v, k) for k, v in cluster_fields.items())

        for old_name, new_name in self._default_cluster_fields_mapping.items():
            if old_name not in cluster_fields_mapping:
                cluster_fields_mapping[old_name] = new_name
                cluster_fields_mapping_inverse[new_name] = old_name

        self._cluster_fields_mapping = cluster_fields_mapping
        self._cluster_fields_mapping_inverse = cluster_fields_mapping_inverse

    def set_hit_dtype(self, hit_dtype):
        ''' Set the data type of the cluster hits array.

        Fields that are not mentioned here are NOT copied into the clustered hits array.
        Clusterizer has to know the hit data type to produce the clustered hit result with the same data types.

        Parameters:
        -----------
        hit_dtype : numpy.dtype or equivalent
            Defines the dtype of the hit array.

        Example:
        --------
        hit_dtype = [("column", np.uint16), ("row", np.uint16)], where
        "column", "row" is the field name of the input hit array.
        '''

        if not hit_dtype:
            hit_dtype = np.dtype([])
        else:
            hit_dtype = np.dtype(hit_dtype)
        cluster_hits_descr = hit_dtype.descr

        # Add default back to description
        for dtype_name, dtype in self._default_cluster_hits_descr:
            if self._hit_fields_mapping[dtype_name] not in hit_dtype.fields:
                cluster_hits_descr.append((dtype_name, dtype))
        self._cluster_hits_descr = cluster_hits_descr
        self._init_arrays(size=0)

    def set_cluster_dtype(self, cluster_dtype):
        ''' Set the data type of the clusters array.

        Parameters:
        -----------
        cluster_dtype : numpy.dtype or equivalent
            Defines the dtype of the cluster array.
        '''
        if not cluster_dtype:
            cluster_dtype = np.dtype([])
        else:
            cluster_dtype = np.dtype(cluster_dtype)
        clusters_descr = cluster_dtype.descr

        for dtype_name, dtype in self._default_clusters_descr:
            if self._cluster_fields_mapping[dtype_name] not in cluster_dtype.fields:
                clusters_descr.append((dtype_name, dtype))

        self._clusters_descr = clusters_descr
        self._init_arrays(size=0)

    def add_hit_field(self, description):
        ''' Adds a field or a list of fields to the cluster result array. Has to be defined as a numpy dtype entry, e.g.: ('parameter', '<i4') '''
        if isinstance(description, list):
            for item in description:
                if len(item) != 2:
                    raise TypeError("Parameter description must be a list of 2-tuples of a string and a dtype.")
                self._cluster_hits_descr.append(item)
        else:
            if len(description) != 2:
                raise TypeError("Parameter description must be a 2-tuple of a string and a dtype.")
            self._cluster_hits_descr.append(description)
        self._init_arrays(size=0)

    def add_cluster_field(self, description):
        ''' Adds a field or a list of fields to the cluster result array. Has to be defined as a numpy dtype entry, e.g.: ('parameter', '<i4') '''
        if isinstance(description, list):
            for item in description:
                if len(item) != 2:
                    raise TypeError("Parameter description must be a list of 2-tuples of a string and a dtype.")
                self._clusters_descr.append(item)
        else:
            if len(description) != 2:
                raise TypeError("Parameter description must be a 2-tuple of a string and a dtype.")
            self._clusters_descr.append(description)
        self._init_arrays(size=0)

    def set_end_of_cluster_function(self, function):
        ''' Adding function to module.
        This is maybe the only way to make the clusterizer to work with multiprocessing.
        '''
        self.cluster_functions._end_of_cluster_function = self._jitted(function)
        self._end_of_cluster_function = function

    def set_end_of_event_function(self, function):
        ''' Adding function to module.
        This is maybe the only way to make the clusterizer to work with multiprocessing.
        '''
        self.cluster_functions._end_of_event_function = self._jitted(function)
        self._end_of_event_function = function

    def set_min_hit_charge(self, value):
        ''' Hits with charge values below this value will be ignored.
        If None, all hits will be used.
        '''
        self._min_hit_charge = value

    def set_max_hit_charge(self, value):
        ''' Hits with charge values above this value will be ignored.
        If None, all hits will be used.
        '''
        self._max_hit_charge = value

    def set_charge_correction(self, value):
        ''' Adding the given value to the hit charge.
        If 0 or None, no offset will be added to the hit charge.
        Note:
        1. The charge correction is olny used when charge_weighted_clustering
           is True.
        2. Charge digitizers of some front-end chips start with a value of 0.
           If the privded data contains hits with chage of 0, charge_correction must
           be set to 1 in this case. Otherwise hits with a charge value of 0
           will not contribute to the charge weighted clustering.
        '''
        if value is None:
            self._charge_correction = 0
        else:
            self._charge_correction = value

    def set_charge_weighted_clustering(self, value):
        ''' If True, the charge value of the hits is used
        to calculate center of gravity of a cluster.
        For correct function, the parameter charge_correction must be set correctly.
        If False, only the arithmetic mean of the hit positions is used
        to calculate the center of a cluster.
        '''
        self._charge_weighted_clustering = bool(value)

    def set_column_cluster_distance(self, value):
        ''' Setting up max. column cluster distance.
        '''
        self._column_cluster_distance = value

    def set_row_cluster_distance(self, value):
        ''' Setting up max. row cluster distance.
        '''
        self._row_cluster_distance = value

    def set_frame_cluster_distance(self, value):
        ''' Setting up max. frame cluster distance.
        '''
        self._frame_cluster_distance = value

    def ignore_same_hits(self, value):
        ''' Whether a duplicate hit in the event with the same column and row is ignored or not.
        '''
        self._ignore_same_hits = value

    def cluster_hits(self, hits, noisy_pixels=None, disabled_pixels=None):
        ''' Cluster given hit array.

        Parameters
        ----------
        hits : numpy.recarray
            Hit data that will be clustered. The hit data contains all information that is required for clustering.
            The hit data must contain only complete events.
        noisy_pixels : list of 2-tuples
            List of 2-tuples of column and row indices of noisy pixels. No cluster will be built when it consists of a single noisy pixel.
            Clusters with two or more noisy pixels are built.
        disabled_pixels : list of 2-tuples
            List of 2-tuples of column and row indices of pixles that will be masked. Masked pixels will not be used during clustering.
        '''
        # Jitting a second time to workaround different bahavior of the installation methods on different platforms (pip install vs. python setup.py).
        # In some circumstances, the Numba compiler can't compile functions that were pickled previously.
        self.cluster_functions._end_of_cluster_function = self._jitted(self._end_of_cluster_function)
        self.cluster_functions._end_of_event_function = self._jitted(self._end_of_event_function)

        n_hits = hits.shape[0]  # Set n_hits to new size

        if (n_hits < int(0.5 * self._cluster_hits.size)) or (n_hits > self._cluster_hits.size):
            self._init_arrays(size=int(1.1 * n_hits))  # oversize buffer slightly to reduce allocations
        else:
            self._assigned_hit_array.fill(0)  # The array represents the assignment of hits to clusters, 0 means not assigned to any cluster
            self._cluster_hit_indices.fill(-1)  # The hit indices of the actual cluster, -1 means not assigned to any hit

        self._clusters.dtype.names = self._unmap_cluster_field_names(self._clusters.dtype.names)  # Reset the data fields from previous renaming
        self._cluster_hits.dtype.names = self._unmap_hit_field_names(self._cluster_hits.dtype.names)  # Reset the data fields from previous renaming
        self._check_struct_compatibility(hits)

        # The hit info is extended by the cluster info; this is only possible by creating a new hit info array and copy data
        for field_name in hits.dtype.fields:
            if field_name in self._hit_fields_mapping_inverse:
                cluster_hits_field_name = self._hit_fields_mapping_inverse[field_name]
            else:
                cluster_hits_field_name = field_name
            if cluster_hits_field_name in self._cluster_hits.dtype.fields:
                self._cluster_hits[cluster_hits_field_name][:n_hits] = hits[field_name]

        # Check if event number is increasing. Otherwise give a warning message.
        if self._last_event_number is not None and hits.shape[0] != 0 and self._cluster_hits["event_number"][0] <= self._last_event_number:
            logging.warning('The event number does not increase with successive chunks.')
        if hits.shape[0] > 1 and not np.all((self._cluster_hits["event_number"][1:n_hits] - self._cluster_hits["event_number"][:n_hits - 1]) >= 0):
            raise RuntimeError('Some values in column "%s" decrease.' % (self._hit_fields_mapping["event_number"],))
        if self._cluster_hits.shape[0] != 0:
            self._last_event_number = self._cluster_hits[-1]["event_number"]

        noisy_pixels_array = np.array([]) if noisy_pixels is None else np.array(noisy_pixels)
        if noisy_pixels_array.shape[0] != 0:
            noisy_pixels_max_range = np.array([max(0, np.max(noisy_pixels_array[:, 0])), max(0, np.max(noisy_pixels_array[:, 1]))])
            noisy_pixels = np.zeros(noisy_pixels_max_range + 1, dtype=bool)
            noisy_pixels[noisy_pixels_array[:, 0], noisy_pixels_array[:, 1]] = 1
        else:
            noisy_pixels = np.zeros((0, 0), dtype=bool)

        disabled_pixels_array = np.array([]) if disabled_pixels is None else np.array(disabled_pixels)
        if disabled_pixels_array.shape[0] != 0:
            disabled_pixels_max_range = np.array([np.max(disabled_pixels_array[:, 0]), np.max(disabled_pixels_array[:, 1])])
            disabled_pixels = np.zeros(disabled_pixels_max_range + 1, dtype=bool)
            disabled_pixels[disabled_pixels_array[:, 0], disabled_pixels_array[:, 1]] = 1
        else:
            disabled_pixels = np.zeros((0, 0), dtype=bool)

#         col_dtype = self._cluster_hits.dtype.fields["column"][0]
#         row_dtype = self._cluster_hits.dtype.fields["row"][0]
#         mask_dtype = {"names": ["column", "row"],
#                       "formats": [col_dtype, row_dtype]}
#         noisy_pixels = np.recarray(noisy_pixels_array.shape[0], dtype=mask_dtype)
#         noisy_pixels[:] = [(item[0], item[1]) for item in noisy_pixels_array]
#         disabled_pixels = np.recarray(disabled_pixels_array.shape[0], dtype=mask_dtype)
#         disabled_pixels[:] = [(item[0], item[1]) for item in disabled_pixels_array]

        n_clusters = self.cluster_functions._cluster_hits(  # Set n_clusters to new size
            hits=self._cluster_hits[:n_hits],
            clusters=self._clusters[:n_hits],
            assigned_hit_array=self._assigned_hit_array[:n_hits],
            cluster_hit_indices=self._cluster_hit_indices[:n_hits],
            min_hit_charge=self._min_hit_charge,
            max_hit_charge=self._max_hit_charge,
            charge_correction=self._charge_correction,
            charge_weighted_clustering=self._charge_weighted_clustering,
            column_cluster_distance=self._column_cluster_distance,
            row_cluster_distance=self._row_cluster_distance,
            frame_cluster_distance=self._frame_cluster_distance,
            ignore_same_hits=self._ignore_same_hits,
            noisy_pixels=noisy_pixels,
            disabled_pixels=disabled_pixels)

        self._cluster_hits.dtype.names = self._map_hit_field_names(self._cluster_hits.dtype.names)  # Rename the data fields for the result
        self._clusters.dtype.names = self._map_cluster_field_names(self._clusters.dtype.names)  # Rename the data fields for the result

        return self._cluster_hits[:n_hits], self._clusters[:n_clusters]

    def _jitted(self, function):
        from numba import njit
        if self.pure_python:
            return function
        else:
            try:
                # test whether the function is already jitted or not
                function.py_func
            except AttributeError:
                # Adding decorator to function
                return njit()(function)
            else:
                # already jitted
                return function

    def _map_hit_field_names(self, dtype_names):  # Maps the hit field names from the internal convention to the external defined one
        unpatched_field_names = list(dtype_names)
        for index, unpatched_field_name in enumerate(unpatched_field_names):
            if unpatched_field_name in self._hit_fields_mapping.keys():
                unpatched_field_names[index] = self._hit_fields_mapping[unpatched_field_name]
        return tuple(unpatched_field_names)

    def _unmap_hit_field_names(self, dtype_names):  # Unmaps the hit field names from the external convention to the internal defined one
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

    def _unmap_cluster_field_names(self, dtype_names):  # Unmaps the cluster field names from the external convention to the internal defined one
        unpatched_field_names = list(dtype_names)
        for index, unpatched_field_name in enumerate(unpatched_field_names):
            if unpatched_field_name in self._cluster_fields_mapping_inverse.keys():
                unpatched_field_names[index] = self._cluster_fields_mapping_inverse[unpatched_field_name]
        return tuple(unpatched_field_names)

    def _check_struct_compatibility(self, hits):
        ''' Takes the hit array and checks if the important data fields have the same data type than the hit clustered array and that the field names are correct.'''
        for key, _ in self._cluster_hits_descr:
            if key in self._hit_fields_mapping_inverse:
                mapped_key = self._hit_fields_mapping_inverse[key]
            else:
                mapped_key = key
            # Only check hit fields that contain hit information
            if mapped_key in ['cluster_ID', 'is_seed', 'cluster_size', 'n_cluster']:
                continue
            if key not in hits.dtype.names:
                raise TypeError('Required hit field "%s" not found.' % key)
            if not self.pure_python and hits.dtype[key] == np.float16:
                raise TypeError('The dtype float16 for hit data filed "%s" is not supported.' % (key,))
            if self._cluster_hits.dtype[mapped_key] != hits.dtype[key] and not np.can_cast(hits.dtype[key], self._cluster_hits.dtype[mapped_key]):
                raise TypeError('The dtype for hit data field "%s" does not match. Got/expected: %s/%s.' % (key, hits.dtype[key], self._cluster_hits.dtype[mapped_key]))
        additional_hit_fields = set(hits.dtype.names) - set([key for key, val in self._cluster_hits_descr])
        if additional_hit_fields:
            logging.warning('Found additional column: %s' % ", ".join(additional_hit_fields))
