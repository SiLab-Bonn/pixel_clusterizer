import numpy as np
from numba import njit, jit
import sys
import logging


from pixel_clusterizer import data_struct


def pprint_array(array):  # just to print the results in a nice way
    offsets = []
    for column_name in array.dtype.names:
        sys.stdout.write(column_name)
        sys.stdout.write('\t')
        offsets.append(column_name.count(''))
    for row in array:
        print('')
        for i, column in enumerate(row):
            sys.stdout.write(' ' * (offsets[i] / 2))
            sys.stdout.write(str(column))
            sys.stdout.write('\t')
    print('')


# Fast functions that are compiled in time via numba
@njit
def _correct_cluster_id(hits, actual_event_number, actual_cluster_id, actual_event_hit_index, cluster_id):
    ' Substracts one from all cluster IDs of the event, starting from cluster id actual_cluster_id'
    for i in range(actual_event_hit_index, hits.shape[0]):
        if _new_event(hits[i].event_number, actual_event_number):  # Stop if new event is reached
            break
        if cluster_id[i] != -1 and cluster_id[i] > actual_cluster_id:
            cluster_id[i] -= 1


@njit
def _new_event(event_number, actual_event_number):
    'Detect a new event by checking if the event number of the actual hit is the actual event number'
    return event_number != actual_event_number


@njit
def _merge_cluster(i, j, hits, cluster, is_seed, cluster_id, max_cluster_charge, next_cluster_id, actual_event_cluster_index):
    # is_seed[j] = 0  # Event hit is not necessarily seed anymore
    if hits[j].charge >= max_cluster_charge:  # Old cluster hit can be the seed, if charge is equal max_cluster_charge to keep lowest index max charge hit seed hit
        max_cluster_charge = hits[j].charge
        is_seed[j] = 1
    actual_cluster_id = cluster_id[j]  # Correct the actual cluster id

    # Merge cluster infos
    cluster[actual_event_cluster_index + cluster_id[j]].n_hits += cluster[actual_event_cluster_index + cluster_id[i]].n_hits  # Add up cluster sizes of the merged cluster
    cluster[actual_event_cluster_index + actual_cluster_id].charge += cluster[actual_event_cluster_index + cluster_id[i]].charge  # Add up cluster charges of the merged cluster

    # Reset wrongly set new cluster value after merge
    cluster[actual_event_cluster_index + cluster_id[i]].n_hits = 0  # Reset wrongly set cluster value
    cluster[actual_event_cluster_index + cluster_id[i]].ID = 0  # Reset wrongly set cluster value
    cluster[actual_event_cluster_index + cluster_id[i]].charge = 0  # Reset wrongly set cluster value

    # Correct cluster IDs of the hit / cluster array
    cluster[actual_event_cluster_index + actual_cluster_id].ID = actual_cluster_id  # Set ID of actual cluster
    cluster_id[i] = actual_cluster_id  # Actual hit belongs to other already existing cluster

    next_cluster_id -= 1  # Actual new cluster ID is not used, since actual hit belongs to already existing cluster

    return max_cluster_charge, next_cluster_id, actual_cluster_id


@njit
def _finish_event(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_event_hit_index, new_actual_event_hit_index, next_cluster_id, actual_event_cluster_index):
    ''' Set hit and cluster information of the last finished event (like number of cluster in this event (n_cluster),  cluster charge ...). '''
    for i in range(actual_event_hit_index, new_actual_event_hit_index):
        # Set hit cluster info that is only known at the end of the event
        # Set it only for hits that are assigned to a cluster (TODO: do not do this anymore, done for compatibility)
        if cluster_id[i] != -1:
            n_cluster[i] = next_cluster_id

    # Normalize cluster position by the charge for center of gravity
    for i in range(actual_event_cluster_index, actual_event_cluster_index + next_cluster_id):
        cluster[i].mean_column /= (cluster[i].charge + cluster[i].n_hits)
        cluster[i].mean_row /= (cluster[i].charge + cluster[i].n_hits)


@njit
def _reset_cluster_hit_indices(actual_cluster_hit_indices, actual_cluster_size):
    for i in range(actual_cluster_size):
        actual_cluster_hit_indices[i] = -1

#@jit
def cluster_hits(hits, cluster, n_hits, x_cluster_distance=1, y_cluster_distance=2, frame_cluster_distance=4, max_n_cluster_hits=30, max_cluster_hit_charge=13):
    # Additional cluster info for the hit array
    cluster_id = np.zeros(shape=hits.shape, dtype=np.int16) - 1  # Cluster ID -1 means hit not assigned to cluster
    is_seed = np.zeros(shape=hits.shape, dtype=np.uint8)  # Seed 1 means hit is seed; lowest index hit with max charge hit is seed, thus there is always only one seed in a cluster
    cluster_size = np.zeros(shape=hits.shape, dtype=np.int16)  # Cluster size of the cluster the hit belongs to
    n_cluster = np.zeros(shape=hits.shape, dtype=np.int16)  # Number of clusters in the event the hit belongs to

    # Temporary variables that are reset for each cluster or event
    actual_event_number, actual_event_hit_index, actual_event_cluster_index, actual_cluster_id, max_cluster_charge, next_cluster_id, actual_cluster_size, actual_cluster_hit_index, seed_index = 0, 0, 0, 0, 0, 0, 0, 0, 0
    actual_cluster_hit_indices = np.zeros(shape=max_n_cluster_hits, dtype=np.int16) - 1  # The hit indices of the actual cluster, -1 means not assigned

    # Outer loop over all hits in the array (refered to as actual hit)
    for i in range(hits.shape[0]):
        if i >= n_hits:
            break

        # Check for new event and reset event variables
        if _new_event(hits[i].event_number, actual_event_number):
            _finish_event(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_event_hit_index, i + 1, next_cluster_id, actual_event_cluster_index)
            actual_event_hit_index = i
            actual_event_cluster_index = actual_event_cluster_index + next_cluster_id
            actual_event_number = hits[i].event_number
            next_cluster_id = 0  # First cluster has ID 1

        # Reset temp array with hit indices of actual cluster for the next cluster
        _reset_cluster_hit_indices(actual_cluster_hit_indices, actual_cluster_size)
        actual_cluster_hit_index = 0

        # Check if actual hit is already asigned to a cluster, if not define new actual cluster containing the actual hit as the first hit
        if cluster_id[i] != -1:  # Hit was already assigned to a cluster in the inner loop, thus skip actual hit
            continue

        # Omit hits with charge > max_cluster_hit_charge
        if hits[i].charge > max_cluster_hit_charge:
            continue

        # Set/reset cluster variables for new cluster
        actual_cluster_size = 1  # actual cluster has one hit so far
        actual_cluster_id = next_cluster_id  # Set actual cluster id
        next_cluster_id += 1  # Create new cluster ID that was not used before
        max_cluster_charge = hits[i].charge  # One hit with max_cluster_charge is seed
        actual_cluster_hit_indices[actual_cluster_hit_index] = i - actual_event_hit_index

        # Set first hit cluster hit info
        is_seed[i] = 1  # First hit of cluster is seed until higher charge hit is found
        seed_index = i
        cluster_id[i] = actual_cluster_id  # Assign actual hit to actual cluster

        # Set cluster info from first hit
        cluster[actual_event_cluster_index + actual_cluster_id].event_number = actual_event_number
        cluster[actual_event_cluster_index + actual_cluster_id].ID = actual_cluster_id
        cluster[actual_event_cluster_index + actual_cluster_id].n_hits = 1
        cluster[actual_event_cluster_index + actual_cluster_id].charge = hits[i].charge
        cluster[actual_event_cluster_index + actual_cluster_id].mean_column += (hits[i].column + 0.5) * (hits[i].charge + 1)
        cluster[actual_event_cluster_index + actual_cluster_id].mean_row += (hits[i].row + 0.5) * (hits[i].charge + 1)
        cluster[actual_event_cluster_index + actual_cluster_id].seed_column = hits[i].column
        cluster[actual_event_cluster_index + actual_cluster_id].seed_row = hits[i].row

        for j in actual_cluster_hit_indices:  # Loop over all hits of the actual cluster; actual_cluster_hit_indices is updated within the loop if new hit are found
            if j == -1:  # There are no more cluster hits found
                break

            actual_inner_loop_hit_index = j + actual_event_hit_index

            # Inner loop over actual event hits (refered to as event hit) and try to find hits belonging to the actual cluster
            for k in range(i + 1, hits.shape[0]):
                # Omit if event hit is already belonging to a cluster
                if cluster_id[k] != -1:
                    continue

                if k >= n_hits:
                    break

                if hits[k].charge > max_cluster_hit_charge:
                    continue

                # Omit if event hit is actual hit (clustering with itself)
                if k == actual_inner_loop_hit_index:
                    continue

                # Stop event hits loop if new event is reached
                if _new_event(hits[k].event_number, actual_event_number):
                    break

                # Check if event hit belongs to actual hit and thus to the actual cluster
                if abs(int(hits[actual_inner_loop_hit_index].column) - int(hits[k].column)) <= x_cluster_distance and abs(int(hits[actual_inner_loop_hit_index].row) - int(hits[k].row)) <= y_cluster_distance and abs(int(hits[actual_inner_loop_hit_index].frame) - int(hits[k].frame)) <= frame_cluster_distance:
                    actual_cluster_size += 1
                    actual_cluster_hit_index += 1
                    actual_cluster_hit_indices[actual_cluster_hit_index] = k - actual_event_hit_index
                    cluster_id[k] = actual_cluster_id  # Add event hit to actual cluster

                    # Add cluster position as sum of all hit positions weighted by the charge (center of gravity)
                    # the position is in the center of the pixel (column = 0 == mean_column = 0.5)
                    cluster[actual_event_cluster_index + actual_cluster_id].mean_column += (hits[k].column + 0.5) * (hits[k].charge + 1)
                    cluster[actual_event_cluster_index + actual_cluster_id].mean_row += (hits[k].row + 0.5) * (hits[k].charge + 1)
                    cluster[actual_event_cluster_index + actual_cluster_id].n_hits += 1
                    cluster[actual_event_cluster_index + actual_cluster_id].charge += hits[k].charge

                    # Check if event hit has a higher or equal charge, then make it the seed hit
                    if hits[k].charge >= max_cluster_charge:
                        # Event hit is seed and not actual hit, thus switch the seed flag
                        is_seed[k] = 1
                        is_seed[seed_index] = 0
                        seed_index = k
                        max_cluster_charge = hits[k].charge
                        # Set new seed hit in the cluster
                        cluster[actual_event_cluster_index + actual_cluster_id].seed_column = hits[k].column
                        cluster[actual_event_cluster_index + actual_cluster_id].seed_row = hits[k].row

        # Set cluster size info for actual cluster hits
        for j in actual_cluster_hit_indices:  # Loop over all hits of the actual cluster; actual_cluster_hit_indices is updated within the loop if new hit are found
            if j == -1:  # there are no more cluster hits found
                break
            cluster_size[j + actual_event_hit_index] = actual_cluster_size

#
# Event hit defines the highest hit charge of the actual cluster
#                     max_cluster_charge = hits[j].charge

        # cluster_size[i] = actual_cluster_size

        # print(i, cluster_id, is_seed)

    # Last event is assumed to be finished at the end of the hit array, thus add info
    _finish_event(hits, cluster, is_seed, n_cluster, cluster_size, cluster_id, actual_event_hit_index, i + 1, next_cluster_id, actual_event_cluster_index)
    return cluster_id, is_seed, cluster_size, n_cluster, actual_event_cluster_index + next_cluster_id


class HitClusterizer(object):

    def __init__(self, n_columns=None, n_rows=None, n_frames=None, n_charges=None, max_hits=10000):
        if any([n_columns, n_rows, n_frames, n_charges]):
            logging.warning('Depreciated: n_columns, n_rows, n_frames, n_charges variables do not have to be defined anymore!')

        self._max_hits = max_hits
        self._x_cluster_distance = 1
        self._y_cluster_distance = 2
        self._frame_cluster_distance = 4

        self.hits_clustered = np.zeros(shape=(self._max_hits, ), dtype=data_struct.ClusterHitInfo)
        self.cluster = np.zeros(shape=(self._max_hits, ), dtype=data_struct.ClusterInfo).view(np.recarray)  # Only recarrays no structured arrays are supported by numba

        self.n_cluster = 0
        self.n_hits = 0

        # Std. settings
        self._create_cluster_hit_info_array = False
        self._max_cluster_hit_charge = value = 13

    def reset(self):
        raise NotImplementedError

    def set_max_hit_charge(self, value):
        self._max_cluster_hit_charge = value

    def set_x_cluster_distance(self, value):
        self._x_cluster_distance = value

    def set_y_cluster_distance(self, value):
        self._y_cluster_distance = value

    def set_frame_cluster_distance(self, value):
        self._frame_cluster_distance = value

    def create_cluster_hit_info_array(self, value=True):
        self._create_cluster_hit_info_array = value

    def add_hits(self, hits):
        # The hit info is extended by the cluster info; this is only possible by creating a new hit info array
        self.hits_clustered['frame'][self.n_hits:hits.shape[0]] = hits['frame']
        self.hits_clustered['column'][self.n_hits:hits.shape[0]] = hits['column']
        self.hits_clustered['row'][self.n_hits:hits.shape[0]] = hits['row']
        self.hits_clustered['charge'][self.n_hits:hits.shape[0]] = hits['charge']
        self.hits_clustered['event_number'][self.n_hits:hits.shape[0]] = hits['event_number']

        self.hits_clustered['cluster_ID'][self.n_hits:hits.shape[0]], self.hits_clustered['is_seed'][self.n_hits:hits.shape[0]], self.hits_clustered['cluster_size'][self.n_hits:hits.shape[0]], self.hits_clustered['n_cluster'][self.n_hits:hits.shape[0]], self.n_cluster = cluster_hits(hits.view(np.recarray), self.cluster, n_hits=hits.shape[0], x_cluster_distance=self._x_cluster_distance, y_cluster_distance=self._y_cluster_distance, frame_cluster_distance=self._frame_cluster_distance, max_n_cluster_hits=30, max_cluster_hit_charge=self._max_cluster_hit_charge)

        self.n_hits += hits.shape[0]

        return self.hits_clustered[:self.n_hits], self.cluster[:self.n_cluster]

    def get_hit_cluster(self):
        hits_clustered = self.hits_clustered[:self.n_hits]
        self.n_hits = 0
        return hits_clustered

    def get_cluster(self):
        cluster = self.cluster[:self.n_cluster]
        # Cluster array is reused, thus has to be set to 0 here
        self.cluster = np.zeros(shape=(self._max_hits, ), dtype=data_struct.ClusterInfo).view(np.recarray)  # Only recarrays no structured arrays are supported by numba
        self.n_cluster = 0
        return cluster

if __name__ == '__main__':
    # create some fake data
    #     hits = np.ones(shape=(20, ), dtype=data_struct.HitInfo)
    #
    #     hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 0, 0, 30, 0
    #     hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 0, 2, 6, 0
    #     hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'] = 0, 6, 30, 0
    #     hits[3]['column'], hits[3]['row'], hits[3]['charge'], hits[3]['event_number'] = 0, 4, 6, 0
    #     hits[4]['column'], hits[4]['row'], hits[4]['charge'], hits[4]['event_number'] = 0, 8, 6, 0
    #     hits[5]['column'], hits[5]['row'], hits[5]['charge'], hits[5]['event_number'] = 0, 1, 30, 0
    #     hits[6]['column'], hits[6]['row'], hits[6]['charge'], hits[6]['event_number'] = 0, 15, 6, 0
    #     hits[7]['column'], hits[7]['row'], hits[7]['charge'], hits[7]['event_number'] = 0, 14, 30, 0
    #     hits[8]['column'], hits[8]['row'], hits[8]['charge'], hits[8]['event_number'] = 0, 16, 6, 0
    #     hits[9]['column'], hits[9]['row'], hits[9]['charge'], hits[9]['event_number'] = 0, 13, 6, 0
    #
    #     hits[10]['column'], hits[10]['row'], hits[10]['charge'], hits[10]['event_number'] = 0, 0, 30, 1
    #     hits[11]['column'], hits[11]['row'], hits[11]['charge'], hits[11]['event_number'] = 0, 2, 6, 1
    #     hits[12]['column'], hits[12]['row'], hits[12]['charge'], hits[12]['event_number'] = 0, 6, 30, 1
    #     hits[13]['column'], hits[13]['row'], hits[13]['charge'], hits[13]['event_number'] = 0, 4, 6, 1
    #     hits[14]['column'], hits[14]['row'], hits[14]['charge'], hits[14]['event_number'] = 0, 8, 6, 1
    #     hits[15]['column'], hits[15]['row'], hits[15]['charge'], hits[15]['event_number'] = 0, 1, 30, 1
    #     hits[16]['column'], hits[16]['row'], hits[16]['charge'], hits[16]['event_number'] = 0, 15, 6, 1
    #     hits[17]['column'], hits[17]['row'], hits[17]['charge'], hits[17]['event_number'] = 0, 14, 30, 1
    #     hits[18]['column'], hits[18]['row'], hits[18]['charge'], hits[18]['event_number'] = 0, 16, 6, 1
    #     hits[19]['column'], hits[19]['row'], hits[19]['charge'], hits[19]['event_number'] = 0, 13, 6, 1

    def create_hits(n_hits, max_column, max_row, max_frame, max_charge):
        hits = np.ones(shape=(n_hits, ), dtype=data_struct.HitInfo)
        for i in range(n_hits):
            hits[i]['event_number'], hits[i]['frame'], hits[i]['column'], hits[i]['row'], hits[i]['charge'] = i / 3, i % max_frame, i % max_column + 1, 2 * i % max_row + 1, i % max_charge
        return hits

    hits = create_hits(n_hits=10, max_column=100, max_row=100, max_frame=1, max_charge=2)

# create some fake data
#     hits = np.ones(shape=(5, ), dtype=data_struct.HitInfo)
    hits_clustered = np.ones(shape=(hits.shape[0], ), dtype=data_struct.ClusterHitInfo)
    cluster = np.zeros(shape=(hits.shape[0], ), dtype=data_struct.ClusterInfo)
#
#     hits[0]['column'], hits[0]['row'], hits[0]['charge'], hits[0]['event_number'] = 0, 0, 30, 0
#     hits[1]['column'], hits[1]['row'], hits[1]['charge'], hits[1]['event_number'] = 0, 1, 6, 0
#     hits[2]['column'], hits[2]['row'], hits[2]['charge'], hits[2]['event_number'] = 0, 2, 30, 0
#     hits[3]['column'], hits[3]['row'], hits[3]['charge'], hits[3]['event_number'] = 0, 3, 6, 0
#     hits[4]['column'], hits[4]['row'], hits[4]['charge'], hits[4]['event_number'] = 0, 8, 6, 0

    pprint_array(hits)

    cluster_id, is_seed, cluster_size, n_cluster, _ = cluster_hits(hits.view(np.recarray), cluster.view(np.recarray), n_hits=hits.shape[0])

    hits_clustered['column'], hits_clustered['row'], hits_clustered['charge'], hits_clustered['event_number'] = hits['column'], hits['row'], hits['charge'], hits['event_number']
    hits_clustered['cluster_ID'] = cluster_id
    hits_clustered['is_seed'] = is_seed
    hits_clustered['cluster_size'] = cluster_size
    hits_clustered['n_cluster'] = n_cluster

    pprint_array(hits_clustered)
    pprint_array(cluster)

#     clusterizer = HitClusterizer()
#     clusterizer.add_hits(hits)

#     pprint_array(clusterizer.get_hit_cluster())
#     pprint_array(clusterizer.get_cluster())
