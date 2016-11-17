''' Fast clustering functions that are compiled in time via numba '''
from numba import njit
from docutils.utils.roman import OutOfRangeError


@njit()
def _new_event(event_number_1, event_number_2):
    'Detect a new event by checking if the event number of the actual hit is the actual event number'
    return event_number_1 != event_number_2


@njit()
def _value_in_array(value, array):
    ''' Check if array contains value.
    Equivalent to np.all(np.in1d(number, array)).
    '''
    for value_array in array:
        if value == value_array:
            return True
    return False


@njit()
def _finish_cluster(hits, cluster, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels):
    ''' Set hit and cluster information of the cluster (e.g. number of hits in the cluster (cluster_size), total cluster charge (charge), ...).
    '''
    cluster_charge = 0
    max_cluster_charge = -1
    total_weighted_column = 0
    total_weighted_row = 0
#     for hit_index in cluster_hit_indices:
#         if hit_index == -1:
#             break
    for i in range(cluster_size):
        hit_index = cluster_hit_indices[i]
        # check for single noisy pixel
        if cluster_size == 1 and _value_in_array(hits[hit_index]['column'], noisy_pixels[0]) and _value_in_array(hits[hit_index]['row'], noisy_pixels[1]):
            return False
        if hits[hit_index]['charge'] > max_cluster_charge:
            seed_hit_index = hit_index
            max_cluster_charge = hits[hit_index]['charge']
        hits[hit_index]['cluster_size'] = cluster_size
        total_weighted_column += hits[hit_index]['column'] * (hits[hit_index]['charge'] + charge_correction)
        total_weighted_row += hits[hit_index]['row'] * (hits[hit_index]['charge'] + charge_correction)
        cluster_charge += hits[hit_index]['charge']
        hits[hit_index]['cluster_ID'] = cluster_id

    hits[seed_hit_index]['is_seed'] = 1

    cluster[cluster_index]["ID"] = cluster_id
    cluster[cluster_index]["n_hits"] = cluster_size
    cluster[cluster_index]["charge"] = cluster_charge
    cluster[cluster_index]['seed_column'] = hits[seed_hit_index]['column']
    cluster[cluster_index]['seed_row'] = hits[seed_hit_index]['row']
    cluster[cluster_index]['mean_column'] = float(total_weighted_column) / (cluster_charge + cluster_size * charge_correction)
    cluster[cluster_index]['mean_row'] = float(total_weighted_row) / (cluster_charge + cluster_size * charge_correction)

    return True


@njit()
def _finish_event(hits, cluster, start_event_hit_index, stop_event_hit_index, start_event_cluster_index, stop_event_cluster_index):
    ''' Set hit and cluster information of the event (e.g. number of cluster in the event (n_cluster), ...).
    '''
    for hit_index in range(start_event_hit_index, stop_event_hit_index):
        hits[hit_index]['n_cluster'] = stop_event_cluster_index - start_event_cluster_index

    for cluster_index in range(start_event_cluster_index, stop_event_cluster_index):
        cluster[cluster_index]['event_number'] = hits[start_event_hit_index]['event_number']


@njit()
def _reset_array(array, value, size=0):
    ''' Sets the cluster hit indices to the std. valie -1. To be able to use this array in the new event
    '''
    if size >= 0:
        for i in range(size):
            array[i] = value
    else:
        for i in range(array.shape[0]):
            array[i] = value


@njit()
def _is_in_max_difference(value_1, value_2, max_difference):
    ''' Helper function to determine the difference of two values that can be np.uints. Works in python and numba mode.
    Circumvents numba bug #1653
    '''
    if value_1 <= value_2:
        return value_2 - value_1 <= max_difference
    return value_1 - value_2 <= max_difference


@njit()
def _end_of_cluster_function(hits, cluster, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels):
    ''' Empty function that can be overwritten with a new function that is called at the end of each cluster
    '''
    return


@njit()
def _end_of_event_function(hits, cluster, start_event_hit_index, stop_event_hit_index, start_event_cluster_index, stop_event_cluster_index):
    ''' Empty function that can be overwritten with a new function that is called at the end of event
    '''
    return


@njit()
def _cluster_hits(hits, cluster, assigned_hit_array, cluster_hit_indices, x_cluster_distance, y_cluster_distance, frame_cluster_distance, max_n_cluster_hits, min_hit_charge, max_hit_charge, ignore_same_hits, noisy_pixels, disabled_pixels):
    ''' Main precompiled function that loopes over the hits and clusters them
    '''
    total_hits = hits.shape[0]

    # Correction for charge weighting
    # Some chips have non-zero charge for a charge value of zero, charge needs to be corrected to calculate cluster center correctly
    if min_hit_charge == 0:
        charge_correction = 1
    else:
        charge_correction = 0

    # Temporary variables that are reset for each cluster or event
    actual_event_hit_index = 0
    start_event_cluster_index = 0
    actual_cluster_size = 0
    actual_event_number = hits[0]['event_number']
    event_cluster_index = 0

    # Outer loop over all hits in the array (referred to as actual hit)
    for i in range(total_hits):

        if assigned_hit_array[i] > 0:  # Hit was already assigned to a cluster in the inner loop, thus skip actual hit
            continue

        # Omit hits with charge < min_hit_charge
        if hits[i]['charge'] < min_hit_charge:
            continue

        # Omit hits with charge > max_hit_charge
        if max_hit_charge != 0 and hits[i]['charge'] > max_hit_charge:
            continue

        if _value_in_array(hits[i]['column'], disabled_pixels[0]) and _value_in_array(hits[i]['row'], disabled_pixels[1]):
            continue

        # Check for new event and reset event variables
        if _new_event(hits[i]['event_number'], actual_event_number):
            _finish_event(
                hits=hits,
                cluster=cluster,
                start_event_hit_index=actual_event_hit_index,
                stop_event_hit_index=i,
                start_event_cluster_index=start_event_cluster_index,
                stop_event_cluster_index=start_event_cluster_index + event_cluster_index)

            # Call end of event function hook
            _end_of_event_function(
                hits=hits,
                cluster=cluster,
                start_event_hit_index=actual_event_hit_index,
                stop_event_hit_index=i,
                start_event_cluster_index=start_event_cluster_index,
                stop_event_cluster_index=start_event_cluster_index + event_cluster_index)

            actual_event_hit_index = i
            start_event_cluster_index = start_event_cluster_index + event_cluster_index
            actual_event_number = hits[i]['event_number']
            event_cluster_index = 0

        # Set/reset cluster variables for new cluster
        # Reset temp array with hit indices of actual cluster for the next cluster
        _reset_array(cluster_hit_indices, -1, actual_cluster_size)
        cluster_hit_indices_index = 0
        cluster_hit_indices[0] = i
        assigned_hit_array[i] = 1
        actual_cluster_size = 1  # actual cluster has one hit so far

        for j in cluster_hit_indices:  # Loop over all hits of the actual cluster; cluster_hit_indices is updated within the loop if new hit are found
            if j < 0:  # There are no more cluster hits found
                break

            actual_inner_loop_hit_index = j + actual_event_hit_index

            for k in range(cluster_hit_indices[0] + 1, total_hits):
                # Stop event hits loop if new event is reached
                if _new_event(hits[k]['event_number'], actual_event_number):
                    break

                # Omit if event hit is actual hit (clustering with itself)
                if k == j:
                    continue

                # Omit if event hit is already belonging to a cluster
                if assigned_hit_array[k] > 0:  # Hit was already assigned to a cluster in the inner loop, thus skip actual hit
                    continue

                if hits[k]['charge'] < min_hit_charge:
                    continue

                if max_hit_charge != 0 and hits[k]['charge'] > max_hit_charge:
                    continue

                if _value_in_array(hits[k]['column'], disabled_pixels[0]) and _value_in_array(hits[k]['row'], disabled_pixels[1]):
                    continue

                # Check if event hit belongs to actual hit and thus to the actual cluster
                if _is_in_max_difference(hits[actual_inner_loop_hit_index]['column'], hits[k]['column'], x_cluster_distance) and _is_in_max_difference(hits[actual_inner_loop_hit_index]['row'], hits[k]['row'], y_cluster_distance) and _is_in_max_difference(hits[actual_inner_loop_hit_index]['frame'], hits[k]['frame'], frame_cluster_distance):
                    if not ignore_same_hits or hits[actual_inner_loop_hit_index]['column'] != hits[k]['column'] or hits[actual_inner_loop_hit_index]['row'] != hits[k]['row']:
                        actual_cluster_size += 1
                        cluster_hit_indices_index += 1
                        if max_n_cluster_hits > 0 and actual_cluster_size > max_n_cluster_hits:
                            raise OutOfRangeError('There are more clusters than specified. Increase max_cluster_hits parameter!')
                        cluster_hit_indices[cluster_hit_indices_index] = k
                        assigned_hit_array[k] = 1

                    else:
                        # TODO: change cluster ID to -2
                        assigned_hit_array[k] = 1

        # check for valid cluster and add it to the array
        if _finish_cluster(
                hits=hits,
                cluster=cluster,
                cluster_size=actual_cluster_size,
                cluster_hit_indices=cluster_hit_indices,
                cluster_index=start_event_cluster_index + event_cluster_index,
                cluster_id=event_cluster_index,
                charge_correction=charge_correction,
                noisy_pixels=noisy_pixels):
            # Call end of cluster function hook
            _end_of_cluster_function(
                hits=hits,
                cluster=cluster,
                cluster_size=actual_cluster_size,
                cluster_hit_indices=cluster_hit_indices,
                cluster_index=start_event_cluster_index + event_cluster_index,
                cluster_id=event_cluster_index,
                charge_correction=charge_correction,
                noisy_pixels=noisy_pixels)
            event_cluster_index += 1

    # Last event is assumed to be finished at the end of the hit array, thus add info
    _finish_event(
        hits=hits,
        cluster=cluster,
        start_event_hit_index=actual_event_hit_index,
        stop_event_hit_index=total_hits,
        start_event_cluster_index=start_event_cluster_index,
        stop_event_cluster_index=start_event_cluster_index + event_cluster_index)

    # Call end of event function hook
    _end_of_event_function(
        hits=hits,
        cluster=cluster,
        start_event_hit_index=actual_event_hit_index,
        stop_event_hit_index=total_hits,
        start_event_cluster_index=start_event_cluster_index,
        stop_event_cluster_index=start_event_cluster_index + event_cluster_index)
    total_clusters = start_event_cluster_index + event_cluster_index
    return total_clusters
