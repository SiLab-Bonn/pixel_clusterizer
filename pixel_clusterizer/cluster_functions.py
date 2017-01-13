''' Fast clustering functions that are compiled in time via numba '''

from numba import njit


@njit()
def _new_event(event_number_1, event_number_2):
    'Detect a new event by checking if the event number of the actual hit is the actual event number'
    return event_number_1 != event_number_2


@njit()
def _pixel_masked(hit, array):
    ''' Checks whether a hit (column/row) is masked or not. Array is 2D array with boolean elements corresponding to pixles indicating whether a pixel is disabled or not.
    '''
    if array.shape[0] > hit["column"] and array.shape[1] > hit["row"]:
        return array[hit["column"], hit["row"]]
    else:
        return False


# @njit()
# def _pixel_masked(hit, array):
#     ''' Checks whether a hit (column/row) is masked or not. Array is an iterable of column/row tuples of disabled pixels.
#     '''
#     for array_value in array:
#         if hit["column"] == array_value["column"] and hit["row"] == array_value["row"]:
#             return True
#     return False


@njit()
def _finish_cluster(hits, clusters, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels, disabled_pixels):
    ''' Set hit and cluster information of the cluster (e.g. number of hits in the cluster (cluster_size), total cluster charge (charge), ...).
    '''
    cluster_charge = 0
    max_cluster_charge = -1
    # necessary for charge weighted hit position
    total_weighted_column = 0
    total_weighted_row = 0

    for i in range(cluster_size):
        hit_index = cluster_hit_indices[i]
        if hits[hit_index]['charge'] > max_cluster_charge:
            seed_hit_index = hit_index
            max_cluster_charge = hits[hit_index]['charge']
        hits[hit_index]['is_seed'] = 0
        hits[hit_index]['cluster_size'] = cluster_size
        # include charge correction in sum
        total_weighted_column += hits[hit_index]['column'] * (hits[hit_index]['charge'] + charge_correction)
        total_weighted_row += hits[hit_index]['row'] * (hits[hit_index]['charge'] + charge_correction)
        cluster_charge += hits[hit_index]['charge']
        hits[hit_index]['cluster_ID'] = cluster_id

    hits[seed_hit_index]['is_seed'] = 1

    clusters[cluster_index]["ID"] = cluster_id
    clusters[cluster_index]["n_hits"] = cluster_size
    clusters[cluster_index]["charge"] = cluster_charge
    clusters[cluster_index]['seed_column'] = hits[seed_hit_index]['column']
    clusters[cluster_index]['seed_row'] = hits[seed_hit_index]['row']
    # correct total charge value and calculate mean column and row
    clusters[cluster_index]['mean_column'] = float(total_weighted_column) / (cluster_charge + cluster_size * charge_correction)
    clusters[cluster_index]['mean_row'] = float(total_weighted_row) / (cluster_charge + cluster_size * charge_correction)

    # Call end of cluster function hook
    _end_of_cluster_function(
        hits=hits,
        clusters=clusters,
        cluster_size=cluster_size,
        cluster_hit_indices=cluster_hit_indices,
        cluster_index=cluster_index,
        cluster_id=cluster_id,
        charge_correction=charge_correction,
        noisy_pixels=noisy_pixels,
        disabled_pixels=disabled_pixels,
        seed_hit_index=seed_hit_index)


@njit()
def _finish_event(hits, clusters, start_event_hit_index, stop_event_hit_index, start_event_cluster_index, stop_event_cluster_index):
    ''' Set hit and cluster information of the event (e.g. number of cluster in the event (n_cluster), ...).
    '''
    for hit_index in range(start_event_hit_index, stop_event_hit_index):
        hits[hit_index]['n_cluster'] = stop_event_cluster_index - start_event_cluster_index

    for cluster_index in range(start_event_cluster_index, stop_event_cluster_index):
        clusters[cluster_index]['event_number'] = hits[start_event_hit_index]['event_number']

    # Call end of event function hook
    _end_of_event_function(
        hits=hits,
        clusters=clusters,
        start_event_hit_index=start_event_hit_index,
        stop_event_hit_index=stop_event_hit_index,
        start_event_cluster_index=start_event_cluster_index,
        stop_event_cluster_index=stop_event_cluster_index)


@njit()
def _hit_ok(hit, min_hit_charge, max_hit_charge):
    ''' Check if given hit is withing the limits.
    '''
    # Omit hits with charge < min_hit_charge
    if hit['charge'] < min_hit_charge:
        return False

    # Omit hits with charge > max_hit_charge
    if max_hit_charge != 0 and hit['charge'] > max_hit_charge:
        return False

    return True


@njit()
def _set_hit_invalid(hit, cluster_id=-1):
    ''' Set values for invalid hit.
    '''
    hit['cluster_ID'] = cluster_id
    hit['is_seed'] = 0
    hit['cluster_size'] = 0


@njit()
def _set_1d_array(array, value, size=-1):
    ''' Set array elemets to value for given number of elements (if size is negative number set all elements to value).
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
def _end_of_cluster_function(hits, clusters, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels, disabled_pixels, seed_hit_index):
    ''' Empty function that can be overwritten with a new function that is called at the end of each cluster
    '''
    pass


@njit()
def _end_of_event_function(hits, clusters, start_event_hit_index, stop_event_hit_index, start_event_cluster_index, stop_event_cluster_index):
    ''' Empty function that can be overwritten with a new function that is called at the end of event
    '''
    pass


@njit()
def _cluster_hits(hits, clusters, assigned_hit_array, cluster_hit_indices, column_cluster_distance, row_cluster_distance, frame_cluster_distance, min_hit_charge, max_hit_charge, ignore_same_hits, noisy_pixels, disabled_pixels):
    ''' Main precompiled function that loopes over the hits and clusters them
    '''
    total_hits = hits.shape[0]
    max_cluster_hits = cluster_hit_indices.shape[0]

    if total_hits != clusters.shape[0]:
        raise ValueError("hits and clusters must be the same size")

    if total_hits != assigned_hit_array.shape[0]:
        raise ValueError("hits and assigned_hit_array must be the same size")

    # Correction for charge weighting
    # Some chips have non-zero charge for a charge value of zero, charge needs to be corrected to calculate cluster center correctly
    if min_hit_charge == 0:
        charge_correction = 1
    else:
        charge_correction = 0

    # Temporary variables that are reset for each cluster or event
    start_event_hit_index = 0
    start_event_cluster_index = 0
    cluster_size = 0
    event_number = hits[0]['event_number']
    event_cluster_index = 0

    # Outer loop over all hits in the array (referred to as actual hit)
    for i in range(total_hits):

        # Check for new event and reset event variables
        if _new_event(hits[i]['event_number'], event_number):
            _finish_event(
                hits=hits,
                clusters=clusters,
                start_event_hit_index=start_event_hit_index,
                stop_event_hit_index=i,
                start_event_cluster_index=start_event_cluster_index,
                stop_event_cluster_index=start_event_cluster_index + event_cluster_index)

            start_event_hit_index = i
            start_event_cluster_index = start_event_cluster_index + event_cluster_index
            event_number = hits[i]['event_number']
            event_cluster_index = 0

        if assigned_hit_array[i] > 0:  # Hit was already assigned to a cluster in the inner loop, thus skip actual hit
            continue

        if not _hit_ok(
                hit=hits[i],
                min_hit_charge=min_hit_charge,
                max_hit_charge=max_hit_charge) or (disabled_pixels.shape[0] != 0 and _pixel_masked(hits[i], disabled_pixels)):
            _set_hit_invalid(hit=hits[i], cluster_id=-1)
            assigned_hit_array[i] = 1
            continue

        # Set/reset cluster variables for new cluster
        # Reset temp array with hit indices of actual cluster for the next cluster
        _set_1d_array(cluster_hit_indices, -1, cluster_size)
        cluster_hit_indices[0] = i
        assigned_hit_array[i] = 1
        cluster_size = 1  # actual cluster has one hit so far

        for j in cluster_hit_indices:  # Loop over all hits of the actual cluster; cluster_hit_indices is updated within the loop if new hit are found
            if j < 0:  # There are no more cluster hits found
                break

            for k in range(cluster_hit_indices[0] + 1, total_hits):
                # Stop event hits loop if new event is reached
                if _new_event(hits[k]['event_number'], event_number):
                    break

                # Hit is already assigned to a cluster, thus skip actual hit
                if assigned_hit_array[k] > 0:
                    continue

                if not _hit_ok(
                        hit=hits[k],
                        min_hit_charge=min_hit_charge,
                        max_hit_charge=max_hit_charge) or (disabled_pixels.shape[0] != 0 and _pixel_masked(hits[k], disabled_pixels)):
                    _set_hit_invalid(hit=hits[k], cluster_id=-1)
                    assigned_hit_array[k] = 1
                    continue

                # Check if event hit belongs to actual hit and thus to the actual cluster
                if _is_in_max_difference(hits[j]['column'], hits[k]['column'], column_cluster_distance) and _is_in_max_difference(hits[j]['row'], hits[k]['row'], row_cluster_distance) and _is_in_max_difference(hits[j]['frame'], hits[k]['frame'], frame_cluster_distance):
                    if not ignore_same_hits or hits[j]['column'] != hits[k]['column'] or hits[j]['row'] != hits[k]['row']:
                        cluster_size += 1
                        if cluster_size > max_cluster_hits:
                            raise IndexError('cluster_hit_indices is too small to contain all cluster hits')
                        cluster_hit_indices[cluster_size - 1] = k
                        assigned_hit_array[k] = 1

                    else:
                        _set_hit_invalid(hit=hits[k], cluster_id=-2)
                        assigned_hit_array[k] = 1

        # check for valid cluster and add it to the array
        if cluster_size == 1 and noisy_pixels.shape[0] != 0 and _pixel_masked(hits[cluster_hit_indices[0]], noisy_pixels):
            _set_hit_invalid(hit=hits[cluster_hit_indices[0]], cluster_id=-1)
        else:
            _finish_cluster(
                hits=hits,
                clusters=clusters,
                cluster_size=cluster_size,
                cluster_hit_indices=cluster_hit_indices,
                cluster_index=start_event_cluster_index + event_cluster_index,
                cluster_id=event_cluster_index,
                charge_correction=charge_correction,
                noisy_pixels=noisy_pixels,
                disabled_pixels=disabled_pixels)
            event_cluster_index += 1

    # Last event is assumed to be finished at the end of the hit array, thus add info
    _finish_event(
        hits=hits,
        clusters=clusters,
        start_event_hit_index=start_event_hit_index,
        stop_event_hit_index=total_hits,
        start_event_cluster_index=start_event_cluster_index,
        stop_event_cluster_index=start_event_cluster_index + event_cluster_index)

    total_clusters = start_event_cluster_index + event_cluster_index
    return total_clusters
