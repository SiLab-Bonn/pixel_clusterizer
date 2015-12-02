# Data structure definitions to be used within cython

cimport numpy as cnp
cnp.import_array()  # if array is used it has to be imported, otherwise possible runtime error

cdef packed struct numpy_hit_info:
    cnp.int64_t event_number  # event number value (unsigned long long: 0 to 18,446,744,073,709,551,615)
    cnp.uint8_t frame  # relative BCID value (unsigned char: 0 to 255)
    cnp.uint16_t column  # column value (unsigned char: 0 to 255)
    cnp.uint16_t row  # row value (unsigned short int: 0 to 65.535)
    cnp.uint16_t charge  # ToT value (unsigned char: 0 to 255)

cdef packed struct numpy_cluster_hit_info:
    cnp.int64_t event_number  # event number value (unsigned long long: 0 to 18,446,744,073,709,551,615)
    cnp.uint8_t frame  # relative BCID value (unsigned char: 0 to 255)
    cnp.uint16_t column  # column value (unsigned char: 0 to 255)
    cnp.uint16_t row  # row value (unsigned short int: 0 to 65.535)
    cnp.uint16_t charge  # ToT value (unsigned char: 0 to 255)
    cnp.int16_t cluster_ID  # the cluster id of the hit
    cnp.uint8_t is_seed  # flag to mark seed pixel
    cnp.uint16_t cluster_size  # the cluster id of the hit
    cnp.uint16_t n_cluster  # the cluster id of the hit

cdef packed struct numpy_cluster_info:
    cnp.int64_t event_number  # event number value (unsigned long long: 0 to 18,446,744,073,709,551,615)
    cnp.uint16_t ID  # the cluster id of the cluster
    cnp.uint16_t size  # sum charge of all cluster hits
    cnp.uint16_t charge  # sum charge of all cluster hits
    cnp.uint16_t seed_column  # column value (unsigned char: 0 to 255)
    cnp.uint16_t seed_row  # row value (unsigned short int: 0 to 65.535)
    cnp.float32_t mean_column  # sum charge of all cluster hits
    cnp.float32_t mean_row  # sum charge of all cluster hits
