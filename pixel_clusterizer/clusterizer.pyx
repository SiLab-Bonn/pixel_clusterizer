# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libcpp cimport bool as cpp_bool  # to be able to use bool variables, as cpp_bool according to http://code.google.com/p/cefpython/source/browse/cefpython/cefpython.pyx?spec=svne037c69837fa39ae220806c2faa1bbb6ae4500b9&r=e037c69837fa39ae220806c2faa1bbb6ae4500b9
from data_struct cimport numpy_hit_info, numpy_cluster_hit_info, numpy_cluster_info

cnp.import_array()  # if array is used it has to be imported, otherwise possible runtime error

cdef extern from "cpp/Basis.h":
    cdef cppclass Basis:
        Basis()

cdef extern from "cpp/Clusterizer.h":
    cdef cppclass HitInfo:
        HitInfo()
    cdef cppclass ClusterHitInfo:
        ClusterHitInfo()
    cdef cppclass ClusterInfo:
        ClusterInfo()
    cdef cppclass Clusterizer(Basis):
        Clusterizer(unsigned int maxCol, unsigned int maxRow) except +
        void setErrorOutput(cpp_bool pToggle)
        void setWarningOutput(cpp_bool pToggle)
        void setInfoOutput(cpp_bool pToggle)
        void setDebugOutput(cpp_bool pToggle)

        void addHits(HitInfo *& rHitInfo, const unsigned int & rNhits) except +
        void getHitCluster(ClusterHitInfo*& rClusterHitInfo, unsigned int& rSize, cpp_bool copy)
        void getCluster(ClusterInfo*& rClusterHitInfo, unsigned int& rSize, cpp_bool copy)

        void createClusterHitInfoArray(cpp_bool toggle)
        void createClusterInfoArray(cpp_bool toggle)

        void setClusterHitInfoArraySize(const unsigned int& rSize)
        void setClusterInfoArraySize(const unsigned int& rSize)

        void setXclusterDistance(const unsigned int & pDx)
        void setYclusterDistance(const unsigned int & pDy)
        void setFrameclusterDistance(const unsigned int & pdFrame)
        void setMinClusterHits(const unsigned int & pMinNclusterHits)
        void setMaxClusterHits(const unsigned int & pMaxNclusterHits)
        void setMaxClusterHitCharge(const unsigned int & pMaxClusterHitCharge)
        void setMaxClusterCharge(const unsigned int & pMaxClusterCharge)

        void setMaxHitCharge(const unsigned int & pMaxHitCharge)

        void getClusterSizeHist(unsigned int & rNparameterValues, unsigned int *& rClusterSize, cpp_bool copy)
        void getClusterChargeHist(unsigned int & rNparameterValues, unsigned int *& rClusterCharge, cpp_bool copy)

        unsigned int getNclusters()

        void reset()
        void test()

cdef cnp.uint32_t* data_32
cdef ClusterHitInfo* cluster_hits
cdef ClusterInfo* cluster_info
cdef unsigned int size = 0
cdef cluster_hit_dt = cnp.dtype([('event_number', '<i8'), ('frame', '<u1'), ('column', '<u2'), ('row', '<u2'), ('charge', '<u2'), ('cluster_ID', '<i2'), ('is_seed', '<u1'), ('cluster_size', '<u2'), ('n_cluster', '<u2')])
cdef cluster_info_dt = cnp.dtype([('event_number', '<i8'), ('ID', '<u2'), ('size', '<u2'), ('charge', '<u2'), ('seed_column', '<u2'), ('seed_row', '<u2'), ('mean_column', 'f4'), ('mean_row', 'f4')])

cdef data_to_numpy_array_uint32(cnp.uint32_t* ptr, cnp.npy_intp N):
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] arr = cnp.PyArray_SimpleNewFromData(1, <cnp.npy_intp*> &N, cnp.NPY_UINT32, <cnp.uint32_t*> ptr)
    return arr

cdef cluster_hit_data_to_numpy_array(void* ptr, cnp.npy_intp N):
    cdef cnp.ndarray[numpy_cluster_hit_info, ndim=1] arr = cnp.PyArray_SimpleNewFromData(1, <cnp.npy_intp*> &N, cnp.NPY_INT8, <void*> ptr).view(cluster_hit_dt)
    arr.setflags(write=False)  # protect the data from python
    return arr

cdef cluster_info_data_to_numpy_array(void* ptr, cnp.npy_intp N):
    cdef cnp.ndarray[numpy_cluster_info, ndim=1] arr = cnp.PyArray_SimpleNewFromData(1, <cnp.npy_intp*> &N, cnp.NPY_INT8, <void*> ptr).view(cluster_info_dt)
    arr.setflags(write=False)  # protect the data from python
    return arr

cdef class HitClusterizer:
    cdef Clusterizer * thisptr  # hold a C++ instance which we're wrapping
    def __cinit__(self, n_columns=1000, n_rows=1000):
        self.thisptr = new Clusterizer(<unsigned int> n_columns, <unsigned int> n_rows)
    def __dealloc__(self):
        del self.thisptr
    def set_debug_output(self, toggle):
        self.thisptr.setDebugOutput(< cpp_bool > toggle)
    def set_info_output(self, toggle):
        self.thisptr.setInfoOutput(< cpp_bool > toggle)
    def set_warning_output(self, toggle):
        self.thisptr.setWarningOutput(< cpp_bool > toggle)
    def set_error_output(self, toggle):
        self.thisptr.setErrorOutput(< cpp_bool > toggle)
    def add_hits(self, cnp.ndarray[numpy_hit_info, ndim=1] hit_info):
        self.thisptr.addHits(< HitInfo *&> hit_info.data, < unsigned int > hit_info.shape[0])
    def get_hit_cluster(self):
        self.thisptr.getHitCluster(<ClusterHitInfo*&> cluster_hits, <unsigned int&> size, <cpp_bool> False)
        if cluster_hits != NULL:
            array = cluster_hit_data_to_numpy_array(cluster_hits, sizeof(ClusterHitInfo) * size)
            return array
    def get_cluster(self):
        self.thisptr.getCluster(<ClusterInfo*&> cluster_info, <unsigned int&> size, <cpp_bool> False)
        if cluster_info != NULL:
            array = cluster_info_data_to_numpy_array(cluster_info, sizeof(ClusterInfo) * size)
            return array
    def create_cluster_hit_info_array(self, value=True):
        self.thisptr.createClusterHitInfoArray(< cpp_bool > value)
    def create_cluster_info_array(self, value=True):
        self.thisptr.createClusterInfoArray(< cpp_bool > value)
    def set_cluster_hit_info_array_size(self, size):
        self.thisptr.setClusterHitInfoArraySize(< const unsigned int &> size)
    def set_cluster_info_array_size(self, size):
        self.thisptr.setClusterInfoArraySize(< const unsigned int &> size)
    def set_x_cluster_distance(self, value):
        self.thisptr.setXclusterDistance(< const unsigned int &> value)
    def set_y_cluster_distance(self, value):
        self.thisptr.setYclusterDistance(< const unsigned int &> value)
    def set_frame_cluster_distance(self, value):
        self.thisptr.setFrameclusterDistance(< const unsigned int &> value)
    def set_min_cluster_hits(self, value):
        self.thisptr.setMinClusterHits(< const unsigned int &> value)
    def set_max_cluster_hits(self, value):
        self.thisptr.setMaxClusterHits(< const unsigned int &> value)
    def set_max_cluster_hit_charge(self, value):
        raise NotImplementedError('This feature does not work right now and is deactivated!')
        self.thisptr.setMaxClusterHitCharge(< const unsigned int &> value)
    def set_max_cluster_charge(self, value):
        raise NotImplementedError('This feature does not work right now and is deactivated!')
        self.thisptr.setMaxClusterCharge(< const unsigned int &> value)
    def set_max_hit_charge(self, value):
        self.thisptr.setMaxHitCharge(<const unsigned int &> value)
    def get_cluster_size_hist(self):
        self.thisptr.getClusterSizeHist(< unsigned int &> size, < unsigned int *&> data_32, < cpp_bool > False)
        if data_32 != NULL:
            return data_to_numpy_array_uint32(data_32, size)
    def get_cluster_charge_hist(self):
        self.thisptr.getClusterChargeHist(< unsigned int &> size, < unsigned int *&> data_32, < cpp_bool > False)
        if data_32 != NULL:
            array = data_to_numpy_array_uint32(data_32, size)
            return array.reshape((20, 100), order='F')  # make linear array to 3d array (col,row,parameter)
    def get_n_clusters(self):
        return < unsigned int > self.thisptr.getNclusters()
    def reset(self):
        self.thisptr.reset()
    def test(self):
        self.thisptr.test()
