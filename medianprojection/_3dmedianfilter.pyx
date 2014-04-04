'''_3dmedianfilter.pyx a median filter projection over a 3-d stack
CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "Python.h":
    ctypedef int Py_intptr_t


cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)

cdef extern from "string.h":
    void *memset(void *, int, int)
    void *memcpy(void *, void *, int)

cdef extern from "numpy/arrayobject.h":
    cdef void import_array()
    
import_array()

DTYPE_UINT32 = np.uint32
DTYPE_BOOL = np.bool
ctypedef np.uint32_t input_t
ctypedef np.int32_t pixel_count_t
ctypedef np.uint8_t mask_t
ctypedef np.float32_t output_t

###########
#
# Histograms
#
# There are five separate histograms for the octagonal filter and
# there are two levels (coarse = 16 values, fine = 256 values)
# per histogram. There are four histograms to maintain per position
# representing the four diagonals of the histogram plus one histogram
# for the straight side (which is used for adding and subtracting)
#
###########

cdef struct HistogramPiece:
    pixel_count_t coarse[16]
    pixel_count_t fine[256]

cdef struct Histogram:
    HistogramPiece top_left     # top-left corner
    HistogramPiece top_right    # top-right corner
    HistogramPiece edge         # leading/trailing edge
    HistogramPiece bottom_left  # bottom-left corner
    HistogramPiece bottom_right # bottom-right corner

# The pixel count has the number of pixels histogrammed in
# each of the five compartments for this position. This changes
# because of the mask
#
cdef struct PixelCount:
    pixel_count_t top_left
    pixel_count_t top_right
    pixel_count_t edge
    pixel_count_t bottom_left
    pixel_count_t bottom_right

ctypedef PixelCount PixelCount_t

#
# A cursor tracking an i/j position within the image
#
cdef struct SCoord:
    np.int32_t i0           # the row offset of the cursor within the data
    np.int32_t j0           # the column index at the start of the raster
    np.int32_t stride_j     # the number of elements between successive j
    input_t *data           # a raster of data for a single i
    np.int32_t mask_stride  # the stride between successive mask elements
    mask_t *mask            # the mask data for the raster
    np.int32_t sign         # positive to add this cursor, negative to subtract
    char *name              # for diagnostics
    

cdef struct Histograms:
    Histogram *histograms       # the histograms for a raster
    PixelCount *pixel_count     # pointer to the pixel count memory
    output_t *output            # pointer to the output array
    np.int32_t column_count     # number of columns represented by this structure
    np.int32_t stripe_length    # number of columns including "radius" before and after
    np.int32_t row_count        # number of rows available in image
    np.int32_t current_column   # the column being processed
    np.int32_t current_row      # the row being processed
    np.int32_t radius           # the "radius" of the octagon
    np.int32_t a_2              # 1/2 of the length of a side of the octagon
    # 
    #
    # The strides are the offsets in the array to the points that need to
    # be added or removed from a histogram to shift from the previous row
    # to the current row.
    # Listed here going clockwise from the trailing edge's top.
    # (-) = needs to be removed
    # (+) = needs to be added
    #
    #          -        -
    #         1.=========2
    #        1.           2
    #       +.             +-   Y
    #      |.               3   |
    #      |.               3   |
    #      -.               X  \|/
    #       5.             4    v
    #        5.           4
    #         +.=========+
    #
    #          x -->
    #
    SCoord last_top_left     # (-) left side of octagon's top - 1 row
    SCoord top_left          # (+) -1 row from trailing edge top
    SCoord last_top_right    # (-) right side of octagon's top - 1 col - 1 row
    SCoord top_right         # (+) -1 col -1 row from leading edge top
    SCoord last_leading_edge # (-) leading edge (right) top stride - 1 row
    SCoord leading_edge      # (+) leading edge bottom stride
    SCoord last_bottom_right # (-) leading edge bottom - 1 col
    SCoord bottom_right      # (+) right side of octagon's bottom - 1 col
    SCoord last_bottom_left  # (-) trailing edge bottom - 1 col
    SCoord bottom_left       # (+) left side of octagon's bottom - 1 col

    np.int32_t row_stride    # stride between one row and the next in the output
    np.int32_t col_stride    # stride between one column and the next in the output
    # The accumulator holds the running histogram
    #
    HistogramPiece accumulator
    #
    HistogramPiece saved_accumulator
    #
    # The running count of pixels in the accumulator
    #
    np.uint32_t accumulator_count
    #
    # The percent of pixels within the octagon whose value is
    # less than or equal to the median-filtered value (e.g. for
    # median, this is 50, for lower quartile it's 25)
    #
    np.int32_t percent
    #
    # last_update_column keeps track of the column # of the last update
    # to the fine histogram accumulator. Short-term, the median
    # stays in one coarse block so only one fine histogram might
    # need to be updated
    #
    np.int32_t last_update_column[16]

cdef class M3DState:
    '''This class holds the state of the median filter operation.
    
    The state contains the cursors for the current raster. The Cython-
    level code processes a single raster. The Python-level code loads
    successive rasters and directs the Cython code to process them.
    
    To process a 3D histogram:
    
    output = np.zeros(input.shape[:2], np.float32)
    
    state = M3DState(output, radius, percent)
    
    while not state.done():
    
        indexes = state.get_raster_indexes()
        
        for index in indexes:
        
            state.install_raster(index, input[index], mask[index])
            
        state.process_raster()
        
    '''
    cdef:
        Histograms histograms
        SCoord *pSCoords[10]
        SCoord *pActiveSCoords[10]
        np.int32_t scoord_indexes[10]
        np.int32_t n_indexes
        Py_intptr_t n_histogram_bytes
        object output
        object histogram_memory
        object pixel_count_memory
        object scoordData
        object scoordMasks
        
    def __init__(
        self, 
        np.ndarray[dtype=output_t, ndim=2,
                   negative_indices = False, mode='c'] output, 
        int radius, int percent):
        '''Initialize the state with the output array
        
        output - a 2-d array of uint8 values. The median filter gets the
                 size of the source from this array.
        
        radius - the radius of the octogon
        
        percent - the per-pixel output is the value at this percentile
        '''
        cdef:
            char *ptr
            np.int32_t offset
        memset(&self.histograms, 0, sizeof(Histograms))
        self.output = output
        self.histograms.output = <output_t *>np.PyArray_DATA(output)
        self.histograms.row_stride = \
            np.PyArray_STRIDE(output, 0) / np.PyArray_ITEMSIZE(output)
        self.histograms.col_stride = \
            np.PyArray_STRIDE(output, 1) / np.PyArray_ITEMSIZE(output)
        self.histograms.row_count = output.shape[0]
        self.histograms.column_count = output.shape[1]
        self.histograms.percent = percent
        self.histograms.stripe_length = \
            self.histograms.column_count + 2*radius + 1
        #
        # Allocate memory for the raster of histograms using a numpy array
        # We align the memory on a cache line for efficient L1 cache access
        # and SIMD access
        #
        self.n_histogram_bytes = sizeof(Histogram) * self.histograms.stripe_length
        self.n_histogram_bytes += 64
        self.histogram_memory = np.zeros(self.n_histogram_bytes, np.uint8)
        ptr = np.PyArray_BYTES(self.histogram_memory)
        offset = (64 - <Py_intptr_t>ptr) % 64
        self.histograms.histograms = <Histogram *>(ptr + offset)
        #
        # Allocate memory for the pixel counts (alignment isn't as crucial)
        #
        n_pixel_count_bytes = sizeof(PixelCount) * self.histograms.stripe_length
        self.pixel_count_memory = np.zeros(
            n_pixel_count_bytes / sizeof(pixel_count_t), np.uint32)
        self.histograms.pixel_count = <PixelCount_t *>np.PyArray_DATA(
            self.pixel_count_memory)
        #
        # Compute the coordinates of the significant points
        #
        # First, the length of a side of an octagon, compared
        # to what we call the radius is:
        #     2*r
        # ----------- =  a
        # (1+sqrt(2))
        #
        # a_2 is the offset from the center to each of the octagon
        # corners
        #
        a = <int>(<np.float64_t>radius * 2.0 / 2.414213)
        a_2 = a / 2
        if a_2 == 0:
            a_2 = 1
        self.histograms.a_2 = a_2
        if radius <= a_2:
            radius = a_2+1
        self.histograms.radius = radius
        self.histograms.current_row = -radius
        #
        # These are the cursors according to the plan
        #
        self.histograms.last_top_left.j0 = -a_2
        self.histograms.last_top_left.i0 = -radius - 1
        self.histograms.last_top_left.sign = -1
        self.histograms.last_top_left.name = "last top left"
        self.pSCoords[0] = &self.histograms.last_top_left
            
        self.histograms.top_left.j0 = -radius
        self.histograms.top_left.i0 = -a_2 - 1
        self.histograms.top_left.sign = 1
        self.histograms.top_left.name = "top left"
        self.pSCoords[1] = &self.histograms.top_left

        self.histograms.last_top_right.j0 = a_2 - 1
        self.histograms.last_top_right.i0 = - radius - 1
        self.histograms.last_top_right.sign = -1
        self.histograms.last_top_right.name = "last top right"
        self.pSCoords[2] = &self.histograms.last_top_right

        self.histograms.top_right.j0 = radius - 1
        self.histograms.top_right.i0 = - a_2 - 1
        self.histograms.top_right.sign = 1
        self.histograms.top_right.name = "top right"
        self.pSCoords[3] = &self.histograms.top_right

        self.histograms.last_leading_edge.j0 = radius
        self.histograms.last_leading_edge.i0 = - a_2 - 1
        self.histograms.last_leading_edge.sign = -1
        self.histograms.last_leading_edge.name = "last leading edge"
        self.pSCoords[4] = &self.histograms.last_leading_edge

        self.histograms.leading_edge.j0 = radius
        self.histograms.leading_edge.i0 = a_2
        self.histograms.leading_edge.sign = 1
        self.histograms.leading_edge.name = "leading edge"
        self.pSCoords[5] = &self.histograms.leading_edge

        self.histograms.last_bottom_right.j0 = radius
        self.histograms.last_bottom_right.i0 = a_2
        self.histograms.last_bottom_right.sign = -1
        self.histograms.last_bottom_right.name = "last bottom right"
        self.pSCoords[6] = &self.histograms.last_bottom_right

        self.histograms.bottom_right.j0 = a_2
        self.histograms.bottom_right.i0 = radius
        self.histograms.bottom_right.sign = 1
        self.histograms.bottom_right.name = "bottom right"
        self.pSCoords[7] = &self.histograms.bottom_right

        self.histograms.last_bottom_left.j0 = -radius-1
        self.histograms.last_bottom_left.i0 = a_2
        self.histograms.last_bottom_left.sign = -1
        self.histograms.last_bottom_left.name = "last bottom left"
        self.pSCoords[8] = &self.histograms.last_bottom_left

        self.histograms.bottom_left.j0 = -a_2-1
        self.histograms.bottom_left.i0 = radius
        self.histograms.bottom_left.sign = 1
        self.histograms.bottom_left.name = "bottom left"
        self.pSCoords[9] = &self.histograms.bottom_left

    def done(self):
        '''Return True if we've processed the last row'''
        return self.histograms.current_row == self.histograms.row_count
        
    def get_raster_indexes(self):
        '''Get the I coordinates of the rasters needed for the next operation
        
        returns a sequence of I indexes of the rasters needed. The caller
        should then call process_raster() with the rasters in the same order
        as in the sequence.
        '''
        self.scoordData = list()
        self.scoordMasks = list()
        rasters = []
        #
        # As a side-effect, we set up to process the raster
        #
        self.histograms.current_column = -self.histograms.radius + 1
        self.n_indexes = 0
        for idx in range(10):
            i = self.pSCoords[idx].i0 + self.histograms.current_row
            if i >=0 and i < self.histograms.row_count:
                if i in rasters:
                    rindx = rasters.index(i)
                else:
                    rasters.append(i)
                self.pActiveSCoords[self.n_indexes] = self.pSCoords[idx]
                self.scoord_indexes[self.n_indexes] = i
                self.n_indexes += 1
        return rasters
        
    def install_raster(
        self, idx, 
        np.ndarray[dtype=input_t, ndim=2, 
                   negative_indices=False, mode='c'] raster, 
        np.ndarray[dtype=np.uint8_t, ndim=1, 
                   negative_indices=False, mode='c'] mask):
        '''Install one of the rasters requested by get_raster_indexes
        
        idx - the index into the array returned by get_raster_indexes
        
        raster - a 2-d array of int32s where the rows are the j along the
                 raster and the columns are counts of occurrences of each of
                 256 levels.
                 
        mask - a 1-d array of uint8s where a 0 indicates a masked element.
        '''
        cdef:
            SCoord *pSCoord
            np.uint32_t rindex
        assert raster.shape[0] >= self.histograms.column_count
        assert raster.shape[1] >= 256
        assert mask.shape[0] >= self.histograms.column_count
        assert raster.strides[1] == sizeof(pixel_count_t)
        self.scoordData.append(raster)
        self.scoordMasks.append(mask)
        for rindex from 0 <= rindex < self.n_indexes:
            if self.scoord_indexes[rindex] == idx:
                pSCoord = self.pActiveSCoords[rindex]
                pSCoord.data = <input_t*>raster.data
                pSCoord.stride_j = \
                    np.PyArray_STRIDE(raster, 0) / np.PyArray_ITEMSIZE(raster)
                pSCoord.mask = <mask_t *>mask.data
                pSCoord.mask_stride = \
                    np.PyArray_STRIDE(mask, 0) / np.PyArray_ITEMSIZE(mask)
            
    def process_raster(self):
        cdef:
            Histograms *ph = &self.histograms
            np.int32_t col
            output_t *ptr
        assert not self.done()
        init_row(ph)
        with nogil:
            if ph.current_row >= 0:
                for col from 0 <= col < ph.column_count:
                    ph.current_column = col
                    update_current_location(ph)
                    accumulate(ph)
                    ptr = ph.output + ph.current_row * ph.row_stride +\
                          col * ph.col_stride
                    find_median(ph, ptr)
            else:
                for col from 0 <= col < ph.column_count:
                    ph.current_column = col
                    update_current_location(ph)
                    accumulate(ph)
            finish_row(ph)
            ph.current_row += 1
    
############################################################################    
#
# <tl,tr,bl,br>_colidx - convert a column index into the histogram
#                        index for a diagonal
#
# The top-right and bottom left diagonals for one row at one column
# become the diagonals for the next column to the right for the next row.
# Conversely, the top-left and bottom right become the diagonals for the
# previous column.
#
# These functions use the current row number to find the index of
# a particular histogram taking this into account. The indices progress
# forward or backward as you go to successive rows.
#
# The histogram array is, in effect, a circular buffer, so the start
# offset is immaterial - we take advantage of this to make sure that
# the numbers computed before taking the modulus are all positive, including
# those that might be done for columns to the left of 0. We add 3* the radius
# here to account for a row of -radius, a column of -radius and a request for
# a column that is "radius" to the left.
#
############################################################################    
cdef inline np.int32_t tl_br_colidx(Histograms *ph, np.int32_t colidx) nogil:
    return cython.cmod(
        colidx + 3*ph.radius + ph.current_row + ph.stripe_length, 
        ph.stripe_length)

cdef inline np.int32_t tr_bl_colidx(Histograms *ph, np.int32_t colidx) nogil:
    return cython.cmod(
        colidx + 3*ph.radius + ph.row_count-ph.current_row + ph.stripe_length,
        ph.stripe_length)

cdef inline np.int32_t leading_edge_colidx(Histograms *ph, np.int32_t colidx) nogil:
    return cython.cmod(colidx + 5*ph.radius + ph.stripe_length, 
                       ph.stripe_length)

cdef inline np.int32_t trailing_edge_colidx(Histograms *ph, np.int32_t colidx) nogil:
    return cython.cmod(colidx + 3*ph.radius - 1 + ph.stripe_length,
                       ph.stripe_length)
#
# add16 - add 16 consecutive integers
#
# Add an array of 16 pixel counts to an accumulator of 16 pixel counts
#
cdef inline void add16(pixel_count_t *dest, pixel_count_t *src) nogil:
    # Hopefully, inline like this uses SIMD if the compiler is at all smart
    dest[0]  += src[0]
    dest[1]  += src[1]
    dest[2]  += src[2]
    dest[3]  += src[3]
    dest[4]  += src[4]
    dest[5]  += src[5]
    dest[6]  += src[6]
    dest[7]  += src[7]
    dest[8]  += src[8]
    dest[9]  += src[9]
    dest[10] += src[10]
    dest[11] += src[11]
    dest[12] += src[12]
    dest[13] += src[13]
    dest[14] += src[14]
    dest[15] += src[15]

cdef inline void sub16(pixel_count_t *dest, pixel_count_t *src) nogil:
    dest[0]  -= src[0]
    dest[1]  -= src[1]
    dest[2]  -= src[2]
    dest[3]  -= src[3]
    dest[4]  -= src[4]
    dest[5]  -= src[5]
    dest[6]  -= src[6]
    dest[7]  -= src[7]
    dest[8]  -= src[8]
    dest[9]  -= src[9]
    dest[10] -= src[10]
    dest[11] -= src[11]
    dest[12] -= src[12]
    dest[13] -= src[13]
    dest[14] -= src[14]
    dest[15] -= src[15]

############################################################################    
#
# accumulate_coarse_histogram - accumulate the coarse histogram
#                               at an index into the accumulator
#
# ph     - the Histograms structure that holds the accumulator
# colidx - the index of the column to add
#
############################################################################    
cdef inline void accumulate_coarse_histogram(Histograms *ph, np.int32_t colidx) nogil:
    cdef:
        int offset

    offset = tr_bl_colidx(ph, colidx)
    if ph.pixel_count[offset].top_right > 0:
        add16(ph.accumulator.coarse, ph.histograms[offset].top_right.coarse)
        ph.accumulator_count += ph.pixel_count[offset].top_right
    offset = leading_edge_colidx(ph, colidx)
    if ph.pixel_count[offset].edge > 0:
        add16(ph.accumulator.coarse, ph.histograms[offset].edge.coarse)
        ph.accumulator_count += ph.pixel_count[offset].edge
    offset = tl_br_colidx(ph, colidx)
    if ph.pixel_count[offset].bottom_right > 0:
        add16(ph.accumulator.coarse, ph.histograms[offset].bottom_right.coarse)
        ph.accumulator_count += ph.pixel_count[offset].bottom_right

############################################################################    
#
# deaccumulate_coarse_histogram - subtract the coarse histogram
#                                 for a given column
#
############################################################################    
cdef inline void deaccumulate_coarse_histogram(Histograms *ph, np.int32_t colidx) nogil:
    cdef:
        int offset
    #
    # The trailing diagonals don't appear until here
    #
    if colidx <= ph.a_2:
        return
    offset = tl_br_colidx(ph, colidx)
    if ph.pixel_count[offset].top_left > 0:
        sub16(ph.accumulator.coarse, ph.histograms[offset].top_left.coarse)
        ph.accumulator_count -= ph.pixel_count[offset].top_left
    #
    # The trailing edge doesn't appear from the border until here
    #
    if colidx > ph.radius:
        offset = trailing_edge_colidx(ph, colidx)
        if ph.pixel_count[offset].edge > 0:
            sub16(ph.accumulator.coarse, ph.histograms[offset].edge.coarse)
            ph.accumulator_count -= ph.pixel_count[offset].edge
    offset = tr_bl_colidx(ph, colidx)
    if ph.pixel_count[offset].bottom_left > 0:
        sub16(ph.accumulator.coarse, ph.histograms[offset].bottom_left.coarse)
        ph.accumulator_count -= ph.pixel_count[offset].bottom_left
    
############################################################################    
#
# accumulate_fine_histogram - accumulate one of the 16 fine histograms
#
############################################################################    
cdef inline void accumulate_fine_histogram(Histograms *ph, 
                                           np.int32_t colidx,
                                           np.uint32_t fineidx) nogil:
    cdef:
        int fineoffset = fineidx * 16
        int offset

    offset = tr_bl_colidx(ph, colidx)
    add16(ph.accumulator.fine+fineoffset, ph.histograms[offset].top_right.fine+fineoffset)
    offset = leading_edge_colidx(ph, colidx)
    add16(ph.accumulator.fine+fineoffset, ph.histograms[offset].edge.fine+fineoffset)
    offset = tl_br_colidx(ph, colidx)
    add16(ph.accumulator.fine+fineoffset, ph.histograms[offset].bottom_right.fine+fineoffset)

############################################################################    
#
# deaccumulate_fine_histogram - subtract one of the 16 fine histograms
#
############################################################################    
cdef inline void deaccumulate_fine_histogram(Histograms *ph, 
                                             np.int32_t colidx,
                                             np.uint32_t fineidx) nogil:
    cdef:
        int fineoffset = fineidx * 16
        int offset

    #
    # The trailing diagonals don't appear until here
    #
    if colidx < ph.a_2:
        return
    offset = tl_br_colidx(ph, colidx)
    sub16(ph.accumulator.fine+fineoffset, ph.histograms[offset].top_left.fine+fineoffset)
    if colidx >= ph.radius:
        offset = trailing_edge_colidx(ph, colidx)
        sub16(ph.accumulator.fine+fineoffset, ph.histograms[offset].edge.fine+fineoffset)
    offset = tr_bl_colidx(ph, colidx)
    sub16(ph.accumulator.fine+fineoffset, ph.histograms[offset].bottom_left.fine+fineoffset)
    return
    
############################################################################    
#
# accumulate - add the leading edge and subtract the trailing edge
#
############################################################################    

cdef inline void accumulate(Histograms *ph) nogil:
    accumulate_coarse_histogram(ph, ph.current_column)
    deaccumulate_coarse_histogram(ph, ph.current_column)

############################################################################    
#
# update_fine - update one of the fine histograms to the current column
#
# The code has two choices:
#    redo the fine histogram from scratch - this involves accumulating
#         the entire histogram from the top_left.x to the top_right.x,
#         the center (edge) histogram from the trailing edge x to the
#         top_left.x and then computing a histogram of all points between
#         the trailing edge top, the point, (top_left.x,trailing edge top.y)
#         and the top_right and the corresponding triangle in the octagon's
#         lower half.
#
#    accumulate and deaccumulate within the fine histogram from the last
#    column computed.
#
#    The code below only implements the accumulate; redo and the code
#    to choose remains to be done.
############################################################################    

cdef inline void update_fine(Histograms *ph, int fineidx) nogil:
    cdef:
        int first_update_column = ph.last_update_column[fineidx]+1
        int update_limit        = ph.current_column+1
        int i
 
    for i from first_update_column <= i < update_limit:
        accumulate_fine_histogram(ph, i, fineidx)
        deaccumulate_fine_histogram(ph, i, fineidx)
    ph.last_update_column[fineidx] = ph.current_column

############################################################################    
#
# update_histogram - update the coarse and fine levels of a histogram
#                    based on addition of one value and subtraction of another
#
# ph         - Histograms pointer (for access to row_count, column_count)
# hist_piece - coarse and fine histogram to update
# pixel_count- pointer to pixel counter for histogram
# last_coord - coordinate and stride of pixel to remove
# coord      - coordinate and stride of pixel to add
# 
############################################################################    
cdef inline void update_histogram(Histograms *ph,
                                  HistogramPiece *hist_piece,
                                  pixel_count_t *pixel_count,
                                  SCoord *last_coord,
                                  SCoord *coord) nogil:
    cdef:
        np.int32_t current_column = ph.current_column
        np.int32_t current_row    = ph.current_row
        np.int32_t column_count   = ph.column_count
        np.int32_t row_count      = ph.row_count
        np.uint8_t value
        np.int32_t stride
        np.int32_t i, il, j, jl
        np.int32_t cidx
        np.int32_t fidx, fidx1
        pixel_count_t acc
        input_t *ptr
        pixel_count_t *cptr
        pixel_count_t *fptr

    jl = last_coord.j0 + current_column
    il = last_coord.i0 + current_row
    j = coord.j0 + current_column
    i = coord.i0 + current_row

    cptr = hist_piece.coarse
    fptr = hist_piece.fine
    if (jl >= 0 and jl < column_count and
        il >= 0 and il < row_count and
        last_coord.mask[last_coord.mask_stride * jl]):
    
        #
        # update the fine and coarse histogram counts at the coordinate
        #
        ptr = last_coord.data + last_coord.stride_j * jl
        fidx = 0
        for cidx from 0 <= cidx < 16:
            acc = 0
            # Inline for SIMD
            fptr[fidx]    -= ptr[fidx]
            fptr[fidx+1]  -= ptr[fidx+1]
            fptr[fidx+2]  -= ptr[fidx+2]
            fptr[fidx+3]  -= ptr[fidx+3]
            fptr[fidx+4]  -= ptr[fidx+4]
            fptr[fidx+5]  -= ptr[fidx+5]
            fptr[fidx+6]  -= ptr[fidx+6]
            fptr[fidx+7]  -= ptr[fidx+7]
            fptr[fidx+8]  -= ptr[fidx+8]
            fptr[fidx+9]  -= ptr[fidx+9]
            fptr[fidx+10] -= ptr[fidx+10]
            fptr[fidx+11] -= ptr[fidx+11]
            fptr[fidx+12] -= ptr[fidx+12]
            fptr[fidx+13] -= ptr[fidx+13]
            fptr[fidx+14] -= ptr[fidx+14]
            fptr[fidx+15] -= ptr[fidx+15]
            acc += ptr[fidx]
            acc += ptr[fidx+1]
            acc += ptr[fidx+2]
            acc += ptr[fidx+3]
            acc += ptr[fidx+4]
            acc += ptr[fidx+5]
            acc += ptr[fidx+6]
            acc += ptr[fidx+7]
            acc += ptr[fidx+8]
            acc += ptr[fidx+9]
            acc += ptr[fidx+10]
            acc += ptr[fidx+11]
            acc += ptr[fidx+12]
            acc += ptr[fidx+13]
            acc += ptr[fidx+14]
            acc += ptr[fidx+15]
            pixel_count[0] -= acc
            cptr[cidx] -= acc
            fidx += 16

    if (j >= 0 and j < column_count and
        i >= 0 and i < row_count and
        coord.mask[coord.mask_stride*j]):
        ptr = coord.data + coord.stride_j * j
        fidx = 0
        for cidx from 0 <= cidx < 16:
            acc = 0
            # Inline for SIMD
            fptr[fidx]    += ptr[fidx]
            fptr[fidx+1]  += ptr[fidx+1]
            fptr[fidx+2]  += ptr[fidx+2]
            fptr[fidx+3]  += ptr[fidx+3]
            fptr[fidx+4]  += ptr[fidx+4]
            fptr[fidx+5]  += ptr[fidx+5]
            fptr[fidx+6]  += ptr[fidx+6]
            fptr[fidx+7]  += ptr[fidx+7]
            fptr[fidx+8]  += ptr[fidx+8]
            fptr[fidx+9]  += ptr[fidx+9]
            fptr[fidx+10] += ptr[fidx+10]
            fptr[fidx+11] += ptr[fidx+11]
            fptr[fidx+12] += ptr[fidx+12]
            fptr[fidx+13] += ptr[fidx+13]
            fptr[fidx+14] += ptr[fidx+14]
            fptr[fidx+15] += ptr[fidx+15]
            acc += ptr[fidx]
            acc += ptr[fidx+1]
            acc += ptr[fidx+2]
            acc += ptr[fidx+3]
            acc += ptr[fidx+4]
            acc += ptr[fidx+5]
            acc += ptr[fidx+6]
            acc += ptr[fidx+7]
            acc += ptr[fidx+8]
            acc += ptr[fidx+9]
            acc += ptr[fidx+10]
            acc += ptr[fidx+11]
            acc += ptr[fidx+12]
            acc += ptr[fidx+13]
            acc += ptr[fidx+14]
            acc += ptr[fidx+15]
            pixel_count[0] += acc
            cptr[cidx] += acc
            fidx += 16

############################################################################    
#
# update_current_location - update the histograms at the current location
#
############################################################################    
#cdef inline void update_current_location(Histograms *ph) nogil:
cdef inline void update_current_location(Histograms *ph) nogil:
    cdef:
        np.int32_t current_column   = ph.current_column
        np.int32_t radius           = ph.radius
        np.int32_t top_left_off     = tl_br_colidx(ph, current_column)
        np.int32_t top_right_off    = tr_bl_colidx(ph, current_column)
        np.int32_t bottom_left_off  = tr_bl_colidx(ph, current_column)
        np.int32_t bottom_right_off = tl_br_colidx(ph, current_column)
        np.int32_t leading_edge_off = leading_edge_colidx(ph, current_column)

    update_histogram(ph, &ph.histograms[top_left_off].top_left,
                     &ph.pixel_count[top_left_off].top_left,
                     &ph.last_top_left,
                     &ph.top_left)

    update_histogram(ph, &ph.histograms[top_right_off].top_right,
                     &ph.pixel_count[top_right_off].top_right,
                     &ph.last_top_right,
                     &ph.top_right)

    update_histogram(ph, &ph.histograms[bottom_left_off].bottom_left,
                     &ph.pixel_count[bottom_left_off].bottom_left,
                     &ph.last_bottom_left,
                     &ph.bottom_left)

    update_histogram(ph, &ph.histograms[bottom_right_off].bottom_right,
                     &ph.pixel_count[bottom_right_off].bottom_right,
                     &ph.last_bottom_right,
                     &ph.bottom_right)

    update_histogram(ph, &ph.histograms[leading_edge_off].edge,
                     &ph.pixel_count[leading_edge_off].edge,
                     &ph.last_leading_edge,
                     &ph.leading_edge)

############################################################################
#
# init_row - initialize the histograms at the start of a row
#
############################################################################

cdef inline void init_row(Histograms *ph):
    cdef:
        np.int32_t tl_off, br_off
        np.int32_t tr_off, bl_off
        np.int32_t i
        np.int32_t col
        np.int32_t radius = ph.radius
        np.int32_t columns = ph.column_count
    #
    # Initialize the starting diagonal histograms to zero.
    #
    tl_off        = tl_br_colidx(ph, columns + radius)
    br_off        = tl_br_colidx(ph, columns + radius)
    tr_off        = tr_bl_colidx(ph, -radius)
    bl_off        = tr_bl_colidx(ph, -radius)

    memset(&ph.histograms[tl_off].top_left, 0, sizeof(HistogramPiece))
    memset(&ph.histograms[br_off].bottom_right, 0, sizeof(HistogramPiece))
    memset(&ph.histograms[tr_off].top_right, 0, sizeof(HistogramPiece))
    memset(&ph.histograms[bl_off].bottom_left, 0, sizeof(HistogramPiece))
    ph.pixel_count[tl_off].top_left     = 0
    ph.pixel_count[br_off].bottom_right = 0
    ph.pixel_count[tr_off].top_right    = 0
    ph.pixel_count[bl_off].bottom_left  = 0
    #
    # Initialize the accumulator (octagon histogram) to zero
    #
    memset(&ph.accumulator, 0, sizeof(ph.accumulator))
    ph.accumulator_count = 0
    for i from 0 <= i < 16:
        ph.last_update_column[i] = -radius-1
    #
    # Update locations and coarse accumulator for the octagon
    # for points before 0
    #
    with nogil:
        for col from -radius <= col < 0:
            ph.current_column = col
            update_current_location(ph)
            accumulate(ph)
        
############################################################################
#
# finish_row - Process the trailing edge of the octagon at the end of a row
#
############################################################################
cdef inline void finish_row(Histograms *ph) nogil:
    cdef:
        np.int32_t col
    for col from ph.column_count <= col <= ph.column_count + ph.radius:
        ph.current_column = col
        update_current_location(ph)
        accumulate(ph)
            
############################################################################
#
# find_median - search the current accumulator for the median
#
############################################################################

@cython.cdivision(True)
cdef inline void find_median(Histograms *ph, output_t *ptr) nogil:
    cdef:
        pixel_count_t pixels_below      # of pixels below the median
        int i
        int j
        int k
        int l
        np.float32_t result
        pixel_count_t accumulator

    i = ph.current_row
    j = ph.current_column
         
    if ph.accumulator_count == 0:
        return
    pixels_below = (ph.accumulator_count * ph.percent + 50) / 100 # +50 for roundoff
    if pixels_below > 0:
        pixels_below -= 1
    accumulator = 0
    for i from 0 <= i <= 16:
        accumulator += ph.accumulator.coarse[i]
        if accumulator > pixels_below:
            break
    accumulator -= ph.accumulator.coarse[i]
    update_fine(ph, i)
    for j from i*16 <= j < (i+1)*16:
        accumulator += ph.accumulator.fine[j]
        if accumulator > pixels_below:
            if accumulator == pixels_below+1 and \
               accumulator < ph.accumulator_count:
                #
                # Find next non-zero value
                #
                for k from j+1 <= k <(i+1)*16:
                    if ph.accumulator.fine[k] > 0:
                        ptr[0] = (<output_t>(j+k)) / 2.0
                        return
                #
                # Check coarse, then update fine to find next
                #
                for k from i+1 <= k <16:
                    if ph.accumulator.coarse[k] > 0:
                        update_fine(ph, k)
                        for l from k*16 <= l < k*16+15:
                            if ph.accumulator.fine[l] > 0:
                                break
                    ptr[0] = (<output_t>(j+l)) / 2.0
                    return
            ptr[0] = j
            return
    ptr[0] = 0
