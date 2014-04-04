'''3dmedianfilter.py - functions for applying filters to images

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
from _3dmedianfilter import M3DState
from scipy.sparse import coo_matrix

def median_filter3d(data, mask, radius, percent=50, xform = None):
    '''Masked three-dimensional median filter with octagonal shape
    
    This median filter computes a 2-d median projection of the cylindrical
    volume within a circle of a given radius in the i,j plane and the projection
    through the k plane. The filter is designed to work with extremely large
    blocks of data - for instance you can pass it an h5py HDF5 file array.
    
    The data must be scaled or ranked to 256 levels as a preprocessing step.
    The output is floating point with the fraction part equal to the ratio
    of number of counts below the median at the median level divided by
    counts at the median level.
    
    data - 3d array of np.uint8 data to be median filtered.
    
    mask - mask of significant pixels in data
    
    radius - the radius of a circle inscribed into the filtering octagon
    
    percent - conceptually, order the significant pixels in the octagon,
              count them and choose the pixel indexed by the percent
              times the count divided by 100. More simply, 50 = median
              
    xform - a function which takes a vector of N data elements and the
            index into the data as input and 
            produces an N by 256 array of level counts.
            
    returns a filtered array.  In areas where the median filter does
      not overlap the mask, the filtered result is undefined, but in
      practice, it will be the lowest value in the valid area.
    '''
    if xform == None:
        if data.dtype.kind == "u" and data.dtype.itemsize == 1:
            normalize = lambda a: np.ascontiguousarray(a, dtype=np.uint8)
        else:
            if mask is None:
                minimum = np.min(data)
                maximum = np.max(data)
            else:
                minimum = np.min(data[mask])
                maximum = np.max(data[mask])
            normalize = lambda a: np.ascontiguousarray(
                255 * (a.astype(np.float32) - minimum) / (maximum - minimum),
                dtype = np.uint8)
        def xform(a, index):
            b = normalize(a)
            i, j = np.mgrid[0:a.shape[0], 0:a.shape[1]]
            i = i.flatten()
            j = j.flatten()
            return coo_matrix(
                (np.ones(i.shape[0], np.int32), (i, b[i, j])),
                shape = (a.shape[0], 256), dtype = np.uint32).toarray()
        
    output = np.zeros(data.shape[:2], np.float32)
    state = M3DState(output, radius, percent)
    while not state.done():
        indexes = state.get_raster_indexes()
        # the rasters and mask_rasters lists  hold onto
        # references of the rasters
        for index in indexes:
            r = xform(data[index], index)
            if mask is not None:
                m = np.ascontiguousarray(mask[index], np.uint8)
            else:
                m = np.ones(data.shape[1], np.uint8)
            state.install_raster(index, r, m)
        state.process_raster()
    return output
