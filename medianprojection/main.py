import argparse
import csv
import numpy as np
import os
import h5py
import sys
import javabridge
import bioformats
import urllib
import tempfile
from scipy.sparse import coo_matrix

from medianfilter import median_filter3d
def main():
    javabridge.start_vm(class_path=bioformats.JARS)
    try:
        src, dest, radius, percentile, stackfile = parse_arguments()
        if h5py.is_hdf5(src):
            h5file = h5py.File(src, mode="r")
            stack = h5file["stack"]
        else :
            with open(src, "rb") as fd:
                rdr = csv.reader(fd)
                header = rdr.next()
                imageset_rows = list(rdr)
                columns = dict([(col.lower(), i) for i, col in enumerate(header)])
                image_sets = []
                if "url" in columns.keys():
                    for row in imageset_rows:
                        url = row[columns["url"]]
                        series = row[columns["series"]] if "series" in columns else 0
                        frame = row[columns["frame"]] if "frame" in columns else 0
                        image_sets.append((url, series, frame))
                    images = make_url_image_reader(image_sets)
                else:
                    for row in imageset_rows:
                        path = os.path.join(row[columns["pathname"]],
                                            row[columns["filename"]])
                        series = row[columns["series"]] if "series" in columns else 0
                        frame = row[columns["frame"]] if "frame" in columns else 0
                        image_sets.append((path, series, frame))
                    images = make_file_image_reader(image_sets)
            if stackfile is None:
                h5fd, h5fn = tempfile.mkstemp(".h5")
                os.close(h5fd)
            else:
                h5fn = stackfile
            h5file = h5py.File(h5fn, "w")
            stack = make_stack(images, h5file)
        if stack.dtype.itemsize == 2:
            #
            # If we have a 16-bit image, we first calculate the median
            # of the top 8 bits, then subtract the median that we find
            # from the image and do another 8 bits.
            # For round 2, it doesn't matter that values are 7 bits above
            # or below the median - they only contribute to the ranking.
            #
            # Since the division rounds down, the median at one level will
            # always be below the actual median
            #
            pixel_type = bioformats.PT_UINT16
            def xform1(a, index):
                i, j = np.mgrid[0:a.shape[0], 0:a.shape[1]]
                i = i.flatten()
                j = j.flatten()
                b = (a[i, j] / (2 ** 8)).astype(np.uint32)
                return coo_matrix(
                    (np.ones(i.shape[0], np.int32), (i, b)),
                    shape = (a.shape[0], 256), dtype = np.uint32).toarray()
            
            result = median_filter3d(stack, None, radius, percentile, xform1)
            result = result.astype(np.uint32) * (2 ** 8)
            def xform2(a, index):
                i, j = np.mgrid[0:a.shape[0], 0:a.shape[1]]
                i = i.flatten()
                j = j.flatten()
                l = result[index ,: ]
                b = (a[i, j]-l[i]).astype(np.int32)
                b[b < 0] = 0
                b[b >= (2**8)] = (2**8)-1
                return coo_matrix(
                    (np.ones(i.shape[0], np.int32), (i, b.astype(np.uint32))),
                    shape = (a.shape[0], 256), dtype = np.uint32).toarray()
            level2 = median_filter3d(stack, None, radius, percentile, xform2)
            result = level2.astype(np.uint32) + result
        else:
            pixel_type = bioformats.PT_UINT8
            def xform(a, index):
                i, j = np.mgrid[0:a.shape[0], 0:a.shape[1]]
                i = i.flatten()
                j = j.flatten()
                return coo_matrix(
                    (np.ones(i.shape[0], np.int32), (i, a[i, j])),
                    shape = (a.shape[0], 256), dtype = np.uint32).toarray()
                
            result = median_filter3d(stack, None, radius, percentile, xform)
        bioformats.write_image(dest, result, pixel_type)
        h5file.close()
        if not h5py.is_hdf5(src) and stackfile is None:
            os.remove(h5fn)
    finally:
        javabridge.kill_vm()

def make_stack(images, h5file):
    assert isinstance(h5file, h5py.File)
    img0 = images.next()
    if img0.ndim == 3:
        img0 = (np.sum(img0[:,:,:3], 2) / 3).astype(img0.dtype)
    img0 = np.atleast_3d(img0) # an NxMx1 array
    if img0.shape[1] > 1024:
        chunk1 = 1024
    else:
        chunk1 = img0.shape[1]
    stack = h5file.create_dataset(
        "stack",
        data=img0,
        chunks=(16, chunk1, 64),
        maxshape=(img0.shape[0], img0.shape[1], None))
    for i, img in enumerate(images):
        if img.ndim == 3:
            img = (np.sum(img[:,:,:3], 2) / 3).astype(img.dtype)
        stack.resize(i+2, 2)
        stack[:, :, (i+1)] = img
    h5file.flush()
    return stack

def make_file_image_reader(image_sets):
    last_path = None
    for path, series, index in image_sets:
        if path != last_path:
            rdr = bioformats.get_image_reader(None, path=path)
            last_path = path
            print "Loading %s" % path
        yield rdr.read(series, index, rescale=False)
        
def make_url_image_reader(image_sets):
    last_url = None
    for url, series, index in image_sets:
        if url != last_url:
            rdr = bioformats.get_image_reader(None, url=url)
            last_url = url
            print "Loading %s" % path
        yield rdr.read(series.index, rescale=False)
    
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute the median projection of a stack of images")
    parser.add_argument(
        "src", metavar="SOURCEFILE",
        type=unicode,
        help="The filename of a .csv file that defines the image stack")
    parser.add_argument(
        "dest", metavar="DESTINATION",
        type=unicode,
        help="The name of the image file to be created")
    parser.add_argument(
        "--radius", "-r",
        dest = "radius", type=int, 
        help="Radius of the median filter",
        default=50)
    parser.add_argument(
        "--percentile", "-p",
        dest = "percentile", type=float,
        help="The percentile of the median filter (50 = median)",
        default=50.
    )
    parser.add_argument(
        "--stackfile", "-s",
        dest="stackfile", type=unicode,
        default=None,
        help="Write the hdf5 stack file here (so we can reuse)")
    args = parser.parse_args()
    return args.src, args.dest, args.radius, args.percentile, args.stackfile

if __name__=="__main__":
    main()
