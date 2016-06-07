import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize

try:
    # Python3 will most likely not be able to load protobuf
    from caffe.proto import caffe_pb2
except:
    import sys
    if sys.version_info >= (3, 0):
        print("Failed to include caffe_pb2, things might go wrong!")
    else:
        raise


## proto / datum / ndarray conversion
def blobproto_to_array(blob, return_diff=False):
    """
    Convert a blob proto to an array. In default, we will just return the data,
    unless return_diff is True, in which case we will return the diff.
    """
    # Read the data into an array
    if return_diff:
        data = np.array(blob.diff)
    else:
        data = np.array(blob.data)

    # Reshape the array
    if blob.HasField('num') or blob.HasField('channels') or blob.HasField('height') or blob.HasField('width'):
        # Use legacy 4D shape
        return data.reshape(blob.num, blob.channels, blob.height, blob.width)
    else:
        return data.reshape(blob.shape.dim)

def array_to_blobproto(arr, diff=None):
    """Converts a N-dimensional array to blob proto. If diff is given, also
    convert the diff. You need to make sure that arr and diff have the same
    shape, and this function does not do sanity check.
    """
    blob = caffe_pb2.BlobProto()
    blob.shape.dim.extend(arr.shape)
    blob.data.extend(arr.astype(float).flat)
    if diff is not None:
        blob.diff.extend(diff.astype(float).flat)
    return blob


def arraylist_to_blobprotovecor_str(arraylist):
    """Converts a list of arrays to a serialized blobprotovec, which could be
    then passed to a network for processing.
    """
    vec = caffe_pb2.BlobProtoVector()
    vec.blobs.extend([array_to_blobproto(arr) for arr in arraylist])
    return vec.SerializeToString()


def blobprotovector_str_to_arraylist(str):
    """Converts a serialized blobprotovec to a list of arrays.
    """
    vec = caffe_pb2.BlobProtoVector()
    vec.ParseFromString(str)
    return [blobproto_to_array(blob) for blob in vec.blobs]


def array_to_datum(arr, label=0):
    """Converts a 3-dimensional array to datum. If the array has dtype uint8,
    the output data will be encoded as a string. Otherwise, the output data
    will be stored in float format.
    """
    if arr.ndim != 3:
        raise ValueError('Incorrect array shape.')
    datum = caffe_pb2.Datum()
    datum.channels, datum.height, datum.width = arr.shape
    if arr.dtype == np.uint8:
        datum.data = arr.tostring()
    else:
        datum.float_data.extend(arr.flat)
    datum.label = label
    return datum


def datum_to_array(datum):
    """Converts a datum to an array. Note that the label is not returned,
    as one can easily get it by calling datum.label.
    """
    if len(datum.data):
        return np.fromstring(datum.data, dtype=np.uint8).reshape(
            datum.channels, datum.height, datum.width)
    else:
        return np.array(datum.float_data).astype(float).reshape(
            datum.channels, datum.height, datum.width)


## Pre-processing

class Transformer:
    """
    Transform input for feeding into a Net.

    Note: this is mostly for illustrative purposes and it is likely better
    to define your own input preprocessing routine for your needs.

    Parameters
    ----------
    net : a Net for which the input should be prepared
    """
    def __init__(self, inputs):
        self.inputs = inputs
        self.transpose = {}
        self.channel_swap = {}
        self.raw_scale = {}
        self.mean = {}
        self.input_scale = {}

    def __check_input(self, in_):
        if in_ not in self.inputs:
            raise Exception('{} is not one of the net inputs: {}'.format(
                in_, self.inputs))

    def preprocess(self, in_, data):
        """
        Format input for Caffe:
        - convert to single
        - resize to input dimensions (preserving number of channels)
        - transpose dimensions to K x H x W
        - reorder channels (for instance color to BGR)
        - scale raw input (e.g. from [0, 1] to [0, 255] for ImageNet models)
        - subtract mean
        - scale feature

        Parameters
        ----------
        in_ : name of input blob to preprocess for
        data : (H' x W' x K) ndarray

        Returns
        -------
        caffe_in : (K x H x W) ndarray for input to a Net
        """
        self.__check_input(in_)
        caffe_in = data.astype(np.float32, copy=False)
        transpose = self.transpose.get(in_)
        channel_swap = self.channel_swap.get(in_)
        raw_scale = self.raw_scale.get(in_)
        mean = self.mean.get(in_)
        input_scale = self.input_scale.get(in_)
        in_dims = self.inputs[in_][2:]
        if caffe_in.shape[:2] != in_dims:
            caffe_in = resize_image(caffe_in, in_dims)
        if transpose is not None:
            caffe_in = caffe_in.transpose(transpose)
        if channel_swap is not None:
            caffe_in = caffe_in[channel_swap, :, :]
        if raw_scale is not None:
            caffe_in *= raw_scale
        if mean is not None:
            caffe_in -= mean
        if input_scale is not None:
            caffe_in *= input_scale
        return caffe_in

    def deprocess(self, in_, data):
        """
        Invert Caffe formatting; see preprocess().
        """
        self.__check_input(in_)
        decaf_in = data.copy().squeeze()
        transpose = self.transpose.get(in_)
        channel_swap = self.channel_swap.get(in_)
        raw_scale = self.raw_scale.get(in_)
        mean = self.mean.get(in_)
        input_scale = self.input_scale.get(in_)
        if input_scale is not None:
            decaf_in /= input_scale
        if mean is not None:
            decaf_in += mean
        if raw_scale is not None:
            decaf_in /= raw_scale
        if channel_swap is not None:
            decaf_in = decaf_in[np.argsort(channel_swap), :, :]
        if transpose is not None:
            decaf_in = decaf_in.transpose(np.argsort(transpose))
        return decaf_in

    def set_transpose(self, in_, order):
        """
        Set the input channel order for e.g. RGB to BGR conversion
        as needed for the reference ImageNet model.

        Parameters
        ----------
        in_ : which input to assign this channel order
        order : the order to transpose the dimensions
        """
        self.__check_input(in_)
        if len(order) != len(self.inputs[in_]) - 1:
            raise Exception('Transpose order needs to have the same number of '
                            'dimensions as the input.')
        self.transpose[in_] = order

    def set_channel_swap(self, in_, order):
        """
        Set the input channel order for e.g. RGB to BGR conversion
        as needed for the reference ImageNet model.
        N.B. this assumes the channels are the first dimension AFTER transpose.

        Parameters
        ----------
        in_ : which input to assign this channel order
        order : the order to take the channels.
            (2,1,0) maps RGB to BGR for example.
        """
        self.__check_input(in_)
        if len(order) != self.inputs[in_][1]:
            raise Exception('Channel swap needs to have the same number of '
                            'dimensions as the input channels.')
        self.channel_swap[in_] = order

    def set_raw_scale(self, in_, scale):
        """
        Set the scale of raw features s.t. the input blob = input * scale.
        While Python represents images in [0, 1], certain Caffe models
        like CaffeNet and AlexNet represent images in [0, 255] so the raw_scale
        of these models must be 255.

        Parameters
        ----------
        in_ : which input to assign this scale factor
        scale : scale coefficient
        """
        self.__check_input(in_)
        self.raw_scale[in_] = scale

    def set_mean(self, in_, mean):
        """
        Set the mean to subtract for centering the data.

        Parameters
        ----------
        in_ : which input to assign this mean.
        mean : mean ndarray (input dimensional or broadcastable)
        """
        self.__check_input(in_)
        ms = mean.shape
        if mean.ndim == 1:
            # broadcast channels
            if ms[0] != self.inputs[in_][1]:
                raise ValueError('Mean channels incompatible with input.')
            mean = mean[:, np.newaxis, np.newaxis]
        else:
            # elementwise mean
            if len(ms) == 2:
                ms = (1,) + ms
            if len(ms) != 3:
                raise ValueError('Mean shape invalid')
            if ms != self.inputs[in_][1:]:
                raise ValueError('Mean shape incompatible with input shape.')
        self.mean[in_] = mean

    def set_input_scale(self, in_, scale):
        """
        Set the scale of preprocessed inputs s.t. the blob = blob * scale.
        N.B. input_scale is done AFTER mean subtraction and other preprocessing
        while raw_scale is done BEFORE.

        Parameters
        ----------
        in_ : which input to assign this scale factor
        scale : scale coefficient
        """
        self.__check_input(in_)
        self.input_scale[in_] = scale


## Image IO

def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.

    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.

    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)


def oversample(images, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.

    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.

    Returns
    -------
    crops : (10*N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10 * len(images), crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirrors
    return crops

def oversample_new(image, crop_dims, resize_dims=None):
    """
    Crop images into the four corners, center, and their mirrored versions.
    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.
    resize_dims : list of the shorter dims for resize before croping.
    Returns
    -------
    crops : (*N x H x W x K) ndarray of crops for number of inputs N.
    """
    short_dim = 0 if image.shape[0] < image.shape[1] else 1
    short = image.shape[short_dim]
    long = image.shape[1-short_dim]
    
    if not resize_dims:
        resize_dims = [short]
    
    N = len(resize_dims) * 36    
    crops = np.empty((N, crop_dims[0], crop_dims[1],
                      image.shape[-1]), dtype=np.float32)
    crop_id = 0
    for resize_dim in resize_dims:
        #resize the image
        if resize_dim == short:
            l = long
            im = image
            new_dim = (resize_dim, l) if short_dim==0 else (l, resize_dim)
        elif resize_dim < short:
            raise ValueError("resize_dim %d samller than short dim %d" %(resize_dim, short))
        else:
            l = int(round(resize_dim*1.0 / short * long))
            assert l >= short
            new_dim = (resize_dim, l) if short_dim==0 else (l, resize_dim)
            im = resize_image(image, new_dim)
        
        #take the left, center and right square of these resized images
        for left in (0, (l-resize_dim)/2, l-resize_dim):
            if short_dim == 0:
                im_square = im[:, left:left+resize_dim, :]
            else:
                im_square = im[left:left+resize_dim, :, :]
                
            #crop 4 corners and center and resize im_square to crop_dims
            # Dimensions and center.
            crop_dims = np.array(crop_dims)
            im_center = np.ones(2) * (resize_dim / 2.0)

            # Make crop coordinates
            h_indices = (0, resize_dim - crop_dims[0])
            w_indices = (0, resize_dim - crop_dims[1])
            crops_ix = np.empty((5, 4), dtype=int)
            curr = 0
            for i in h_indices:
                for j in w_indices:
                    crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
                    curr += 1
            crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
                -crop_dims / 2.0,
                 crop_dims / 2.0
            ])
            # Extract crops
            for crop in crops_ix:
                crops[crop_id] = im_square[crop[0]:crop[2], crop[1]:crop[3], :]
                crop_id += 1
            # Resize im_square to crop_dims
            crops[crop_id] = resize_image(im_square, crop_dims)
            crop_id += 1
    assert crop_id == N/2
    crops[N/2:N] = crops[0:N/2, :, ::-1, :]  # flip for mirrors
    return crops

def oversample_10(image, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.

    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.

    Returns
    -------
    crops : (10*N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(image.shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10, crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
#    for im in images:
    for crop in crops_ix:
        crops[ix] = image[crop[0]:crop[2], crop[1]:crop[3], :]
        ix += 1
    crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirrors
    return crops
    
if __name__ == "__main__":
    im = load_image("/home/cc/mycaffe/examples/images/cat.jpg")
    im = resize_image(im, (300, 256))
    crops = oversample_new(im, (224,224), [256,288,320,352])
    for i in range(len(crops)):
        imsave("crops/crop_%d.jpg"%i, crops[i])