import array
import functools
import numpy
import operator
import struct
from intake.source.base import DataSource, Schema


class MNISTImagesPlugin(DataSource):
    name = "mnist_images"
    container = 'numpy'
    version = '0.0.1'
    partition_access = False

    datasets_url = 'http://yann.lecun.com/exdb/mnist/'

    def __init__(self, part, metadata=None):
        """Read MNIST images and labels

        Parameters
        ----------
        part: str
            'train' or 'test' for the two sections of the data.

        Returns
        -------
        After discovery, the files will have been downloaded and uncompressed
        into the Intake cache and the attributes "labels" and "images" will
        be populated. You could select these with the ``read_partition``
        method, as with canonical Intake data source use, or just access
        the attributes directly.
        """
        super(MNISTImagesPlugin, self).__init__(metadata=metadata)
        files = {'train': ['t10k-labels-idx1-ubyte.gz',
                           't10k-images-idx3-ubyte.gz'],
                 'test': ['train-labels-idx1-ubyte.gz',
                          'train-images-idx3-ubyte.gz']}
        self.lfile, self.ifile = [self.datasets_url + f for f in files[part]]
        self.labels = None
        self.images = None

    def _get_schema(self):
        if self.labels is None:
            lfile = self._get_cache(self.lfile)[0]
            ifile = self._get_cache(self.ifile)[0]

            self.labels = parse_idx(open(lfile[0], 'rb'))
            self.images = parse_idx(open(ifile[0], 'rb'))
        return Schema(datashape=None,
                      dtype=self.images.dtype,
                      shape=self.images.shape,
                      npartitions=1,
                      extra_metadata={})

    def read_partition(self, i):
        """We don't really have partitions, but can select labels or images"""
        self._get_schema()
        if isinstance(i, (tuple, list)):
            i = i[0]
        if i == 'labels':
            return self.labels
        elif i == 'images':
            return self.images
        else:
            raise KeyError("Select 'labels', or 'images'")

    def read(self):
        """The "data" here is the images"""
        self._get_schema()
        return self.images


def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.

    https://github.com/datapythonista/mnist/blob/master/mnist/__init__.py
    Creadit: @datapythonista, Marc Garcia

    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse

    endian : str
        Byte order of the IDX file. See [1] for available options

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file

    1. https://docs.python.org/3/library/struct.html
        #byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise RuntimeError('Invalid IDX file, '
                           'file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise RuntimeError('Invalid IDX file, '
                           'file must start with two zero bytes. '
                           'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise RuntimeError('Unknown data type '
                           '0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise RuntimeError('IDX file has wrong number of items. '
                           'Expected: %d. Found: %d' % (expected_items,
                                                          len(data)))

    return numpy.array(data).reshape(dimension_sizes)
