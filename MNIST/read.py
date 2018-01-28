import struct
import array
import functools
import operator
import numpy as np
def fun_read(fd):
	    DATA_TYPES = {0x08: 'B',  # unsigned byte
		          0x09: 'b',  # signed byte
			  0x0b: 'h',  # short (2 bytes)
			  0x0c: 'i',  # int (4 bytes)
		          0x0d: 'f',  # float (4 bytes)
			  0x0e: 'd'}  # double (8 bytes)

		
	    header=fd.read(4)
	    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
	    if zeros != 0:
		raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
		                     'Found 0x%02x' % zeros)
	    try:
		data_type = DATA_TYPES[data_type]
	    except KeyError:
		raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

	    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
		                            fd.read(4 * num_dimensions))

	    data = array.array(data_type, fd.read())
	    data.byteswap()  # looks like array.array reads data as little endian

	    expected_items = functools.reduce(operator.mul, dimension_sizes)
	    if len(data) != expected_items:
		raise IdxDecodeError('IDX file has wrong number of items. '
		                     'Expected: %d. Found: %d' % (expected_items, len(data)))

	    return np.array(data).reshape(dimension_sizes)

def load_train_data():

            with open('Data/train-images.idx3-ubyte','rb') as fd:
            	return fun_read(fd)

def load_train_labels():
	    with open('Data/train-labels.idx1-ubyte','rb') as fd:
	    	return fun_read(fd)

def load_test_data():
	  
	   with open('Data/t10k-images.idx3-ubyte','rb') as fd:
	    	return fun_read(fd)

def load_test_labels():
	   with open('Data/t10k-labels.idx1-ubyte','rb') as fd:
	    	return fun_read(fd)

	   with open('t10k-labels.idx1-ubyte','rb') as fd:
	    	return fun_read(fd)


