import unittest

from nsdmd import model
from nsdmd import dmdio

import numpy as np
import os



class TestFunctions(unittest.TestCase):
    def test_io(self):
        filename = 'example_data.txt'
        testdata = dmdio.load_data(filename)
        self.assertTrue(issubclass(type(testdata), np.ndarray))


print("testcode works fine")
