import unittest


from nsdmd import model
from nsdmd import dmdio

import os



class TestFunctions(unittest.TestCase):
    def test_io(self):
        filename = 'example_data.txt'
        testdata = dmdio.load_data(filename)
        self.assertTrue(issubclass(type(testdata), np.ndarray))
        
        
    def text_likelihood(self):
        testdata = dmdio.load_data('example_data.txt')
        theta_range = np.linspace(0,0.1,3)
        
        for i in rannge(len(theta_range)):
            vrot[i]=model.model_NFW(theta,x)
        self.assertTrue(np.argmax(vrot)>0 and np.argmax(vrot) <1000 )

            
            
            
            



print("testcode works fine")