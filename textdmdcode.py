import unittest
import os
import numpy as np


from nsdmd import model
from nsdmd import dmdio




class TestFunctions(unittest.TestCase):
    def test_io(self):
        filename = 'example_data.txt'
        testdata_x,testdata_xerr,testdata_y,testdata_yerr = dmdio.load_data(filename)
        self.assertTrue(issubclass(type(testdata_x), np.ndarray))
        
        
        
    def test_prior_transform_NFW(self):
        testdata = dmdio.load_data('example_data.txt')
        theta0_range = np.linspace(0,0.1,3)
        theta1_range = np.linspace(0,0.1,3)
        priorRange=np.ones((len(theta0_range),len(theta1_range)))
        

        for i in range(len(theta0_range)):
            for j in range(len(theta1_range)):
                
                theta = theta0_range[i], theta1_range[j]

                priorRange[i,j]= np.argmax(model.prior_transform_NFW(theta))
            
        self.assertTrue(np.argmax(priorRange)<10000 and np.argmax(priorRange)>-1)
        

        
        
     
    def test_model_NFW(self):
        
        theta0_range = np.linspace(6,1,10)
        vrot=np.ones(len(theta0_range))
        
        for i in range(len(theta0_range)):
            
            theta = theta0_range[i], 10
            vrot[i]=model.model_NFW(theta,11)
                    
        self.assertTrue(np.argmax(vrot)>-1000)
        
        
        


if __name__ == '__main__':
    unittest.main()            
            