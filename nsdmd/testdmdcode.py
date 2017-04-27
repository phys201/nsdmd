import unittest
import os
import numpy as np
import nsdmd

class TestFunctions(unittest.TestCase):
    def test_io(self):
        '''
        Tests that the nsdmd.io functions are working and that the loaded data is a numpy array
        '''
        filename = 'dataref1.txt'
        totalpath = nsdmd.get_example_data_file_path(filename, data_dir='data')
        testdata_x,testdata_xerr,testdata_y,testdata_yerr = nsdmd.load_data(totalpath)
        self.assertTrue(issubclass(type(testdata_x), np.ndarray))
              
    def test_prior_transform_NFW(self):
        filename = 'dataref1.txt'
        totalpath = nsdmd.get_example_data_file_path(filename, data_dir='data')
        testdata = nsdmd.load_data(totalpath)
        theta0_range = np.linspace(0,0.1,3)
        theta1_range = np.linspace(0,0.1,3)
        priorRange=np.ones((len(theta0_range),len(theta1_range)))
        

        for i in range(len(theta0_range)):
            for j in range(len(theta1_range)):
                
                theta = theta0_range[i], theta1_range[j]

                priorRange[i,j]= np.argmax(nsdmd.prior_transform_NFW(theta))
            
        self.assertTrue(np.argmax(priorRange)<10000 and np.argmax(priorRange)>-1)
        
    def test_model_NFW(self):
        
        theta0_range = np.linspace(6,1,10)
        vrot=np.ones(len(theta0_range))
        
        for i in range(len(theta0_range)):
            
            theta = theta0_range[i], 10
            vrot[i]=nsdmd.model_NFW(theta,11)
                    
        self.assertTrue(np.argmax(vrot)>-1000)
        
if __name__ == '__main__':
    unittest.main()            
            
