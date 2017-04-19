import unittest


from nsdmd import model
from nsdmd import dmdio

import os



class TestFunctions(unittest.TestCase):
    def test_io(self):
        filename = 'example_data.txt'
        testdata = dmdio.load_data(filename)
        self.assertTrue(issubclass(type(testdata), np.ndarray))
        
        
        
    def test_model_NFW(self):
        
        theta_range = np.linspace(0,0.1,3)
        
        for i in rannge(len(theta_range)):
            vrot[i]=model.model_NFW(theta,x)
        self.assertTrue(np.argmax(vrot)>0 and np.argmax(vrot) <1000)
        
        
        
        
    def test_loglike_NFW(self):
                
        theta0_range = np.linspace(0,0.1,3)
        theta1_range = np.linspace(0,0.1,3)


        for i in rannge(len(theta0_range)):
            for j in rannge(len(theta1_range)):
                
                theta = theta0, theta1 
                minusChiSq = model.prior_loglike_NFW(theta)
                
        self.assertTrue(np.argmax(minusChiSq)<0)
        
        

        
    def test_prior_transform_NFW(self):
        testdata = dmdio.load_data('example_data.txt')
        theta0_range = np.linspace(0,0.1,3)
        theta1_range = np.linspace(0,0.1,3)


        for i in rannge(len(theta0_range)):
            for j in rannge(len(theta1_range)):
                
                theta = theta0, theta1 

                priorRange= model.prior_transform_NFW(theta)
            
        self.assertTrue(np.argmax(priorRange)<1000 and np.argmax(priorRange)>1000 )


            
            



print("testcode works fine")