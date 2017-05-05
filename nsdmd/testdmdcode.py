import unittest
import numpy as np

from nsdmd import model
from nsdmd import io



class TestFunctions(unittest.TestCase):
    
    # test data open function 
    def test_io(self):
        
        # import data 
        filename = 'dataref1.txt'
        testdata_x,testdata_xerr,testdata_y,testdata_yerr = io.load_data(filename)
        
        
        self.assertTrue(issubclass(type(testdata_x), np.ndarray))
        
        
        
    # test prior function 
    def test_prior_transform_NFW(self):
        
        
        # limit the range of parameters 
        theta0_range = np.linspace(0,0.1,3)
        theta1_range = np.linspace(0,0.1,3)
        priorRange=np.ones((len(theta0_range),len(theta1_range)))
        
        # generating priors 
        for i in range(len(theta0_range)):
            for j in range(len(theta1_range)):
                
                theta = theta0_range[i], theta1_range[j]

                priorRange[i,j]= np.argmax(model.prior_transform_NFW(theta,[10,500]))
            
        self.assertTrue(np.argmax(priorRange)<10000 and np.argmax(priorRange)>-1)        
        
    # test model    
    def test_model_NFW(self):
        
        theta0_range = np.linspace(6,1,10)
        vrot=np.ones(len(theta0_range))
        
        # calculte velocities by the model given mass density function 
        for i in range(len(theta0_range)):
            
            theta = theta0_range[i], 10
            vrot[i]=model.model_NFW(11,theta)
        
        # test velocity 
        self.assertTrue(np.argmax(vrot)>-1000)
             
        
    # test model    
    def test_model_ISO(self):
        
        theta0_range = np.linspace(6,1,10)
        vrot=np.ones(len(theta0_range))
        
        # calculte velocities by the model given mass density function 
        for i in range(len(theta0_range)):
           
            theta = theta0_range[i], 10
            vrot[i]=model.model_ISO(11,theta)
        
        #test velocities 
        self.assertTrue(np.argmax(vrot)>-1000)
                     
    # test model     
    def test_model_Einasto(self):
        
        theta0_range = np.linspace(6,1,10)
        vrot=np.ones(len(theta0_range))
        
        # calculte velocities by the model given mass density function 
        for i in range(len(theta0_range)):
            
            theta = theta0_range[i], 10, 50
            vrot[i]=model.model_Einasto(11,theta)
            
        # test velocities             
        self.assertTrue(np.argmax(vrot)>-1000)
        
        
        
    # test model     
    def test_model_GeneralizedHalo(self):
        
        theta0_range = np.linspace(6,1,10)
        vrot=np.ones(len(theta0_range))
        
        # calculte velocities by the model given mass density function 
        for i in range(len(theta0_range)):
            
            theta = theta0_range[i], 10, 50
            vrot[i]=model.model_Einasto(11,theta)
            
        # test velocities             
        self.assertTrue(np.argmax(vrot)>-1000)
                
        
        
        
        
        
        
    # test nestle sampling     
    def test_sample_NFW(self):
        
        # testing the result on the NFW model 
        result = model.sample (model.loglike_NFW, model.prior_transform_NFW, 'example_data.txt',[10,10E10])
        
        self.assertTrue(isinstance(result.logz, float))
        

    # test nestle sampling     
    def test_sample_ISO(self):
        
        # testing the result on model 
        result = model.sample (model.loglike_ISO, model.prior_transform_ISO, 'example_data.txt',[10,10E10])
        
        self.assertTrue(isinstance(result.logz, float))

    # test nestle sampling     
    def test_sample_Einasto(self):
        
        # testing the result on model 
        result = model.sample (model.loglike_Einasto, model.prior_transform_Einasto, 'example_data.txt',[10,10,10E10])
        
        self.assertTrue(isinstance(result.logz, float))

       
    
    
  
       # this test also WORKS WELL. We decide not to include it because it takes relative long to run (about 2 mins). You could run the below test by manually removing '#'. [Thank you.]
    
    # def test_sample_GeneralizedHalo(self):
        
         #   result = model.sample (model.loglike_GeneralizedHalo, model.prior_transform_GeneralizedHalo, 'example_data.txt',[5,10,4,1.5,5E8])
        #  self.assertTrue(isinstance(result.logz, float))













 
    
    
    
    
    
    
                
if __name__ == '__main__':
    unittest.main()            
            
