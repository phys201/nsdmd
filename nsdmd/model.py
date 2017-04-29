import numpy as np
import nestle
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import corner

from scipy.integrate import quad


# Define model
def model_NFW(theta, x):
    
    # Calculate the mass between 0 and data_x by integrating the NFW distribution.
    #
    # Note: the mass defined here does not include the normalization constant rho0 (kg/kpc^3).
    # The units of a are kpc.
    # The units of the "mass" calculated here are thus kpc^3.
    a = theta[0]
    mass = 4.*np.pi*(a**3)*(np.log((a+x)/a)-x/(a+x))
        
    # Calculate the rotation velocity: vrot = theta[1]*sqrt(mass/x)
    # The units of sqrt(mass/x) are kpc.
    # The rotation velocity is equal to sqrt(G*rho0)*sqrt((M/rho0)/x) where M is the mass enclosed, 
    # rho0 is the normalization constant of the mass distribution, and x is the distance at which we
    # calculate vrot.
    # When we determine theta[1], we determine sqrt(G*rho0).
    # The units of vrot are km/s.
    # The units of sqrt(mass/x) are kpc.
    # The units of theta[1] are thus (km/s)/kpc = (10^3 m)/s/(3.086E19 m) = 3.24E-17 1/s.
    # Since theta[1] = sqrt(G*rho0) we can now determine rho0: rho0 = theta[1]^2/G.
    # The units on the right-hand side are: (3.24E-17 1/s)^2/(m^3/(kg s^2)) = (3.24E-17)^2 kg/(m^3)
    # To convert from kg/m^3 to kg/kpc^3, we multiply by (3.086E19)^3
    # The normalization constant rho0 is thus (theta1[1]^2)/6.67E-11 * ((3.24E-17)^2 * (3.086E19)^3 kg/(kpc)^3 = 
    # (theta1[1]^2)*4.625E35 kg/(kpc)^3 = 2.312E5 Msun/(kpc)^3.
    return theta[1]*np.sqrt(mass/x) 
   
    
    
    
    
    
    # Define a likelihood function
def loglike_NFW(theta):
    
    # Calculate the mass between 0 and data_x by integrating the NFW distribution.
    a = theta[0]
    mass = 4.*np.pi*(a**3)*(np.log((a+data_x)/a-data_x/(a+data_x)))
        
    # Calculate the rotation velocity.
    vrot = theta[1]*np.sqrt(mass/data_x) 
    
    # The y variable is the rotational velocity.
   

    # Calculate chisq
    chisq= np.sum(((data_y - vrot) / data_yerr)**2)
    return -chisq / 2.



# It is a transformation from a space where variables are independently and uniformly distributed between 0 and 1 to the parameter space of interest. 
# 


def prior_transform_NFW(theta):
    
    # theta[0] in the range of [0,10] and theta[1] in the range of [0,300]
    # return  np.array([10, 300]) * theta
    return  np.array([10*theta[0],500*theta[1]])

def show_results(loglike_NFW, prior_transform_NFW, n): 
    # n is the dim of theta; for NFW model, n =2 
    result = nestle.sample(loglike_NFW, prior_transform_NFW, 2)

    print ('log evidence')
    print (result.logz)

    print ('numerical (sampling) error on logz')
    print (result.logzerr)   
       
    print ('array of sample parameters')
    print (result.samples)  
       
    print ('array of weights associated with each sample')
    print (result.weights)
    
    
    p, cov = nestle.mean_and_cov(result.samples, result.weights)

    print("core radius a = {0:5.2f} +/- {1:5.2f} kpc".format(p[0], np.sqrt(cov[0, 0])))
    print("normalization factor = {0:5.2f} +/- {1:5.2f}".format(p[1], np.sqrt(cov[1, 1])))
    print("Halo density normalization constant = {0:5.2e} +/- {1:5.2e} Msun/kpc^3".format(2.312E5*p[1], 2.312E5*np.sqrt(cov[1,
                                                                                                                            1])))

    # Note: in order to convert the model to units of Msun/kpc^3 we multiply its value by 2.312E5.
    # See comments in the model definition for details.
    print("Halo density in our solor system = {0:5.2e} Msun/kpc^3.".format(2.312E5*model_NFW(p, 8)))

    # Note: 1 Msun/kpc^3 = 3.817E-2 (GeV/c^2)/m^3 = 3.817E-5 (GeV/c^2)/(dm^3)
    # 1 dm^3 = 1 liter.
    # 3 WIMPS/liter would be 300 GeV/c^2/liter
    print("Halo density in our solor system = {0:5.2e} GeV/c^2/liter.".format(3.817E-5*2.312E5*model_NFW(p, 8)))

    plt.figure()
    plt.errorbar(data_x,data_y,data_yerr,data_xerr,fmt='*')
    plt.xlabel("r (kpc)")
    plt.ylabel('V (km/s)')
    plt.title("The measured rotational speed of the interstellar medium as a fucntion of the galactocentric radius")
    plt.plot([5.,200.],model_NFW(p, np.array([5.,200.])))
    plt.show()

    fig = corner.corner(result.samples, weights=result.weights, labels=['a', 'rho0'],
                        range=[0.99999, 0.99999], bins=30)
    plt.show()

    return 0 


def model_Isothermal(theta, x):
    a = theta[0]
    mass = 4.*np.pi*(a**3)*(np.log((a+x)/a)-x/(a+x))      
    return theta[1]*np.sqrt(mass/x)   
  

def loglike_IsothermalProfile_ref1 (theta):
    #theta[1] is the constant proportional factor 
    density_IsothermalProfile =   1/( 1+data_x_ref1 /theta[0] )**2 
    y=theta[1] * 1
    chisq = np.sum(((data_y_ref1 - y) / data_yerr_ref1)**2)
    return -chisq / 2.

def loglike_IsothermalProfile_ref2 (theta):
    #theta[1] is the constant proportional factor 
    density_IsothermalProfile =   1/( 1+data_x_ref2 /theta[0] )**2 
    y=theta[1] * 1
    chisq = np.sum(((data_y_ref2 - y) / data_yerr_ref2)**2)
    return -chisq / 2.

def prior_transform_Isothermal(theta):
    
    # theta[0] in the range of [0,10] and theta[1] in the range of [0,300]
    return  np.array([20, 300]) * theta
    #return  np.array([10*theta[0],500*theta[1]])
    












def model_Einasto(theta, x):
    a = theta[0]
    mass = 4.*np.pi*(a**3)*(np.log((a+x)/a)-x/(a+x))      
    vrot = theta[1]*np.sqrt(mass/x)   
    return vrot

def loglike_EinastoProfile_ref1(theta):
    #theta[2] is the constant proportional factor 
    density_EinastoProfile = np.exp( -  theta[0]*  data_x_ref1**theta[1] ) 
    y=theta[2] *1
    chisq = np.sum(((data_y_ref1 - y) / data_yerr_ref1)**2)
    return -chisq / 2.

def loglike_EinastoProfile_ref2(theta):
    #theta[2] is the constant proportional factor 
    density_EinastoProfile = np.exp( -  theta[0]*  data_x_ref2**theta[1] ) 
    y=theta[2] *1
    chisq = np.sum(((data_y_ref2 - y) / data_yerr_ref2)**2)
    return -chisq / 2.

def prior_transform_Einasto(theta):
    
    # theta[0] in the range of [0,10] and theta[1] in the range of [0,300]
    return  np.array([20, 300]) * theta
    #return  np.array([10*theta[0],500*theta[1]])
    
    
    
    
    
def model_GeneralizedDM(theta, x):
    a = theta[0]
    mass = 4.*np.pi*(a**3)*(np.log((a+x)/a)-x/(a+x))      
    vrot = theta[1]*np.sqrt(mass/x)   
    return vrot

def loglike_GeneralizedDMProfile_ref1(theta):
    #theta[4] is the constant proportional factor 
    density_GeneralizedDMProfile = 1/(    (data_x_ref1 /theta[0])**theta[1] *( 1+ (data_x_ref1 /theta[0])**theta[2] )**2 ) **((theta[3] - theta[1])/theta[2] )
    y=theta[4] * 1
    chisq = np.sum(((data_y_ref1 - y) / data_yerr_ref1)**2)
    return -chisq / 2.

def loglike_GeneralizedDMProfile_ref2(theta):
    #theta[4] is the constant proportional factor 
    density_GeneralizedDMProfile = 1/(    (data_x_ref2 /theta[0])**theta[1] *( 1+ (data_x_ref2 /theta[0])**theta[2] )**2 ) **((theta[3] - theta[1])/theta[2] )
    y=theta[4] * 1
    chisq = np.sum(((data_y_ref2 - y) / data_yerr_ref2)**2)
    return -chisq / 2.

def prior_transform_GeneralizedDM(theta):
    
    # theta[0] in the range of [0,10] and theta[1] in the range of [0,300]
    return  np.array([20, 300]) * theta
    #return  np.array([10*theta[0],500*theta[1]])

