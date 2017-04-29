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
    vrot = theta[1]*np.sqrt(mass/x) 
    
    return vrot
    
    
    
    
    
    # Define a likelihood function
def loglike_NFW(theta):
    
    # Calculate the mass between 0 and data_x by integrating the NFW distribution.
    a = theta[0]
    mass = 4.*np.pi*(a**3)*(np.log((a+data_x)/a-data_x/(a+data_x)))
        
    # Calculate the rotation velocity.
    vrot = theta[1]*np.sqrt(mass/data_x) 
    
    # The y variable is the rotational velocity.
    y = vrot

    # Calculate chisq
    chisq= np.sum(((data_y - y) / data_yerr)**2)
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







# It is a transformation from a space where variables are independently and uniformly distributed between 0 and 1 to the parameter space of interest. 
# 

def prior_transform_NFW(theta):
    
    # theta[0] in the range of [0,10] and theta[1] in the range of [0,300]
    return  np.array([20, 500]) * theta

# Define the model we use to describe the data.

def model_ISO(theta, x):
    
    # Calculate the mass between 0 and data_x by integrating the NFW distribution.
    #
    # Note: the mass defined here does not include the normalization constant rho0 (kg/kpc^3).
    # The units of a are kpc.
    # The units of the "mass" calculated here are thus kpc^3.
    a = theta[0]
    mass = 4.*np.pi*(a**3)*(x/a - np.arctan(x/a))
        
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
    vrot = theta[1]*np.sqrt(mass/x) 
    
    return vrot


# Define a likelihood function
def loglike_ISO_ref1(theta):
    
    # Calculate the mass between 0 and data_x by integrating the NFW distribution.
    a = theta[0]
    mass = 4.*np.pi*(a**3)*(data_x_ref1/a - np.arctan(data_x_ref1/a))
        
    # Calculate the rotation velocity.
    vrot = theta[1]*np.sqrt(mass/data_x_ref1) 
        
    # The y variable is the rotational velocity.
    y = vrot
    
    # Calculate chisq
    chisq= np.sum(((data_y_ref1 - y) / data_yerr_ref1)**2)
    return -chisq / 2.

def loglike_ISO_ref2(theta):
    
    # Calculate the mass between 0 and data_x by integrating the NFW distribution.
    a = theta[0]
    mass = 4.*np.pi*(a**3)*(data_x_ref2/a - np.arctan(data_x_ref2/a))
        
    # Calculate the rotation velocity.
    vrot = theta[1]*np.sqrt(mass/data_x_ref2) 
        
    # The y variable is the rotational velocity.
    y = vrot
    
    # Calculate chisq
    chisq= np.sum(((data_y_ref2 - y) / data_yerr_ref2)**2)
    return -chisq / 2.


# It is a transformation from a space where variables are independently and uniformly distributed between 0 and 1 to the parameter space of interest. 
# 

def prior_transform_ISO(theta):
    
    # theta[0] in the range of [0,10] and theta[1] in the range of [0,300]
    return  np.array([5, 1000]) * theta













# Define the model we use to describe the data.
# The Einasto model.

#
# Start with defining the Einasto desity function.

def rho_Einasto(x,a,n):
    
    # Calculate the Einasto density.
    # The units of a are kpc.
    # n is dimensionless
    # The units of x are kpc.
    
    # Calculate the density.
    rho = np.exp(-2.*n*(x/a)**(1./n) - 1.)

    # Return the density
    return rho


def integrand_Einasto(x,a,n):
    
    # Calculate the integrand for mass integration.
    # This is x**2 * rho
    
    # Calculate the density.
    rho = rho_Einasto(x,a,n)

    # Return the density
    return rho*x**2


def mass_Einasto(theta, x):
    
    # Calculate the mass between 0 and data_x by integrating the Einasto distribution.
    #
    # Note: the mass defined here does not include the normalization constant rho0 (kg/kpc^3).
    # The units of a are kpc.
    # n is dimensionless
    # The units of the "mass" calculated here are thus kpc^3.
    a = theta[0]
    n = theta[1]
    
    mass, err = quad(integrand_Einasto,0,x,args=(a,n))
    mass = 4.*np.pi*mass
        
    return mass


def model_Einasto(theta, x):
    
    # Calculate the mass between 0 and data_x by integrating the Einasto distribution.
    #
    # Note: the mass defined here does not include the normalization constant rho0 (kg/kpc^3).
    # The units of a are kpc.
    # n is dimensionless
    # The units of the "mass" calculated here are thus kpc^3.
    a = theta[0]
    n = theta[1]
    
    mass, err = quad(integrand_Einasto,0,x,args=(a,n))
    mass = 4.*np.pi*mass
        
    # Calculate the rotation velocity: vrot = theta[1]*sqrt(mass/x)
    # The units of sqrt(mass/x) are kpc.
    # The rotation velocity is equal to sqrt(G*rho0)*sqrt((M/rho0)/x) where M is the mass enclosed, 
    # rho0 is the normalization constant of the mass distribution, and x is the distance at which we
    # calculate vrot.
    # When we determine theta[1], we determine sqrt(G*rho0).
    # The units of vrot are km/s.
    # The units of sqrt(mass/x) are kpc.
    # The units of theta[2] are thus (km/s)/kpc = (10^3 m)/s/(3.086E19 m) = 3.24E-17 1/s.
    # Since theta[1] = sqrt(G*rho0) we can now determine rho0: rho0 = theta[1]^2/G.
    # The units on the right-hand side are: (3.24E-17 1/s)^2/(m^3/(kg s^2)) = (3.24E-17)^2 kg/(m^3)
    # To convert from kg/m^3 to kg/kpc^3, we multiply by (3.086E19)^3
    # The normalization constant rho0 is thus (theta1[1]^2)/6.67E-11 * ((3.24E-17)^2 * (3.086E19)^3 kg/(kpc)^3 = 
    # (theta1[1]^2)*4.625E35 kg/(kpc)^3 = 2.312E5 Msun/(kpc)^3.
    vrot = theta[2]*np.sqrt(mass/x) 
    
    return vrot







# Define a likelihood function
def loglike_Einasto_ref1(theta):
    
    # Set chisq to zero.
    chisq = 0.

    # Note: we use this loop to determine chisq since mass_Einasto has a problem 
    # when data_x_ref1 is used as an argument.
    for index in range(len(data_x_ref1)):
        mass = mass_Einasto(theta,data_x_ref1[index])
        vrot = theta[2]*np.sqrt(mass/data_x_ref1[index]) 
        y = vrot
        chisq = chisq + ((data_y_ref1[index] - y) / data_yerr_ref1[index])**2
        
    return -chisq / 2.


# Define a likelihood function
def loglike_Einasto_ref2(theta):
    
    # Set chisq to zero.
    chisq = 0.

    # Note: we use this loop to determine chisq since mass_Einasto has a problem 
    # when data_x_ref2 is used as an argument.
    for index in range(len(data_x_ref2)):
        mass = mass_Einasto(theta,data_x_ref2[index])
        vrot = theta[2]*np.sqrt(mass/data_x_ref2[index]) 
        y = vrot
        chisq = chisq + ((data_y_ref2[index] - y) / data_yerr_ref2[index])**2
        
    return -chisq / 2.



# It is a transformation from a space where variables are independently and uniformly distributed between 0 and 1 to the parameter space of interest. 
# 

def prior_transform_Einasto(theta):
    
    # theta[0] and theta[1] in the range of [0,10] and theta[1] in the range of [0,500]
    return  np.array([10, 10, 500]) * theta



