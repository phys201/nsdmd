import numpy as np
import nestle
import math

from scipy.integrate import quad
from . import io


#  NFW model  ---------------------------------------------------------------------------------------------------------------
def model_NFW(theta, x):
    
    
    
    """
    define the model 
    
    """
    
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
def loglike_NFW(theta,data):
    
    
    """
    data: data_x,data_xerr,data_y,data_yerr
    
    
    """
    data_x, data_xerr, data_y, data_yerr = data


    
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


def prior_transform_NFW( theta,priorRange):
    
    
    """
    a:  theta[0] in the range of [0,a]  (10 the value used )     
    b:  theta[1] in the range of [0,b]  (500 the value used )
    theta: para 
    
    """
    a,b = priorRange[0],priorRange[1]
    
    return  np.array([a*theta[0],b*theta[1]])


# end of NFW model ----------------------------------------------------------------------------



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
def loglike_ISO_ref1(theta,data):
    
    data_x_ref1, data_xerr_ref1, data_y_ref1, data_yerr_ref1 = data


    
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


# It is a transformation from a space where variables are independently and uniformly distributed between 0 and 1 to the parameter space of interest. 
# 

def prior_transform_ISO(theta,priorRange):

    """
    a:  theta[0] in the range of [0,a]  (10 the value used )     
    b:  theta[1] in the range of [0,b]  (500 the value used )
    theta: para 
    
    """
    
    a,b = priorRange[0],priorRange[1]    
    
    
    # theta[0] in the range of [0,10] and theta[1] in the range of [0,300]
    return  np.array([a, b]) * theta


# end of ISO model ------------------------------------------------------------------------------







# Einasto model ------------------------------------------------------------------------------------

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
def loglike_Einasto_ref1(theta,data):
    
    
    data_x_ref1, data_xerr_ref1, data_y_ref1, data_yerr_ref1 = data


    
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


# It is a transformation from a space where variables are independently and uniformly distributed between 0 and 1 to the parameter space of interest. 
# 

def prior_transform_Einasto(theta,priorRange):
    
    """
    a:  theta[0] in the range of [0,a]  (10 the value used )     
    b:  theta[1] in the range of [0,b]  (10 the value used )
    c:  theta[2] in the range of [0,c]  (500 the value used )
    theta: para 
    
    """
    a,b,c = priorRange[0],priorRange[1],priorRange[2]       
    
    
    # theta[0] and theta[1] in the range of [0,10] and theta[1] in the range of [0,500]
    return  np.array([a, b, c]) * theta







# end of Einasto model ------------------------------------------------------------------------------------
































def sample (loglike_model, prior_transform_model, datafile,priorRange):
    
    """
    this function calls the loglikihood calculation function and the prior calculation function AND calculates the nestle results 
    
    
    loglike_model :function returns loglikelihood interms of parameters 
    
    prior_transform_model: function  returns prior interms of parameters 
    
    
    prior range : an arrage which specifies the limits of prior for different parameters eg:  prior range = [rangeForTheta[0],rangeForTheta[1],...]
    
    """
    
    
    data_x,data_xerr,data_y,data_yerr = io.load_data(datafile)
    
    #n: number of parameters, len(priorRange)
    n=len(priorRange) 





    def new_loglike_model(theta):
        return loglike_model(theta, (data_x,data_xerr,data_y,data_yerr))
        
    def new_prior_transform_model(theta):
        return prior_transform_model(theta,priorRange)
    
    result = nestle.sample(new_loglike_model, new_prior_transform_model, n)
    
    
    print ('log evidence')
    print (result.logz)

    print ('numerical (sampling) error on logz')
    print (result.logzerr)   
       
    print ('array of sample parameters')
    print (result.samples)  
       
    print ('array of weights associated with each sample')
    print (result.weights)
    
    
    
    
    
    
    return result 






