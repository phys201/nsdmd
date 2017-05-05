import numpy as np
import nestle
from . import io

from scipy.special import gammainc
from scipy.special import hyp2f1

# Define the halo models to be used
#
# For each model the following functions are defined:
#   rho_MODEL: returns the density at position x.
#   model_MODEL: returns the rotation velovity at position x.
#   loglike_MODEL: returns -chisq/2
#   prior_transform_MODEL: define the parameter space of interest for the variables being used.


###############################################################################################
# Model 1: NFW model


# Density profile.
def rho_NFW(x, theta):
    
    # Calculate the density of the NFW profile at position x.
    #
    # Note: the mass defined here does not include the normalization constant rho0 (kg/kpc^3).
    # The units of a are kpc.
    # The units of rho is Msun/(kpc)^3.
    #
    a = theta[0]
    rho0 = theta[1]

    # Calculate the density
    density = rho0/((x/a)*(1 + x/a)**2)
    
    return density



# Enclosed mass.
def mass_NFW(x, theta):
    
    # Calculate the mass enclosed within radius x.
    #
    # The units of a are kpc.
    # The units of rho is Msun/(kpc)^3.
    a = theta[0]
    rho0 = theta[1]

    # Calculate the enclosed mass.
    # The enclosed mass is in units of solar masses.
    mass = 4.*np.pi*(a**3)*rho0*(np.log((a+x)/a)-x/(a+x))
        
    return mass



# Rotational velocity.
def model_NFW(x, theta):
    
    # Calculate the rotation velovity at position x.

    # Define required constants.
    G = 6.67E-11    # Gravitational constant.  Units: m^3/kg/s^2.
    kpc = 3.086E19  # Conversion factor from kpc to m.
    Msun = 2.0E30   # Conversion factor from solar mass to kg.
    
    # Calculate the enclosed mass.
    # The enclosed mass is in units of solar masses.
    mass = mass_NFW(x, theta)
        
    # Calculate the rotation velocity: vrot = sqrt(G*mass/x)
    vrot = np.sqrt(G*mass*Msun/(x*kpc))

    # Convert velocity from m/s to km/s.
    vrot = vrot/1000.
    
    return vrot



# Likelihood function.
def loglike_NFW(theta,data):

    data_x, data_xerr, data_y, data_yerr = data


    # Calculate the rotation velocity.
    vrot = model_NFW(data_x, theta)
        
    # Calculate chisq
    chisq = np.sum(((data_y - vrot) / data_yerr)**2)

    # Return -chisq/2,
    return -chisq / 2.



# Transform the space where variables are independently and uniformly distributed between 0 and 1
# to the parameter space of interest. 
def prior_transform_NFW(theta,priorRange):
    
    
    a,b = priorRange[0],priorRange[1]


    
    # theta[0] in the range of [0,10] and theta[1] in the range of [0,10E10]
    return  np.array([a*theta[0],b*theta[1]])





###############################################################################################
# Model 2: Isothermal model


# Density profile.
def rho_ISO(x, theta):
    
    # Calculate the density of the NFW profile at position x.
    #
    # Note: the mass defined here does not include the normalization constant rho0 (kg/kpc^3).
    # The units of a are kpc.
    # The units of rho is Msun/(kpc)^3.
    #
    a = theta[0]
    rho0 = theta[1]

    # Calculate the density
    density = rho0/(1 + (x/a)**2)
    
    return density



# Enclosed mass.
def mass_ISO(x, theta):
    
    # Calculate the mass enclosed within radius x.
    #
    # The units of a are kpc.
    # The units of rho is Msun/(kpc)^3.
    a = theta[0]
    rho0 = theta[1]

    # The enclosed mass is in units of solar masses.
    mass = 4.*np.pi*(a**3)*rho0*(x/a - np.arctan(x/a))
            
    return mass



# Rotational velocity.
def model_ISO(x, theta):
    
    # Calculate the rotation velovity as position x.

    # Define required constants.
    G = 6.67E-11    # Gravitational constant.  Units: m^3/kg/s^2.
    kpc = 3.086E19  # Conversion factor from kpc to m.
    Msun = 2.0E30   # Conversion factor from solar mass to kg.

    # The enclosed mass is in units of solar masses.
    mass = mass_ISO(x, theta)
        
    # Calculate the rotation velocity: vrot = sqrt(G*mass/x)
    vrot = np.sqrt(G*mass*Msun/(x*kpc))

    # Convert velocity from m/s to km/s.
    vrot = vrot/1000.
    
    return vrot



# Likelihood function.
def loglike_ISO(theta,data):
    
    
    data_x, data_xerr, data_y, data_yerr = data


    # Calculate the rotation velocity.
    vrot = model_ISO(data_x, theta)
        
    # Calculate chisq
    chisq = np.sum(((data_y - vrot) / data_yerr)**2)

    # Return -chisq/2,
    return -chisq / 2.



# Transform the space where variables are independently and uniformly distributed between 0 and 1
# to the parameter space of interest. 
def prior_transform_ISO(theta,priorRange):
    
    # theta[0] in the range of [0,10] and theta[1] in the range of [0,10E10]
    a,b = priorRange[0],priorRange[1]


    
    return   np.array([a*theta[0],b*theta[1]])




###############################################################################################
# Model 3: Einasto model


# Define the model we use to describe the data.
# The Einasto model.

#
# Start with defining the Einasto desity function.
def rho_Einasto(x,theta):
    
    # Calculate the Einasto density.
    # The units of a are kpc.
    # n is dimensionless
    # The units of rho are Msun/(kpc)^3
    # The units of x are kpc.
    a = theta[0]
    n = theta[1]
    rho0 = theta[2]
    
    # Calculate the density.
    # The density in in units of GMsun/(kpc)^3.
    rho = rho0*np.exp(-2.*n*((x/a)**(1./n) - 1.))

    # Return the density
    return rho


# Enclosed mass.
def mass_Einasto(x, theta):
    
    # Calculate the mass enclosed within radius x.
    #
    # The units of a are kpc.
    # n is dimensionless
    # The units of rho are GMsun/(kpc)^3
    # The units of x are kpc.
    a = theta[0]
    n = theta[1]
    rho0 = theta[2]

    # Find the enclosed mass.
    mass = 4.*np.pi*rho0*(a**3)*np.exp(2.0*n)*((2.0*n)**(-3.0*n))*gammainc((3.0*n),(x/a))
        
    return mass


# Rotational velocity.
def model_Einasto(x, theta):
    
    # Calculate the rotational velovity at position x.

    # Define required constants.
    G = 6.67E-11    # Gravitational constant.  Units: m^3/kg/s^2.
    kpc = 3.086E19  # Conversion factor from kpc to m.
    Msun = 2.0E30   # Conversion factor from solar mass to kg.

    # Determine the enclosed mass.
    # Note: the enclosed mass is in units of solar masses.
    mass = mass_Einasto(x, theta)
        
    # Calculate the rotation velocity: vrot = sqrt(G*mass/x)
    vrot = np.sqrt(G*mass*Msun/(x*kpc))

    # Convert velocity from m/s to km/s.
    vrot = vrot/1000.
    
    return vrot



# Define a likelihood function
def loglike_Einasto(theta,data):
    
    data_x, data_xerr, data_y, data_yerr = data


    # Calculate the rotation velocity.
    vrot = model_Einasto(data_x, theta)
        
    # Calculate chisq
    chisq = np.sum(((data_y - vrot) / data_yerr)**2)
        
    return -chisq / 2.



# It is a transformation from a space where variables are independently and uniformly distributed between 0 and 1 to the parameter space of interest. 
# 
def prior_transform_Einasto(theta,priorRange):
    
    a,b,c = priorRange[0],priorRange[1],priorRange[2]

    # theta[0] and theta[1] in the range of [0,10] and theta[1] in the range of [0,10E10]
    return   np.array([a*theta[0],b*theta[1],c*theta[2]])



###############################################################################################
# Model 4: GEneralized halo model

#
# Start with defining the GeneralizedHalo desity function.
def rho_GeneralizedHalo(x,theta):
    
    # Calculate the GeneralizedHalo density.
    # The units of a are kpc.
    # alpha, beta, gamma are dimensionless
    # The units of x are kpc.
    a = theta[0]
    alpha = theta[1]
    beta = theta[2]
    gamma = theta[3]
    rho0 = theta[4]
    
    # Calculate the density.
    rho = rho0/(((x/a)**gamma)*(1. + (x/a)**alpha))**((beta - gamma)/alpha)
    
    # Return the density
    return rho



def mass_GeneralizedHalo(x, theta):
    
    # Calculate the mass between 0 and data_x by integrating the GeneralizedHalo distribution.
    #
    # The units of a are kpc.
    # alpha, beta, gamma are dimensionless
    # The units of x are kpc.
    a = theta[0]
    alpha = theta[1]
    beta = theta[2]
    gamma = theta[3]
    rho0 = theta[4]

    # Calculate the enclosed mass.
    y = x/a

    ai = gamma - 2.0
    bi = alpha
    ci = (beta - gamma)/alpha

    mass = -4.*np.pi*(a**3)*rho0*(y**(1.0 - ai))*(hyp2f1((1.0 - ai)/bi, ci, (1.0 + (1.0 - ai)/bi), -(y**bi)))/(-1.0 + ai)

    return mass



# Rotational velocity.
def model_GeneralizedHalo(x, theta):
    
    # Calculate the rotational velovity at position x.

    # Define required constants.
    G = 6.67E-11    # Gravitational constant.  Units: m^3/kg/s^2.
    kpc = 3.086E19  # Conversion factor from kpc to m.
    Msun = 2.0E30   # Conversion factor from solar mass to kg.

    # Calculate the enclosed mass.    
    mass = mass_GeneralizedHalo(x, theta)
        
    # Calculate the rotation velocity
    vrot = np.sqrt(G*mass*Msun/(x*kpc)) 

    # Convert velocity from m/s to km/s.
    vrot = vrot/1000.
    
    return vrot



# Define a likelihood function
def loglike_GeneralizedHalo(theta,data):
    
    data_x, data_xerr, data_y, data_yerr = data

    
    # Calculate the rotation velocity.
    vrot = model_GeneralizedHalo(data_x, theta)
        
    # Calculate chisq
    chisq = np.sum(((data_y - vrot) / data_yerr)**2)
    
    return -chisq / 2.



# It is a transformation from a space where variables are independently and uniformly distributed between 0 and 1 to the parameter space of interest. 
def prior_transform_GeneralizedHalo(theta,priorRange):
    
    
    a,b,c,d,e = priorRange[0],priorRange[1],priorRange[2],priorRange[3],priorRange[4]


    
    # Define mean and width of prior distributions.
    mu = np.array([a,b,c,d,e])             # ([5.,10.,4.,1.5,5E8])
    
    sigma = np.array([a,b,c/4,d,e])                     #([5.,10.,1.0,1.5,5E8])
    return mu + (2.*sigma*theta - sigma)











def sample (loglike_model, prior_transform_model, datafile,priorRange):
    
    """
The function runs nested sampling. The function can be run on 4 different models. The user just needs to specify the model's loglikelihood and prior. 
----------    
The following input parameters are required:
    
loglike_model: function calculates likelihood. The name of the loglike_model needs to be specified. 
prior_transform_model: function calculates the prior. The name of the prior transformation function needs to be specified. 
datafile: datafile with format has been discussed in above. 
priorRange: an array which specifies the limits of unifrom prior for different parameters eg: priorRange =[rangeForTheta[0],rangeForTheta[1],...]
  
----------
Here are some example commands for running the fuction for 4 different models.    

    model.sample (model.loglike_NFW, model.prior_transform_NFW, 'DMdataref1.txt',[10,10E10])
    model.sample (model.loglike_ISO, model.prior_transform_ISO, 'DMdataref1.txt',[10,10E10])
    model.sample (model.loglike_Einasto, model.prior_transform_Einasto, 'DMdataref1.txt',[10,10,10E10])
    model.sample (model.loglike_GeneralizedHalo, model.prior_transform_GeneralizedHalo, 'DMdataref1.txt',[5.,10.,4.,1.5,5E8])
-----------
As shown above, different models take in different loglike_model, prior_transform_model and priorRange. 

For NFW model: 
loglike_model: it's named as 'model.loglike_NFW'
prior_transform_model: it's named as 'model.prior_transform_NFW'
datafile: any datafile that has the correct 4 columns format. 
priorRange: an arrange of 2 elements. Recommand to use [10,10e10]


For ISO model: 
loglike_model: it's named as 'model.loglike_ISO'
prior_transform_model: it's named as 'model.prior_transform_ISO'
datafile: any datafile that has the correct 4 columns format. 
priorRange: an arrange of 2 elements. Recommand to use [10,10e10]

For Einasto model: 
loglike_model: it's named as 'model.loglike_Einasto'
prior_transform_model: it's named as 'model.prior_transform_Einasto'
datafile: any datafile that has the correct 4 columns format. 
priorRange: an arrange of 3 elements. Recommand to use [10,10,10e10]

For GeneralizedHalo model: 
loglike_model: it's named as 'model.loglike_GeneralizedHalo'
prior_transform_model: it's named as 'model.prior_transform_GeneralizedHalo'
datafile: any datafile that has the correct 4 columns format. 
priorRange: an arrange of 5 elements. Recommand to use [5.,10.,4.,1.5,5e8]

 
    """
    
    data_file = io.get_data_file_path(datafile)
    data_x,data_xerr,data_y,data_yerr = io.load_data(data_file)
    
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













