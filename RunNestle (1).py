import numpy as np

import matplotlib.pyplot as plt

import corner

import nestle

from nsdmd import HaloModels

# Define a function to run nested sampling.

def run_nested_sampling(loglike_Name,prior_Name,nrOfPar):
    # The following input parameters are required:
    #
    # loglike_Name: name of the likelihood function.
    # prior_Name: name of the prior transformation function.
    # nrOfPar: number of parameters.
    
    # Run nested samping.
    result_Nestle = nestle.sample(loglike_Name, prior_Name, nrOfPar)

    # Print a brief summary of the results    
    print ('Number of iterations = {0:d}'.format(result_Nestle.niter))
    
    print ('log evidence: = {0:5.2f}'.format(result_Nestle.logz))
            
    # Print the results of the sampling.
    if loglike_Name.__name__ == "loglike_NFW":
        results_nested_sampling(result_Nestle, 1)

    if loglike_Name.__name__ == "loglike_ISO":
        results_nested_sampling(result_Nestle, 2)

    if loglike_Name.__name__ == "loglike_Einasto":
        results_nested_sampling(result_Nestle, 3)

    if loglike_Name.__name__ == "loglike_GeneralizedHalo":
        results_nested_sampling(result_Nestle, 4)

    return result_Nestle


# Define a function to summarize the results of nested sampling

def results_nested_sampling(result_Nestle, modelNr):
    # The following input parameters are required:
    #
    # results_Nestle: the results of the nested sampling
    # modelNr: the number of the model being analyzed.
    #
    # Note: modelNr = 1: NWF model
    #       modelNR = 2:
    #       modelNR = 3:
    #       modelNr = 4:

    
    # Analyze the results of the sampling.
    #
    # Get the results of the fit and the uncertainties in the fit parameters.
    p_fit, cov_fit = nestle.mean_and_cov(result_Nestle.samples, result_Nestle.weights)


    # Start with printing the results of the fit.
    # Note: different models have different number of fit parameters and we have
    # to take this into consideration.
    if modelNr == 1:
        # Model number 1: NFW
        # Fit parameters: a and rho0

        # Print information on quality of the fit
        print ('Number of degrees of freedom = {0:d}'.format(data_x.size - 2))
        print ('Chi squared per dof = {0:5.2f}'.format(-2.0*HaloModels.loglike_NFW(p_fit)/(data_x.size - 2.)))
        print("")
        
        # Print fit parameters.
        print("Results based on fits:")
        print("Core radius a = {0:5.2f} +/- {1:5.2f} kpc".format(p_fit[0], np.sqrt(cov_fit[0, 0])))
        print("Normalization factor = {0:5.2e} +/- {1:5.2e} Msun/kpc^3".format(p_fit[1], np.sqrt(cov_fit[1, 1])))
        print("")

        # Calculate the local halo density (we assume the sun is located at 8 kpc).
        print("Halo density in our solor system = {0:5.2e} Msun/kpc^3.".format(HaloModels.rho_NFW(8, p_fit)))

        # Note: 1 Msun/kpc^3 = 3.817E-2 (GeV/c^2)/m^3 = 3.817E-5 (GeV/c^2)/(dm^3)
        # 1 dm^3 = 1 liter.
        # 3 WIMPS/liter would be 300 GeV/c^2/liter
        print("Halo density in our solor system = {0:5.2e} (GeV/c^2)/liter.".format(3.817E-5*HaloModels.rho_NFW(8,p_fit)))
        print("")
        
        # Calculate the total enclosed dark-matter mass within 200 and 1000 kpc.
        print("Total dark matter mass enclosed within 200 kpc = {0:5.2e} Msun".format(HaloModels.mass_NFW(200, p_fit)))
        print("Total dark matter mass enclosed within 1000 kpc = {0:5.2e} Msun".format(HaloModels.mass_NFW(1000, p_fit)))


        plt.figure()
        plt.errorbar(data_x,data_y,yerr=data_yerr,fmt='*')
        plt.xlabel("r (kpc)")
        plt.ylabel('V (km/s)')
        plt.title("Results of using the NFW model to fit the DM rotational velocity distribution")
        xplot = [5+5*i for i in range(40)]
        yplot = [HaloModels.model_NFW(xplot[i], p_fit) for i in range(40)]
        plt.plot(xplot,yplot)
        plt.show()

        fig = corner.corner(result_Nestle.samples, weights=result_Nestle.weights, labels=['a', 'rho0'],
                            range=[0.99999, 0.99999], bins=30)
        plt.show()


    if modelNr == 2:
        # Model number 1: ISO
        # Fit parameters: a and rho0

        # Print information on quality of the fit
        print ('Number of degrees of freedom = {0:d}'.format(data_x.size - 2))
        print ('Chi squared per dof = {0:5.2f}'.format(-2.0*HaloModels.loglike_ISO(p_fit)/(data_x.size - 2.)))
        print("")
        
        # Print fit parameters.
        print("Results based on fits:")
        print("Core radius a = {0:5.2f} +/- {1:5.2f} kpc".format(p_fit[0], np.sqrt(cov_fit[0, 0])))
        print("Normalization factor = {0:5.2e} +/- {1:5.2e} Msun/kpc^3".format(p_fit[1], np.sqrt(cov_fit[1, 1])))
        print("")
        
        # Calculate the local halo density (we assume the sun is located at 8 kpc).
        print("Halo density in our solor system = {0:5.2e} Msun/kpc^3.".format(HaloModels.rho_ISO(8, p_fit)))

        # Note: 1 Msun/kpc^3 = 3.817E-2 (GeV/c^2)/m^3 = 3.817E-5 (GeV/c^2)/(dm^3)
        # 1 dm^3 = 1 liter.
        # 3 WIMPS/liter would be 300 GeV/c^2/liter
        print("Halo density in our solor system = {0:5.2e} (GeV/c^2)/liter.".format(3.817E-5*HaloModels.rho_ISO(8,p_fit)))
        print("")
        
        # Calculate the total enclosed dark-matter mass within 200 and 1000 kpc.
        print("Total dark matter mass enclosed within 200 kpc = {0:5.2e} Msun".format(HaloModels.mass_ISO(200, p_fit)))
        print("Total dark matter mass enclosed within 1000 kpc = {0:5.2e} Msun".format(HaloModels.mass_ISO(1000, p_fit)))


        plt.figure()
        plt.errorbar(data_x,data_y,yerr=data_yerr,fmt='*')
        plt.xlabel("r (kpc)")
        plt.ylabel('V (km/s)')
        plt.title("Results of using the ISO model to fit the DM rotational velocity distribution")
        xplot = [5+5*i for i in range(40)]
        yplot = [HaloModels.model_ISO(xplot[i], p_fit) for i in range(40)]
        plt.plot(xplot,yplot)
        plt.show()

        fig = corner.corner(result_Nestle.samples, weights=result_Nestle.weights, labels=['a', 'rho0'],
                            range=[0.99999, 0.99999], bins=30)
        plt.show()


    if modelNr == 3:
        # Model number 1: Einasto
        # Fit parameters: a, n, and rho0

         # Print information on quality of the fit
        print ('Number of degrees of freedom = {0:d}'.format(data_x.size - 3))
        print ('Chi squared per dof = {0:5.2f}'.format(-2.0*HaloModels.loglike_Einasto(p_fit)/(data_x.size - 3.)))
        print("")
        
        # Print fit parameters.
        print("Results based on fits:")
        print("Core radius a = {0:5.2f} +/- {1:5.2f} kpc".format(p_fit[0], np.sqrt(cov_fit[0, 0])))
        print("Einasto index n = {0:5.2f} +/- {1:5.2f}".format(p_fit[1], np.sqrt(cov_fit[1, 1])))
        print("Normalization factor = {0:5.2e} +/- {1:5.2e} Msun/kpc^3".format(p_fit[2], np.sqrt(cov_fit[2, 2])))
        print("")
        
        # Calculate the local halo density (we assume the sun is located at 8 kpc).
        print("Halo density in our solor system = {0:5.2e} Msun/kpc^3.".format(HaloModels.rho_Einasto(8, p_fit)))

        # Note: 1 Msun/kpc^3 = 3.817E-2 (GeV/c^2)/m^3 = 3.817E-5 (GeV/c^2)/(dm^3)
        # 1 dm^3 = 1 liter.
        # 3 WIMPS/liter would be 300 GeV/c^2/liter
        print("Halo density in our solor system = {0:5.2e} (GeV/c^2)/liter.".format(3.817E-5*HaloModels.rho_Einasto(8,p_fit)))
        print("")
        
        # Calculate the total enclosed dark-matter mass within 200 and 1000 kpc.
        print("Total dark matter mass enclosed within 200 kpc = {0:5.2e} Msun".format(HaloModels.mass_Einasto(200, p_fit)))
        print("Total dark matter mass enclosed within 1000 kpc = {0:5.2e} Msun".format(HaloModels.mass_Einasto(1000, p_fit)))

        plt.figure()
        plt.errorbar(data_x,data_y,yerr=data_yerr,fmt='*')
        plt.xlabel("r (kpc)")
        plt.ylabel('V (km/s)')
        plt.title("Results of using the Einasto model to fit the DM rotational velocity distribution")
        xplot = [5+5*i for i in range(40)]
        yplot = [HaloModels.model_Einasto(xplot[i], p_fit) for i in range(40)]
        plt.plot(xplot,yplot)
        plt.show()

        fig = corner.corner(result_Nestle.samples, weights=result_Nestle.weights, labels=['a', 'n', 'rho0'],
                            range=[0.99999, 0.99999, 0.99999], bins=30)
        plt.show()


    if modelNr == 4:
        # Model number 1: Generalized Halo
        # Fit parameters: a, alpha, beta, gamma, and rho0

        # Print information on quality of the fit
        print ('Number of degrees of freedom = {0:d}'.format(data_x.size - 5))
        print ('Chi squared per dof = {0:5.2f}'.format(-2.0*HaloModels.loglike_GeneralizedHalo(p_fit)/(data_x.size - 5.)))
        print("")
        
        # Print fit parameters.
        print("Results based on fits:")
        print("Core radius a = {0:5.2f} +/- {1:5.2f} kpc".format(p_fit[0], np.sqrt(cov_fit[0, 0])))
        print("alpha = {0:5.2f} +/- {1:5.2f}".format(p_fit[1], np.sqrt(cov_fit[1, 1])))
        print("beta = {0:5.2f} +/- {1:5.2f}".format(p_fit[2], np.sqrt(cov_fit[2, 2])))
        print("gamma = {0:5.2f} +/- {1:5.2f}".format(p_fit[3], np.sqrt(cov_fit[3, 3])))
        print("Normalization factor = {0:5.2e} +/- {1:5.2e} Msun/kpc^3".format(p_fit[4], np.sqrt(cov_fit[4, 4])))
        print("")
        
        # Calculate the local halo density (we assume the sun is located at 8 kpc).
        print("Halo density in our solor system = {0:5.2e} Msun/kpc^3.".format(HaloModels.rho_GeneralizedHalo(8, p_fit)))

        # Note: 1 Msun/kpc^3 = 3.817E-2 (GeV/c^2)/m^3 = 3.817E-5 (GeV/c^2)/(dm^3)
        # 1 dm^3 = 1 liter.
        # 3 WIMPS/liter would be 300 GeV/c^2/liter
        print("Halo density in our solor system = {0:5.2e} (GeV/c^2)/liter.".format(3.817E-5*HaloModels.rho_GeneralizedHalo(8,p_fit)))
        print()
        
        # Calculate the total enclosed dark-matter mass within 200 and 1000 kpc.
        print("Total dark matter mass enclosed within 200 kpc = {0:5.2e} Msun".format(HaloModels.mass_GeneralizedHalo(200, p_fit)))
        print("Total dark matter mass enclosed within 1000 kpc = {0:5.2e} Msun".format(HaloModels.mass_GeneralizedHalo(1000, p_fit)))

        plt.figure()
        plt.errorbar(data_x,data_y,yerr=data_yerr,fmt='*')
        plt.xlabel("r (kpc)")
        plt.ylabel('V (km/s)')
        plt.title("Results of using the Einasto model to fit the DM rotational velocity distribution")
        xplot = [5+5*i for i in range(40)]
        yplot = [HaloModels.model_GeneralizedHalo(xplot[i], p_fit) for i in range(40)]
        plt.plot(xplot,yplot)
        plt.show()

        fig = corner.corner(result_Nestle.samples, weights=result_Nestle.weights, labels=['a', 'alpha', 'beta', 'gamma', 'rho0'],
                            range=[0.99999, 0.99999, 0.99999, 0.99999, 0.99999], bins=30)
        plt.show()



    return



