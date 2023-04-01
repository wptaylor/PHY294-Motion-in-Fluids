#Parse files in all folders

import os
import numpy as np
import matplotlib.pyplot as plt


#Get all folders in this directory (excluding "pics")
def get_folders():
    folders = []
    for folder in os.listdir(os.getcwd()):
        if folder != 'Unedited' and os.path.isdir(folder):
            folders.append(folder)
    return folders

#Get all text files files in a subfolder of this directory
def get_files(folder):
    files = []
    for file in os.listdir(os.getcwd() + '/' + folder):
        if file.endswith('.txt'):
            files.append(file)
    return files


#Parse text file into a numpy array, using loadtxt. Skip first two rows.
def parse_file(file, folder):
    #Get file path
    path = folder + '/' + file
    #Parse file
    data = np.loadtxt(path, skiprows=2)
    #Return data
    return data, path


#Overlay multiple plots on the same graph. Include labels. Include vertical lines for the endpoints of the interval of greatest average.
def plot_data(data, labels):
        #Get number of plots
    n = len(data)
    #Loop through all plots
    for i in range(n):
        #Plot data
        plt.plot(data[i][:,0], data[i][:,1], label=labels[i])
    #Add legend
    plt.legend()
    #Show plot
    plt.show()
    #Save plot in pics folder
    plt.savefig('pics/' + labels[0] + '.png')
    #Close window
    plt.close()



#Given a text file of two columns, remove all rows with trailing zeros in the second column. Ignore first two rows.
def remove_trailing_zeros(file, folder):
    #Get file path
    path = folder + '/' + file
    #Parse file
    data = np.loadtxt(path, skiprows=2)
    #Get number of rows
    n = len(data)
    #Loop through all rows in reverse order
    for i in range(n-1, -1, -1):
        #If second column is zero, delete row
        if data[i,1] == 0:
            data = np.delete(data, i, 0)
    #Save file
    np.savetxt(path, data, fmt='%.3f', delimiter='\t', header='Time\tPosition', comments='')


#Compute the average slope of a linear numpy array.
def compute_average_slope(data):
    #Get number of rows
    n = len(data)
    #Compute average slope
    average_slope = (data[n-1,1] - data[0,1]) / (data[n-1,0] - data[0,0])
    #Return average slope
    return average_slope


#Computer a least-squares fit to a linear numpy array.
def compute_least_squares_fit(data):
    #Get number of rows
    n = len(data)
    #Compute least-squares fit
    x = data[:,0]
    y = data[:,1]
    A = np.vstack([x, np.ones(n)]).T
    m, c = np.linalg.lstsq(A, y)[0]
    #Return slope and intercept
    return m, c


#Compute the R-squared value of a linear numpy array against a least-squares fit.
def compute_r_squared(data):
    #Get number of rows
    n = len(data)
    #Compute least-squares fit
    x = data[:,0]
    y = data[:,1]
    A = np.vstack([x, np.ones(n)]).T
    m, c = np.linalg.lstsq(A, y)[0]
    #Compute R-squared
    y_fit = m * x + c
    SS_res = np.sum((y - y_fit)**2)
    SS_tot = np.sum((y - np.mean(y))**2)
    R_squared = 1 - (SS_res / SS_tot)
    #Return R-squared
    return R_squared


#Fit a quadratic function to a numpy array.
def fit_quadratic(data):
    #Get number of rows
    n = len(data)
    #Fit quadratic function
    x = data[:,0]
    y = data[:,1]
    A = np.vstack([x**2, x, np.ones(n)]).T
    a, b, c = np.linalg.lstsq(A, y)[0]
    #Return coefficients
    return a, b, c

#Fit a square root function to a numpy array.
def fit_square_root(data):
    #Get number of rows
    n = len(data)
    #Fit square root function
    x = data[:,0]
    y = data[:,1]
    A = np.vstack([np.sqrt(x), np.ones(n)]).T
    a, b = np.linalg.lstsq(A, y)[0]
    #Return coefficients
    return a, b


#Compute the reduced chi-squared value of a numpy array against fitted values. Include error bars.
def compute_reduced_chi_squared(data, data_error, fitting_data):
    #Get number of rows
    n = len(data)
    #Compute chi-squared
    chi_squared = 0
    for i in range(n):
        chi_squared += ((data[i,1] - fitting_data[i,1]) / data_error[i,1])**2
    #Compute reduced chi-squared
    reduced_chi_squared = chi_squared / (n - 3)
    #Return reduced chi-squared
    return reduced_chi_squared



#Main function
def main():

    #Remove trailing zeros from all files in all folders
    '''
    folders = get_folders()
    for folder in folders:
        files = get_files(folder)
        for file in files:
            remove_trailing_zeros(file, folder)
    '''
    
    '''
    #Plot all files in all folders
    #Get all folders
    folders = get_folders()
    #Loop through all folders
    for folder in folders:
        #Get all files
        files = get_files(folder)
        #Loop through all files
        data = []
        labels = []
        for file in files:
            #Parse file
            data.append(parse_file(file, folder)[0])
            #Add label
            labels.append(file[:-4])
        #Plot data
        plot_data(data, labels)
    '''
    

    #Compute least-squares fit of all files in all folders. Sort by value of R-squared. Print file name, slope, intercept, and R-squared.
    #Also print the minumum R-squared value and the corresponding file name.
    #Get all folders
    folders = get_folders()
    #Loop through all folders
    R_squared_min = 1
    folder_min = ''
    file_min = ''

    average_vterm = []
    standard_error_mean = []

    for folder in folders:
        #Get all files
        files = get_files(folder)
        #Loop through all files
        data = []
        labels = []
        for file in files:
            #Parse file
            data.append(parse_file(file, folder)[0])
            #Add label
            labels.append(file[:-4])
        #Compute least-squares fit
        m = []
        c = []
        R_squared = []
        for i in range(len(data)):
            m.append(compute_least_squares_fit(data[i])[0])
            c.append(compute_least_squares_fit(data[i])[1])
            R_squared.append(compute_r_squared(data[i]))
        #Sort by R-squared
        R_squared, m, c, labels = (list(t) for t in zip(*sorted(zip(R_squared, m, c, labels))))
        #Print results
        print(folder)
        for i in range(len(labels)):
            print(labels[i], m[i], c[i], R_squared[i])
        print('')
        #Find minimum R-squared
        if R_squared[0] < R_squared_min:
            R_squared_min = R_squared[0]
            folder_min = folder
            file_min = labels[0]

        average_vterm.append(np.mean(m))
        standard_error_mean.append(np.std(m)/np.sqrt(len(m)))

    print('Minimum R-squared: ' + str(R_squared_min) + ' in ' + folder_min + '/' + file_min + '.txt')

    

    D = 94
    D_uncertainty = 2

    d_uncertainty = 0.01

    Glycerine_means = average_vterm[:5]
    Glycerine_errors = [standard_error_mean[i] * 1.96 for i in range(5)]
    Glycerine_d = [1.53, 2.57, 3.13, 4.47, 6.15]

    Water_means = average_vterm[5:]
    Water_errors = [standard_error_mean[i] * 1.96 for i in range(5,10)]
    Water_d = [2.36, 3.12, 3.90, 4.72, 6.30]


    G_v_corrs = [Glycerine_means[i]/(1-2.104*(Glycerine_d[i]/D) + 2.089*(Glycerine_d[i]/D)**2) for i in range(len(Glycerine_means))]
    G_v_corrs_errors = [Glycerine_errors[i]/(1-2.104*(Glycerine_d[i]/D) + 2.089*(Glycerine_d[i]/D)**2) for i in range(len(Glycerine_means))]

    G_p = 1.26*(10**-3)/(100**3)
    G_nug = 9.34*(10**-4)/(100)
    
    G_reynolds = [100*G_p*Glycerine_means[i]*Glycerine_d[i]/G_nug for i in range(len(Glycerine_means))]
    print('Glycerine Reynolds numbers: ' + str(G_reynolds))

    #Fit a quadratic function to G_v_corrs vs Glycerine_d
    G_v_corrs_fit = fit_quadratic(np.column_stack((Glycerine_d, G_v_corrs)))

    #Determine the reduced chi-squared value using compute_reduced_chi_squared
    G_v_corrs_reduced_chi_squared = compute_reduced_chi_squared(np.column_stack((Glycerine_d, G_v_corrs)), np.column_stack((Glycerine_d, G_v_corrs_errors)), np.column_stack((Glycerine_d, G_v_corrs_fit[0]*np.array(Glycerine_d)**2 + G_v_corrs_fit[1]*np.array(Glycerine_d) + G_v_corrs_fit[2])))
    print('Reduced chi-squared: ' + str(G_v_corrs_reduced_chi_squared))

    #Plot this fit
    x = np.linspace(1, 6.5, 100)
    y = G_v_corrs_fit[0]*x**2 + G_v_corrs_fit[1]*x + G_v_corrs_fit[2]
    plt.plot(x, y, label='Quadratic Fit')

    #Plot G_v_corrs vs Glycerine_d
    plt.errorbar(Glycerine_d, G_v_corrs,  yerr=G_v_corrs_errors, fmt='o', label='Glycerine', capsize=4, markersize=4)
    plt.title('Teflon Beads in Glycerine, v vs. d')
    plt.xlabel('d (mm)')
    plt.ylabel('v (mm/s)')
    plt.show()

    #Plot the residuals
    plt.errorbar(Glycerine_d, G_v_corrs - (G_v_corrs_fit[0]*np.array(Glycerine_d)**2 + G_v_corrs_fit[1]*np.array(Glycerine_d) + G_v_corrs_fit[2]), yerr=G_v_corrs_errors, fmt='o', capsize=4, markersize=4)
    #Plot a horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Teflon Beads in Glycerine, Residuals')
    plt.xlabel('d (mm)')
    plt.ylabel('v (mm/s)')
    plt.show()

    #Do the same for water except use a square root fit
    W_v_corrs = [Water_means[i]/(1-2.104*(Water_d[i]/D) + 2.089*(Water_d[i]/D)**2) for i in range(len(Water_means))]
    W_v_corrs_errors = [Water_errors[i]/(1-2.104*(Water_d[i]/D) + 2.089*(Water_d[i]/D)**2) for i in range(len(Water_means))]
    W_v_corrs_fit = fit_square_root(np.column_stack((Water_d, W_v_corrs)))
    W_v_corrs_reduced_chi_squared = compute_reduced_chi_squared(np.column_stack((Water_d, W_v_corrs)), np.column_stack((Water_d, W_v_corrs_errors)), np.column_stack((Water_d, W_v_corrs_fit[0]*np.sqrt(np.array(Water_d)) + W_v_corrs_fit[1])))
    print('Reduced chi-squared: ' + str(W_v_corrs_reduced_chi_squared))
    x = np.linspace(2, 6.5, 100)
    y = W_v_corrs_fit[0]*np.sqrt(x) + W_v_corrs_fit[1]
    plt.plot(x, y, label='Square Root Fit')
    plt.errorbar(Water_d, W_v_corrs,  yerr=W_v_corrs_errors, fmt='o', label='Water', capsize=4, markersize=4)
    plt.title('Nylon Beads in Water, v vs. d')
    plt.xlabel('d (mm)')
    plt.ylabel('v (mm/s)')
    #plt.legend("Square Root Fit")
    plt.show()
    plt.errorbar(Water_d, W_v_corrs - (W_v_corrs_fit[0]*np.sqrt(np.array(Water_d)) + W_v_corrs_fit[1]), yerr=W_v_corrs_errors, fmt='o', capsize=4, markersize=4)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Nylon Beads in Water, Residuals')
    
    plt.xlabel('d (mm)')
    plt.ylabel('v (mm/s)')
    plt.show()

    print("Water: ",Water_means)
    print("Corrected: ",W_v_corrs)

    W_p = 1*(10**-6)
    W_nug = 1*(10**-4)/(100)
    
    W_reynolds = [W_p*W_v_corrs[i]*Water_d[i]/W_nug for i in range(len(Water_means))]
    print('Water Reynolds numbers: ' + str(W_reynolds))


    #Do water again but with a quadratic fit
    W_v_corrs_fit = fit_quadratic(np.column_stack((Water_d, W_v_corrs)))
    W_v_corrs_reduced_chi_squared = compute_reduced_chi_squared(np.column_stack((Water_d, W_v_corrs)), np.column_stack((Water_d, W_v_corrs_errors)), np.column_stack((Water_d, W_v_corrs_fit[0]*np.array(Water_d)**2 + W_v_corrs_fit[1]*np.array(Water_d) + W_v_corrs_fit[2])))
    print('Reduced chi-squared: ' + str(W_v_corrs_reduced_chi_squared))
    x = np.linspace(2, 6.5, 100)
    y = W_v_corrs_fit[0]*x**2 + W_v_corrs_fit[1]*x + W_v_corrs_fit[2]
    plt.plot(x, y, label='Quadratic Fit')
    plt.errorbar(Water_d, W_v_corrs,  yerr=W_v_corrs_errors, fmt='o', label='Water', capsize=4, markersize=4)
    plt.title('Nylon Beads in Water, v vs. d (Quadratic Fit)')
    plt.xlabel('d (mm)')
    plt.ylabel('v (mm/s)')
    plt.show()
    plt.errorbar(Water_d, W_v_corrs - (W_v_corrs_fit[0]*np.array(Water_d)**2 + W_v_corrs_fit[1]*np.array(Water_d) + W_v_corrs_fit[2]), yerr=W_v_corrs_errors, fmt='o', capsize=4, markersize=4)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Nylon Beads in Water, Residuals (Quadratic Fit))')
    plt.xlabel('d (mm)')
    plt.ylabel('v (mm/s)')
    plt.show()

    print("Glycerine: ",G_v_corrs_errors)
    print("Water: ", W_v_corrs_errors)


    #Open the file "Gtrial 11.txt"
    file = open('Gtrial 11.txt', 'r')
    # Read the data into a numpy array
    data = np.loadtxt(file, skiprows=2)
    file.close()
    #Plot the data. No error bars
    plt.plot(data[:,0], data[:,1], 'o', markersize=4)
    plt.show()

# Run program
if __name__ == '__main__':
    main()
