# Setting up required packages
# FOLLOW GITHUB INSTRUCTIONS OR MAY NOT WORK 

import juliacall
import numpy as np
import fit_modeler
import evolve_binary
import pandas as pd
import time
import surrogate
from astropy import constants as const
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.set_printoptions(legacy='1.13')

    # Setting up empty lists to read McFACTS output data files names
    # Uploading surrogate file to run Post Newtonian Methods on Binary Black Holes Mergers

    gal = [str(i).zfill(3) for i in range(100)]
    obj = [str(i) for i in range(100)]
    runs = [[] for _ in range(len(gal))]
    print(runs)
    for i in range(len(gal)):
        for j in range(len(obj)):
            temp = '/Users/sray/Documents/Saavik_Barry/test_mcfacts/runs/gal' + gal[i] + '/output_bh_binary_' + obj[j] + '.dat'
            runs[i].append(temp)
    print(gal)
    print(obj)
    print(runs)
    surrogate = fit_modeler.GPRFitters.read_from_file(f"surrogate.joblib")

    # Setting up a function to estimate the BBH seperations allowing for faster run times

    def new_bin_sep(freq, i):
        active_bin = pd.read_csv(i, delimiter=' ', header = 0)
        mass_1 = active_bin['mass_1']
        mass_2 = active_bin['mass_2']
        print(mass_1)
        G = const.G.value
        M_dot = 2e30
        grav_freq = freq
        
        #print('mass_1: ', active_bin['mass_1'])
        #print('grav_freq: ', active_bin['gw_freq'])
        
        a_bin = ((G * (mass_1 + mass_2) * M_dot * np.pi**2) / grav_freq**2) ** (1/3)
        #a_bin = ((G * (mass_1 + mass_2) * M_dot) / (16 * grav_freq**2 * np.pi**2))**1/3
        a_bin_rg = a_bin / 1.5e11
        return a_bin_rg

    # Bulk of calculations in which McFACTS outputs are loaded in 
    # Once those outputs are loaded in the cell will check for mergers and continuously monitor the status of each binary system as the program runs
    # once a merger is detected, the cell will call the Post Newtonian methods and calculate the final parameters of the remant black hole 
    # The final parameters include: remnant mass in units of solar masses, remnant spin in a dimensionless quantity, and remnant kick velocity in units of c
    # Run time for Apple MacBook Air with M3 chip, 16GB RAM, and 8 cores is approximately 11min.

    total_mass_final = []
    total_spin_final = []
    total_velocity_final = []

    num_of_obj_mergers = []
    num_of_gal_mergers = []


    for i in range(len(gal)):
        merger_counter_gal = []
        total_start_mass1 = []
        total_start_mass2 = []
        total_start_spin1 = []
        total_start_spin2 = []
        print('===================================   NEW GALAXY   =================================')
        for j in range(len(obj)):
            if j == 0:
                continue
            merger_counter_obj = []
            
            #The following logic systems use the current binary as the base for checks
            active_bin = pd.read_csv(runs[i][j], delimiter=' ', header = 0)
            print('----------------     NEW TIMESTEP     ----------------')
            mass_final = []
            spin_final = []
            velocity_final = []
            starting_mass1 = []
            starting_mass2 = []
            starting_spin1 = []
            starting_spin2 = []
            
            # This checks if the current timestep is empty
            if active_bin.empty:
                print('Galaxy ', gal[i], ' and run ', obj[j])
                print('file is empty. Checking for previous binaries and moving onto next system')
                act_bin_prev = pd.read_csv(runs[i][j-1], delimiter=' ', header = 0)
                
                bh_prev = act_bin_prev['id_num']
                bh = active_bin['id_num']
                
                #This loop checks if there are any previously generated binaries
                # If there are any previously merged binaries, then this must mean those binaries have merged
                #  since the current timestep is empty.
                if bh_prev.empty:
                    print('no previously merged binaries')
                else:
                    for values in bh_prev:
                        print('binary ', values, ' has merged...beginning surrogate model')
                        #merger_counter_obj.append(obj[j])
                        for i in range(len(act_bin_prev['mass_1'])):
                            mass_1 = act_bin_prev['mass_1'][i]
                            mass_2 = act_bin_prev['mass_2'][i]
                            spin_1_mag = act_bin_prev['spin_1'][i]
                            spin_2_mag = act_bin_prev['spin_2'][i]
                            spin_angle_1 = act_bin_prev['spin_angle_1'][i]
                            spin_angle_2 = act_bin_prev['spin_angle_2'][i]
                            phi_12 = spin_angle_2 - spin_angle_1
                            # This should be in units of mass_1 + mass_2
                            bin_sep = 1000
                            bin_inc = [0, 0, 1]
                            bin_phase = 0
                            # These next three are used to correct the remnant velocity;
                            # If they are None, no correction is applied.
                            bin_orb_a = None
                            mass_SMBH = None
                            spin_SMBH = None

                            surrogate = fit_modeler.GPRFitters.read_from_file(f"surrogate.joblib")

                            start = time.time()
                            M_f, spin_f, v_f = evolve_binary.evolve_binary(
                                mass_1,
                                mass_2,
                                spin_1_mag,
                                spin_2_mag,
                                spin_angle_1,
                                spin_angle_2,
                                phi_12,
                                bin_sep,
                                bin_inc,
                                bin_phase,
                                bin_orb_a,
                                mass_SMBH,
                                spin_SMBH,
                                surrogate,
                                verbose=True,
                            )
                            end = time.time()
                            
                            run_time = end - start

                            mass_final.append(M_f)
                            spin_final.append(spin_f)
                            velocity_final.append(v_f)
                            
                        print("M_f = ", M_f)
                        print("spin_f = ", spin_f)
                        print("v_f = ", v_f)
                        print("Merger took ", run_time, " seconds")
            
            else:
                print('binary found at galaxy ', gal[i], ' and binary ', obj[j])
                act_bin_prev = pd.read_csv(runs[i][j-1], delimiter=' ', header = 0)
                
                mass_1 = act_bin_prev['mass_1']
                mass_2 = act_bin_prev['mass_2']
                spin_1 = act_bin_prev['spin_1']
                spin_2 = act_bin_prev['spin_2']
                spin_angle_1 = act_bin_prev['spin_angle_1']
                spin_angle_2 = act_bin_prev['spin_angle_2']
                
                bh_prev = act_bin_prev['id_num'].values
                bh = active_bin['id_num'].values
                
                #This loop check to see if any binaries have formed
                for value in bh:
                    if act_bin_prev.empty:
                        print('binary ', value, ' has formed')
                    elif value not in bh_prev:
                        print('binary ', value, ' has formed')
                    else:
                        print('binary ', value, ' has not yet mergered')
                
                for value in bh_prev:
                    if value not in bh:
                        print('binary ', value, ' has merged...beginning surrogate model')
                        #merger_counter_obj.append(obj[j])
                        for k in range(len(act_bin_prev['mass_1'])):
                            mass_1 = act_bin_prev['mass_1'][k]
                            mass_2 = act_bin_prev['mass_2'][k]
                            spin_1_mag = act_bin_prev['spin_1'][k]
                            spin_2_mag = act_bin_prev['spin_2'][k]
                            spin_angle_1 = act_bin_prev['spin_angle_1'][k]
                            spin_angle_2 = act_bin_prev['spin_angle_2'][k]
                            phi_12 = spin_angle_2 - spin_angle_1
                            # This should be in units of mass_1 + mass_2
                            bin_sep = 1000
                            bin_inc = [0, 0, 1]
                            bin_phase = 0
                            # These next three are used to correct the remnant velocity;
                            # If they are None, no correction is applied.
                            bin_orb_a = None
                            mass_SMBH = None
                            spin_SMBH = None

                            surrogate = fit_modeler.GPRFitters.read_from_file(f"surrogate.joblib")

                            start = time.time()
                            M_f, spin_f, v_f = evolve_binary.evolve_binary(
                                mass_1,
                                mass_2,
                                spin_1_mag,
                                spin_2_mag,
                                spin_angle_1,
                                spin_angle_2,
                                phi_12,
                                bin_sep,
                                bin_inc,
                                bin_phase,
                                bin_orb_a,
                                mass_SMBH,
                                spin_SMBH,
                                surrogate,
                                verbose=True,
                            )
                            end = time.time()
                            
                            run_time = end - start

                            mass_final.append(M_f)
                            spin_final.append(spin_f)
                            velocity_final.append(v_f)
                            
                            starting_mass1.append(mass_1)
                            starting_mass2.append(mass_2)
                            starting_spin1.append(spin_1)
                            starting_spin2.append(spin_2)
                            
                        print("M_f = ", M_f)
                        print("spin_f = ", spin_f)
                        print("v_f = ", v_f)
                        print("Merger took ", run_time, " seconds")
            #merger_counter_gal.append(merger_counter_obj)
                        
            #merger_counter_gal.append(merger_counter_obj)
            #print('galaxy ', gal[i], ' had mergers at timesteps ', merger_counter_gal)
            total_mass_final.append(mass_final)
            total_spin_final.append(spin_final)
            total_velocity_final.append(velocity_final)
            total_start_mass1.append(starting_mass1)
            total_start_mass2.append(starting_mass2)
            total_start_spin1.append(starting_spin1)
            total_start_spin2.append(starting_spin2)

    print(total_mass_final)
    print(total_spin_final)
    print(total_velocity_final)
    print(total_start_mass1)
    print(total_start_mass2)
    print(total_start_spin1)
    print(total_start_spin2)
                        

    # Taking the final values and truncating necesssary values for plotting 

    reduced_mass_final = []
    reduced_spin_final = []
    reduced_velocity_final = []

    for i in range(len(total_mass_final)):
        if total_mass_final[i] != []:
            reduced_mass_final.append(total_mass_final[i])
    for i in range(len(total_spin_final)):
        if total_spin_final[i] != []:
            reduced_spin_final.append(total_spin_final[i])
    for i in range(len(total_velocity_final)):
        if total_velocity_final[i] != []:
            reduced_velocity_final.append(total_velocity_final[i])
        
    print(reduced_mass_final[2])
    print(reduced_spin_final[2])
    print(reduced_velocity_final[2])
            

    plot_mass = []
    for i in range(len(reduced_mass_final)):
        for j in range(len(reduced_mass_final[i])):
            plot_mass.append(reduced_mass_final[i][j])
    #print(plot_mass)

    plot_spin = []
    for i in range(len(reduced_spin_final)):
        for j in range(len(reduced_spin_final[i])):
            plot_spin.append(reduced_spin_final[i][j])
    #print(plot_spin)

    plot_velocity = []
    for i in range(len(reduced_velocity_final)):
        for j in range(len(reduced_velocity_final[i])):
            plot_velocity.append(reduced_velocity_final[i][j])
    #print(plot_velocity)


    # Converting spin_1 magnitude values into cartesian values
    spin_mag = [[] for _ in range(len(plot_spin))] 

    for i in range(len(spin_mag)):
        for j in range(len(plot_spin[i])):
            spin_mag[i] = np.sqrt(plot_spin[i][0]**2 + plot_spin[i][1]**2 + plot_spin[i][2]**2)
    print(spin_mag)

    # Converting spin_1 magnitude values into cartesian values
    vel_mag = [[] for _ in range(len(plot_velocity))] 

    for i in range(len(vel_mag)):
        vel_mag[i] = np.linalg.norm(plot_velocity[i]) * 100
    print(vel_mag)

    # Counting the numer of mergers that occured
    num_mergers = range(len(plot_mass))

    plt.hist(plot_mass, num_mergers)
    plt.xlabel('Remnant ' r'Mass ($M_{\odot}$)')
    plt.ylabel('Number of mergers')
    plt.xlim([10, 120])
    plt.title('MNumber of Mergers per Remnant Mass')
    plt.show()

    plt.scatter(plot_mass, vel_mag)
    plt.xlabel('Remnant ' r'Mass ($M_{\odot}$)')
    plt.ylabel('Velocity (%c)')
    plt.title('Merger Velocities per Merger Mass')
    plt.show()

    plt.scatter(plot_mass, spin_mag)
    plt.xlabel('Remnant ' r'Mass ($M_{\odot}$)')
    plt.ylabel('Spin')
    plt.title('Merger Spin per Merger Mass')
    plt.show()
