# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 13:41:23 2018

@author: DanielM
"""

from neuron import h, gui  # gui necessary for some parameters to h namespace
import numpy as np
import net_tunedrev
from burst_generator_inhomogeneous_poisson import inhom_poiss
import os
import argparse
import scipy.stats as stats
from analysis_main import time_stamps_to_signal
tsts = time_stamps_to_signal

# Handle command line inputs
pr = argparse.ArgumentParser(description='Local pattern separation paradigm')
pr.add_argument('-runs',
                nargs=3,
                type=int,
                help='start stop range for the range of runs',
                default=[0, 1, 1],
                dest='runs')
pr.add_argument('-savedir',
                type=str,
                help='complete directory where data is saved',
                default=os.getcwd(),
                dest='savedir')
pr.add_argument('-scale',
                type=int,
                help='standard deviation of gaussian distribution',
                default=1000,
                dest='input_scale')
pr.add_argument('-seeds',
                nargs=3,
                type=int,
                help='standard deviation of gaussian distribution',
                default=[100, 107, 1],
                dest='seeds')
pr.add_argument('-pp_weight',
                type=float,
                help='standard deviation of gaussian distribution',
                default=1e-3,
                dest='pp_weight')
pr.add_argument('-poiss_rate',
                type=float,
                help='standard deviation of gaussian distribution',
                default=10,
                dest='poiss_rate')

args = pr.parse_args()
runs = range(args.runs[0], args.runs[1], args.runs[2])
savedir = args.savedir
input_scale = args.input_scale
#seed = args.seeds

# Where to search for nrnmech.dll file. Must be adjusted for your machine.
dll_files = [("C:\\Users\\DanielM\\Repos\\models_dentate\\"
              "dentate_gyrus_Santhakumar2005_and_Yim_patterns\\"
              "dentategyrusnet2005\\nrnmech.dll"),
             "C:\\Users\\daniel\\Repos\\nrnmech.dll",
             ("C:\\Users\\Holger\\danielm\\models_dentate\\"
              "dentate_gyrus_Santhakumar2005_and_Yim_patterns\\"
              "dentategyrusnet2005\\nrnmech.dll"),
             ("C:\\Users\\Daniel\\repos\\"
              "dentate_gyrus_Santhakumar2005_and_Yim_patterns\\"
              "dentategyrusnet2005\\nrnmech.dll"),
              ("/home/daniel/repos/pyDentate/mechs_7-6_linux/"
               "x86_64/.libs/libnrnmech.so")]

for x in dll_files:
    if os.path.isfile(x):
        dll_dir = x
print("DLL loaded from: " + dll_dir)
h.nrn_load_dll(dll_dir)

for seed in range(args.seeds[0], args.seeds[1], args.seeds[2]):
    # Seed the numpy random number generator for replication
    np.random.seed(seed)
    
    # Randomly choose target cells for the PP lines
    gauss_gc = stats.norm(loc=1000, scale=input_scale)
    gauss_bc = stats.norm(loc=12, scale=(input_scale/2000.0)*24)
    pdf_gc = gauss_gc.pdf(np.arange(2000))
    pdf_gc = pdf_gc/pdf_gc.sum()
    pdf_bc = gauss_bc.pdf(np.arange(24))
    pdf_bc = pdf_bc/pdf_bc.sum()
    GC_indices = np.arange(2000)
    start_idc = np.random.randint(0, 1999, size=400)

    PP_to_GCs = []
    for x in start_idc:
        curr_idc = np.concatenate((GC_indices[x:2000], GC_indices[0:x]))
        PP_to_GCs.append(np.random.choice(curr_idc, size=100, replace=False,
                                          p=pdf_gc))
    
    PP_to_GCs = np.array(PP_to_GCs)
    
    BC_indices = np.arange(24)
    start_idc = np.array(((start_idc/2000.0)*24), dtype=int)
    
    PP_to_BCs = []
    for x in start_idc:
        curr_idc = np.concatenate((BC_indices[x:24], BC_indices[0:x]))
        PP_to_BCs.append(np.random.choice(curr_idc, size=1, replace=False,
                                          p=pdf_bc))
    
    PP_to_BCs = np.array(PP_to_BCs)
    
    # Generate temporal patterns for the 100 PP inputs
    np.random.seed(seed)
    temporal_patterns = inhom_poiss(rate=args.poiss_rate)

# Start the runs of the model
    for run in runs:
        print(f"Start Running {run}")
        nw = net_tunedrev.TunedNetwork(seed, temporal_patterns[0+run:24+run],
                                       PP_to_GCs[0+run:24+run],
                                       PP_to_BCs[0+run:24+run],
                                       pp_weight=args.pp_weight)
    
        # Attach voltage recordings to all cells
        nw.populations[0].voltage_recording(range(2000))
        nw.populations[1].voltage_recording(range(60))
        nw.populations[2].voltage_recording(range(24))
        nw.populations[3].voltage_recording(range(24))
        # Run the model
        """Initialization for -2000 to -100"""
        h.cvode.active(0)
        dt = 0.1
        h.steps_per_ms = 1.0/dt
        h.finitialize(-60)
        h.t = -2000
        h.secondorder = 0
        h.dt = 10
        while h.t < -100:
            h.fadvance()

        h.secondorder = 2
        h.t = 0
        h.dt = 0.1

        """Setup run control for -100 to 1500"""
        h.frecord_init()  # Necessary after changing t to restart the vectors
        while h.t < 600:
            h.fadvance()
        print("Done Running {run}")

        save_data_name = (f"{str(nw)}_"
                          f"{seed}_"
                          f"{run:03d}_"
                          f"{args.poiss_rate:04f}_"
                          f"{args.pp_weight:03f}")

        if run == 0:
            fig = nw.plot_aps(time=600)
            tuned_fig_file_name =save_data_name
            nw.save_ap_fig(fig, args.savedir, tuned_fig_file_name)
    
        pp_lines = np.empty(400, dtype = np.object)
        pp_lines[0+run:24+run] = temporal_patterns[0+run:24+run]
        
        curr_pp_ts = np.array(tsts(pp_lines, dt_signal=0.1, t_start=0, t_stop=600), dtype = np.bool)
        curr_gc_ts = np.array(tsts(nw.populations[0].get_properties()['ap_time_stamps'], dt_signal=0.1, t_start=0, t_stop=600), dtype = np.bool)
        curr_mc_ts = np.array(tsts(nw.populations[1].get_properties()['ap_time_stamps'], dt_signal=0.1, t_start=0, t_stop=600), dtype = np.bool)
        curr_bc_ts = np.array(tsts(nw.populations[2].get_properties()['ap_time_stamps'], dt_signal=0.1, t_start=0, t_stop=600), dtype = np.bool)
        curr_hc_ts = np.array(tsts(nw.populations[3].get_properties()['ap_time_stamps'], dt_signal=0.1, t_start=0, t_stop=600), dtype = np.bool)
           
     
        np.savez(args.savedir + os.path.sep + "population_vectors_" + save_data_name,
                 pp_ts=np.array(curr_pp_ts).sum(axis=1),
                 gc_ts=np.array(curr_gc_ts).sum(axis=1),
                 mc_ts=np.array(curr_mc_ts).sum(axis=1),
                 bc_ts=np.array(curr_bc_ts).sum(axis=1),
                 hc_ts=np.array(curr_hc_ts).sum(axis=1))
        
        del curr_pp_ts, curr_gc_ts, curr_mc_ts, curr_hc_ts, curr_bc_ts
        del nw
