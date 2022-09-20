# -*- coding: utf-8 -*-
"""
This module imports a folder from IMES and processes the data.
"""
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import cm

# change matplotlib settings to make plots look nicer
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.major.width'] = 3

def plot_setup(labels=None, fsize=20, setlimits=False,
               limits=[0,1,0,1], title='', legend=False,
               colorbar=False, save=False, filename='plot.jpg',
               size=None):
    """Creates a custom plot configuration to make graphs look nice.
    This can be called with matplotlib for setting axes labels,
    titles, axes ranges, and the font size of plot labels.
    This should be called between plt.plot() and plt.show() commands."""
    if labels is not None:
        plt.xlabel(str(labels[0]), fontsize=fsize)
        plt.ylabel(str(labels[1]), fontsize=fsize)
    plt.title(title, fontsize=fsize)
    fig = plt.gcf()
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    if colorbar:
        plt.colorbar()
    if legend:
        plt.legend()#fontsize=fsize-4)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        fig.savefig(filename, dpi=500, bbox_inches='tight')
        plt.tight_layout()


def imes_dict(folder):
    """Place IMES files into a dictionary based on their data type."""
    # get list of all files in folder
    files = glob(folder+'/*')
    # create dictionary to hold all file designations
    d = {'all': files}
    # get list of all qcm spectra files
    d['qcm_spec'] = [f for f in files if 'qcm_n=' in f]
    # get qcm time step info
    d['qcm_params'] = [f for f in files if 'qcm_params' in f][0]
    # get main pressure file
    d['main'] = [f for f in files if 'main_df' in f][0]
    # get impedance file
    d['eis'] = [f for f in files if 'eis.csv' in f][0]
    # read main df data to get total time span of experimental data
    main_df = pd.read_csv(d['main'])
    d['timespan'] = (main_df['time'].max() - main_df['time'].min())/60
    return d

def process_eis(eis_file):
    """Process EIS data from IMES eis filename."""
    df = pd.read_csv(eis_file).dropna()
    eis = {'freq': np.array(df.iloc[:, 0]),
           'z': df[[i for i in list(df) if i.startswith('z_')]],
           'phi': df[[i for i in list(df) if i.startswith('phase_')]],
           'rez': df[[i for i in list(df) if i.startswith('rez_')]],
           'imz': df[[i for i in list(df) if i.startswith('imz_')]]}
    return eis
    

def process_qcm(filelist):
    """Process QCM data from IMES qcm filename."""
    qcm = {'freq': {},
           'rs': {},
           'xs': {}}
    # loop over qcm spectra file
    for f in filelist:
        n = f.split('qcm_n=')[1].split('_')[0]
        df = pd.read_csv(f).dropna()
        freq = np.array(df.iloc[:, 0])
        qcm['freq'][n] = freq
        qcm['rs'][n] = df[[i for i in list(df) if 'rs__' in i]]
        qcm['xs'][n] = df[[i for i in list(df) if 'xs__' in i]]
        
        # normalize each column
        qcm['rs'][n] = (qcm['rs'][n]-qcm['rs'][n].min())/(
                qcm['rs'][n].max()-qcm['rs'][n].min())
    return qcm

def plot_rh(f):
    """Plot RH sequence over time using the main filename from IMES."""
    df = pd.read_csv(f)
    rh = np.array(df['rh_setpoint'].dropna())
    times = np.array((df['time']-df['time'].min())/60)[:-1]
    rh[0] = 0
    rh[-1] = 0
    
    plt.fill(times, rh, facecolor='blue', alpha=0.8)
    plt.xlim((0, 11))
    plt.ylim((0, 100))
    #plt.xticks([0, 5, 10])
    
    plot_setup(labels=['Time (hours)', 'Relative humidity (%)'])
    fig = plt.gcf()
    fig.savefig(os.path.join(save_img_dir, 'rh.jpg'),
                dpi=500, bbox_inches='tight')
    plt.show()



def plot_qcm_params(qcm_params_filename, timespan):
    """Plot QCM parameters from the IMES qcm params file."""
    df = pd.read_csv(qcm_params_filename).dropna()
    df['time'] = timespan
    colors = cm.rainbow(np.linspace(0, 1, num=9))[::-1]
    # plot delta F
    i = 0
    deltaf_mat = np.empty((len(df), 0))
    for col in list(df):
        if 'f_' in col:
            n = col.split('f_')[1]
            deltaf = (df[col] - df[col].iloc[0])/1e3
            deltaf_mat = np.column_stack((deltaf_mat, deltaf/int(n)))
            plt.plot(timespan, deltaf, marker='o', c=colors[i], label=n)
            i += 1
    #plt.legend()
    plot_setup(labels=['Time (hours)', '$\Delta$f (kHz/cm$^{2}$)'],
                       legend=True)
    plt.xlim([0, 11])
    fig = plt.gcf()
    fig.savefig(os.path.join(save_img_dir, 'df_trace.jpg'),
                dpi=500, bbox_inches='tight')
    plt.show()
    
    # plot delta D
    i = 0
    deltad_mat = np.empty((len(df), 0))
    for col in list(df):
        if 'd_' in col:
            n = col.split('d_')[1]
            deltad = (df[col] - df[col].iloc[0])*1e6
            deltad_mat = np.column_stack((deltad_mat, deltad))
            plt.plot(timespan, deltad, marker='o', c=colors[i], label=n)
            i += 1
    #plt.legend()
    plot_setup(labels=['Time (hours)', '$\Delta$D (x10$^{-6}$)'],
                       legend=True)
    plt.xlim([0, 11])
    fig = plt.gcf()
    fig.savefig(os.path.join(save_img_dir, 'dd_trace.jpg'),
                dpi=500, bbox_inches='tight')
    plt.show()
    
    
    # plot matrix of delta F
    extent = [0, 11, 17, 1]
    plt.imshow(deltaf_mat.T*1e3, interpolation='gaussian', cmap='jet_r',
               extent=extent, aspect='auto', vmax=0, vmin=-300)
    plt.xticks([0, 5, 10])
    plt.yticks([1, 5, 9, 13, 17])
    plt.colorbar(ticks=[-300, -200, -100, 0])
    plot_setup(labels=['Time (hours)', 'Harmonic'],
                       title='$\Delta$f/n (kHz/cm$^{2}$)')
    fig = plt.gcf()
    fig.savefig(os.path.join(save_img_dir, 'df_mat.jpg'),
                dpi=500, bbox_inches='tight')
    plt.show()
    
    
    # plot matrix of delta D
    plt.imshow(deltad_mat.T, interpolation='gaussian', cmap='jet',
               extent=extent, aspect='auto')
    plt.xticks([0, 5, 10])
    plt.yticks([1, 5,  9, 13, 17])
    plt.colorbar(ticks=[0, 100, 200, 300])
    plot_setup(labels=['Time (hours)', 'Harmonic'],
                       title='$\Delta$D (x10$^{-6}$)')
    fig = plt.gcf()
    fig.savefig(os.path.join(save_img_dir, 'dd_mat.jpg'),
                dpi=500, bbox_inches='tight')
    plt.show()
    
    
    



# %% ======================= read in data ==========================


# designate data folder
folder = '/home/eric/Documents/2020-03-17_cupcts_eis_qcm'

# set directory in which to save images
save_img_dir = '/home/eric/Desktop'

# get dictionary of all IMES files
d = imes_dict(folder)

# plot RH data
plot_rh(d['main'])

# read in electrochemical impedance spectroscopy data
eis = process_eis(d['eis'])

# read in qcm data
qcm = process_qcm(d['qcm_spec'])

# get time axis for qcm data
qcm_timespan = np.linspace(0, d['timespan'], num=len(list(qcm['rs']['1'])))

# plot qcm parameters
plot_qcm_params(d['qcm_params'], qcm_timespan)



# %% ====================== plot impedance data =====================

EIS = True
if EIS:
    
    # plot EIS data
    eis_timespan = np.linspace(0, d['timespan'], num=len(list(eis['z'])))
    # plot eis Z
    eis_extent = [eis_timespan[0],
                 eis_timespan[-1],
                 eis['freq'][0],
                 eis['freq'][-1]]
    plt.imshow(np.log10(eis['z']),
               cmap='jet',
               extent=eis_extent,
               origin='lower',
               interpolation='gaussian',
               vmin=2, vmax=8)
    plot_setup(labels=['Time (hours)', 'Frequency (Hz)'],
                       title='Log(Z) (M$\Omega$)')
    plt.yscale('log')
    plt.colorbar(ticks=[2, 4, 6, 8])
    fig = plt.gcf()
    fig.savefig(os.path.join(save_img_dir, 'eis_z.jpg'),
               dpi=500, bbox_inches='tight')
    plt.show()
    
    
    # plot eis phase
    plt.imshow(eis['phi'],
               cmap='jet_r',
               extent=eis_extent,
               origin='lower',
               interpolation='gaussian',
               vmin=-90, vmax=0)
    plot_setup(labels=['Time (hours)', 'Frequency (Hz)'],
                       title='Phase (deg)')
    plt.yscale('log')
    plt.colorbar(ticks=[0, -30, -60, -90])
    fig = plt.gcf()
    fig.savefig(os.path.join(save_img_dir, 'eis_phase.jpg'),
                dpi=500, bbox_inches='tight')
    plt.show()



# %% ====================== plot qcm data ===============================
QCM_SPEC=True
if QCM_SPEC:
    harmonics = [1,3,5,7,9,11,13,15,17]
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i+1)
        mat = qcm['rs'][str(harmonics[i])][300:-320]
        
        plt.imshow(mat,
                   origin='lower',
                   aspect='auto',
                   interpolation='gaussian',
                   cmap='jet')
        plot_setup(title=str(harmonics[i]*5)+ ' MHz',
                   size=(10,10))
        #plt.axis('off')
        plt.tick_params(bottom='off',
                        left='off', labelleft='off',
                        labelbottom='off')
    
    fig.savefig(os.path.join(save_img_dir, 'qcm_maps.jpg'),
                dpi=500, bbox_inches='tight')
    plt.tight_layout()
    plt.show()








