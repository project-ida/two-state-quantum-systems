# Functions from previous tutorials that are helpful for 05 tutorial

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from itertools import product
import matplotlib.pyplot as plt
from qutip import *


def make_df_for_energy_scan(label_param, min_param, max_param, num_param, num_levels):
    """
    Creates a dataframe to hold energy level information from many Hamiltonians that differ only by some paratmer. 
    
    See https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/04-spin-boson-model.ipynb#4.2---Stationary-states for where this function was first created
    
    Parameters
    ----------
    label_param : String, Label for the scanning parameter
    min_param   : Number, Minumum value of the scanning parameter
    max_param   : Number, Maximum value of the scanning parameter
    max_param   : Integer, Number of values of scanning parameter
    num_levels  : Integer, Number of energy levels to be calculated
    
    Returns
    -------
    df : Dataframe,  of strings
    
    Examples
    --------
    >>> df = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, H.shape[0])
    
    """
    # creates an empty dictionary to store the row/column information
    d = {}
    
    # creates array of parameter values that we want to scan through
    param_values = np.linspace(min_param, max_param, num_param)
    
    # stores the parameter scan label and values (this will soon become the first column in the dataframe)
    d[label_param] = param_values
    
    # creates empty columns to store the eigenvalues for the different levels later on
    # num_levels will be the number of rows of H (or any of the operators that make up H)
    for i in range(num_levels):
        d[f"level_{i}"] = np.zeros(num_param)
     
    # creates the dataframe
    df = pd.DataFrame(data=d)
    
    return df


def make_braket_labels(list_of_states):
    """
    Creates 2 lists of strings to be used in labels for plot when bras and kets are required.
    
    See https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/04-spin-boson-model.ipynb#4.3---Structure-of-the-Hamiltonian for where this function was first created.
    
    Parameters
    ----------
    list_of_states : list of strings, or list of tuples containing strings
    
    Returns
    -------
    bra_labels : list of strings
    ket_labels : list of strings
    
    Examples
    --------
    >>> bra_labels, ket_labels = make_braket_labels([("+","+"), ("-","+")])
    
    """
    bra_labels = ["$\langle$"+', '.join(map(str,n))+" |" for n in list_of_states]
    ket_labels = ["| "+', '.join(map(str,n))+"$\\rangle$" for n in list_of_states]
    return bra_labels, ket_labels



def simulate(H, psi0, times):
    """
    Solves the time independent Schrödinger equation
    
    See https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/04-spin-boson-model.ipynb#4.5---Down-conversion for where this function was first created
    
    Parameters
    ----------
    H     :  QuTiP object, Hamiltonian for the system you want to simulate
    psi0  :  QuTiP object, Initial state of the system
    times :  1D numpy array, Times to evaluate the state of the system (best to use use np.linspace to make this) 

    
    Returns
    -------
    P   : numpy array [i,j], Basis state (denoted by i) occupation probabilities at each time j
    psi : numpy array [i,j], Basis state (denoted by i) values at each time j
    
    Examples
    --------
    >>> P, psi = simulate(sigmaz() + sigmax(), basis(2, 0),  np.linspace(0.0, 20.0, 1000) )
    
    """
    num_states = H.shape[0]
    
    # create placeholder for values of amplitudes for different states
    psi = np.zeros([num_states,times.size], dtype="complex128")
     # create placeholder for values of occupation probabilities for different states
    P = np.zeros([num_states,times.size], dtype="complex128")
    
    evals, ekets = H.eigenstates()
    psi0_in_H_basis = psi0.transform(ekets)

    for k in range(0,num_states):
        amp = 0
        for i in range(0,num_states):
            amp +=  psi0_in_H_basis[i][0][0]*np.exp(-1j*evals[i]*times)*ekets[i][k][0][0]
        psi[k,:] = amp
        P[k,:] = amp*np.conj(amp)
    return P, psi


def prettify_states(states, mm_list=None):
    """
    Takes an array of QuTiP states and returns a pandas dataframe that makes it easier to compare the states side by side 
    
    
    Parameters
    ----------
    states     :  Numpy array of QuTiP objects
    mm_list    :  list strings, or list of tuples containing strings, these will become row labels to label basis states

    
    Returns
    -------
    df   : pandas dataframe
    
    """
    pretty_states = np.zeros([states[0].shape[0],len(states)], dtype="object")
    
    for j, state in enumerate(states):
        for i, val in enumerate(state):
            pretty_states[i,j] = f"{val[0,0]:.1f}"
    if (mm_list == None):
        df = pd.DataFrame(data=pretty_states)
    else:
        df = pd.DataFrame(data=pretty_states, index=mm_list)
            
    return df


def plot_sim(times, P, labels=None, ylabel="Probability", xlabel="Time", legend="right"):
    """
    Plots simulation results over time
    
    
    Parameters
    ----------
    P      :  List containing ket vectors of 1D numpy arrays
    times  :  1D numpy array
    labels :  List of strings, labels to be used for the plot legend

    
    """
    f = plt.figure(figsize=(10,8))
    ax = f.add_subplot(1, 1, 1)
    
    if (labels == None):
        for i in range(0,len(P)):
            ax.plot(times, P[i][:], label=f"{i}")
    else:
        for i in range(0,len(P)):
            ax.plot(times, P[i][:], label=f"{labels[i]}")
            
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    ax.legend(loc=legend)
    
    return

def expectation(operator, states):
    """
    Calculates the expectation of an operator given a array of states
    
    
    Parameters
    ----------
    operator      :  QuTiP operator
    states        :  2D numpy array, get this from output of the simulate function

    
    """
    
    operator_matrix = operator.full()
    operator_expect = []
    for i in range(0,shape(states)[1]):
        e = np.conj(states[:,i])@ (operator_matrix @ states[:,i])
        operator_expect.append(e)
    return operator_expect