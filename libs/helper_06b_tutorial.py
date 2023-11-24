# Functions from previous tutorials that are helpful for 05 tutorial

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from itertools import product
from fractions import Fraction
from qutip.cy.piqs import j_min, j_vals, m_vals


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


def make_braket_labels(n_list, output_fractions = True):
    """
    Creates 2 lists of strings to be used in labels for plot when bras and kets are required.
    
    See https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/04-spin-boson-model.ipynb#4.3---Structure-of-the-Hamiltonian for where this function was first created. The function has since been adapted.
    
    Parameters
    ----------
    n_list             : list of tuples, Each tuple contains strings or numbers that describe the state
    output_fractions   : Boolean, If true, the function will try and turn numbers into fractions, e.g. 0.5 -> 1/2
    
    Returns
    -------
    bra_labels : list of strings
    ket_labels : list of strings
    
    Examples
    --------
    >>> bra_labels, ket_labels = make_braket_labels([("+","+"), ("-","+")])
    
    """
    
    def Fraction_no_fail(val):
        try:
            val = Fraction(val)
        except:
            pass
        return val
    
    bra_labels = []
    ket_labels = []
    
    if (output_fractions == True):
        
        bra_labels = []
        ket_labels = []
        
        for label in n_list:
            s = []
            for l in label:
                s.append(str(Fraction_no_fail(str(l))))
            
            bra_labels.append("$\langle$" + ', '.join(s) + " |")
            ket_labels.append("| "+ ', '.join(s) +"$\\rangle$")
        
    else:
        
        bra_labels = ["$\langle$"+', '.join(map(str,n))+" |" for n in n_list]
        ket_labels = ["| "+', '.join(map(str,n))+"$\\rangle$" for n in n_list]
        
    
    return bra_labels, ket_labels


# This function takes a list of QuTiP states and puts them in a dataframe to make them easier to visually compare
def prettify_states(states, mm_list=None):
    pretty_states = np.zeros([states[0].shape[0],len(states)], dtype="object")
    
    for j, state in enumerate(states):
        for i, val in enumerate(state):
            pretty_states[i,j] = f"{val[0,0]:.1f}"
    if (mm_list == None):
        df = pd.DataFrame(data=pretty_states)
    else:
        df = pd.DataFrame(data=pretty_states, index=mm_list)
            
    return df

def j_states_list(num_tss):
    i=0
    
    jm_list = []   # This will be the ordered list of the basis states
    j_index = {}   # This will be a python doctionary to allow us to easily find the rows/columns for a specific j

    # Get the j values for 2 TSS and order them high to low
    js = j_vals(num_tss)[::-1]
    
    for j in js:
        j_index[j] = []
        # for each j value get the different possible m's and order them high to low
        ms = m_vals(j)[::-1]
        for m in ms:
            j_index[j].append(i)
            jm_list.append((j,m))
            i+=1
    return j_index, jm_list


def simulate(H, psi0, times):
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