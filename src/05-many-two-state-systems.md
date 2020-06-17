---
jupyter:
  jupytext:
    formats: ipynb,src//md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<a href="https://colab.research.google.com/github/project-ida/two-state-quantum-systems/blob/master/05-many-two-state-systems.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/05-many-two-state-systems.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>


# 5 - Many two state systems


> TODO: Intro

```python
# Libraries
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import Image
import gif
import numpy as np
import pandas as pd
from qutip import *
from qutip.piqs import *
from qutip.cy.piqs import j_min, j_vals, m_vals
import warnings
warnings.filterwarnings('ignore')
from itertools import product
import os
from fractions import Fraction
```

```python
spins = spin_algebra(2)
```

```python
spins[2]
```

```python
A=0.1
```

```python
H = spins[2][0] + spins[2][1] + A*spins[0][0] + A*spins[0][1]
```

```python
H
```

```python
def make_df_for_energy_scan(label_param, min_param, max_param, num_param, num_levels):
    
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
```

```python
df = make_df_for_energy_scan("$\delta$/A", -40,40, 100, spins[2][0].shape[0])
```

```python
spins[2][1]+spins[2][0]

```

```python
A=0.1
for i, row in df.iterrows():
    H =  row[ "$\delta$/A"]*A*spins[2][0] + row[ "$\delta$/A"]*A*spins[2][1] + A*spins[0][0] + A*spins[0][1]
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\delta$/A",figsize=(10,8),legend=True, 
        title="Stationary states ($\omega=1$, $U=0.2$)     (Fig 2)");
plt.ylabel("Energy");
```

```python
delE = 1
```

```python
A = 0.01
```

```python
H = delE*spins[2][0] + delE*spins[2][1] + A*spins[0][0] + A*spins[0][1]
```

```python
H
```

```python
evals, ekets = H.eigenstates()
```

```python
evals
```

```python
ekets
```

```python
J = jspin(2, basis="uncoupled")
```

```python
Jx = J[0]
Jy = J[1]
Jz = J[2]
```

```python
H = delE*Jz + A*Jx
```

```python
H
```

```python
J2 = Jx**2 + Jy**2 + Jz**2
```

```python
J2
```

```python
commutator(H,J2)
```

```python
J2.transform(ekets)
```

```python
J = jspin(2)
```

```python
Jx = J[0]
Jy = J[1]
Jz = J[2]
```

```python
A = 0.01
```

```python
H = delE*Jz + A*Jx
```

```python
H
```

```python
J2 = Jx**2 + Jy**2 + Jz**2
```

```python
J2
```

```python
from qutip.cy.piqs import j_min, j_vals, m_vals
```

```python

```

```python

```

```python
N = 2
i=0

nm_list = []
j_index = {}

js = j_vals(N)[::-1]
for j in js:
    j_index[j] = []
    ms = m_vals(j)[::-1]
    for m in ms:
        j_index[j].append(i)
        nm_list.append((j,m))
        i+=1
```

```python
nm_list
```

```python
def make_braket_labels(nm_list):
    bra_labels = ["$\langle$"+str(n)+", "+str(m)+" |" for (n,m) in nm_list]
    ket_labels = ["| "+str(n)+", "+str(m)+"$\\rangle$" for (n,m) in nm_list]
    return bra_labels, ket_labels
```

```python
bra_labels, ket_labels = make_braket_labels(nm_list)
```

```python

```

```python

```

```python
df = make_df_for_energy_scan("$\delta$/A", -4,4, 100, Jz1.shape[0])
```

```python
A=0.1
for i, row in df.iterrows():
    H =  row[ "$\delta$/A"]*A*Jz1 + A*Jx1
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\delta$/A",figsize=(10,8),legend=True, 
        title="Stationary states ($\omega=1$, $U=0.2$)     (Fig 2)");
plt.ylabel("Energy");
```

[Wolfram alpha eigenvalues](https://www.wolframalpha.com/input/?i=%7B%7B1%2Ca%2C0%7D%2C%7Ba%2C0%2Ca%7D%2C%7B0%2Ca%2C-1%7D%7D+eigenvalues)

```python

```

```python

```

```python
J = jspin(2)
```

```python
Jx = J[0]
Jz = J[2]
```

```python

```

```python
def j_states_list(num_tss):
    i=0
    
    jm_list = []
    j_index = {}

    js = j_vals(N)[::-1]
    
    for j in js:
        j_index[j] = []
        ms = m_vals(j)[::-1]
        for m in ms:
            j_index[j].append(i)
            jm_list.append((j,m))
            i+=1
    return j_index, jm_list
```

```python
j_index, jm_list = j_states_list(2)
```

```python
j=0
Jz_j = Jz.extract_states(j_index[j])
Jx_j = Jx.extract_states(j_index[j])
```

```python
max_bosons = 2
number   = tensor(num(max_bosons+1), qeye(Jz_j.shape[0]))
Jz_tensor = tensor(qeye(max_bosons+1), Jz_j)
Jx_tensor = tensor(qeye(max_bosons+1), Jx_j)
a = tensor(destroy(max_bosons+1), qeye(Jz_j.shape[0]))

interaction  =    (a.dag() + a) * Jx_tensor

bosons = number+0.5
```

```python
from qutip.dimensions import (
    type_from_dims, flatten, unflatten, enumerate_flat, deep_remove, deep_map, is_scalar, collapse_dims_oper)
```

```python
(a.dag() + a)
```

```python
Jz_tensor
```

```python
tensor_contract(Jz_tensor)
```

```python
test = (a.dag() + a)*Jx_tensor
```

```python
test.dims[0][0]
```

```python

```

```python

```

```python
collapse_dims_oper
```

```python
interaction.dims = [[3,1],[3,1]]
```

```python
interaction
```

```python
def j_states_list(num_tss):
    i=0
    
    jm_list = []
    j_index = {}

    js = j_vals(num_tss)[::-1]
    
    for j in js:
        j_index[j] = []
        ms = m_vals(j)[::-1]
        for m in ms:
            j_index[j].append(i)
            jm_list.append((j,m))
            i+=1
    return j_index, jm_list
```

```python
def make_braket_labels(njm_list):
    bra_labels = ["$\langle$"+str(n)+", "+str(Fraction(j))+", "+str(Fraction(m))+" |" for (n,j,m) in njm_list]
    ket_labels = ["| "+str(n)+", "+str(Fraction(j))+", "+str(Fraction(m))+"$\\rangle$" for (n,j,m) in njm_list]
    return bra_labels, ket_labels
```

```python
def make_operators(num_tss, max_bosons, j, parity=0):
    
    Js = jspin(num_tss)
    Jx = Js[0]
    Jz = Js[2]
    
    j_index, jm_list = j_states_list(num_tss)
    
    num_ms = len(m_vals(j))
    Jz = Jz.extract_states(j_index[j])
    Jx = Jx.extract_states(j_index[j])
    jm_list = [jm_list[i] for i in j_index[j]]
    
    
    a        = tensor(destroy(max_bosons+1), qeye(num_ms))     # tensorised boson destruction operator
    number   = tensor(num(max_bosons+1), qeye(num_ms))         # tensorised boson number operator
    Jz       = tensor(qeye(max_bosons+1), Jz)                  # tensorised sigma_x operator 1
    Jx       = tensor(qeye(max_bosons+1), Jx)                  # tensorised sigma_x operator 1
    
    bosons         =   (number+0.5)                                # boson energy operator
    interaction  =    (a.dag() + a) * Jx                        # interaction energy operator
    
    if(num_ms==1):
        interaction.dims = [[max_bosons+1,1],[max_bosons+1,1]]
    
    M = tensor(qeye(max_bosons+1),qdiags(m_vals(j)[::-1],0))    # M operator
    
    if((2*j)%2==0):
        P = (1j*np.pi*M).expm()*(1j*np.pi*number).expm()                  # parity operator 
    else:
        P = 1j*(1j*np.pi*M).expm()*(1j*np.pi*number).expm() 
    
    
    # map from QuTiP number states to |n,±, ±> states
    possible_ns = range(0, max_bosons+1)
    njm_list = [(n,j,m) for (n,(j,m)) in product(possible_ns, jm_list)]
    
    # only do parity extraction if a valid parity is being used
    if (parity==1) | (parity==-1):
        p           = np.where(P.diag()==parity)[0]
        
        Jz     = Jz.extract_states(p)
        Jx     = Jx.extract_states(p)
        bosons          = bosons.extract_states(p)
        number          = number.extract_states(p)
        interaction   = interaction.extract_states(p)
        P               = P.extract_states(p)
        njm_list         = [njm_list[i] for i in p]
    
    
    return Jz, bosons, interaction, number, njm_list, P
```

```python

```

```python
n = 5
```

```python
Jx = jspin(n,"x")
```

```python
Jx
```

```python
Jp = jspin(n,"+")
Jm = jspin(n,"-")
```

```python
(Jp+Jm)
```

```python
j_index, jm_list = j_states_list(3)
```

```python
j_index
```

```python
jm_list
```

```python
j_vals(3)
```

```python
Js = jspin(3)
Jx = Js[0]
Jz = Js[2]
```

```python
j = 0.5
```

```python
Jz = Jz.extract_states(j_index[j])
```

```python
Jz
```

```python
Jz, bosons, interaction, number, njm_list, P = make_operators(6, 10, 3, 0)

df = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, Jz.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*Jz + 1*bosons + 0.2*interaction*2
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals 
```

```python
df.plot(x="$\Delta E$",figsize=(10,8),legend=False, ylim=[-0.5,5.5],
        title="Stationary states ($\omega=1$, $U=0.2$)     (Fig 2)");
plt.ylabel("Energy");
```

```python
## EVEN

Jz, bosons, interaction, number, njm_list, P = make_operators(6, 10, 3, 1)

df_even = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, Jz.shape[0])

for i, row in df_even.iterrows():
    H =  row["$\Delta E$"]*Jz + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df_even.iloc[i,1:] = evals 
```

```python
## ODD

Jz, bosons, interaction, number, njm_list, P = make_operators(6, 10, 3, -1)

df_odd = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, Jz.shape[0])

for i, row in df_odd.iterrows():
    H =  row["$\Delta E$"]*Jz + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df_odd.iloc[i,1:] = evals 
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6), sharey=True)


df_odd.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False, 
        title="Odd stationary states ($\omega=1$, $U=0.2$)     (Fig 4)",  ax=axes[0]);

df_even.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False, 
        title="Even stationary states ($\omega=1$, $U=0.2$)     (Fig 5)",  ax=axes[1]);

axes[0].set_ylabel("Energy");
```

```python
bra_labels, ket_labels = make_braket_labels(njm_list)
```

```python
f, ax = hinton(H, xlabels=ket_labels, ylabels=bra_labels)
ax.tick_params(axis='x',labelrotation=90,)
ax.set_title("Matrix elements of H     (Fig 3)");
```

```python
df.plot(x="$\Delta E$",figsize=(10,8),legend=True, ylim=[-0.5,5.5],
        title="Stationary states ($\omega=1$, $U=0.2$)     (Fig 2)");
plt.ylabel("Energy");
```

```python

```

```python

```

```python

```

```python
jm_list
```

```python
 possible_ns = range(0, max_bosons+1)
```

```python
max_bosons = 2
```

```python
njm_list = [(n,j,m) for (n,(j,m)) in product(possible_ns, jm_list)]
```

```python
njm_list
```

```python
product([range(0, max_bosons+1),range(0, max_bosons+1)])
```

```python
df = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, Jz_tensor.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*Jz_tensor + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals 
```

```python
df.plot(x="$\Delta E$",figsize=(10,8),legend=True, ylim=[-0.5,5.5],
        title="Stationary states ($\omega=1$, $U=0.2$)     (Fig 2)");
plt.ylabel("Energy");
```

```python
Mt = tensor(qeye(max_bosons+1),qdiags(m_vals(1)[::-1],0))
```

```python
Mt
```

```python
qeye(3)
```

```python
m_vals(1)[::-1]*qeye(3)
```

```python
P = (1j*np.pi*Mt).expm()*(1j*np.pi*number).expm()
```

```python
P
```

```python
commutator(P,H)
```

```python
def make_operators(max_bosons, parity=0, J):
    
    a        = tensor(destroy(max_bosons+1), qeye(2), qeye(2))     # tensorised boson destruction operator
    number   = tensor(num(max_bosons+1), qeye(2), qeye(2))         # tensorised boson number operator
    sx1      = tensor(qeye(max_bosons+1), sigmax(), qeye(2))       # tensorised sigma_x operator 1
    sx2      = tensor(qeye(max_bosons+1), qeye(2), sigmax())       # tensorised sigma_x operator 2
    sz1      = tensor(qeye(max_bosons+1), sigmaz(), qeye(2))        # tensorised sigma_z operator 1 
    sz2      = tensor(qeye(max_bosons+1), qeye(2), sigmaz())        # tensorised sigma_z operator 2
    sy1      = tensor(qeye(max_bosons+1), sigmay(), qeye(2))       # tensorised sigma_x operator 1
    sy2      = tensor(qeye(max_bosons+1), qeye(2), sigmay())       # tensorised sigma_x operator 2
    
    two_state_1    =    1/2*sz1                                    # two state system energy operator 1
    two_state_2    =    1/2*sz2                                    # two state system energy operator 2
    bosons         =   (number+0.5)                                # boson energy operator
    interaction_1  =    (a.dag() + a) * sx1                        # interaction energy operator 1
    interaction_2  =    (a.dag() + a) * sx2                        # interaction energy operator 2 
    
    P = sz1*sz2*(1j*np.pi*number).expm()                           # parity operator 
    
    # map from QuTiP number states to |n,±, ±> states
    possible_ns = range(0, max_bosons+1)
    possible_ms = ["+","-"]
    nm_list = [(n,m1,m2) for (n,m1,m2) in product(possible_ns, possible_ms, possible_ms)]
    
    # only do parity extraction if a valid parity is being used
    if (parity==1) | (parity==-1):
        p           = np.where(P.diag()==parity)[0]
        
        two_state_1     = two_state_1.extract_states(p)
        two_state_2     = two_state_2.extract_states(p)
        bosons          = bosons.extract_states(p)
        number          = number.extract_states(p)
        interaction_1   = interaction_1.extract_states(p)
        interaction_2   = interaction_2.extract_states(p)
        P               = P.extract_states(p)
        nm_list         = [nm_list[i] for i in p]
    
    
    return two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list, P
```
