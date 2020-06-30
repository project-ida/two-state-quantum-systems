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

<a href="https://colab.research.google.com/github/project-ida/two-state-quantum-systems/blob/master/05-excitation-transfer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/05-excitation-transfer.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>


# 5 - Excitation transfer


> TODO: Description

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
# Functions created in 04 tutorial

from libs.helper_05_tutorial import *
```

> TODO: Need to motivate how the `make_operators`function needs to be modified

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
    
    j_index, jm_list = j_states_list(num_tss)
        
    try:
        j_index[j]
    except:
        raise Exception(f"j needs to be one of {j_vals(num_tss)}")
    
    Js = jspin(num_tss)
    Jx = Js[0]
    Jz = Js[2]
    

    
    num_ms = len(m_vals(j))
    Jz = Jz.extract_states(j_index[j])
    Jx = Jx.extract_states(j_index[j])
    jm_list = [jm_list[i] for i in j_index[j]]
    
    
    a        = tensor(destroy(max_bosons+1), qeye(num_ms))     # tensorised boson destruction operator
    number   = tensor(num(max_bosons+1), qeye(num_ms))         # tensorised boson number operator
    Jz       = tensor(qeye(max_bosons+1), Jz)                  # tensorised sigma_x operator 1
    Jx       = tensor(qeye(max_bosons+1), Jx)                  # tensorised sigma_x operator 1
    
    bosons         =   (number+0.5)                                # boson energy operator
    interaction  =    (a.dag() + a) * 2*Jx                        # interaction energy operator
    
    if(num_ms==1):
        interaction.dims = [[max_bosons+1,1],[max_bosons+1,1]]
    
    M = tensor(qeye(max_bosons+1),qdiags(m_vals(j)[::-1],0))    # M operator
    
    if((2*j)%2==0):
        P = (1j*np.pi*M).expm()*(1j*np.pi*number).expm()                  # parity operator 
    else:
        P = 1j*(1j*np.pi*M).expm()*(1j*np.pi*number).expm() 
    
    
    # map from QuTiP number states to |n,J, M> states
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
J= 3, M = 3

+++

J= 4, M = 4

++++
```

```python

```

```python

```

```python
Jz, bosons, interaction, number, njm_list, P = make_operators(6, 10, 3, 0)

df = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, Jz.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*Jz + 1*bosons + 0.2*interaction
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

Jz, bosons, interaction, number, njm_list, P = make_operators(10, 20, 5, 1)

df_even = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, Jz.shape[0])

for i, row in df_even.iterrows():
    H =  row["$\Delta E$"]*Jz + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df_even.iloc[i,1:] = evals 
```

```python
## ODD

Jz, bosons, interaction, number, njm_list, P = make_operators(10, 20, 5, -1)

df_odd = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, Jz.shape[0])

for i, row in df_odd.iterrows():
    H =  row["$\Delta E$"]*Jz + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df_odd.iloc[i,1:] = evals 
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6), sharey=True)


df_odd.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False, 
        title="Odd stationary states ($N=10$, $J=5$, $\omega=1$, $U=0.2$)     (Fig 4)",  ax=axes[0]);

df_even.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False, 
        title="Even stationary states ($N=10$, $J=5$, $\omega=1$, $U=0.2$)     (Fig 5)",  ax=axes[1]);

axes[0].set_ylabel("Energy");
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
def make_operators(max_bosons):
    
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
    
    # map from QuTiP number states to |n,±, ±> states
    possible_ns = range(0, max_bosons+1)
    possible_ms = ["+","-"]
    nm_list = [(n,m1,m2) for (n,m1,m2) in product(possible_ns, possible_ms, possible_ms)]
    
    
    return two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list
```

```python
def make_braket_labels(nm_list):
    bra_labels = ["$\langle$"+str(n)+", "+str(m1)+", "+str(m2)+" |" for (n,m1,m2) in nm_list]
    ket_labels = ["| "+str(n)+", "+str(m1)+", "+str(m2)+"$\\rangle$" for (n,m1,m2) in nm_list]
    return bra_labels, ket_labels
```

```python
two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list = make_operators(
    max_bosons=4)
```

```python
bra_labels, ket_labels = make_braket_labels(nm_list)
```

### Energy level landscape $U=0$


Just as in the previous tutorial, let's start with zero coupling to the bosons field to get a sense of the energy level landscape.

```python
df = make_df_for_energy_scan("$\Delta E$", -4,4, 201, two_state_1.shape[0])
```

```python
for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[-0.5,5.5],legend=False, 
        title="Stationary states ($\omega=1$, $U=0$)     (Fig 1)");
plt.ylabel("Energy");
```

>TODO: chat about the levels to give some intuition


Let's see how the energy levels cross and anti-cross when we switch the coupling on

```python
df = make_df_for_energy_scan("$\Delta E$", -4,4, 201, two_state_1.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.2*interaction_1 + 0.2*interaction_2
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[-0.5,5.5],legend=False, 
        title="Stationary states ($\omega=1$, $U=0.2$)     (Fig 2)");
plt.ylabel("Energy");
```

There are clearly some anti-crossings and also what look like crossings. From what we learnt last time about parity, we can suspect that the crossings correspond to universes that can't interact with each other.

We can make a guess for what parity might be for the cast of 2 TSS. To simplify visuals, we'll reduce the number of bosons.

```python
two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list = make_operators(
    max_bosons=2)

bra_labels, ket_labels = make_braket_labels(nm_list)
```

```python
sz1 = 2*two_state_1          
sz2 = 2*two_state_2 
parity = sz1*sz2*(1j*np.pi*number).expm()
```

```python
f, ax = hinton(parity, xlabels=ket_labels, ylabels=bra_labels)
ax.tick_params(axis='x',labelrotation=90,)
ax.set_title("Matrix elements of H     (Fig 3)");
```

Parity is either 1 or -1 very much like last time.

Taking an example Hamiltonian, we can check whether this "parity" commutes and if it does then it's conserved and we can use it to make out separate universes 

```python
H =  1*two_state_1+ 1*two_state_2 + 1*bosons + 0.2*interaction_1 + 0.2*interaction_2
```

```python
commutator(H,parity)
```

Lovely, we have a good definition of parity.

Let's augment make_operators as we did before.


## Separate parity universes

```python
def make_operators(max_bosons, parity=0):
    
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

Let's make separate energy level diagrams for even and odd parity

```python
# ODD PARITY

two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list, parity = make_operators(
    max_bosons=4, parity=-1)

df_odd = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state_1.shape[0])

for i, row in df_odd.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.2*interaction_1 + 0.2*interaction_2
    evals, ekets = H.eigenstates()
    df_odd.iloc[i,1:] = evals 
```

```python
# EVEN PARITY

two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list, parity = make_operators(
    max_bosons=4, parity=1)

df_even = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state_1.shape[0])

for i, row in df_even.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.2*interaction_1 + 0.2*interaction_2
    evals, ekets = H.eigenstates()
    df_even.iloc[i,1:] = evals 
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6), sharey=True)


df_odd.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False, 
        title="Odd stationary states ($\omega=1$, $U=0.2$)     (Fig 4)",  ax=axes[0]);

df_even.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False, 
        title="Even stationary states ($\omega=1$, $U=0.2$)     (Fig 5)",  ax=axes[1]);

axes[0].set_ylabel("Energy");
```

What's interesting about these diagrams is that it appears like we might still have some crossings. 

This is because parity isn't the only thing that's conserved - the total spin is as well.


## Total Spin

```python
def make_operators(max_bosons, parity=0):
    
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
    
    S2 = (sx1/2+sx2/2)**2 + (sy1/2+sy2/2)**2 + (sz1/2+sz2/2)**2    # Total "spin"^2 operator
    
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
        S2              = S2.extract_states(p)
        P               = P.extract_states(p)
        nm_list         = [nm_list[i] for i in p]
    
    
    return two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list, P, S2
```

```python
two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list, parity, S2 = make_operators(
    max_bosons=4, parity=-1)


spin = S2.eigenstates()[1]
p   = np.where(S2.eigenenergies()==2)[0]


df_odd = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, p.shape[0])
# df_odd = make_df_for_energy_scan("$\Delta E$", 2.75,2.8, 201, p.shape[0])


for i, row in df_odd.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.2*interaction_1 + 0.2*interaction_2
    test = H.transform(S2.eigenstates()[1])
    test = test.extract_states(p)
    evals, ekets = test.eigenstates()
    df_odd.iloc[i,1:] = evals 
```

```python
df_odd.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False,  figsize=(8,6),
        title="Odd stationary states ($\omega=1$, $U=0.2$)     (Fig 10)");
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
H = 1*two_state_1+ 1*two_state_2 + 1*bosons
```

```python
f, ax = hinton(H, xlabels=ket_labels, ylabels=bra_labels)
ax.tick_params(axis='x',labelrotation=90,)
ax.set_title("Matrix elements of H     (Fig 5)");
```

```python
f, ax = hinton(S2, xlabels=ket_labels, ylabels=bra_labels)
ax.tick_params(axis='x',labelrotation=90,)
ax.set_title("Matrix elements of H     (Fig 5)");
```

```python
f, ax = hinton(parity, xlabels=ket_labels, ylabels=bra_labels)
ax.tick_params(axis='x',labelrotation=90,)
ax.set_title("Matrix elements of H     (Fig 5)");
```

```python
H = 3*two_state_1+ 3*two_state_2 + 1*bosons + 0.001*interaction_1 + 0.001*interaction_2
```

```python
nm_list
```

```python
psi0 = basis(len(nm_list), 1)
```

```python
times = np.linspace(0.0, 20000000.0, 10000) # simulation time
P, psi = simulate(H, psi0, times)
```

```python
plt.figure(figsize=(10,8))
for i in range(0,P.shape[0]):
    plt.plot(times, P[i,:], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title("($\Delta E=3$, $\omega=1$, $U=0.001$)     (Fig 12)")
plt.show();
```

```python
two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list, parity, S2 = make_operators(max_bosons=4)
```

```python
bra_labels, ket_labels = make_braket_labels(nm_list)
```

```python
df = make_df_for_energy_scan("$\Delta E$", -4,4, 201, two_state_1.shape[0])
```

```python
for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.001*interaction_1 + 0.001*interaction_2
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[-0.5,5.5],legend=False, 
        title="Stationary states ($\omega=1$, $U=0.001$)     (Fig 2)");
plt.ylabel("Energy");
```

```python
df = make_df_for_energy_scan("$\Delta E$", 2,4, 201, two_state_1.shape[0])
for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.001*interaction_1 + 0.001*interaction_2
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[-0.5,5.5],legend=False, 
        title="Stationary states ($\omega=1$, $U=0.001$)     (Fig 2)");
plt.ylabel("Energy");
```

```python
df = make_df_for_energy_scan("$\Delta E$", -4,4, 201, two_state_1.shape[0])
for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.2*interaction_1 + 0.2*interaction_2
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[-0.5,5.5],legend=False, 
        title="Stationary states ($\omega=1$, $U=0.001$)     (Fig 2)");
plt.ylabel("Energy");
```

```python
df = make_df_for_energy_scan("$\Delta E$", 2.75,2.8, 201, two_state_1.shape[0])
for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.2*interaction_1 + 0.2*interaction_2
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[0.47,0.6],legend=False, 
        title="Stationary states ($\omega=1$, $U=0.001$)     (Fig 2)");
plt.ylabel("Energy");
```

```python

```

```python
# ODD PARITY

two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list, parity, S2 = make_operators(max_bosons=4, parity=-1)

df_odd = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state_1.shape[0])

for i, row in df_odd.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.001*interaction_1 + 0.001*interaction_2
    evals, ekets = H.eigenstates()
    df_odd.iloc[i,1:] = evals 
```

```python
# EVEN PARITY

two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list, parity, S2 = make_operators(max_bosons=4, parity=1)

df_even = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state_1.shape[0])

for i, row in df_even.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.001*interaction_1 + 0.001*interaction_2
    evals, ekets = H.eigenstates()
    df_even.iloc[i,1:] = evals 
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6), sharey=True)


df_odd.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False, 
        title="Odd stationary states ($\omega=1$, $U=0.001$)     (Fig 9)",  ax=axes[0]);

df_even.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False, 
        title="Even stationary states ($\omega=1$, $U=0.001$)     (Fig 10)",  ax=axes[1]);

axes[0].set_ylabel("Energy");
```

```python

```

```python
# ODD PARITY

two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list, parity, S2 = make_operators(max_bosons=4, parity=-1)

df_odd = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state_1.shape[0])

for i, row in df_odd.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.1*interaction_1 + 0.1*interaction_2
    evals, ekets = H.eigenstates()
    df_odd.iloc[i,1:] = evals 
```

```python
# EVEN PARITY

two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list, parity, S2 = make_operators(max_bosons=4, parity=1)

df_even = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state_1.shape[0])

for i, row in df_even.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.1*interaction_1 + 0.1*interaction_2
    evals, ekets = H.eigenstates()
    df_even.iloc[i,1:] = evals 
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6), sharey=True)


df_odd.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False, 
        title="Odd stationary states ($\omega=1$, $U=0.1$)     (Fig 9)",  ax=axes[0]);

df_even.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False, 
        title="Even stationary states ($\omega=1$, $U=0.1$)     (Fig 10)",  ax=axes[1]);

axes[0].set_ylabel("Energy");
```

```python
df_even.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False,  figsize=(8,6),
        title="Even stationary states ($\omega=1$, $U=0.1$)     (Fig 10)");
```

```python
two_state_1, two_state_2, bosons, interaction_1, interaction_2, number, nm_list, parity, S2 = make_operators(max_bosons=4, parity=-1)


spin = S2.eigenstates()[1]
p   = np.where(S2.eigenenergies()==2)[0]


df_odd = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, p.shape[0])
# df_odd = make_df_for_energy_scan("$\Delta E$", 2.75,2.8, 201, p.shape[0])





for i, row in df_odd.iterrows():
    H =  row["$\Delta E$"]*two_state_1+ row["$\Delta E$"]*two_state_2 + 1*bosons + 0.001*interaction_1 + 0.001*interaction_2
    test = H.transform(S2.eigenstates()[1])
    test = test.extract_states(p)
    evals, ekets = test.eigenstates()
    df_odd.iloc[i,1:] = evals 
```

```python
nm_list
```

```python
df_odd.plot(x="$\Delta E$",ylim=[-0.5,5.5],legend=False,  figsize=(8,6),
        title="Odd stationary states ($\omega=1$, $U=0.1$)     (Fig 10)");
```

```python
df_odd.plot(x="$\Delta E$",figsize=(10,8),ylim=[0.47,0.6],legend=False, 
        title="Stationary states ($\omega=1$, $U=0.001$)     (Fig 2)");
plt.ylabel("Energy");
```

>TODO: Need to look at 

https://www2.ph.ed.ac.uk/~ldeldebb/docs/QM/lect15.pdf - addition of angular momentum

https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients

http://www.lassp.cornell.edu/clh/p654/MM-Lec0.pdf

https://quantummechanics.ucsd.edu/ph130a/130_notes/node312.html

```python
sigmax()**2+sigmay()**2+sigmaz()**2
```

```python
sx1 = tensor(sigmax()/2, qeye(2))
sy1 = tensor(sigmay()/2, qeye(2))
sz1 = tensor(sigmaz()/2, qeye(2))
```

```python
sx2 = tensor(qeye(2), sigmax()/2)
sy2 = tensor(qeye(2), sigmay()/2)
sz2 = tensor(qeye(2), sigmaz()/2)
```

```python
S2 = (sx1+sx2)**2 + (sy1+sy2)**2 + (sz1+sz2)**2
```

```python
hinton(S2)
```

```python
S2
```

```python
S2.eigenstates()
```

```python
1/np.sqrt(2)
```

```python

```

```python

```
