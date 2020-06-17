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

# Functions created in 04 tutorial

from libs.helper_05_tutorial import *
```

```python
spins = spin_algebra(2)
```

```python
spins[2][0]
```

```python
spins[2]
```

```python
spins[2][0]*tensor(basis(2,1), basis(2,1))
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
Jx = spins[0][0] + spins[0][1]
Jy = spins[1][0] + spins[1][1]
Jz = spins[2][0] + spins[2][1]
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

```

```python

```

[Wolfram alpha eigenvalues](https://www.wolframalpha.com/input/?i=%7B%7B1%2Ca%2C0%7D%2C%7Ba%2C0%2Ca%7D%2C%7B0%2Ca%2C-1%7D%7D+eigenvalues)

```python

```

```python

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

```

```python

```

```python
def make_braket_labels(jm_list):
    bra_labels = ["$\langle$"+str(Fraction(j))+", "+str(Fraction(m))+" |" for (j,m) in jm_list]
    ket_labels = ["| "+str(Fraction(j))+", "+str(Fraction(m))+"$\\rangle$" for (j,m) in jm_list]
    return bra_labels, ket_labels
```

```python

```

```python

```

```python

```

```python
def make_operators(num_tss, j):
    
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
    
    
    
    return Jz, Jx, jm_list
```

```python
Jz, Jx, jm_list = make_operators(6, 3)
```

```python
df = make_df_for_energy_scan("$\Delta E$/A", -4,4, 100, Jz.shape[0])
```

```python
A=0.1
for i, row in df.iterrows():
    H =  row[ "$\Delta E$/A"]*A*Jz +  A*Jx
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\Delta E$/A",figsize=(10,8),legend=True, 
        title="$H=\Delta E \ J_z + A \ J_x$    (N=6, J=3, A=0.1) ");
plt.ylabel("Energy");
```

```python
bra_labels, ket_labels = make_braket_labels(jm_list)
```

```python
# Function created in 01 tutorial to make plotting of states calculated from `sesolve` easier
def states_to_df(states,times):
    psi_plus = np.zeros(len(times),dtype="complex128")  # To store the amplitude of the |+> state
    psi_minus = np.zeros(len(times),dtype="complex128") # To store the amplitude of the |-> state

    for i, state in enumerate(states):
        psi_plus[i] = state[0][0][0]
        psi_minus[i] = state[1][0][0]

    return pd.DataFrame(data={"+":psi_plus, "-":psi_minus}, index=times)
```

```python
delta = 0.001
A = 0.1

H0 = A*Jx

evals, estates = H0.eigenstates()

H1 =  delta*Jz

H = [H0,[H1,'cos(w*t)']]

times = np.linspace(0.0, 10000.0, 1000) 

# psi0=basis(7,0)
psi0 = estates[6]

result = sesolve(H, psi0, times, args={'w':A})

# result = sesolve(H0, basis(7,0), times)


```

```python
num_states = result.states[0].shape[0]
psi = np.zeros([num_states,times.size], dtype="complex128")
P = np.zeros([num_states,times.size], dtype="complex128")

for i, state in enumerate(result.states):
    transformed_state = state.transform(estates)
    psi[:,i] = np.transpose(transformed_state)
    P[:,i] = np.abs(psi[:,i]*np.conj(psi[:,i]))
```

```python
plt.figure(figsize=(10,8))
for i in range(0,P.shape[0]):
    plt.plot(times, P[i,:], label=f"E_level_{i}")
plt.ylabel("Probability")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title("$H =A \ J_x + \delta \ J_z \  \cos (\omega t)$     (N=6, J=3, A=0.1, $\omega = 0.1$, $\delta=0.001$)")
plt.show();
```

```python
H0 = A*Jx
```

```python
evals, estates = H0.eigenstates()
```

```python
evals
```

```python
estates
```

```python
plot_fock_distribution(estates[2])
```

```python

```

```python
jm_list
```

```python

```
