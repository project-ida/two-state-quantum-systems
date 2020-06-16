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
import warnings
warnings.filterwarnings('ignore')
from itertools import product
import os
```

```python
spins = spin_algebra(2)
```

```python
spins[2]
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
Jz1 = Jz.extract_states(j_index[1])
Jx1 = Jx.extract_states(j_index[1])
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
