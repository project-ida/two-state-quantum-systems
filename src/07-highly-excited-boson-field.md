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

<a href="https://colab.research.google.com/github/project-ida/two-state-quantum-systems/blob/master/05-highly-excited-boson-field.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/05-highly-excited-boson-field.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>


# 5 - Highly excited boson field


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
import warnings
warnings.filterwarnings('ignore')
from itertools import product
import os

# Functions created in 04 tutorial

from libs.helper_05_tutorial import *

H_latex = "$H = (\Delta E / 2) \sigma_z + \hbar\omega(a^{{\dagger}}a +1/2) + U( a^{{\dagger}} + a )\sigma_x$"

```

If we are only interested in high boson numbers and don't want to incur the cost of simulation the low ones as well then we need to modify `make_operators` from the last tutorial.

```python

def make_operators(max_bosons, min_bosons, parity=0):
    
    num_boson_states = max_bosons - min_bosons
    
    a       = tensor(destroy(num_boson_states+1 ,min_bosons), qeye(2))   # tensorised boson destruction operator
    number  = tensor(num(num_boson_states+1, min_bosons), qeye(2))       # tensorised boson number operator
    sx      = tensor(qeye(num_boson_states+1), sigmax())                 # tensorised sigma_x operator
    sz      = tensor(qeye(num_boson_states+1),sigmaz())                  # tensorised sigma_z operator
    
    two_state     =  1/2*sz                                             # two state system energy operator
    bosons       =  (number+0.5)                                        # boson energy operator
    interaction  = (a.dag() + a) * sx                                   # interaction energy operator      
    
    P = sz*(1j*np.pi*number).expm()                                     # parity operator 
    
    # map from QuTiP number states to |n,±> states
    possible_ns = range(min_bosons, max_bosons+1)
    possible_ms = ["+","-"]
    nm_list = [(n,m) for (n,m) in product(possible_ns, possible_ms)]

    
    # only do parity extraction if a valid parity is being used
    if (parity==1) | (parity==-1):
        p           = np.where(P.diag()==parity)[0]
        
        two_state   = two_state.extract_states(p)
        bosons      = bosons.extract_states(p)
        number      = number.extract_states(p)
        interaction = interaction.extract_states(p)
        nm_list     = [nm_list[i] for i in p]
    
    
    return two_state, bosons, interaction, number, nm_list
```

We'll start with `min_bosons` = 2400 and go up to `max_bosons` = 2800 and look at even parity.

```python
two_state, bosons, interaction, number, nm_list = make_operators(2800,2400,1)

df = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

We'll only plot the levels that are away from plotting close to energies associated with the max and min boson numbers. This is so that we don't get any boundary effects in the levels. 

We didn't worry about this last time as it would be too much to talk about. But we need to get more serious about it now.

```python
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[2500,2700],legend=False, 
        title=f"Stationary states ($\omega=1$, $U=0.2$), n_max={min_bosons+200}, n_min ={min_bosons}      (Fig 1)");
plt.ylabel("Energy");
```

Something seems strange - the levels appear to "thin" out as we go up in energy. It's hard to see at this scale so let's create an animation that zooms in and slides up the axis

```python

@gif.frame
def plot(ymin,ymax):
    df.plot(x="$\Delta E$",figsize=(10,8),ylim=[ymin,ymax],legend=False, 
            title=f"Stationary states ($\omega=1$, $U=0.2$), n_max=2800, n_min =2400");
    plt.ylabel("Energy");

num_frames = int((2675-2525)/1)

frames = []

ylims = np.array([2500,2525])
for i in range(num_frames):
    frame = plot(ylims[0], ylims[1])
    frames.append(frame)
    ylims+=1
gif.save(frames, "./img/level_scan.gif", duration=500)    
```

```python
os.rename("./img/level_scan.gif", "./img/level_scan.gif.png")
```

```python
Image(filename="./img/level_scan.gif.png")
```

I don't understand what is going on in this animation at all.


Let's make a few snap shots of the animation to study more.

```python jupyter={"source_hidden": true}
min_bosons = 2400

two_state, bosons, interaction, number, nm_list = make_operators(min_bosons+200,min_bosons,1)

df = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
    
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[min_bosons+100,min_bosons+120],legend=False, 
        title=f"Stationary states ($\omega=1$, $U=0.2$), n_max={min_bosons+200}, n_min ={min_bosons}      (Fig 2)");
plt.ylabel("Energy");
```

```python jupyter={"source_hidden": true}
min_bosons = 2450

two_state, bosons, interaction, number, nm_list = make_operators(min_bosons+200,min_bosons,1)

df = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
    
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[min_bosons+100,min_bosons+120],legend=False, 
        title=f"Stationary states ($\omega=1$, $U=0.2$), n_max={min_bosons+200}, n_min ={min_bosons}      (Fig 2)");
plt.ylabel("Energy");
```

```python jupyter={"source_hidden": true}
min_bosons = 2500

two_state, bosons, interaction, number, nm_list = make_operators(min_bosons+200,min_bosons,1)

df = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
    
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[min_bosons+100,min_bosons+120],legend=False, 
        title=f"Stationary states ($\omega=1$, $U=0.2$), n_max={min_bosons+200}, n_min ={min_bosons}      (Fig 2)");
plt.ylabel("Energy");
```

```python jupyter={"source_hidden": true}
min_bosons = 2600

two_state, bosons, interaction, number, nm_list = make_operators(min_bosons+200,min_bosons,1)

df = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
    
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[min_bosons+100,min_bosons+120],legend=False, 
        title=f"Stationary states ($\omega=1$, $U=0.2$), n_max={min_bosons+200}, n_min ={min_bosons}      (Fig 2)");
plt.ylabel("Energy");
```

Let's try and repeat that last figure but changing the max and min number of bosons - do we still get the same pattern?

```python
two_state, bosons, interaction, number, nm_list = make_operators(2900,2500,1)

df = make_df_for_energy_scan("$\Delta E$", -4, 4, 201, two_state.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
    
df.plot(x="$\Delta E$",figsize=(10,8),ylim=[2700,2720],legend=False, 
        title=f"Stationary states ($\omega=1$, $U=0.2$), n_max=2900, n_min =2500      (Fig 2)");
plt.ylabel("Energy");
```

We do get the same pattern which suggests it's not a boundary effect

```python
two_state, bosons, interaction, number, nm_list = make_operators(600,500,1)

df = make_df_for_energy_scan("$\Delta E$", -10, 10, 201, two_state.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
    

```

```python
df.plot(x="$\Delta E$",figsize=(10,8),legend=False, 
        title=f"Stationary states ($\omega=1$, $U=0.2$),     (Fig 2)");
plt.ylabel("Energy");
```

```python
two_state, bosons, interaction, number, nm_list = make_operators(3100,3000,1)

df = make_df_for_energy_scan("$\Delta E$", -40, 40, 201, two_state.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\Delta E$",figsize=(10,8),legend=False, ylim=[3020,3080],
        title=f"Stationary states ($\omega=1$, $U=0.2$)     (Fig 1)");
plt.ylabel("Energy");
```

```python
two_state, bosons, interaction, number, nm_list = make_operators(3100,3000,1)

df = make_df_for_energy_scan("$\Delta E$", -40, 40, 201, two_state.shape[0])

for i, row in df.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\Delta E$",figsize=(10,8),legend=False, ylim=[3020,3080],
        title=f"Stationary states ($\omega=1$, $U=0.2$)     (Fig 1)");
plt.ylabel("Energy");
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
two_state, bosons, interaction, number, nm_list = make_operators(1000,0,1)

df_no_coupling = make_df_for_energy_scan("$\Delta E$", -20, 20, 201, two_state.shape[0])
df_coupling = make_df_for_energy_scan("$\Delta E$", -20, 20, 201, two_state.shape[0])


```

```python
for i, row in df_no_coupling.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0*interaction
    evals, ekets = H.eigenstates()
    df_no_coupling.iloc[i,1:] = evals
```

```python
for i, row in df_coupling.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0.2*interaction
    evals, ekets = H.eigenstates()
    df_coupling.iloc[i,1:] = evals
```

```python jupyter={"outputs_hidden": true}
fig, axes = plt.subplots( figsize=(15,800), sharey=True)


df_no_coupling.plot(x="$\Delta E$",legend=False, ylim=[-10,1015],
        title="Stationary states ($\omega=1$, $U=0$)     (Fig 9)",  ax=axes, alpha=0.3);

plt.gca().set_prop_cycle(None)

df_coupling.plot(x="$\Delta E$",legend=False, ylim=[-10,1015],
        title="Stationary states ($\omega=1$, $U=0.2$)     (Fig 10)",  ax=axes);

axes.yaxis.set_major_locator(ticker.MultipleLocator(5))

axes.set_ylabel("Energy");
```

```python
fig, axes = plt.subplots( figsize=(10,8), sharey=True)


df_no_coupling.plot(x="$\Delta E$",legend=False,ylim=[0,20], ax=axes,
        title="Stationary states ($\omega=1$, $U=0$)     (Fig 9)", alpha=0.3);

plt.gca().set_prop_cycle(None)

df_coupling.plot(x="$\Delta E$",legend=False, ylim=[0,20], ax=axes,
        title=f"Stationary states ($\omega=1$, $U=0.2$),     (Fig 1)");

plt.ylabel("Energy");
```

```python
fig, axes = plt.subplots( figsize=(10,8), sharey=True)


df_no_coupling.plot(x="$\Delta E$",legend=False,ylim=[-0.5,5.5],xlim=[-4,4], ax=axes,
        title="Stationary states ($\omega=1$, $U=0$)     (Fig 9)", alpha=0.3);

plt.gca().set_prop_cycle(None)

df_coupling.plot(x="$\Delta E$",legend=False, ylim=[-0.5,5.5],xlim=[-4,4], ax=axes,
        title=f"Stationary states ($\omega=1$, $U=0.2$),     (Fig 1)");

plt.ylabel("Energy");


```

### Let's look at n=9 with $\Delta E=0$ and see what we find

```python
two_state, bosons, interaction, number, nm_list = make_operators(100,0,1)
```

```python
bra_labels, ket_labels = make_braket_labels(nm_list)
```

```python
H =  1*bosons + 0.2*interaction
```

```python
psi0 = basis(len(nm_list), 9)
```

```python
times = np.linspace(0.0, 10.0, 1000)
P, psi = simulate(H, psi0, times)
```

```python
plt.figure(figsize=(10,8))
# for i in range(0,P.shape[0]):
for i in range(0,15):
    plt.plot(times, P[i,:], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title(f" {H_latex}    ($\Delta E=0$, $\omega=1$, $U=0.2$) ")
plt.show();
```

```python
hamiltonian_expect = expectation(H,psi)
two_state_expect = expectation(0*two_state,psi)
bosons_expect = expectation(1*bosons,psi)
interaction_expect = expectation(0.2*interaction,psi)
```

```python
plt.figure(figsize=(10,8))
plt.plot(times, hamiltonian_expect, label="$H$ - Total Hamiltonian")
plt.plot(times, two_state_expect, label="$(\Delta E/2) \sigma_z$ - TSS")
plt.plot(times, bosons_expect, label="$\hbar\omega(a^{{\dagger}}a +1/2)$ - bosons")
plt.plot(times, interaction_expect, label="$U( a^{{\dagger}} + a )\sigma_x$ - interaction")

plt.ylabel("Energy")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title(f"Expectation values for {H_latex}    ($\Delta E=0$, $\omega=1$, $U=0.2$)")
plt.show();
```

```python

```

### Let's look at n=0 with $\Delta E=11$ and see what we find

```python
two_state, bosons, interaction, number, nm_list = make_operators(100,0,1)
```

```python
H =  11*two_state+1*bosons + 0.2*interaction
```

```python
bra_labels, ket_labels = make_braket_labels(nm_list)
```

```python
state_number = 11
psi0 = basis(len(nm_list), state_number)
nm_list[state_number]
```

```python
times = np.linspace(0.0, 10.0, 1000)
P, psi = simulate(H, psi0, times)
```

```python
plt.figure(figsize=(10,8))
# for i in range(0,P.shape[0]):
for i in range(0,15):
    plt.plot(times, P[i,:], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title(f" {H_latex}    ($\Delta E=11$, $\omega=1$, $U=0.2$) ")
plt.show();
```

```python
two_state, bosons, interaction, number, nm_list = make_operators(100,0,1)
```

```python
df_no_coupling_100 = make_df_for_energy_scan("$\Delta E$", -20, 20, 201, two_state.shape[0])

for i, row in df_no_coupling_100.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0*interaction
    evals, ekets = H.eigenstates()
    df_no_coupling_100.iloc[i,1:] = evals


```

```python
# NOTE: THIS CELL TAKES ABOUT 2 MIN TO RUN

dfs = []                        # This will hold the dataframes for each value of U
Us = np.linspace(0, 2, 40)    # This will give us 40 frames in our animation from U=0 to U=0.6

for U in Us:
    
    df = make_df_for_energy_scan("$\Delta E$", -20, 20, 201, two_state.shape[0])

    for i, row in df.iterrows():
        H =  row["$\Delta E$"]*two_state + 1*bosons + U*interaction
        evals, ekets = H.eigenstates()
        df.iloc[i,1:] = evals 
        
    dfs.append(df)
```

```python
# Uses the gif library from https://github.com/maxhumber/gif
@gif.frame
def plot(df,j):
    fig, axes = plt.subplots( figsize=(10,8), sharey=True)

    df_no_coupling_100.plot(x="$\Delta E$",legend=False,ylim=[0,20], ax=axes,
            title=f"Stationary states for $H = (\Delta E / 2) \sigma_z + \hbar\omega(a^{{\dagger}}a +1/2) + U( a^{{\dagger}} + a )\sigma_x$     ($\omega=1$, $U={Us[j]:.3f}$) ", alpha=0.3);

    plt.gca().set_prop_cycle(None)

    df.plot(x="$\Delta E$",legend=False, ylim=[0,20], ax=axes,
            title=f"Stationary states for $H = (\Delta E / 2) \sigma_z + \hbar\omega(a^{{\dagger}}a +1/2) + U( a^{{\dagger}} + a )\sigma_x$     ($\omega=1$, $U={Us[j]:.3f}$) ");


    plt.ylabel("Energy");
    
    
frames = []
for j, df in enumerate(dfs):
    frame = plot(df, j)
    frames.append(frame)
gif.save(frames, "./img/05-enegy-levels-100.gif", duration=250)
```

```python
os.rename("./img/05-enegy-levels-100.gif", "./img/05-enegy-levels-100.gif.png")
```

```python
Image(filename="./img/05-enegy-levels-100.gif.png")
```

```python
H =  11*two_state+1*bosons + 0.8101*interaction

bra_labels, ket_labels = make_braket_labels(nm_list)
```

```python
state_number = 0
psi0 = basis(len(nm_list), state_number)
nm_list[state_number]
```

```python
times = np.linspace(0.0, 10.0, 1000)
P, psi = simulate(H, psi0, times)
```

```python
plt.figure(figsize=(10,8))
# for i in range(0,P.shape[0]):
for i in range(0,15):
    plt.plot(times, P[i,:], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title(f" {H_latex}    ($\Delta E=11$, $\omega=1$, $U=0.2$) ")
plt.show();
```

```python
df_coupling_100 = make_df_for_energy_scan("$\Delta E$", -20, 20, 201, two_state.shape[0])

```

```python
for i, row in df_coupling_100.iterrows():
    H =  row["$\Delta E$"]*two_state + 1*bosons + 0.8101*interaction
    evals, ekets = H.eigenstates()
    df_coupling_100.iloc[i,1:] = evals
```

```python
fig, axes = plt.subplots( figsize=(10,8), sharey=True)


df_no_coupling_100.plot(x="$\Delta E$",legend=False,ylim=[0,20], ax=axes,
        title="Stationary states ($\omega=1$, $U=0$)     (Fig 9)", alpha=0.3);

plt.gca().set_prop_cycle(None)

df_coupling_100.plot(x="$\Delta E$",legend=False, ylim=[0,20], ax=axes,
        title=f"Stationary states ($\omega=1$, $U=0.8101$),     (Fig 1)");

plt.ylabel("Energy");
```

```python

```
