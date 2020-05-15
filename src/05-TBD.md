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

<a href="https://colab.research.google.com/github/project-ida/two-state-quantum-systems/blob/matt-sandbox/04-spin-boson-model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/matt-sandbox/04-spin-boson-model.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>


# Spin boson model


> TODO: intro

```python
import numpy as np
import pandas as pd
from qutip import *

from itertools import product

from scipy.optimize import minimize_scalar
from scipy.signal import argrelextrema

%matplotlib inline
import matplotlib.pyplot as plt
import plotly.express as px

warnings.filterwarnings('ignore')
```

## How do the energy levels change with varying coupling?

```python
E = 1.0                   # two level energy difference
N = 11                    # number of phonon quanta needed to exite the atom
E_phonon = E / N          # phonon energy
M_min =  1000             # Min mode number to simulation
M_max =  1300             # Max mode number to simulation
ng = 500                  # number of different coupling strengths to try out (need 100 to reproduce SJByrnes Moiré pattern)
min_coupling = 0.006*E    # min atom phonon coupling
max_coupling = 0.03*E     # maximum atom phonon coupling
parity = "even"           # "all", "even" or "odd"

M = M_max - M_min                           # Number of modes to simulate
num_states = 2*M if parity=="all" else M    # Number of states needed to represent the system
```

```python
a  = tensor(destroy(M,M_min), qeye(2))     # phonon destruction operator
sm = tensor(qeye(M), sigmam())             # sigma_minus operator which is often called a lowering operator
sz = tensor(qeye(M),sigmaz())              # z component of the "spin" of the two level system

two_state     =  E*sz/2                         # two state system energy
phonons       =  E_phonon*(a.dag()*a+0.5)       # phonon field energy
interaction   = (a.dag() + a) * (sm + sm.dag()) # interaction energy - needs to be multiplied by coupling constant in final H

num           = a.dag()*a  # phonon number operator
spin          = sz/2       # z component of spin
```

```python
#inspired by SJB code https://github.com/sbyrnes321/cf/blob/1a34a461c3b15e26cad3a15de3402142b07422d9/spinboson.py#L56
if parity != "all":
    S=1/2
    possible_ns = range(M_min, M_max)
    possible_ms = - (np.arange(2*S+1) - S)
    Smn_list = product([S], possible_ns, possible_ms)

    if parity == "even":
        mn_from_index = [(n-M_min,int(np.abs(m-0.5))) for (S,n,m) in Smn_list if (S+m+n) % 2 == 0]
    elif parity == "odd":
        mn_from_index = [(n-M_min,int(np.abs(m-0.5))) for (S,n,m) in Smn_list if (S+m+n) % 2 == 1]

    subset_idx = []
    for s in mn_from_index:
        subset_idx.append(state_number_index([M,2],s))
    
    # Labels for hinton plots in case we want to plot it later (use xlabels=ket_labels, ylabels = bra_labels)
    bra_labels = ["$\langle$"+str(n)+", "+str(m)+"|" for (n,m) in mn_from_index]
    ket_labels = ["|"+str(n)+", "+str(m)+"$\\rangle$" for (n,m) in mn_from_index]


    # http://qutip.org/docs/latest/apidoc/classes.html?highlight=extract_states#qutip.Qobj.extract_states
    two_state    = two_state.extract_states(subset_idx) 
    phonons      = phonons.extract_states(subset_idx) 
    interaction  = interaction.extract_states(subset_idx) 
    num          = (a.dag()*a).extract_states(subset_idx)
    spin         = spin.extract_states(subset_idx)
    
    
```

```python
d = {"coupling":np.linspace(0,max_coupling,ng)}
for i in range(num_states):
    d[f"level_{i}"] = np.zeros(ng)
    
df = pd.DataFrame(data=d)

# We'll create some dataframes to store expectation values for: 
df_num = pd.DataFrame(data=d) # phonon number
df_sz = pd.DataFrame(data=d)  # z component of spin
df_int = pd.DataFrame(data=d) # interaction energy
```

```python
for index, row in df.iterrows():
    # c.f. https://coldfusionblog.net/2017/07/09/numerical-spin-boson-model-part-1/
    H = two_state + phonons + row.coupling*interaction
    evals, ekets = H.eigenstates()
    df.iloc[index,1:] = np.real(evals/E_phonon)
    
    # We'll also calculate some expectation values so we don't have to do it later
    df_num.iloc[index,1:] = expect(num,ekets)           # phonon number
    df_sz.iloc[index,1:] = expect(spin,ekets)           # z component of spin
    df_int.iloc[index,1:] = expect(interaction,ekets)   # interaction energy
    
```

```python
df.plot(x="coupling",ylim=[1146,1154],figsize=(10,6),legend=False);
plt.ylabel("Energy ($\hbar\omega$)");


# It can be easier to initially explore with an interactive plot like the one below (it does take more memory though).
# melt = df.melt(id_vars=["coupling"],var_name="level",value_name="energy")
# fig = px.line(melt,x="coupling",y="energy",color="level",width=900,height=600)
# fig.layout.showlegend = False 
# fig.show()
```

```python
df[["coupling","level_150","level_149"]].plot(x="coupling",figsize=(10,6));
plt.ylabel("Energy ($\hbar\omega$)");
```

## How do the anti-crossings change with varying coupling?


The anti-crossings manifest when the difference in the energy levels is at its minimum

```python
df_diff = df.drop('coupling', axis=1).diff(axis=1).dropna(axis=1)
df_diff["coupling"] = df["coupling"]
```

```python
df_diff.plot(x="coupling",ylim=[0,3],figsize=(10,6),legend=False);
plt.ylabel("$\Delta E$ ($\hbar\omega$)");

# It can be easier to initially explore with an interactive plot like the one below (it does take more memory though).
# melt_diff = df_diff.melt(id_vars=["coupling"],var_name="level",value_name="energy")
# fig = px.line(melt_diff,x="coupling",y="energy",color="level",width=900,height=600)
# fig.layout.showlegend = False 
# fig.show()
```

We can see that there are many times when the differences between the levels almost goes to zero. Let's look at two adjacent differences, level_150 and level_149

```python
df_diff[["coupling","level_150","level_149"]].plot(x="coupling",figsize=(10,6));
plt.ylabel("$\Delta E$ ($\hbar\omega$)");
```

To investigate these minimums, i.e. the enery gap at the anti-crossings, we can first start by using `argrelextrema` to get find the multiple local minimums.


Let's join the levels together to make it more convenient.

```python
df_diff_subset = df_diff[["coupling","level_150","level_149"]]
```

```python
df_diff_subset["min"] =  df_diff_subset[["level_150","level_149"]].min(axis=1)
```

```python
df_diff_subset["level_min"] = df_diff_subset[["level_150","level_149"]].idxmin(axis=1).str.split("_",expand = True)[1]
```

```python
df_diff_subset[["coupling","level_150","level_149","min"]].plot(x="coupling",figsize=(10,6));
plt.ylabel("$\Delta E$ ($\hbar\omega$)");
```

```python
argmin = argrelextrema(df_diff_subset["min"].values, np.less)[0]
anti_crossing = df_diff_subset.iloc[argmin][["coupling","min","level_min"]]
anti_crossing["g"] = anti_crossing["coupling"]*np.sqrt((M_min+M_max)/2)
anti_crossing.reset_index(inplace=True,drop=True)
```

```python
anti_crossing.plot.line(x="g",y="min",logy=True,figsize=(10,6),ylim=[0.0001,0.2]);
```

We can consider the above as our first "guess" at the value of coupling where the anti-crossings are.

We can go further and use `minimize_scalar` to find a more precise value of both the coupling at the anti-crossing and the energy separation at that point. 

```python
# Define a function which returns the energy difference between two levels for a given coupling
def ev(g,i):
    H = two_state + phonons + g*interaction
    evals, ekets = H.eigenstates()
    return evals[i] - evals[i-1] 
```

```python
dg = (max_coupling - min_coupling)/ng
```

```python
for index, row in anti_crossing.iterrows():
    res = minimize_scalar(ev,args=int(row["level_min"]),bracket=[row["coupling"]-dg, row["coupling"]+dg])
    anti_crossing.loc[index, "coupling"] = res.x
    anti_crossing.loc[index, "min"] = res.fun
anti_crossing["g"] = anti_crossing["coupling"]*np.sqrt((M_min+M_max)/2)
```

```python
anti_crossing.plot.line(x="g",y="min",logy=True,figsize=(10,6), ylim=[1e-7,1e-2], xlim=[0,1],grid=True,marker=".");
plt.ylabel("$\Delta E_{min}$ ($\hbar\omega$)");
```

## Investigating phonon number


We will now look at the difference in phonon number when going from one energy level to the next. We might expect it to change by one...we'll see.

We already calculated the expectation value of the phonon number for the energy eigenstates, so we just need to plot them.

```python
df_num[["coupling","level_150","level_149"]].head()
```

We see that even the zero coupling, levels 149 and 150 differ in phonon number by 9. How does this change as the coupling changes?

```python
df_num[["coupling","level_150","level_149"]].plot(x="coupling",figsize=(10,6));
plt.ylabel("<N>");
```

As we approach the anti-crossing it appears that the levels "exchange" quanta with each other, 13, 15, 17 etc.



## Expectation values side by side


Let's compare the expected values of phonon number, spin value and interaction energy side by side with energy of level 150 to see whether anything interesting strikes us.

```python
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15,10))
df[["coupling","level_150",]].plot(x="coupling",ax=axes[0]);
df_num[["coupling","level_150"]].plot(x="coupling",ax=axes[1]);
df_sz[["coupling","level_150"]].plot(x="coupling",ax=axes[2]);
df_int[["coupling","level_150"]].plot(x="coupling",ax=axes[3]);
axes[0].set_ylabel("Energy ($\hbar\omega$)")
axes[1].set_ylabel("<N>")
axes[2].set_ylabel("<$s_z$>");
axes[3].set_ylabel("<int>");
```

```python
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15,10))
df[["coupling","level_149",]].plot(x="coupling",ax=axes[0]);
df_num[["coupling","level_149"]].plot(x="coupling",ax=axes[1]);
df_sz[["coupling","level_149"]].plot(x="coupling",ax=axes[2]);
df_int[["coupling","level_149"]].plot(x="coupling",ax=axes[3]);
axes[0].set_ylabel("Energy ($\hbar\omega$)")
axes[1].set_ylabel("<N>")
axes[2].set_ylabel("<$s_z$>");
axes[3].set_ylabel("<int>");
```

## Simulating up/down conversation

```python
anti_crossing
```

```python
H = two_state + phonons + anti_crossing.loc[0].coupling*interaction
```

```python
evals, ekets = H.eigenstates()

fig, axes = plt.subplots(1, 2, figsize=(12,5))
plot_fock_distribution(ekets[150], title="150th Eigenstate", ax=axes[0])
plot_fock_distribution(ekets[149],title="149th Eigenstate", ax=axes[1])
axes[0].set_xlim(140,160)
axes[1].set_xlim(140,160)
fig.tight_layout()
```

We cam see that the 149th and 159th eigenstates are mostly made up of the 143rd and 156th base states. We can check this.

```python
print( np.abs(ekets[150][143])**2, np.abs(ekets[150][156])**2)
```

We can check what these base states correspond to by referring back to our ket labels

```python
print ( ket_labels[143], ket_labels[156])
```

```python
# psi_150_minus = tensor(basis(M, 156), basis(2, 1))  
# P_150_minus = psi_150_minus*psi_150_minus.dag() 

# psi_149_plus = tensor(basis(M, 148), basis(2, 1)) 
# P_149_plus = psi_149_plus*psi_149_plus.dag() 

# psi_151_plus = tensor(basis(M, 151), basis(2, 0)) 
# P_151_plus = psi_151_plus*psi_151_plus.dag() 


# psi_150_minus    = psi_150_minus.extract_states(subset_idx) 
# P_150_minus    = P_150_minus.extract_states(subset_idx) 

# psi_149_plus    = psi_149_plus.extract_states(subset_idx) 
# P_149_plus    = P_149_plus.extract_states(subset_idx) 

# psi_151_plus    = psi_151_plus.extract_states(subset_idx) 
# P_151_plus    = P_151_plus.extract_states(subset_idx) 


psi_143_plus = basis(M,143)
psi_156_minus = basis(M,156)

P_143_minus = psi_143_plus*psi_143_plus.dag() 
P_156_minus = psi_156_minus*psi_156_minus.dag() 


times = np.linspace(0.0, 100, 10000)      # simulation time


H = two_state + phonons + anti_crossing.loc[0].coupling*interaction

result = sesolve(H, psi_156_minus, times,[P_143_minus,P_156_minus])

```

```python
plt.plot(times, result.expect[0], label="143")
plt.plot(times, result.expect[1], label="156")
plt.legend(loc="right")
plt.show();
```

```python
P = []

for i in range(0,M):
    psi = basis(M,i)
    P.append(psi*psi.dag())
```

```python
result = sesolve(H, psi_156_minus, times,P)

```

```python
plt.figure(figsize=(15,6))

for i in range(150,159):
    plt.plot(times, result.expect[i], label=f"{i}")
    
plt.ylabel("Probability")
plt.legend(loc="right")
plt.show();
```

```python
ket_labels[155:159]
```

```python

```
