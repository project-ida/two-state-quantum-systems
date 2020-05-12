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

```python
# Libraries
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import Image
import gif
import numpy as np
import pandas as pd
from qutip import *
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')
from itertools import product

from scipy.optimize import minimize_scalar
from scipy.signal import argrelextrema
```

Let's remind ourselves of the Hamiltonian that we've been working with:

$$H =  A \sigma_z + \hbar\omega\left(a^{\dagger}a +\frac{1}{2}\right) + \frac{\delta}{2}\left( a^{\dagger} + a \right)\sigma_x$$

A and $\delta$ were chosen because of the path of discovery that we took to get here. From now on it will be convenient for us to re-write the Hamiltonian as:

$$H =  \frac{\Delta E}{2} \sigma_z + \hbar\omega\left(a^{\dagger}a +\frac{1}{2}\right) + U\left( a^{\dagger} + a \right)\sigma_x$$

where we recognise  $\Delta E$ as the difference in energy of our two state system which we will set as $\Delta E = 1$.

```python
delta_E = 1.0  
```

Last time we saw how this Hamiltonian gave rise to the physics of spontaneous emission when the boson energy matched the two state transition energy, i.e. $\Delta E = \omega$. We will continue to assume that in this tutorial.


```python
omega = delta_E
```

To explore more, we will relax the assumption we have made up to now of small coupling and instead make the coupling the same as the transition energy

```python
U = delta_E
```

How does such a strongly coupled system behave? Let's see.

We need to choose the maximum number of bosons we want to simulate. We'll start small just keep things simple.

```python
max_bosons = 4
```

We now need to construct the Hamiltonian using the `tensor` function in the same way as in the previous tutorial.

```python
a  = tensor(destroy(max_bosons+1), qeye(2))     # tensorised boson destruction operator
sx = tensor(qeye(max_bosons+1), sigmax())       # tensorised sigma_x operator
sz = tensor(qeye(max_bosons+1),sigmaz())        # tensorised sigma_z operator

two_state     =  delta_E/2*sz                          # two state system energy
bosons       =  omega*(a.dag()*a+0.5)          # bosons field energy
interaction  = (a.dag() + a) * sx             # interaction energy

H = two_state + bosons + U*interaction
```

<!-- #region jupyter={"source_hidden": true} -->
One of the most instructive things to do when you get a new Hamiltonian is to visualise it. QuTiP offers a function called [`hinton`](http://qutip.org/docs/latest/apidoc/functions.html?highlight=hinton#qutip.visualization.hinton) for just such a purpose.
<!-- #endregion -->

```python
f, ax = hinton(H)
ax.tick_params(axis='x',labelrotation=90)
ax.set_title("Matrix elements of H     (Fig 1)");
```

The colour and size of the squares in Fig 1 give you a measure of the how large different matrix elements are. The off diagonal elements arise solely from the interaction part of the Hamiltonian - this is what allows one state to (in a sense) "mutate" into another.

We'll study Fig 1 in more detail shortly, but for now I want to draw you attention to the labels for the rows and columns. For example, $|3, 0 \rangle$ represents 3 bosons and the two state system in the 0 state. The two state system numbers are handled somewhat confusingly in QuTiP, namely opposite to what you'd expect $0 \rightarrow +$ and $1\rightarrow -$. It will be helpful to have a way to map these QuTiP states to something more immediately recognisable, i.e. $|3, + \rangle$.

```python
possible_ns = range(0, max_bosons+1)
possible_ms = ["+","-"]
nm_list = [(n,m) for (n,m) in product(possible_ns, possible_ms)]
```

```python
nm_list
```

We can create some nice labels corresponding to the `nm_list`. This will make things like the hinton plot a lot easier to understand.

```python
bra_labels = ["$\langle$"+str(n)+", "+str(m)+" |" for (n,m) in nm_list]
ket_labels = ["| "+str(n)+", "+str(m)+"$\\rangle$" for (n,m) in nm_list]
```

```python
f, ax = hinton(H, xlabels=ket_labels, ylabels=bra_labels)
ax.tick_params(axis='x',labelrotation=90,)
ax.set_title("Matrix elements of H     (Fig 2)");
```

That's better!

If we now take a closer look at the structure of the Hinton diagram we can see some interesting features:

```python
print("                Matrix elements of H     (Fig 3)")
Image(filename='parity.png') 
```

If we imagine starting a simulation with 0 bosons and the two state system in its + state, i.e. |0,+>, then Fig 3 suggests that:
1. there are connections (albeit indirect) from |0,+> to many different states with many more bosons, e.g. $|0,+> \rightarrow |1,-> \rightarrow |2,+> \rightarrow |3,-> \rightarrow |4,+> ...$. This implies that spontaneous emission of a single boson (as we saw in the last tutorial) isn't the only possibility
2. there are some states that are not accessible at all if we start in the |0,+> state

Let's see if these features arise in the simulation and if so what do they mean?

```python
psi0 = tensor(basis(max_bosons+1, 0), basis(2, 0))
```

In the previous tutorials we have been using QuTiP's [`sesolve`](http://qutip.org/docs/latest/apidoc/functions.html#module-qutip.sesolve) to simulate the system. `sesolve` solves the Schrödinger equation. This was convenient as for us when we were getting started - we only needed a single line of code to run the simulation. It was especially useful when we introduced a time dependent perturbation to our two state Hamiltonian in Tutorial 2. However, `sesolve` will cause us problems as we increase the number of bosons that we want to simulate - the simulation will take too long to run.

Technically, we don't actually need a special solver like `sesolve` when dealing with time-independent problems (like ours). The business of solving the Schrödinger equation can be reduced to a problem of finding the eigenvalues and eigenvectors of the Hamiltonian.

Let's see how it works and then go through an example:

1. Transform initial state $\psi_0$ into a new basis defined by the eigenvectors (aka eigenkets) of the Hamiltonian i.e. the states of constant energy (represented here by $|i>$)
  - $\psi_0 = \underset{i}{\Sigma}   <i|\psi_0> |i>$
  -  $<i|\psi_0> = $ `psi0.transform(ekets)[i]`
2. Evolve each part of the state according to its eigenfrequency (aka eigenvalues) $\omega_i$
  - $\psi (t)= \underset{i}{\Sigma}  <i|\psi_0> e^{-i\omega_i t}\ |i>$
  - $\omega_i =$ `evals[i]`
3. Transform the evolved state back into the basis we started with (represented here by $|k>$)
  - $\psi (t)= \underset{i,k}{\Sigma}  <i|\psi_0> e^{-i\omega_i t}\ <k|i>|k>$
  - $<k|i> = $ `ekets[i][k]`


Let's see this in action.

**Step 1**:

```python
evals, ekets = H.eigenstates()
psi0_in_H_basis = psi0.transform(ekets)
```

```python
psi0_in_H_basis
```

This way of representing $\psi_0$ shows us that $|0,+>$ is mainly a mixture of the 3rd and 4th energy states. QuTiP has a convenient way of visualising the probabilities associated with such a state using [`plot_fock_distribution`](http://qutip.org/docs/latest/apidoc/functions.html?highlight=plot_fock_distribution#qutip.visualization.plot_fock_distribution)

```python
plot_fock_distribution(psi0_in_H_basis, title=f" |0,+> in constant energy basis     (Fig 4)")
plt.xlim(-1,10);
```

Continuing to follow the procedure, we have:

$\psi_0 = \underset{i}{\Sigma}  <i|\psi_0> |i> \\
\ \ \ \ = 0 |0> + 0.479 |1> + 0 |2> - 0.607 |3> ...$

**Step 2:**


The frequencies are given by the eigenvalues of the Hamiltonian:

```python
evals
```

and so (dropping the zero terms from step 1) the evolved state becomes:

$\psi (t)= \underset{i}{\Sigma}  <i|\psi_0> e^{-i\omega_i t}\ |i> \\
\ \ \ \ =  0.479 e^{-i (-0.497)t}|1> +-0.607 e^{-i 0.837t} |3> ...$


**Step 3:**

Taking only the $|1>$ part form step 2 above for the sake of brevity, we only need to look at `ekets[1]`

```python
ekets[1]
```

Then:

$0.479 e^{-i (-0.497)t}|1> \rightarrow 0.479 e^{-i (-0.497)t}0.479|0'> + 0.479 e^{-i (-0.497)t}(-0.754)|3'> + 0.479 e^{-i (-0.497)t}0.421|4'> ...$

where the prime in $|n'>$ indicates the original basis and not the energy basis. We can relabel these states to be the more familiar $|n,\pm>$ using the list we made earlier:

```python
nm_list
```

From this we see that:

$|0'> = |0,+>$,  

$|3'> = |1,->$ 

$|4'> = |2,+>$

and so we have:

$0.479 e^{-i (-0.497)t}|1> \rightarrow 0.479 e^{-i (-0.497)t}0.479|0,+>  + 0.479 e^{-i (-0.497)t}(-0.754)|1,-> + 0.479 e^{-i (-0.497)t}0.421|2,+> ...$


All of the above can be automated by making a function that we can reuse again and again:

```python
def simulate(H, psi0, times):
    evals, ekets = H.eigenstates()
    psi0_in_H_basis = psi0.transform(ekets)
    psi = np.zeros([(max_bosons+1)*2,times.size], dtype="complex128")
    P = np.zeros([(max_bosons+1)*2,times.size], dtype="complex128")
    for k in range(0,(max_bosons+1)*2):
        amp = 0
        for i in range(0,(max_bosons+1)*2):
            amp +=  psi0_in_H_basis[i][0][0]*np.exp(-1j*evals[i]*times)*ekets[i][k][0][0]
        psi[k,:] = amp
        P[k,:] = amp*np.conj(amp)
    return P, psi
```

```python
times = np.linspace(0.0, 14.0, 100)
P, psi = simulate(H, psi0, times)
```

```python
plt.figure(figsize=(10,8))
for i in range(0,(max_bosons+1)*2):
    plt.plot(times, P[i,:], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.legend(loc="right")
plt.title("(Fig 5)")
plt.show();
```

Fig 5 is the equivalent of Fig 2 from the previous tutorial - the only significant difference is the coupling is now much larger. 

Whereas previously we saw only the Rabi oscillation between $|0,+>$ and $|1,->$, now we see much more. There is significant probability of finding the system with more than 1 boson - as was suggested in the Hinton diagram.

But, how is such a thing possible? Doesn't it violate conservation of energy? After all, 2 or more bosons have more energy than the + state we started in. 

As you might suspect, the answer is no, energy is conserved, the "missing" energy comes from the interaction term in the Hamiltonian. We can see this explicitly by looking at the expectation values various parts of the Hamiltonian. QuTiP does allow us to do this using the [`expect`](http://qutip.org/docs/latest/guide/guide-states.html#expectation-values) function.



It works in the following way for the expectation of the Hamiltonian H:

`expect(H, state)`

We have the values of the `state` at every time step from the `A` output of our `simulate` function. We must however turn those values into a QuTiP quantum object ([`Qobj`](http://qutip.org/docs/latest/guide/guide-basics.html#the-quantum-object-class)) that has dimensions compatible with the interaction operator. For example, the dimensions of $\psi_0$ are:

```python
psi0.dims
```

We can use the `dims` of $\psi_0$ to create a compatible `Qobj` in the following way (e.g. for time step 10)

```python
d = [[5, 2], [1, 1]]
state = Qobj(psi[:,10],dims=d)
```

The expectation value of the interaction energy at time step 10 is then:

```python
expect(H, state)
```

We could automate this process, but it turns out that creating the `Qobj` for every time step can be very slow. We will instead directly calculate the expectation value using matrix multiplication, i.e.

$<H> = \psi^{\dagger}H\psi = \psi^{\dagger} @ (H @\psi) $

Where @ is the matrix multiplication operator and $\dagger$ in this context means taking the complex conjugate. For the 10th time step we have:

```python
np.conj(psi[:,10])@ (H @ psi[:,10])
```

Let's automate this process using a function.

```python
def expectation(operator, states):
    operator_expect = []
    for i in range(0,shape(states)[1]):
        e = np.conj(states[:,i])@ (operator @ states[:,i])
        operator_expect.append(e)
    return operator_expect
```

We can now see how the different parts of the Hamiltonian change overtime

```python
hamiltonian_expect = expectation(H,psi)
two_state_expect = expectation(two_state,psi)
bosons_expect = expectation(bosons,psi)
interaction_expect = expectation(U*interaction,psi)
```

```python
plt.figure(figsize=(10,8))
plt.plot(times, hamiltonian_expect, label="total hamiltonian")
plt.plot(times, two_state_expect, label="two-state")
plt.plot(times, bosons_expect, label="bosons")
plt.plot(times, interaction_expect, label="interaction")

plt.ylabel("Energy")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title(f"Expectation values for parts of the Hamiltoian (U={U})     (Fig 6)")
plt.show();
```

Fig 6 confirms that the total energy (the blue line) is constant - energy conservation is not violated. We can also see that indeed it's the lowering of interaction energy that's allowing the boson energy to get higher than we might expect by only considering the two state energy and the bosons.


It's helpful to compare this to the case from the previous tutorial, i.e. when the coupling is weak.

```python
U = delta_E/200
H = two_state + bosons + U*interaction
times = np.linspace(0.0, 1400.0, 10000)
P, psi = simulate(H, psi0, times)
```

```python
hamiltonian_expect = expectation(H,psi)
two_state_expect = expectation(two_state,psi)
bosons_expect = expectation(bosons,psi)
interaction_expect = expectation(U*interaction,psi)
```

```python
plt.figure(figsize=(10,8))
plt.plot(times, hamiltonian_expect, label="total hamiltonian")
plt.plot(times, two_state_expect, label="two-state")
plt.plot(times, bosons_expect, label="bosons")
plt.plot(times, interaction_expect, label="interaction")

plt.ylabel("Energy")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title(f"Expectation values for parts of the Hamiltoian (U={U})     (Fig 7)")
plt.show();
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
def states_to_df(states,times):
    data = {}
    for i in range(0,states[0].shape[0]):
        which_mode = divmod(i,2)
        if which_mode[1] == 0:
            two_state = "+"
        elif which_mode[1] == 1:
            two_state = "-"
        data[str(which_mode[0])+" , "+two_state] = np.zeros(len(times),dtype="complex128")
    
    for i, state in enumerate(states):
        for j, psi in enumerate(state):
            which_mode = divmod(j,2)
            if which_mode[1] == 0:
                two_state = "+"
            elif which_mode[1] == 1:
                two_state = "-"
            data[str(which_mode[0])+" , "+two_state][i] = psi[0][0]

    return pd.DataFrame(data=data, index=times)
```

```python
A[:,5]
```

```python
result.states[5]
```

```python
psi0 = tensor(basis(max_bosons+1, 0), basis(2, 0))  # No bosons and two-state system is in the higher energy + state

times = np.linspace(0.0, 14.0, 100)      # simulation time

result = sesolve(H, psi0, times)
df_coupled =  states_to_df(result.states, times)
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
df_coupled.plot(title="Real part of amplitudes Re($\psi$)     (Fig 1)", ax=axes[0]);
(df_coupled.abs()**2).plot(title="Probabilities $|\psi|^2$     (Fig 2)", ax=axes[1]);
```

```python
plt.figure(figsize=(10,8))
for i in range(0,(max_bosons+1)*2):
    plt.plot(times, A[i,:], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.legend(loc="right")
plt.title("(Fig 5)")
plt.show();
```

```python
P = sz*(1j*np.pi*a.dag()*a).expm()
```

```python
(1j*np.pi*a.dag()*a).expm()
```

```python
P
```

```python
f, ax = hinton(H)
ax.tick_params(axis='x',labelrotation=90)
ax.set_title("Matrix elements of H     (Fig 1)");
```

```python
commutator(H,P)
```

```python
possible_ms
```

```python
psi0 = tensor(basis(max_bosons+1, 0), basis(2, 0))
```

```python
evals, ekets = H.eigenstates()
psi0_in_H_basis = psi0.transform(ekets)
```

```python
times = np.linspace(0.0, 14.0, 100)
```

```python
P = []

for k in range(0,(max_bosons+1)*2):
    amp = 0
    for i in range(0,max_bosons+1):
        amp +=  psi0_in_H_basis[i][0][0]*np.exp(-1j*evals[i]*times)*ekets[i][k][0][0]
    P.append(amp*np.conj(amp))
```

```python
plt.figure(figsize=(10,8))
for i in range(0,(max_bosons+1)*2):
    plt.plot(times, P[i], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.legend(loc="right")
plt.show();
```

```python
ket_labels
```

```python

```

```python

```

```python
f, ax = hinton(H)
ax.tick_params(axis='x',labelrotation=90)
ax.set_title("Matrix elements of H     (Fig 1)");
```

```python
evals, ekets = H.eigenstates()
```

```python
evals
```

```python

```

```python

```

```python

```

```python

```

## Calculate energy eigenvalues for different couplings

```python
delta_E = 1.0            # two level energy difference
N = 2                    # number of phonon quanta needed to exite the atom
omega = delta_E / N          # phonon energy
max_bosons =  100             # Max mode number to simulation
num_U = 100                  # number of different coupling strengths to try out (need 100 to reproduce SJByrnes Moiré pattern)
U_min = 0.01    # min atom phonon coupling
U_max = 0.8*delta_E     # maximum atom phonon coupling
```

```python
a  = tensor(destroy(max_bosons+1), qeye(2))     # tensorised boson destruction operator
sx = tensor(qeye(max_bosons+1), sigmax())             # tensorised sigma_x operator
sz = tensor(qeye(max_bosons+1),sigmaz())              # tensorised sigma_z operator

two_state     =  delta_E/2*sz                          # two state system energy
bosons       =  omega*(a.dag()*a+0.5)          # bosons field energy
interaction   = (a.dag() + a) * sx     # interaction energy (needs to be multiplied by coupling constant)

number           = a.dag()*a  # phonon number operator
spin          = sz/2       # z component of spin
```

```python
H = two_state + bosons + interaction
```

```python
# f, ax = hinton(H)
# ax.tick_params(axis='x',labelrotation=90)
# ax.set_title("Matrix elements of H     (Fig 1)");
```

```python
parity = "even"
```

```python
#inspired by SJB code https://github.com/sbyrnes321/cf/blob/1a34a461c3b15e26cad3a15de3402142b07422d9/spinboson.py#L56
if parity != "all":
    S=1/2
    possible_ns = range(0, max_bosons+1)
    possible_ms = - (np.arange(2*S+1) - S)
    Smn_list = product([S], possible_ns, possible_ms)

    if parity == "even":
        mn_from_index = [(n,int(np.abs(m-0.5))) for (S,n,m) in Smn_list if (S+m+n) % 2 == 0]
    elif parity == "odd":
        mn_from_index = [(n,int(np.abs(m-0.5))) for (S,n,m) in Smn_list if (S+m+n) % 2 == 1]

    subset_idx = []
    for s in mn_from_index:
        subset_idx.append(state_number_index([max_bosons+1,2],s))
    
    # Labels for hinton plots in case we want to plot it later (use xlabels=ket_labels, ylabels = bra_labels)
    bra_labels = ["$\langle$"+str(n)+", "+str(m)+"|" for (n,m) in mn_from_index]
    ket_labels = ["|"+str(n)+", "+str(m)+"$\\rangle$" for (n,m) in mn_from_index]

```

It will be helpful for us to be able to search for the index of a particular state by the (n,m) numbers

```python
def index_from_nm(n,m): 
    try:
        return [item for item in mn_from_index].index((n,m))
    except:
        print("ERROR: State doesn't exist or has the wrong parity ")
```

```python
    # http://qutip.org/docs/latest/apidoc/classes.html?highlight=extract_states#qutip.Qobj.extract_states
    two_state    = two_state.extract_states(subset_idx) 
    bosons      = bosons.extract_states(subset_idx) 
    interaction  = interaction.extract_states(subset_idx) 
    number          = (a.dag()*a).extract_states(subset_idx)
    spin         = spin.extract_states(subset_idx)
```

```python
# H = two_state + bosons + interaction
# f, ax = hinton(H, xlabels=ket_labels, ylabels=bra_labels)
# ax.tick_params(axis='x',labelrotation=90)
# ax.set_title("Matrix elements of H     (Fig 1) EVEN");
```

```python
d = {"coupling":np.linspace(U_min,U_max,num_U)}
for i in range((max_bosons+1)):
    d[f"level_{i}"] = np.zeros(num_U)
    
df = pd.DataFrame(data=d)


# We'll create some dataframes to store expectation values for: 
df_num = pd.DataFrame(data=d) # phonon number
df_sz = pd.DataFrame(data=d)  # z component of spin
df_int = pd.DataFrame(data=d) # interaction energy
```

```python
for index, row in df.iterrows():
    # c.f. https://coldfusionblog.net/2017/07/09/numerical-spin-boson-model-part-1/
    H = two_state + bosons + row.coupling*interaction
    evals, ekets = H.eigenstates()
    df.iloc[index,1:] = np.real(evals/omega)
    
    # We'll also calculate some expectation values so we don't have to do it later
    df_num.iloc[index,1:] = expect(number,ekets)           # phonon number
    df_sz.iloc[index,1:] = expect(spin,ekets)           # z component of spin
    df_int.iloc[index,1:] = expect(row.coupling*interaction,ekets)   # interaction energy
```

```python
df.plot(x="coupling",ylim=[-6,20],figsize=(10,6),legend=False);
plt.ylabel("Energy ($\hbar\omega$)");
```

## More detail on energy levels 7 and 8

```python
level_number_1 = 2
level_number_2 = 3
```

```python
level_label_1 = f"level_{level_number_1}"
level_label_2 = f"level_{level_number_2}"
```

```python
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15,10))
df[["coupling",level_label_1,level_label_2]].plot(x="coupling",ax=axes[0]);
df_num[["coupling",level_label_1,level_label_2]].plot(x="coupling",ax=axes[1]);
df_sz[["coupling",level_label_1,level_label_2]].plot(x="coupling",ax=axes[2]);
df_int[["coupling",level_label_1,level_label_2]].plot(x="coupling",ax=axes[3]);
axes[0].set_ylabel("Energy ($\hbar\omega$)")
axes[1].set_ylabel("<N>")
axes[2].set_ylabel("<$s_z$>");
axes[3].set_ylabel("<int>");
```

```python
df_num[["coupling",level_label_1,level_label_2]].plot(x="coupling",figsize=(10,6));
plt.ylabel("<N>");
```

## Find the anti-crossing points


### Start with rough calculation of the anti-crossing coupling

```python
df_diff = df.drop('coupling', axis=1).diff(axis=1).dropna(axis=1)
df_diff["coupling"] = df["coupling"]
```

```python
level_label = level_label_2

df_diff_subset = df_diff[["coupling",level_label]]
df_diff_subset["min"] =  df_diff_subset[[level_label]].min(axis=1)
df_diff_subset["level_min"] = df_diff_subset[[level_label]].idxmin(axis=1).str.split("_",expand = True)[1]

argmin = argrelextrema(df_diff_subset["min"].values, np.less)[0]
anti_crossing = df_diff_subset.iloc[argmin][["coupling","min","level_min"]]
anti_crossing.reset_index(inplace=True,drop=True)
```

```python
anti_crossing
```

### Now more precise calculation of anti-crossing couplings

```python
# Define a function which returns the energy difference between two levels for a given coupling
def ev(U,i):
    H = two_state + bosons + U*interaction
    evals, ekets = H.eigenstates()
    return evals[i] - evals[i-1] 
```

```python
dU = (U_max - U_min)/num_U
```

```python
for index, row in anti_crossing.iterrows():
    res = minimize_scalar(ev,args=int(row["level_min"]),bracket=[row["coupling"]-dU, row["coupling"]+dU])
    anti_crossing.loc[index, "coupling"] = res.x
    anti_crossing.loc[index, "min"] = res.fun
```

```python
anti_crossing
```

## Simulation of the first anti-crossing


### First let's look at some expectation values at the anti-crossing

```python
H = two_state + bosons + anti_crossing.loc[0]["coupling"]*interaction
# H = two_state + bosons + 0.15*interaction
# H = two_state + bosons + 0.531938*interaction # for N=1
# H = two_state + bosons + 0.08103160077219432*interaction # for N=11
evals, ekets = H.eigenstates()
```

```python
print("state", "energy", "number", "spin")
for i in range(level_number_1-3,level_number_2+3):
    print(i, evals[i]/omega, expect(number,ekets[i]), expect(spin,ekets[i]))
```

We see above that levels 7 and 8 are almost identical which confirms what we see in the figures above


### What are the anti-crossing eigenstates made of?

```python
fig, axes = plt.subplots(1, 2, figsize=(12,5))
plot_fock_distribution(ekets[level_number_1], title=f"{level_number_1} Eigenstate", ax=axes[0])
plot_fock_distribution(ekets[level_number_2],title=f"{level_number_2} Eigenstate", ax=axes[1])
axes[0].set_xlim(-1,25)
axes[1].set_xlim(-1,25)
fig.tight_layout()
```

The Energy eigenstates that come together at the anti-crossing are mostly made up of states numbered:

```python
P_eigenstate = ekets[level_number_1].full()*np.conj(ekets[level_number_1].full())
P_eigenstate = np.hstack(P_eigenstate)
eigenstate_composition = P_eigenstate.argsort()[-2:][::-1]
print(eigenstate_composition[0], ",", eigenstate_composition[1])
```

What do these numbered states corresponds to? We can use the `mn_from_index` to find the n and m numbers.

```python
print ( mn_from_index[eigenstate_composition[0]], ",", mn_from_index[eigenstate_composition[1]])
```

### What does the most dominate part look like in basis of the eigenstates?


We should create the state by using `tensor` function and the extracting the states that have the wrong parity. We can however create the state directly from the `basis` function using the index we found above.

```python
i0 = eigenstate_composition[0]
```

```python
psi0 = basis(max_bosons+1, i0)
```

```python
psi0_in_H_basis = psi0.transform(ekets)
```

```python
plot_fock_distribution(psi0_in_H_basis, title=f"{ket_labels[i0]} in H basis")
plt.xlim(-1,20);
```

so |1,0> (ie 1 bosons and lower state for two state system) is mixture of eigenstates 7 and 8.

Therefore if we start the system off in |1,0> we can expect it to rabi oscillate between 7 and 8 which should give us some oscilation between |1,0> and |14,1>. 

Let's see

```python
P = []

for i in range(0,max_bosons+1):
    psi = basis(max_bosons+1,i)
    P.append(psi*psi.dag())
```

```python
times = np.linspace(0.0, 1000.0, 10000)      # simulation time

result = sesolve(H, psi0, times,P)
```

```python
plt.figure(figsize=(10,6))
for i in range(0,17):
    plt.plot(times, result.expect[i], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.legend(loc="right")
plt.show();


```

Not much osccilation at all!

Maybe we need to wait longer. Let's manually do a time varying state otherwise it will take a very long time.

> TODO: explain the method below

```python
times = np.linspace(0.0, 1000000.0, 10000)
```

```python
P = []

for k in range(0,max_bosons+1):
    amp = 0
    for i in range(0,max_bosons+1):
        amp +=  psi0_in_H_basis[i][0][0]*np.exp(-1j*evals[i]*times)*ekets[i][k][0][0]
    P.append(amp*np.conj(amp))
```

```python
plt.figure(figsize=(10,8))
for i in range(0,20):
    plt.plot(times, P[i], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.legend(loc="right")
plt.show();
```

Seems like we'd have to wait a really long time to see the Rabi oscillation.


### How sensitive is the anti-crossing?

```python

# Us = np.linspace(anti_crossing.loc[0]["coupling"]/10,2*anti_crossing.loc[0]["coupling"],40)

Us_min = anti_crossing.loc[0]["coupling"] - anti_crossing.loc[0]["coupling"]/5000
Us_max = anti_crossing.loc[0]["coupling"] + anti_crossing.loc[0]["coupling"]/5000
Us = np.linspace(Us_min,Us_max,21)
Ps = []
for U in Us:
    H = two_state + bosons + U*interaction
    evals, ekets = H.eigenstates()
    times = np.linspace(0.0, 1000000.0, 10000)

    psi0 = basis(max_bosons+1, i0)
    psi0_in_H_basis = psi0.transform(ekets)

    P = []

    for k in range(0,max_bosons+1):
        amp = 0
        for i in range(0,max_bosons+1):
            amp +=  psi0_in_H_basis[i][0][0]*np.exp(-1j*evals[i]*times)*ekets[i][k][0][0]
        P.append(amp*np.conj(amp))
    Ps.append(P)
```

```python
psi0_level = []
for U in Us:
    H = two_state + bosons + U*interaction
    evals, ekets = H.eigenstates()
    psi0_in_H_basis = psi0.transform(ekets)
    P0 = psi0_in_H_basis.full()*np.conj(psi0_in_H_basis.full())
    P0 = np.hstack(P0)
    psi0_level.append(P0.argmax())
```

```python
@gif.frame
def plot(P,j):
    plt.figure(figsize=(10,8))
    for i in range(0,20):
        plt.plot(times, P[i], label=f"{ket_labels[i]}")
    plt.ylabel("Probability")
    plt.xlabel("Time")
    plt.legend(loc="right")
    plt.title(f"U = {Us[j]}")
```

```python
@gif.frame
def plot2(P,j):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
    for i in range(0,20):
        axes[0].plot(times, P[i], label=f"{ket_labels[i]}")
    axes[0].set_ylabel("Probability")
    axes[0].set_xlabel("Time")
    axes[0].legend(loc="right")
    deltaU = (Us[j]-anti_crossing.loc[0]["coupling"])/anti_crossing.loc[0]["coupling"]*100
    axes[0].set_title(f"$\Delta U / U_{{anti\\times}}$ = {deltaU:.3f}%")
    df.plot(x="coupling",ylim=[-6,20],legend=False,ax=axes[1], title=f"Tracking energy level of {ket_labels[i0]} state");
    axes[1].set_ylabel("Energy ($\hbar\omega$)");
    axes[1].plot(Us[j],np.interp(Us[j],df["coupling"], df[f"level_{psi0_level[j]}"]),"ok")
```

```python
frames = []
for j, P in enumerate(Ps):
    frame = plot2(P, j)
    frames.append(frame)
gif.save(frames, "anti-crossing-approach.gif", duration=500)
```

```python
Image(filename="./anti-crossing-approach.gif")
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
def index_from_nm(n,m): 
    try:
        return [item for item in nm_list].index((n,m))
    except:
        print("ERROR: State doesn't exist")
```

```python

```
