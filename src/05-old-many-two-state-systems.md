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

```

```python
# Libraries
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import Image, Math
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

def prettify_states(states, mm_list=None):
    pretty_states = np.zeros([states[0].shape[0],len(states)], dtype="object")
    
    for j, state in enumerate(states):
        x = []
        for i, val in enumerate(state):
            pretty_states[i,j] = f"{val[0,0]:.1f}"
    if (mm_list == None):
        df = pd.DataFrame(data=pretty_states)
    else:
        df = pd.DataFrame(data=pretty_states, index=mm_list)
            
    return df
```

As soon as we start adding more than one TSS things get quite complicated. In order to give us an intuition for how such systems behave, we will take inspiration from Tutorials 1 and 2.





Let's start simple and look at 2 TSS. We can describe this system by the different possible combinations of the + and -, namely:
- |+,+>
- |+,->
- |-,+>
- |-,->

So, 2 TSS is actually a 4 state system. Mathematically these 4 states can be represented as vectors of length 4 using the following basis:

$$
|+, +> = \begin{bmatrix}
 1   \\
 0   \\
 0   \\
 0   \\
 \end{bmatrix}, 
|+, -> = \begin{bmatrix}
 0   \\
 1   \\
 0   \\
 0   \\
\end{bmatrix}, 
|-, +> = \begin{bmatrix}
 0   \\
 0   \\
 1   \\
 0   \\
\end{bmatrix}, 
|-, -> = \begin{bmatrix}
 0   \\
 0   \\
 0   \\
 1   \\
\end{bmatrix}
$$

How to we create these states in QuTiP?

In [Tutorial 3](https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/03-a-two-state-system-in-a-quantised-field.ipynb#3.5---Describing-coupled-systems-in-QuTiP) we learnt to describe such states| by using the tensor product. For example, we would create the |+,-> state by doing:

```python
tensor(basis(2,0), basis(2,1))
```

and we can keep track of which basis states corresponds to which row by using the `product` function that we introduced in the previous tutorial. Specifically:

```python
possible_ms = ["+","-"]
mm_list = [m for m in product(possible_ms, possible_ms)]
mm_list
```

So, the row number 1 of the state vector refers to the basis state:

```python
mm_list[1]
```

How does such a system behave?


## Independent TSS

In [tutorial 1](https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/01-an-isolated-two-state-system.ipynb#1---An-isolated-two-state-system), we motivated that a single isolated TSS, whose states have the same energy $E_0$ and are coupled with some strength $A$, can be described by a Hamiltonian of the form:

$$
H = E_0 I - A \sigma_x
$$

where we now know that we can set $E_0=0$ (without loss of generality) leaving us with:

$$
H = - A \sigma_x
$$

We found that (due to the coupling) the |+> and |-> basis states were not actually states of constant energy (i.e. stationary states). This gave rise to Rabi oscillations in the probabilities.

Let's refresh our memory and re-simulate this case.

Before we do this, we will make a minor adjustment to the Hamiltonian.

You may recall that there are mathematical similarities between a TSS and a spin $1/2$ particle. When considering many TSS, we will find it invaluable to refer to well known spin results, such as conservation of angular momentum, to help us solve problems. In light of this, we will introduce a factor of $1/2$ into the Hamiltonian:

$$
H = - A \frac{1}{2}\sigma_{x}
$$

so that we can rewrite the Hamiltonian in terms of spin operators, (denoted by $S$), for a [spin $1/2$ particle](https://en.wikipedia.org/wiki/Spin-%C2%BD#Observables), i.e.

$$
H = - A S_{x}
$$


```python
A=0.1
H = -A*sigmax()/2

times = np.linspace(0.0, 99.0, 1000) 

psi0=basis(2,0)

P, psi = simulate(H, psi0, times)
```

```python
bra_labels, ket_labels = make_braket_labels(["+","-"])
```

```python
plt.figure(figsize=(10,8))
for i in range(0,P.shape[0]):
    plt.plot(times, P[i,:], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title(" Fig 1")
plt.show();
```

The Rabi frequency (and hence the difference in energy of the stationary states) is now given by $A$ (rather than $2A$ from tutorial 1) because we introduced the $1/2$ into the Hamiltonian.

Now back to 2 TSS. If these are considered to be independent, then we should be able to work out the probabilities from above by simply multiplying the probabilities together, e.g. P(|+,+>) = P(|+>)P(|+>):

```python
P2 = np.zeros([4,times.size], dtype="complex128")
P2[0,:] = P[0,:]*P[0,:]
P2[1,:] = P[0,:]*P[1,:]
P2[2,:] = P[1,:]*P[0,:]
P2[3,:] = P[1,:]*P[1,:]
```

```python
bra_labels, ket_labels = make_braket_labels(mm_list)
```

```python
plt.figure(figsize=(10,8))
for i in range(0,P2.shape[0]):
    plt.plot(times, P2[i,:], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title("Fig 2")
plt.show();
```

What about the energies of the stationary states? We can have a guess by thinking that the energy difference of the levels being A. This means we can have ±A/2 for the levels. So we can have both upper giving A/2+A/2 = A, both lower giving -A or both opposite giving 0.

What about the stationary states themselves? We could probably work it out, but at this point it might be easier to try and figure out the Hamiltonian and calculate from there.

<!-- #region -->
A natural starting point for the Hamiltonian of $N$  independent TSS is then


$$
H = - A \overset{N}{\underset{n=1}{\Sigma}} S_{n x}
$$

Because spin represents angular momentum, the combination of spin operators above is mathematically the same as how one would create the [total angular momentum operators](https://www2.ph.ed.ac.uk/~ldeldebb/docs/QM/lect15.pdf) - denoted by $J$, e.g. $J_x = \overset{N}{\underset{n=1}{\Sigma}} S_{n x}$. The Hamiltonian can then be written more compactly as:

$$
H = - A J_{x}
$$


Let's see if we can reproduce the above figure from this Hamiltonian.

QuTiP has a nice function to generate the J operators for any given number of TSS.
<!-- #endregion -->

```python
J = jspin(2, basis="uncoupled")
```

```python
A=0.1

H = -A*J[0]


times = np.linspace(0.0, 99.0, 1000) 

psi0=basis(4,0)

P, psi = simulate(H, psi0, times)
```

```python
bra_labels, ket_labels = make_braket_labels(mm_list)
```

```python
plt.figure(figsize=(10,8))
for i in range(0,P2.shape[0]):
    plt.plot(times, P[i,:], label=f"{ket_labels[i]}")
plt.ylabel("Probability")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title("Fig 3")
plt.show();
```

Great, that seemed to work! Let's take a look at the stationary states by calculating the eigenvalues and eigenstates

```python
evals, ekets = H.eigenstates()
```

```python
evals
```

The eigenvalues are as we predicted.

```python
prettify_states(ekets, mm_list)
```

Although the stationary states look a bit complicated they are actually just combinations of the 2 stationary states from the single TSS, i.e.

- |+> + |->
- |+> - |->

For example, column 0 (the state corresponding to energy = -0.1) is made from:

$$(|+> + \ |->) \otimes (|+> + \ |->) = |+,+> +\ |+,-> + \ |-,+> + \ |-,->$$

> TODO: Be more explicit with the other states

What else can we learn from this system?

Just like in Tutorial 2, we can perturb the system and see what happens.



## Perturbing 2 TSS


We imagine some kind of perturbing field that affects all the TSS in the same way.

We can extend the Hamiltonian to include a perturbation $\delta$ in a similar way to above:

$$
H = - A J_{x} + \delta J_{z}
$$

where here $J_z = \overset{N}{\underset{n=1}{\Sigma}} S_{n z}$


We proceed as we have done several times by looking for the stationary states of the system. When the system is in one of these states it will remain there for all time. Such states are described by a single constant energy.



We'll calculate the eigenvalues of the Hamiltonian (i.e the energies) and see how they depend on the perturbation strength $\delta$. When we did this in Tutorial 2 we discovered an avoided crossing (aka anti-crossing) when the perturbation was zero - this was due to the coupling between the states splitting the energy levels apart.

Let's see what we find.

```python
J = jspin(2, basis="uncoupled")
```

```python
df = make_df_for_energy_scan("$\delta$/A", -4,4, 100, J[0].shape[0]) 
```

```python
for i, row in df.iterrows():
    H = - A*J[0] + row[ "$\delta$/A"]*A*J[2]

    evals, ekets = H.eigenstates()
    df.iloc[i,1:] = evals
```

```python
df.plot(x="$\delta$/A",figsize=(10,8),legend=True, 
        title="Stationary states for $H=-A\Sigma S_{nx} + \delta \Sigma S_{nz}$   (A=0.1, N=2)     (Fig 4)");
plt.ylabel("Energy");
```

We see that level_0 and level_3 show a similar dependency on $\delta$ (much like what we saw in tutorial 2). What is most interesting is level_1 and level_2 whose energy does not show any dependence on $\delta$. Let's explore what the eigenstates look like for the $\delta/A=4$ case (the last value calculated in the above loop)

```python
evals
```

```python
prettify_states(ekets, mm_list)
```

What is most striking is that the state corresponding to level_2 (column 2) is the same as when we had $\delta = 0$. Then it was denoted as level_1.

What does this mean?

Let's try and answer this questions by using a technique we employed in tutorial 2, namely using a resonant time dependent perturbation, i.e.

$$
H = - A J_{x} + \delta J_{z} \cos(\omega t)
$$

with $\omega = A$.

The idea is to start the system off in a stationary state of the unperturbed system (with $\delta=0$) - we will start with of on the middle energies, specifically state(|+,-> + |-,+>) - and see what happens. We know that when the system depends explicitly on time, the energy of the system in not conserved so we expect the state to not be fixed in time. We also saw this effect explicitly in Tutorial 2.

```python
J = jspin(2, basis="uncoupled")
```

```python

delta = 0.001
A = 0.1

H0 = -A*J[0]


evals, ekets = H0.eigenstates()

H1 =  delta*J[2]


H = [H0,[H1,'cos(w*t)']]

times = np.linspace(0.0, 20000.0, 1000) 

psi0 = ekets[1]

result = sesolve(H, psi0, times, args={'w':A})

```

Now that we have simulated the system, it is convenient to transform the state vector into the basis consisting of stationary states of $H_0$. This removes fast oscillations arising from the Rabi oscillations.

```python
num_states = result.states[0].shape[0]
psi = np.zeros([num_states,times.size], dtype="complex128")
P = np.zeros([num_states,times.size], dtype="complex128")

for i, state in enumerate(result.states):
    transformed_state = state.transform(ekets)
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
plt.title("$H =A \ J_x + \delta \ J_z \  \cos (\omega t)$     (A=0.1, $\omega = 0.1$, $\delta=0.001$)   (Fig 5)")
plt.show();
```

Just like in tutorial 2, we see that oscillations on a timescale related to $\delta$. Specifically, now that we have incorporated the factor of $1/2$ into the Hamiltonian, the oscillation in this case is $\delta/2$

```python
2*np.pi/(delta)
```

We see several things that are interesting:
1. At around 3000 we see that there is a 50:50 chance of the system being in the upper or lower energy level. We can think of this as the chance of the combined 2 TSS undergoes stimulated absorption or emission.
2. Throughout the simulation there is zero chance of the system moving to level 1.

On 2, When we look at level 1, we find that it's the state that is unaffected by $\delta$ that we spoke of earlier.

```python
prettify_states(ekets, mm_list)
```

It seems like state 1 is unable to couple to the other 3. Let's see this explicitly by starting the simulation off in the other zero energy state (|+,-> - |-,+>).

```python
delta = 0.001
A = 0.1

H0 = -A*J[0]

evals, ekets = H0.eigenstates()

H1 =  delta*J[2]


H = [H0,[H1,'cos(w*t)']]

times = np.linspace(0.0, 20000.0, 1000) 

psi0 = ekets[1]

result = sesolve(H, psi0, times, args={'w':A})

```

```python
num_states = result.states[0].shape[0]
psi = np.zeros([num_states,times.size], dtype="complex128")
P = np.zeros([num_states,times.size], dtype="complex128")

for i, state in enumerate(result.states):
    transformed_state = state.transform(ekets)
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
plt.title("$H =A \ J_x + \delta \ J_z \  \cos (\omega t)$     (A=0.1, $\omega = 0.1$, $\delta=0.001$)    (Fig 6)")
plt.show();
```

This result is striking. The constant probability implies that, in this state, the combined system is incapable of absorbing or emitting radiation - it is stuck in a zero energy when you might expect it to drop down to the ground state.

How is this possible? Let's look at the maths to start with and then look at physics. 

Let's look at the Hamilton $ H = - A J_{x} + \delta J_{z}$ in the basis of stationary states of the unperturbed Hamiltonian $H_0= - A J_{x} $

```python
H = H0+H1
H
```

```python
H.transform(ekets)
```

What can we see from this transformed Hamiltonian

We see that state 1 (|+,-> - |-,+>) lives in its own universe.  We see this because row 1 and column 1 are identically zero - this means no other states can interact with it.

What's the physics behind this?

It's related to conservation of angular momentum. 


## Angular momentum


> TODO: Chat about angular momentum, including links and how do do the adding up and j

```python
J2 = J[0]*J[0] + J[1]*J[1] + J[2]*J[2] 
J2
```

```python
commutator(H,J2)
```

```python
evalsJ, eketsJ = J2.eigenstates()
```

```python
evalsJ
```

> TODO: Singlet vs triplet state chat

```python
prettify_states(eketsJ, mm_list) # cf https://quantummechanics.ucsd.edu/ph130a/130_notes/node312.html
```

```python
J2.transform(eketsJ)
```

We see things are grouped into different blocks according to their $j$ value.

Within each $j$ group, we can assign a value of $m$ according to $\frac{1}{2}\Sigma (n_+ - n_-)$ giving the total z "angular momentum". We can most conveniently see this by looking at $J_z$ in the $J^2$ basis

```python
J[2].transform(eketsJ)
```

And so we can write (ignoring normalisations):

- |+,-> - |-,+> = |0,0>
- |+,+> = |1,1/2>
- |+,-> + |-,+> = |1,0>
- |-,->  = |1,-1>


QuTiP has a function that can represent the J's in this way - in the so called Dicke basis. We have used it already `jspin()` - this time we won't use the "basis" parameter.

```python
J = jspin(2)
```

Let's take a look at $J^2$ and $J_z$

```python
J2 = J[0]*J[0] + J[1]*J[1] + J[2]*J[2] 
J2
```

```python
J[2]
```

Almost like what we calculated above, only the order of things has been switched. In this Dicke basis states are enumerated from high to low in both j and m.

What does the Hamiltonian look like in this basis.

```python
H0 = -A*J[0]
H1 =  delta*J[2]
H = H0 +  H1
H
```

Straight away we see the isolation of the |0,0> state even before we transform into the stationary sates of $H_0$.

Let's just confirm that we get back the same Hamilton just before the start of this section by transforming this Hamiltonian into the basis associated with the stationary states of $H_0$.

```python
evals, ekets = H0.eigenstates()
H.transform(ekets)
```

This is exactly the same as we obtained earlier - as it should be.

What is most convenient about being in the $J$ basis is that we can perform a similar trick to tutorial 2. Specifically, we found that the system described by $H =  - A \sigma_x + \delta\sigma_z$ could be described by $H =   A\sigma_z +\delta \sigma_x$ after we changed basis into the stationary states of the unperturbed Hamiltonian $H = - A \sigma_x$.

We can do something similar, $H = -AJ_z + \delta J_x$

```python
H2 = -A*J[2] + delta*J[0]
H2
```

Other than reordering of the columns, this is the same as we got by transforming using the eigenstates approach.

So, in general we can say that many TSS can be described by 

$$H = -AJ_z + \delta J_x \cos(\omega t)$$

(we have only shown this for 2 TSS but in general this is true for arbitrary number)

Let's try this out.

```python
delta = 0.001
A = 0.1

H0 = -A*J[2]

# evals, ekets = H0.eigenstates()

H1 =  delta*J[0]


H = [H0,[H1,'cos(w*t)']]

times = np.linspace(0.0, 20000.0, 1000) 

psi0 = basis(4,0)

result = sesolve(H, psi0, times, args={'w':A})

```

```python
num_states = result.states[0].shape[0]
psi = np.zeros([num_states,times.size], dtype="complex128")
P = np.zeros([num_states,times.size], dtype="complex128")

for i, state in enumerate(result.states):
#     transformed_state = state.transform(ekets)
    psi[:,i] = np.transpose(state)
    P[:,i] = np.abs(psi[:,i]*np.conj(psi[:,i]))
```

```python
plt.figure(figsize=(10,8))
for i in range(0,P.shape[0]):
    plt.plot(times, P[i,:], label=f"E_level_{i}")
plt.ylabel("Probability")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title("$H =A \ J_x + \delta \ J_z \  \cos (\omega t)$     (A=0.1, $\omega = 0.1$, $\delta=0.001$)    (Fig 6)")
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

## Dicke basis

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
def make_braket_labels(jm_list):
    bra_labels = ["$\langle$"+str(Fraction(j))+", "+str(Fraction(m))+" |" for (j,m) in jm_list]
    ket_labels = ["| "+str(Fraction(j))+", "+str(Fraction(m))+"$\\rangle$" for (j,m) in jm_list]
    return bra_labels, ket_labels
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
delta = 0.001
A = 0.1

H0 = A*Jx

evals, estates = H0.eigenstates()

H1 =  delta*Jz

H = [H0,[H1,'cos(w*t)']]

times = np.linspace(0.0, 20000.0, 1000) 

#psi0=basis(4,0)
psi0 = estates[0]

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
plot_fock_distribution(estates[6])
```

```python

```

```python
jm_list
```

```python

```

```python

```

```python

```

```python

```

[Wolfram alpha eigenvalues](https://www.wolframalpha.com/input/?i=%7B%7B1%2Ca%2C0%7D%2C%7Ba%2C0%2Ca%7D%2C%7B0%2Ca%2C-1%7D%7D+eigenvalues)




https://www2.ph.ed.ac.uk/~ldeldebb/docs/QM/lect15.pdf

https://ocw.mit.edu/courses/physics/8-05-quantum-physics-ii-fall-2013/lecture-notes/MIT8_05F13_Chap_10.pdf

https://www.ks.uiuc.edu/Services/Class/PHYS480/qm_PDF/chp6.pdf

https://quantummechanics.ucsd.edu/ph130a/130_notes/node312.html

```python

```