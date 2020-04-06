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

<a href="https://colab.research.google.com/github/project-ida/two-state-quantum-systems/blob/matt-sandbox/03-a-two-state-system-in-a-quantised-field.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/matt-sandbox/03-a-two-state-system-in-a-quantised-field.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>


# 3 - A two state system in a quantised field


> TODO: Intro

```python
# Libraries
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qutip import *
import warnings
warnings.filterwarnings('ignore')
```

## 3.1 - Recap


We have previously looked at a two state system whose states are allowed to couple to each other with strength $A$. This coupling resulted in a splitting of the states of constant energy. When we perturbed the energy of those states by an amount $\pm \delta$ we found that a natural way to represent the Hamiltonian is

$$
H = \begin{bmatrix}
 A  &  \delta  \\
 \delta  &  -A  \\
\end{bmatrix} = A\sigma_z +\delta \sigma_x
$$

The base states being used to represent this system are the stationary states of the unperturbed system ($\delta=0$) that we describe by:

$$
|+> = \begin{bmatrix}
 1   \\
 0   \\
 \end{bmatrix}, 
|-> = \begin{bmatrix}
 0   \\
 1   \\
\end{bmatrix}
$$

where |+>, |-> correspond to the higher and lower energy states respectively. 

```python
plus = basis(2, 0)
minus = basis(2, 1)
```

In the last tutorial, we made the perturbation $\delta$ time dependent ($\sin{\omega t}$) and discovered a resonance effect. Even when the perturbation was small, the two level system could be made to oscillate (see [Rabi cycle](https://en.wikipedia.org/wiki/Rabi_cycle)) between the upper and lower energy state when i.e. $\omega = 2A$ - this is the physical basis for stimulated emission.


## 3.2 - What is a quantum field?


So far we have considered the environment to be unaffected by the two-state system. This has been a convenient approximation but naturally leaves some bits of important physics out e.g. spontaneous emission.

To capture this missing physics we must think of the environment as a field (actually maybe many fields, but let's not over complicate things for now). For example the electric field $E$ - a continuous thing (a vector thing) that exists at all points in space and time i.e $E(r,t)$. To properly describe the interaction of our quantised two-state system with such a field, we are forced to quantise the field (in some sense) as well.

But what does quantising a field mean? Answering this question in a completely satisfactory way will take us down the rather deep rabbit hole of [Quantum Field Theory](https://en.wikipedia.org/wiki/Quantum_field_theory) and [Lagrangian mechanics](https://en.wikipedia.org/wiki/Lagrangian_(field_theory%29) - we will not go there today! For now, we will simply summarise the most important bits (which are by no means self evident) that will help us to explore some of the physics using QuTiP. The following might still seem a bit alien but I promise we'll get to some calculations soon.


### 3.2.1 - Fields as harmonic oscillators

The Lagrangian for a field (and resulting Hamiltonian that's of direct interest to us) can represented in a way that's mathematically equivalent to a set of independent harmonic oscillators. This is actually a classical result that comes from:
- The requirements of relativity and that the field equations (like Maxwell's equations) are linear
- The field being represented as a sum of plane waves like:
  
$$
\underset{k}{\sum} a_k(t)e^{i(k\cdot r)}
$$

For more information, see [Classical Mechanics](https://en.wikipedia.org/wiki/Classical_Mechanics_(Goldstein_book)) (Section 13.6) by Goldstein and also [Student Friendly Quantum Field Theory](https://www.quantumfieldtheory.info/) (Section 3.2.3) by Kaluber


### 3.2.2 - Quantising the field

Quantisation comes from treating the classical oscillators (labeled by their mode number k) as if they are [quantum harmonic oscillators](https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator) whose energy is known to take on discrete values:
$$
E_{k,n} = \left(n + \frac{1}{2} \right)\hbar \omega_k
$$

where $k=0,1,2,3 ...$ and $\omega \propto k$. This quantisation is technically accomplished by:
- Treating the amplitudes of the field $a_k$ as operators and not simply complex numbers
- Applying a form of Heisenberg's uncertainty principle ([Canonical commutation relation](https://en.wikipedia.org/wiki/Canonical_commutation_relation))to the $a_k$'s, i.e.
$$
[a_k a_k^{\dagger}] \equiv a_k a_k^{\dagger} - a_k^{\dagger}a_k = 1
$$

The Hamiltonian for the quantised field then looks like

$$
H = \underset{k}{\sum} \hbar\omega_k\left(a_k^{\dagger}a_k +\frac{1}{2}\right)
$$ 

and we can then identify the eigenvalues of the operator $a_k^{\dagger}a_k$ as equal to $n$ (by comparing with the form of $E_{k,n}$). To understand the deeper meaning of $a_k^{\dagger}$ and $a_k$ we must first understand how to interpret $n$.


### 3.2.3 - The meaning of $n$

When we first encounter the quantum harmonic oscillator, it is usually to model a particle moving in a quadratic potential well. There, we think of $n$ as simply a label for the energy state of the particle with energy $\left(n + \frac{1}{2} \right)\hbar \omega$. 

In quantum field theory, $n$ is interpreted as a particle number. As an example, take the $k=3$ mode with $n=2$. This means the field has 2 particles with energy $\hbar \omega_3$. These particles are what people call bosons and they have different names depending on the field being described, e.g. photons, phonons etc.

The particle number interpretation suggests that a natural way to describe the states of a quantum field is by the number of bosons in each mode, i.e. $|n_{k_1}, n_{k_2}, n_{k_3}, \cdots>$ - this is called a [Fock state](https://en.wikipedia.org/wiki/Fock_state). For simplicity, consider that there is only one mode. We can represent the different states of the field as:

$$
|0> = \begin{bmatrix}
 1   \\
 0   \\
 0   \\
 \vdots
 \end{bmatrix}, \ \ \ 
|1> = \begin{bmatrix}
 0   \\
 1   \\
 0   \\
 \vdots
\end{bmatrix}, \ \ \
|2> = \begin{bmatrix}
 0   \\
 0   \\
 1   \\
 \vdots
\end{bmatrix}
$$

In QuTiP, we can create Fock states using the same `basis` function that we used previously to represent our two-states system. For example, to represent a field with a single mode, with a maximum capacity of 4 bosons, that currently occupied by only 2 bosons, we can write:

```python
two = basis(5, 2)
two
```

Using these fock states, we can identify the operator $a_k^{\dagger}a_k$ as having eigenvalues equal to the the number of bosons in mode k and having eigenvectors equal to the fock states, e.g.

$$
a_k^{\dagger}a_k |2> = 2 |2>
$$

We call $a_k^{\dagger}a_k$ the number operator.

But what about the individual $a_k^{\dagger}$ and $a_k$ operators? How do we understand them? How can we construct them?


### 3.2.4 - The meaning of $a_k^{\dagger}$ and $a_k$
Once we understand $a_k^{\dagger}$ and $a_k$ we are in a position to build the Hamiltonian and start doing some simulations with QuTiP. 

Let's apply the number operator to a state that is not obviously one of its eigenstates. e.g. $a_k^{\dagger}|2>$

$$
a_k^{\dagger}a_k (a_k^{\dagger}|2>) = a_k^{\dagger}a_k a_k^{\dagger} |2> \overset{[a_k a_k^{\dagger}]=1}{=} a_k^{\dagger}(a_k^{\dagger}a_k+1) |2> = a_k^{\dagger}(2+1) |2> = 3a_k^{\dagger}|2>
$$

So, the number of bosons in the state $a_k^{\dagger}|2>$ is 3. In effect the $a_k^{\dagger}$ operator has created a new boson so we call it a **creation** operator. We can perform a similar calculation with the $a_k$ operator to find it reduces the number of bosons and so we call it an **annihilation** or **destruction** operator.

QuTiP allows us to construct the creation and destruction operators using the [`create` and `destroy` functions](http://qutip.org/docs/latest/apidoc/functions.html?highlight=create#qutip.operators.create)

```python
a = destroy(5)
a_dag = create(5) # we could also use a.dag()
```

Applying creation to our *two* state gives

```python
a_dag*two
```

Once we normalise this state we can see immediately that $|2>$ has become state $|3>$ under the operation of $a_k^{\dagger}$

```python
(a_dag*two).unit()
```

Similarly we can see that $|2>$ becomes $|1>$ under the $a_k$ operator

```python
(a*two).unit()
```

Now we are ready to construct the Hamiltonian.


## 3.3 - The Hamiltonian for a quantum field


We have seen that to describe a quantum field we need to construct a Hamiltonian of the form:

$$
H = \underset{k}{\sum} \hbar\omega_k\left(a_k^{\dagger}a_k +\frac{1}{2}\right)
$$

where the sum is over all the possible modes of the field and $a_k^{\dagger}$ and $a_k$ operators create and destroy phonons in the k'th mode.

Let's continue to consider only a single mode and let's also continue to only consider a maximum of 4 bosons in that mode. For simplicity we'll set the energy of the mode $\omega$ = 1 (recall $\hbar=1$ in QuTiP)

```python
omega = 1
max_bosons = 4

a = destroy(max_bosons+1)

H = omega*(a.dag()*a+0.5)
```

```python
H
```

We'll see later on that, as the Hamiltonian becomes more complicated, it can be helpful to visualise it. QuTiP has a built in way to do this using a [`hinton` diagram](http://qutip.org/docs/latest/apidoc/functions.html#qutip.visualization.hinton)

```python
f, ax = hinton(H)
ax.tick_params(axis='x',labelrotation=90)
ax.set_title("Matrix elements of H     (Fig 1)");
```

Because the Hamiltonian is diagonal there is no coupling between the different number states. Without doing any further calculation we can therefore say that if we start out with e.g. 3 bosons in the mode then we'll continue to have 3 bosons in the mode indefinitely. This is the same type of behaviour that we saw in the isolated two state system (see section 1.1).

We do expect that bosons can get created and destroyed as a result of interaction with another system, e.g. when an electron makes a transition between different energy levels. Let's see how we can model that.


## 3.4 - Coupling to a quantum field

<!-- #region -->
In general, finding the coupling between two quantum things requires expressing everything as its own field (even the two state system!) and imposing a certain symmetry on the Lagrangian of those combined fields. The interaction term in the Lagrangian (and resulting Hamiltonian) pops out as a consequence, how lovely! This is known as [gauge theory](https://en.wikipedia.org/wiki/Gauge_theory) and is an even deeper rabbit hole than quantum field theory. Needless to say, we shall not be going deeper into that today either. If you are curious I recommend  chapter 10 of [Explorations in Mathematical Physics by Koks](https://www.bookdepository.com/Explorations-Mathematical-Physics-Don-Koks/9780387309439) for an intro.

So, what can we say about the interaction of our two-state system with a quantised field without getting lost in rigor? *(not that rigor isn't important, but it will slow us down too much at the moment)*


We can make a guess at the interaction term by recalling the Hamiltonian from [Tutorial 02](https://github.com/project-ida/two-state-quantum-systems/blob/master/02-perturbing-a-two-state-system.ipynb):

$$
H = \begin{bmatrix}
 A  &  \delta  \\
 \delta  &  -A  \\
\end{bmatrix} = A\sigma_z +\delta \sigma_x
$$

We interpreted $\delta$ as being related to the strength of a perturbing field. Considering only a single mode of our now quantised field, its strength can be written as the operator $a^{\dagger} + a$ (coming from the requirement that our field be real) and we can then postulate the following interaction term:

$$V\left( a^{\dagger} + a \right)\sigma_x$$

where $V$ is a coupling constant. What we've essentially created is an interaction term that closely resembles that of an [electric](https://en.wikipedia.org/wiki/Electric_dipole_transition) and [magnetic](https://en.wikipedia.org/wiki/Magnetic_dipole_transition) dipole.


The overall Hamiltonian can then be written as:

$$H =  A \sigma_z + \hbar\omega\left(a^{\dagger}a +\frac{1}{2}\right) + V\left( a^{\dagger} + a \right)\sigma_x$$

The only remaining problem is figuring out how to make the QuTiP representations for the field and the two-state system compatible. Right now, we represent the two state system by something like

$$
|+> = \begin{bmatrix}
 1   \\
 0   \\
 \end{bmatrix}
$$

and the field is represented by something like

$$
|2> = \begin{bmatrix}
 0   \\
 0   \\
 1   \\
 0   \\
 \end{bmatrix}
$$

They clearly don't have the same dimension. Luckily QuTiP will come to our rescue.
<!-- #endregion -->

## 3.5 - Describing coupled systems in QuTiP
> TODO: Tensor product description


## 3.6 - Exploring the coupled two-state and field Hamiltonian


> TODO: Need some chat here


Let's now look at the resonant case when $2A = \omega$

```python
V = 0.1
A = 0.1
omega = 2*A
M = 4
```

```python
a  = tensor(destroy(M+1), qeye(2))     # boson destruction operator
sx = tensor(qeye(M+1), sigmax())             
sz = tensor(qeye(M+1),sigmaz())              

two_state     =  A*sz                         # two state system energy
bosons       =  omega*(a.dag()*a+0.5)       # bosons field energy
interaction   = (a.dag() + a) * sx           # interaction energy - needs to be multiplied by coupling constant in final H


H = two_state + bosons + V*interaction
```

Now that we've created these tensor products, it can be hard to interpret the Hamiltonian just by looking at the matrix without some labels. This is when the Hinton diagram we introduced earlier is super helpful.

```python
f, ax = hinton(H)
ax.tick_params(axis='x',labelrotation=90)
ax.set_title("Matrix elements of H     (Fig 2)");
```

QuTiP is smart and knows how to label things in a nice way. **|n,m>** means the following:
- **n** is the number of bosons
- **m** tells us what the state of the two state system is - 0 for higher energy state that we normally call |+> and 1 for the lower energy |-> state.


>TODO: Maybe some more chat about the hinton diagram...maybe selection rules, aka partiy?


## 3.7 - Spontaneous emission


>TODO: Intro chat

```python
# helper function
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

Start with no bosons and in the higher energy state of the two state system and see what happens

```python
psi0 = tensor(basis(M+1, 0), basis(2, 0))

times = np.linspace(0.0, 70.0, 1000) # simulation time

result = sesolve(H, psi0, times)
df_coupled =  states_to_df(result.states, times)
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
df_coupled.plot(title="Real part of amplitudes Re($\psi$)     (Fig 3)", ax=axes[0]);
(df_coupled.abs()**2).plot(title="Probabilities $|\psi|^2$     (Fig 4))", ax=axes[1]);
```

We see above that even though we start with the field "empty" of bosons (blue line), after a while, the field is likely to have a single boson at the expense of the energy in the two state which transitions to the lower state (red line) - this is where spontaneous emission comes from.


> TODO: Maybe calculate rates of emission / Einstein coefficients

```python

```
