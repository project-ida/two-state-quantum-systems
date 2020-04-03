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


We have previously looked at a two state system whose base states **|+>** and **|->** were represented as

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


```python
plus = basis(2, 0)
minus = basis(2, 1)
```

and whose energies $E_0$ were identical. When we considered that coupling between the states could occur (with strength $A$), the hamiltonian for the system could then be represented as

$$
H = \begin{bmatrix}
 E_0  &  -A  \\
 -A  &  E_0  \\
\end{bmatrix} = E_0 I - A \sigma_x
$$


Upon investigating the time evolution of states using the above Hamiltonian, we have seen that the stationary states of the system (those of constant energy) are not |+> and |->, but instead

$\frac{|+> + \,\  |->}{\sqrt{2}}$ - in phase (a.k.a symmetric) - lower energy state

$\frac{|+> - \,\  |->}{\sqrt{2}}$ - out of phase (a.k.a anti-symmetric) - higher energy state

```python
in_phase = (plus + minus).unit()
out_phase = (plus - minus).unit()
```

and we have seen that the coupling effectively splits the energy of the two states.

In quantum mechanics classes we often talk about transitions between energy levels - such as those in the two level system. Such transitions can only be accomplished by connecting the system to the "environment". 


In the last tutorial we considered the environment to have a particular effect on our two state system, namely to directly perturb the energy of the states by an amount $\pm \delta$. Our modified Hamiltonian then took the form:



$$
H = \begin{bmatrix}
 E_0 + \delta  &  -A  \\
 -A  &  E_0 - \delta  \\
\end{bmatrix} = E_0 I - A \sigma_x + \delta\sigma_z
$$


*nb. The physical system we imagined was that of a particle with a dipole moment exposed an electric field.*

When the perturbation $\delta$ was time dependent ($\sin{\omega t}$) we discovered a resonance effect. Even when the perturbation was small, the two level system could be made to oscillate (see [Rabi cycle](https://en.wikipedia.org/wiki/Rabi_cycle)) between the upper and lower energy state when i.e. $\omega = 2A$ - this is the physical basis for stimulated emission.


## 3.2 - Quantum fields


So far we have considered the environment to be unaffected by the two-state system. This has been a convenient approximation but naturally leaves some bits of important physics out.

To capture this missing physics we must think of the environment as a field (actually maybe many fields, but let's not over complicate things for now). For example the electric field $E$ - a continuous thing (a vector thing) that exists at all points in space and time i.e $E(r,t)$. To properly describe the interaction of our quantised two-state system with such a field, we are forced to quantise the field (in some sense) as well.

But what does quantising a field mean? Answering this question in a completely satisfactory way will take us down the rather deep rabbit hole of [Quantum Field Theory](https://en.wikipedia.org/wiki/Quantum_field_theory) and [Lagrangian mechanics](https://en.wikipedia.org/wiki/Lagrangian_(field_theory)) - we will not go there today! For now, we will simply summarise the most important bits (which are by no means self evident) that will help us to explore some of the physics using QuTiP. The following might still seem a bit alien but I promise we'll get to some calculations soon.


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

It is now possible to identify the operator $a_k^{\dagger}a_k$ as having eigenvalues equal to the the number of bosons in mode k and having eigenvectors equal to the fock states, e.g.

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

Once we normalise this state we can see immediately that $|2>$ has become state $|3>$ under the $a_k^{\dagger}$ operator

```python
(a_dag*two).unit()
```

Similarly we can see that $|2>$ becomes $|1>$ under the $a_k$ operator

```python
(a*two).unit()
```

Ok, let's take a look at our Hamiltonian


## 3.3 - The Hamiltonian for a quantum field


We have seen that to describe a quantum field we need to construct a Hamiltonian of the form:

$$
H = \underset{k}{\sum} \hbar\omega_k\left(a_k^{\dagger}a_k +\frac{1}{2}\right)
$$

where the sum is over all the possible modes of the field and $a_k^{\dagger}$ and $a_k$ operators create and destroy phonons in the k'th mode.

Let's continue to consider only a single mode and let's also continue to only consider a maximum of 4 bosons in that mode. For simplicity we'll set the energy of the mode $\hbar\omega$ = 1.

```python
E_boson = 1
max_bosons = 4

a = destroy(max_bosons+1)

H = E_boson*(a.dag()*a+0.5)
```

```python
H
```

Because the Hamiltonian is diagonal there is no coupling between the different number states. Without doing any further calculation we can therefore say that if we start out with e.g. 3 bosons in the mode then we'll continue to have 3 bosons in the mode indefinitely. This is the same type of behaviour that we saw in the isolated two state system (see section 1.1).

We do expect that bosons can get created and destroyed as a result of interaction with another system, e.g. when the electrons in an transition between different energy levels. Let's see how we can model that.


## 3.4 - Coupling to a quantum field


We can make a guess at the interaction term by recalling the Hamiltonian from [tutorial 02](https://github.com/project-ida/two-state-quantum-systems/blob/master/02-perturbing-a-two-state-system.ipynb)

$$
H = \begin{bmatrix}
 A  &  \delta  \\
 \delta  &  -A  \\
\end{bmatrix} = A\sigma_z +\delta \sigma_x
$$

We interpreted $\delta$ as being related to the strength of a perturbing field. Considering only a single mode of our now quantised field, its strength can be written as the operator $a^{\dagger} + a$ - this comes from the requirement that our field be real.

We that then postulate that the interaction term be written as $V\left( a^{\dagger} + a \right)\sigma_x$, where $V$ is a coupling constant that determines how strongly the two-state system interacts with the field.

In general, finding the coupling between two quantum things required the use of guage theory looking for symmetries etc etc.

The overall Hamiltonian is then:


$$H =  A \sigma_z + \hbar\omega\left(a^{\dagger}a +\frac{1}{2}\right) + V\left( a^{\dagger} + a \right)\sigma_x$$


```python

```
