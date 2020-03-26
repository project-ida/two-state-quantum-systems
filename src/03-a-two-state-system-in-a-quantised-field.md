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


## 3.2 - Quantising the field


So far we have considered the environment to be unaffected by the two state system. This has been a convenient approximation but naturally leaves some bits of important physics out. We are now going to explore what those are.

```python

```
