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

# Making sense of the total spin operator (and the energy level analogy)

$S$ or $J$ is always thought as the combination of many total spin operators, $S^2$ should have a clear matrix form. 
In this notebook we investigate the behavior of $S^2$ and the analogy that people make when **dealing with [energy level states](https://coldfusionblog.net/2014/05/19/introduction-to-superradiance/)**

```python
%matplotlib inline
import numpy as np
import pandas as pd
from qutip import *
import plotly.express as px
import matplotlib.pyplot as plt
```

```python
sup=Qobj([[1],[0]])  #warming up with spin up and down
sdw=Qobj([[0],[1]])
sdw
```

```python
sx=0.5*Qobj([[0, 1],[1, 0]])      #also 0.5*sigmax()
sy=0.5*Qobj([[0, -1j],[+1j, 0]])
sz=0.5*Qobj([[1,0],[0,-1]])

```

Once we created the conventional Spin projection matrices $S_i$, we can make the $S^2$

```python
s2=sx**2 + sy**2 + sz**2
s2
```

```python
sup=fock(2,1)      #"excited" fock state that conventionally is spin down here

sup.dag()*sz*sup   #explicit expectation value
expect(sz,sup)     #Qutip function for expectation value
```

## Now we are ready to make a tensor product and produce a simple hamiltonian
$$H = \sum_ {i}S_z ^{(i)}$$

In this representation:  $$ 1/2 \otimes 1/2$$

```python
sz_1=tensor(sz,qeye(2))
sz_2=tensor(qeye(2),sz)
H = sz_1 + sz_2
H
```

Now $H$ is the total $S_z=S_z^{(i)} +S_z^{(2)}$


***N.B. : in QuTip, excited 2D (fock) states are those with spin down***

```python
ssz=H
psi=tensor(basis(2,0), basis(2,0))
psi
```

## Average value of $S_z$ in a composite state (tensor product state)

```python
expect(ssz, psi)
```

### A general quantum state of the vector product of two spin 1/2 particles ($ 1/2 \otimes 1/2$), constructed as a linear combination

Let's see how it works and how it behaves when feeded to the Hamiltonian

```python
uu=tensor(basis(2,0), basis(2,0))
ud=tensor(basis(2,0), basis(2,1))    #all combinations
du=tensor(basis(2,1), basis(2,0))
dd=tensor(basis(2,1), basis(2,1))

a=1j   #coefficients for all combinations
b=1
c=0
d=0

psig= (a*uu + b*ud + c*du + d*dd).unit()
psig


```

### Average value of $S_z$ in a general state

```python
psig.dag() * ssz * psig   
```

# Now we want to find the matrix representation of the operator $S^2$

According to these [lecture notes](https://ocw.mit.edu/courses/physics/8-05-quantum-physics-ii-fall-2013/lecture-notes/MIT8_05F13_Chap_10.pdf), eq. 2.9, the representatin we are intereste in is
$$1 \oplus 0 = 1/2 \otimes 1/2$$ 
we want find the right matrix representation for a sum of angular momenta as described in eq. 2.19:

$$ S^2 = (S^{(1)} + S^{(2)})^2 = (S^{(1)})^2 + (S^{(2)})^2 + 2S^{(1)}\cdot S^{(2)} $$
$$  = (S^{(1)})^2 + (S^{(2)})^2 + 2S_z^{(1)} S_z^{(2)} + S_-^{(1)} S_+^{(2)}  + S_+^{(1)} S_-^{(2)} $$

where, to be fully explicit, $S^{(1)}=S \otimes  1$

While, for calculating the square $(S^{(1)})^2$ we follow the intuitive rule of equation 1.26 [here](https://ocw.mit.edu/courses/physics/8-05-quantum-physics-ii-fall-2013/lecture-notes/MIT8_05F13_Chap_09.pdf)
Then it follows that

$$ S^2 = S_x^2 + S_y^2 + S_z^2  $$

which is simpler, since we already found the components of the total angular momentum (spin).

```python

```

```python
ssz
ssx= tensor(sx,qeye(2)) + tensor(qeye(2),sx)
ssy= tensor(sy,qeye(2)) + tensor(qeye(2),sy)

ss2= ssx**2 + ssy**2 + ssz**2 # + 2* tensor(sx,qeye(2)) * tensor(qeye(2),sx)  +  tensor(sigmam(),qeye(2)) * tensor(qeye(2),sigmap()) + tensor(sigmap(),qeye(2)) * tensor(qeye(2),sigmam())   
ss2  #here is the total spin operator  
```

This operator should allow to understand when a state is in an eigenstate of the total angular momentum: as you see it is not diagonal, it means that in this representation $|1>\otimes |2>$ needs to be rotated in order to have a diagonal (degenerate) $S^2$

```python
evals, ekets = ss2.eigenstates()
evals
```

```python
ekets[0]
```

```python
(ud.dag()*ekets[0])*(ekets[0].dag()*ud)
```

# $1 \oplus 0 = 1/2 \otimes 1/2$  : we should find the right representation for this

```python
jmat(0.5, 'z')
```

```python
sz
```

```python
kk=(tensor(sx,qeye(2)) + tensor(qeye(2),sx))**2
kk
```

```python
kk=tensor(sx,qeye(2))**2 + tensor(qeye(2),sx)**2
kk
```

```python

```

```python

```
