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

# Making sense of the total spin operator in the energy levels analogy

$S$ is always thought as the combination of many total spin operators, $S^2$ should have a clear matrix form. 
In this notebook we investigate the behavior of $S^2$ and the analogy that people make with [energy level states](https://coldfusionblog.net/2014/05/19/introduction-to-superradiance/)

```python
%matplotlib inline
import numpy as np
import pandas as pd
from qutip import *
import plotly.express as px
import matplotlib.pyplot as plt
```

```python
sup=Qobj([[1],[0]])
sdw=Qobj([[0],[1]])
sdw
```

```python
sx=0.5*Qobj([[0, 1],[1, 0]])
sx
```

```python
sy=0.5*Qobj([[0, -1j],[+1j, 0]])
sy
```

```python
sz=0.5*Qobj([[1,0],[0,-1]])
sz
```

Once we created the conventional Spin projection matrices $S_i$, we can make the $S^2$

```python
s2=sx*sx + sy*sy + sz*sz
s2
```

```python
sup=fock(2,1)

sup.dag()*sz*sup
```

## Now we are ready to make a tensor product and produce a simple hamiltonian
$$H = \sum_ {i}S_z ^{(i)}$$

```python
sz_1=tensor(sz,qeye(2))
sz_2=tensor(qeye(2),sz)
H = sz_1 + sz_2
H
```

But $H$ now seems to be the total $S_z=S_z^{(i)} +S_z^{(2)}$


## N.B. : in QuTip, excited 2D (fock) states are those with spin down

```python
ssz=H
psi=tensor(basis(2,0), basis(2,0))
psi
```

## Average value of $S_z$ in a composite state (tensor product state)

```python
psi.dag() * ssz * psi
```

## A general quantum state of the vector product of two spin 1/2 particles as a linear combination

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

## Average value of $S_z$ in a general state

```python
psig.dag() * ssz * psig   
```

# Now we want to find the matrix representation of the operator $S^2$



```python
sigmam()
```

```python
ssz
ssx= tensor(sx,qeye(2)) + tensor(qeye(2),sx)
ssy= tensor(sy,qeye(2)) + tensor(qeye(2),sy)

ss2= ssx**2 + ssy**2 + ssz**2 # + 2*( tensor(sx,qeye(2)) * tensor(qeye(2),sx)  +  tensor(sy,qeye(2)) * tensor(qeye(2),sy) + tensor(sz,qeye(2)) * tensor(qeye(2),sz)   )
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
jmat(0, '-')
```

```python

```
