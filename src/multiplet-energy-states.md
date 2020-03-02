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

## Now we are ready to make a tensor product and produce a simple hamiltonian
$$H = \sum_ {i}S_z ^{(i)}$$

```python
sz_1=tensor(sz,qeye(2))
sz_2=tensor(qeye(2),sz)
H = sz_1 + sz_2
H
```

I would like to make the total spin $S^2$ operator, but I've got some difficoulties

```python

```

```python

```
