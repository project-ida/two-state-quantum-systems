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

<!-- #region colab_type="text" id="ncowyAh9D6Nm" -->
**Load a few helpers to pretty print Latex**
<!-- #endregion -->

```python colab={} colab_type="code" id="7lf_yUbz-g13"
from google.colab.output._publish import javascript
mathjaxurl = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/latest.js?config=default"
```

```python colab={} colab_type="code" id="unBkf_G5-jlf"
import numpy as np
from sympy import * # not sure when to use import sympy as sp
import sympy as sp
sp.init_printing(use_latex='mathjax') 
# this is the default printing and in colab it only works if the above mathjaxurl is loaded in the cell
```

<!-- #region colab_type="text" id="GMihUnr6EHfd" -->
**Starting SJB code**
<!-- #endregion -->

```python colab={} colab_type="code" id="mv_xTHN_idCP"
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from numpy.linalg import eigh
from math import sqrt
```

```python colab={} colab_type="code" id="evqSz6uKkCHq"
def S_plus_coef(S,m):
    """S+|S,m> = (BLANK) * ħ |S,m+1>. Calculate BLANK."""
    assert (S+m) % 1 == 0
    return sqrt((S-m) * (S+m+1))
def S_minus_coef(S,m):
    """S-|S,m> = (BLANK) * ħ |S,m-1>. Calculate BLANK."""
    assert (S+m) % 1 == 0
    return sqrt((S+m) * (S-m+1))
```

```python colab={} colab_type="code" id="owNWMqchkHt9"
def H_and_basis(S, min_n, max_n, sector, V, ħω, ΔE):
       
    """Construct the Hamiltonian matrix, and a table of contents defining the
    basis it is written in.
    INPUTS:
    * S, the total spin describing the collection of 2-level systems.
    * max_n, min_n, the range of possible phonon counts that we are modeling.
      (inclusive.)
    * sector is either 'odd' or 'even' or 'both' depending on parity of S+m+n
      (since the interaction conserves parity we can model each separately)
    * V, ΔE, ħω, the three parameters in the Hamiltonian (units of energy)
    OUTPUTS:
    * H, the Hamiltonian matrix
    * index_from_Smn, a dictionary with the property that
      index_from_Smn[(S,m,n)] gives the index of the row / column in H
      corresponding to (S,m,n).
    * Smn_from_index, a list in which Smn_from_index[i] is the (S,m,n)
      describing the i'th row / column in H.
    """
    ### "Table of contents" for basis for H matrix
    possible_ns = range(min_n, max_n+1)
    assert ((2*S) % 1 == 0) and (S >= 0)
    possible_ms = np.arange(2*S+1) - S
    Smn_list = product([S], possible_ms, possible_ns)
    if sector == 'even':
        Smn_from_index = [(S,m,n) for (S,m,n) in Smn_list if (S+m+n) % 2 == 0]
    elif sector == 'odd':
        Smn_from_index = [(S,m,n) for (S,m,n) in Smn_list if (S+m+n) % 2 == 1]
    else:
        assert sector == 'both'
        Smn_from_index = [(S,m,n) for (S,m,n) in Smn_list if (S+m+n) % 1 == 0]
    index_from_Smn = {Smn:index for (index,Smn) in enumerate(Smn_from_index)}

    ### Fill in entries of H matrix
    H = np.zeros(shape=(len(Smn_from_index), len(Smn_from_index)))

    # H = ΔE·Sz/ħ + ħ·ω0(a†·a + ½) + V(a† + a)(S+ + S-)/ħ
    for i,(S,m,n) in enumerate(Smn_from_index):
        # ΔE·Sz/ħ term ... note that Sz|S,m> = ħm|S,m>
        H[i,i] += ΔE * m
        # ħ·ω0(a†·a + ½) term
        H[i,i] += ħω * (n + 1/2)
        # V·a†·S+/ħ term
        if (S,m+1,n+1) in index_from_Smn:
            H[index_from_Smn[(S,m+1,n+1)],i] = S_plus_coef(S,m) * sqrt(n+1) * V
        # V·a·S+/ħ term
        if (S,m+1,n-1) in index_from_Smn:
            H[index_from_Smn[(S,m+1,n-1)],i] = S_plus_coef(S,m) * sqrt(n) * V
        # V·a†·S-/ħ term
        if (S,m-1,n+1) in index_from_Smn:
            H[index_from_Smn[(S,m-1,n+1)],i] = S_minus_coef(S,m) * sqrt(n+1) * V
        # V·a·S-/ħ term
        if (S,m-1,n-1) in index_from_Smn:
            H[index_from_Smn[(S,m-1,n-1)],i] = S_minus_coef(S,m) * sqrt(n) * V
    return H, index_from_Smn, Smn_from_index, Smn_list
```

<!-- #region colab_type="text" id="EV_DngCWHG1p" -->
**I've taken the plotting function code out into the main program, to simplify and look at variables more easily**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 225} colab_type="code" id="IvKr9ygKC27V" outputId="f1c1c284-9b99-4317-9db8-1bf6e2a514a2"
if True:        
    ΔE = 1 # shouldn't matter
    S = 0.5
    min_n = 0
    max_n = 100
    num_quanta = 11    
    sector = 'both'
    V_max = ΔE/5
    ħω=ΔE/num_quanta
    
    #### DOWN TO ZERO PHONONS

    """Plot all the eigenvalues"""
    V_list = np.linspace(0, V_max, num=100)
    energies_array = []
    all_H = list() # added a list for storing H each loop iteration below
    for V in V_list: # changed by FM in next line: get all variables back so we can look at them
        H,index_from_Smn,Smn_from_index, Smn_list= H_and_basis(S, min_n, max_n, sector, V, ħω, ΔE)
        energies,_ = eigh(H)
        energies_array.append(energies)
        all_H.append(H)
    energies_array = np.array(energies_array)
    plt.figure(figsize=(12,12))
    for i in range(np.shape(H)[1]):
        plt.plot(V_list / ΔE, energies_array[:,i]/ħω)
    plt.xlabel('V / ΔE')
    plt.ylabel('Eigenstate energy / ħω')
    plt.title('All energy levels')

    plt.ylim(-7, 60)
    plt.xlim(0, 1/5)
    plt.tight_layout()
```

<!-- #region colab_type="text" id="-5Nhh2W8I7qB" -->
**Just looking at some variables below**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="fBWkJMdqEI77" outputId="4507506b-4c91-4d5a-8301-026e29e50fa4"
len(index_from_Smn)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 84} colab_type="code" id="os5LJvykDGHl" outputId="fc503c4f-63f5-45c6-fd41-3adf1867e380"
javascript(url=mathjaxurl) # so this needs to be done for every cell where we output

index_from_Smn
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="NLKt3bedI2CD" outputId="0f5360b7-2828-4220-a65a-1881739ed398"
len(energies_array)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 374} colab_type="code" id="845dFm5lr3ll" outputId="d6ba338f-7a3e-4679-c161-cba1d6aa73b8"
energies_array[0]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="7RHHGW44TDLQ" outputId="6756b917-e321-491c-9566-4f3b8d29ea31"
len(all_H)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 386} colab_type="code" id="Q-yixvmXTG8K" outputId="a87db098-137d-4ec8-b7d7-cf1c7f8f4682"
H[99]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 187} colab_type="code" id="OUGAth7XhMQB" outputId="1439bd6c-d793-4468-990f-b0b94f97dcb6"
d = np.diag(H)
d/ħω
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} colab_type="code" id="_1BZSEI5hq7Y" outputId="13ff8db7-2bc6-45e7-bf90-835faa8c5d5d"
plt.plot(d/ħω)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="pgHUTzpQJOGH" outputId="8f4ddd41-d782-434b-933d-7d47726e01a9"
H.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 238} colab_type="code" id="yHWWcVdWJZVr" outputId="11ac1677-7de6-4a37-cac1-d85a473ffa4f"
H
```

```python colab={} colab_type="code" id="M7aNooBaeRUc"
H_sp = sp.Matrix(H)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="Xwrbpw17eC_Z" outputId="924c0043-3c81-4f7e-e55b-333cfef9eef6"
javascript(url=mathjaxurl) # so this needs to be done for every cell where we output

H_sp
```

```python colab={"base_uri": "https://localhost:8080/", "height": 459} colab_type="code" id="eRKb9tA8cvMq" outputId="39d920c0-1d1c-41af-870c-0cb7376ea30f"
energies
```

```python colab={} colab_type="code" id="v_JM3xoSlQGg"

```
