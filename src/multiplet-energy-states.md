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

**S** is always thought as the combination of many total spin operators, **S^2** should have a clear matrix form. 
In this notebook we investigate the behavior of **S^2** and the analogy that people make with [energy level states](https://coldfusionblog.net/2014/05/19/introduction-to-superradiance/)

```python
%matplotlib inline
import numpy as np
import pandas as pd
from qutip import *
import plotly.express as px
import matplotlib.pyplot as plt
```

```python
q
```
