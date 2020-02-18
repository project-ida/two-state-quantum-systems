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

<a href="https://colab.research.google.com/github/project-ida/two-state-quantum-systems/blob/master/02-perturbing-a-two-state-system.ipynb" target="_parent\"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


# 2 - Perturbing a two state system


In this tutorial we are going to explore what happens if we connect a two state system to the "outside world". Or, put another way, what happens when we perturb a two state system?

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qutip import *

## note, if you start getting errors when using pandas with complex numbers then update Pandas
## - there was a bug that's been recently fixed https://github.com/pandas-dev/pandas/issues/27484
```

## 2.1 Static perturbation

<!-- #region -->
Last time we looked at an isolated two state system whose energies were identical. The hamiltonian for this system looked like this


$$
H = \begin{bmatrix}
 E_0  &  0  \\
 0  &  E_0  \\
\end{bmatrix} = E_0 I
$$

where $I$ is the identity matrix.
<!-- #endregion -->

When we allowed the possibility that the two states could be coupled (qm tunneling) i.e. the Hamiltonian looks like:
$$
H = \begin{bmatrix}
 E_0  &  -A  \\
 -A  &  E_0  \\
\end{bmatrix} = E_0 I - A \sigma_x
$$

We discovered that the two energy states split apart, $E_0+A$ and $E_0-A$.

Now we are going to explore how this coupled two state system changes when we perturb it.




Now we introduce a perturbation to the energy of the two states which differentiates between the two states. E.g. Applying an electric field to a molecule with a permanent dipole moment.

$$
H = \begin{bmatrix}
 E_0 + \delta  &  -A  \\
 -A  &  E_0 - \delta  \\
\end{bmatrix} = E_0 I - A \sigma_x + \delta\sigma_z
$$

```python
def states_to_df(states,times):
    psi_plus = np.zeros(len(times),dtype="complex128")  # To store the amplitude of the |+> state
    psi_minus = np.zeros(len(times),dtype="complex128") # To store the amplitude of the |-> state

    for i, state in enumerate(states):
        psi_plus[i] = state[0][0][0]
        psi_minus[i] = state[1][0][0]

    return pd.DataFrame(data={"+":psi_plus, "-":psi_minus}, index=times)
```

```python
plus = basis(2, 0)
minus = basis(2, 1)

in_phase = (plus + minus).unit()
out_phase = (plus - minus).unit()
```

```python
E0 = 1.0
delta = 0.01
A = 0.1

H = E0*qeye(2) - A*sigmax() + delta*sigmaz()

times = np.linspace(0.0, 70.0, 1000) 

# First let's get the evolution of the state when initialised as "in phase"
result = sesolve(H, in_phase, times)
df =  states_to_df(result.states, times)

```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
df.plot(title="Real part of amplitudes Re($\psi$)", ax=axes[0]);
(df.abs()**2).plot(title="Probabilities $|\psi|^2$", ax=axes[1]);
```

In phase state $|+> + \,\  |->$  no longer a stationary state. This is not too surprising. We have in effect raised the energy of the plus state and lowered the energy of the minus state so things have changed.

We can see what the true stationary states should be:

```python
H.eigenstates()
```

The lower energy state is symmetric as before and we have less of the plus state and more of the minus state which creates the lowest energy.

Let's see how the energy of the stationary states changes as the perturbation increases


```python
n_deltas = 50
deltas = delta*np.array(range(0,n_deltas))
upper = np.zeros(n_deltas)
lower = np.zeros(n_deltas)

for i, d in enumerate(deltas):
    H = E0*qeye(2) - A*sigmax() + d*sigmaz()
    E = H.eigenenergies()
    upper[i] = E[1]
    lower[i] = E[0]
Energies = pd.DataFrame(data={"up":upper, "low":lower, "$\delta$/A":deltas/A})
```

```python
Energies.plot(x="$\delta$/A", title="Energy", figsize=(7,6));
plt.plot((deltas/A),(E0+deltas),'k--')
plt.plot((deltas/A),(E0-deltas),'k--',label="$E_0 \pm \delta$");
plt.legend()
```

As the perturbation increases, the coupling becomes less and less important. We can see this in the energy which approaches $E_0 \pm \delta$, i.e. no dependence on $A$ at all.

The exact energy takes the form $E_0 \pm \sqrt{A^2 + \delta^2}$ (link out to formal solution for this system)

We will now consider the case when the purturbation is small, i.e $\delta/A \ll 1$. In this case the energies are approximately

$$
E_I = E_0 + A +\frac{\delta^2}{2A} \\
E_{II} = E_0 - A -\frac{\delta^2}{2A}
$$

As a side note, one might hope to be able to calcualte the above approximate energies using [first order perturbation theory](https://en.wikipedia.org/wiki/Perturbation_theory_(quantum_mechanics)#First_order_corrections) (see also more compact explanation [here](https://math.stackexchange.com/a/626736)), but the $\delta^2$ tells you this isn't possible.

You can explicity see this by using QuTip to calculate the [matrix element](http://qutip.org/docs/latest/guide/guide-basics.html?highlight=matrix%20element#functions-operating-on-qobj-class) between the perturbation $\delta\sigma_z$ and the unperturbed eigenvectors.


```python
delta*sigmaz().matrix_element(in_phase,in_phase)
```

## 2.2 Time dependent perturbation


Now let's consider a tiem dependent perturbation of the form $\delta\cos(\omega t)$. With QuTiP, we can add [time dependence in several ways](http://qutip.org/docs/latest/guide/dynamics/dynamics-time.html#function-based-time-dependence)


### Off resonance

```python
E0 = 1.0
delta = 0.01
A = 0.1

H0 = E0*qeye(2) - A*sigmax() 

H1 =  delta*sigmaz()

H = [H0,[H1,'cos(0.25*t)']]

times = np.linspace(0.0, 700.0, 1000) 

result = sesolve(H, in_phase, times)
df =  states_to_df(result.states, times)

```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
df.plot(title="Real part of amplitudes Re($\psi$)", ax=axes[0]);
(df.abs()**2).plot(title="Probabilities $|\psi|^2$", ax=axes[1]);
```

```python
def change_basis_to_df(states, times, new_basis, new_basis_labels):
    psi_new_basis_0 = np.zeros(len(times),dtype="complex128")  # To store the amplitude of the new_basis_0 state
    psi_new_basis_1 = np.zeros(len(times),dtype="complex128") # To store the amplitude of the new_basis_0 state

    for i, state in enumerate(states):
        transformed_state = state.transform(new_basis)
        psi_new_basis_0[i] = transformed_state[0][0][0]
        psi_new_basis_1[i] = transformed_state[1][0][0]

    return pd.DataFrame(data={new_basis_labels[0]:psi_new_basis_0, new_basis_labels[1]:psi_new_basis_1}, index=times)
```

```python
df = change_basis_to_df(result.states, times, [in_phase,out_phase], ["in_phase", "out_phase"])
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
df.plot(title="Real part of amplitudes Re($\psi$)", ax=axes[0]);
(df.abs()**2).plot(title="Probabilities $|\psi|^2$", ax=axes[1]);
```

### Resonance

```python
E0 = 1.0
delta = 0.01
A = 0.1

H0 = E0*qeye(2) - A*sigmax() 

H1 =  delta*sigmaz()

H = [H0,[H1,'cos(0.2*t)']]

times = np.linspace(0.0, 700.0, 1000) 

result = sesolve(H, in_phase, times)
df =  states_to_df(result.states, times)

```

```python
df = change_basis_to_df(result.states, times, [in_phase,out_phase], ["in_phase", "out_phase"])
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
df.plot(title="Real part of amplitudes Re($\psi$)", ax=axes[0]);
(df.abs()**2).plot(title="Probabilities $|\psi|^2$", ax=axes[1]);
```

We can see the period even clearer now, it's determined by the size of $\delta$

$$
T = \frac{2\pi}{\delta} \approx 630
$$

```python

```
