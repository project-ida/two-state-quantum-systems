---
jupyter:
  jupytext:
    formats: ipynb,src//md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<a href="https://colab.research.google.com/github/project-ida/two-state-quantum-systems/blob/master/08-accelerating-quantum-processes.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/08-accelerating-quantum-processes.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>


# 8 - Accelerating quantum processes


Last time, we saw hints of being able to accelerate quantum processes like spontaneous emission and excitation transfer by "delocalising" excitations across many TLS. 

In this tutorial, we'll see just how much speed-up we can get by developing a description of many TLS (called "Dicke states") that allows us to set up delocalised excitations easily and also helps us avoid getting stuck in a computational bottleneck. 

We're covering a lot of ground today so we've split everything up into the following sections:

1. Recap
2. Angular momentum
3. Superradiance
4. Supertransfer
5. Superradiance vs Supertransfer

```python
# Libraries and helper functions

%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import Image

import numpy as np
from itertools import product
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from qutip import *
from qutip.piqs import *
from qutip.cy.piqs import j_min, j_vals, m_vals

from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# The helper file below brings functions created in previous tutorials
from libs.helper_07_tutorial import *
```

## 8.1 -  Recap

<!-- #region -->
So far, we've described a combined TLS & quantum field using a notation like, e.g. $|0,+,-,-,- \rangle$. The first number (in this case $0$) tells us the number of bosons present (this is often referred to as a [Fock state](https://en.wikipedia.org/wiki/Fock_state)), and the $\pm$ tell us what state each of the $N$ TLS are in ($+$ excited, $-$ ground). This is a complete description in the sense that every configuration of the system can be described as a mixture of these states. For example, a single excitation delocalised across 4 TLS with no bosons can be described by:

$\Psi_0 \sim | 0, +, -, -, - \rangle + | 0, -, +, -, - \rangle + | 0, -, -, +, - \rangle + | 0, -, -, -, + \rangle $


The issue with this description is that, for a specific number of bosons, there are $2^N$ possibilities for the state of the TLS and that means it becomes infeasible to simulate more than about 10 TLS.

Because delocalised excitations are of most interest to us today, we don't actually need a lot of the detail that the complete description holds. Superficially, we'd be quite happy with a simpler description like $| n, n_+ \rangle$ - where $n$ is the boson number and $n_+$ is the number of excitations. There would then only be $N+1$  possibilities for the state of the TLS in this case - this is much more favourable from a computational perspective. 

Let's see if we can make this simpler description rigorous enough to help us with simulating many TLS.
<!-- #endregion -->

## 8.2 - Angular momentum $J$ 


Creating a description of delocalised excitations is not quite as simple as $| n, n_+ \rangle$. For example, the following delocalised states $\Psi_1$ and $\Psi_2$ contain the same number of delocalised excitations but they're different:

$\Psi_1 \sim | 0, +, - \rangle + | 0, -, + \rangle $

```python
psi_1 = basis([2,2], [0,1]) + basis([2,2], [1,0])
psi_1 = psi_1.unit()
psi_1
```

$\Psi_2 \sim | 0, +, - \rangle - | 0, -, + \rangle $

```python
psi_2 = basis([2,2], [0,1]) - basis([2,2], [1,0])
psi_2 = psi_2.unit()
psi_2
```

What makes these states physically different is related to what we normally think of as their angular momentum. We're therefore going to need to add some angular momentum information to our states. 

Ultimately, our states are going to need to look like:

$\Psi \sim | n, j, m \rangle$

Where:
- $n$ - boson number = $0,1,2,3,...$
- $j$ - [total angular momentum quantum number](https://en.wikipedia.org/wiki/Total_angular_momentum_quantum_number) = $0,\frac{1}{2},1,\frac{3}{2},2,\frac{5}{2}, 3, ...$
- $m$ - [magnetic quantum number](https://en.wikipedia.org/wiki/Magnetic_quantum_number) = $j, (j-1), (j-2), ..., -(j-2), -(j-1), -j$

It's not obvious where all those numbers come from and how they can be mapped to e.g. 4 TLS with a single delocalised excitation? So, let's figure that out.


### Angular momentum numbers

Although we don't explicitly have a description of angular momentum in our TLS, you may recall from [tutorial 2](https://github.com/project-ida/two-state-quantum-systems/blob/master/02-perturbing-a-two-state-system.ipynb) that our system is mathematically equivalent to spin 1/2 particles which do have angular momentum. We can also see this explicitly in the language we've been using to describe our Hamiltonian:

$$H =  \Delta E J_{Nz} + \hbar\omega\left(a^{\dagger}a +\frac{1}{2}\right) + U\left( a^{\dagger} + a \right)2J_{Nx}$$

```python
H_latex = "$H = \Delta E J_{Nz} + \hbar\omega(a^{{\dagger}}a +1/2) + U( a^{{\dagger}} + a )2J_{Nx}$ "
```

where:

- The [total angular momentum operators](https://www2.ph.ed.ac.uk/~ldeldebb/docs/QM/lect15.pdf) ($J$) for $N$ TLS:

$$J_{Nx} = \overset{N}{\underset{n=1}{\Sigma}} S_{n x} \,\,\,\,\,\, J_{Ny} = \overset{N}{\underset{n=1}{\Sigma}} S_{n y} \,\,\,\,\,\, J_{Nz} = \overset{N}{\underset{n=1}{\Sigma}} S_{n z}$$

- The spin operators ($S$) for a [spin 1/2 particle](https://en.wikipedia.org/wiki/Spin-%C2%BD#Observables):

$$
S_x = \frac{1}{2}\sigma_x \,\,\,\,\,\, S_y = \frac{1}{2}\sigma_y \,\,\,\,\,\, S_z = \frac{1}{2}\sigma_z
$$


We'll continue to talk about angular momentum here to keep as sense of familiarity. Eventually we'll move to a more abstract way of thinking about $J$, but that can wait.

So far, we've seen that the x and z "components" of the total angular momentum operator $J_x$ and $J_z$ are used in the Hamiltonian. I use quote marks around "component" because this is vector language which is not obviously applicable to operators. It turns out, however, that we can (in some sense) treat the angular momentum operator as a vector (see [spinors](https://en.wikipedia.org/wiki/Spinors_in_three_dimensions)). We can create the squared "magnitude" of the total angular momentum operator ($J^2$) much like we would a vector by summing of the squares of the components.

Let's see this for the case of 2 TLS.

```python
Jx, Jy, Jz = jspin(2, basis="uncoupled")
J2 = Jx*Jx + Jy*Jy + Jz*Jz
```

What does this operator tell us about how we might go about differentiating between states like $\Psi_1$ and $\Psi_2$ that have the same amount of delocalised excitation?

If $\Psi_1$ and $\Psi_2$ are eigenstates of $J^2$ (i.e. $J^2 \Psi = \lambda \Psi$) then those states have specific, well defined, angular momentum that's characterised by the constant (eigenvalue) $\lambda$. That constant could then be used to label our states.

```python
J2*psi_1
```

```python
J2*psi_2
```

We can therefore see that:
- $J^2 \Psi_1 = 2 \Psi_1 \implies$  $\lambda = 2$
- $J^2 \Psi_2 = 0 \Psi_2 \implies$ $\lambda = 0$


Although not immediately obvious, these eigenvalues of $J^2$ [always have the form $j(j+1)$](https://www.feynmanlectures.caltech.edu/II_34.html#Ch34-S7), where $j$ is either an integer or half integer. 

```python
evalsJ, eketsJ = J2.eigenstates()
```

```python
evalsJ
```

By just looking at these eigenvalues of $J^2$, we know there must be:
- 1 state with $j=0$ (we've seen that already with $\Psi_2$)
- 3 states with $j=1$ (we've seen one of those with $\Psi_1$)


You might wonder how we're able to have 3 states with the same angular momentum number $j$? They have different $J_z$ - what is known as the "magnetic quantum number" often given the label $m$. 

In quantum mechanics, it has been found experimentally that [angular momentum is quantised](https://www.feynmanlectures.caltech.edu/II_34.html#Ch34-S7) in the sense that when its z component is measured it can only take values $m\hbar$ where $m = j, (j-1), (j-2), ..., -(j-2), -(j-1), -j$. We can see this explicitly by looking at the eigenvalues of $J_z$.

```python
evalsM, eketsM = Jz.eigenstates()
```

```python
evalsM
```

<!-- #region -->
For $N=2$, the TLS can therefore be described in terms of angular momentum by giving 2 numbers $|j,m\rangle$:
- $| 0, 0 \rangle$ - this is $\Psi_1$
- $|1, -1 \rangle$
- $|1, 0 \rangle$ - this is $\Psi_2$
- $|1, 1\rangle$

This is actually a complete description because for 2 TLS there are only 4 states, $| -, - \rangle$, $| +, - \rangle$, $| -, + \rangle$, $| +, + \rangle$. This is not the case for $N>2$.  In general, we lose the ability to describe every state uniquely when we use this angular momentum description. In other words, there can be many states with the same $j,m$ values (degenerate states). When you enumerate all the states in the angular momentum description, there are $\sim N^2$ possibilities compared to the $2^N$ we've been working with up to now. This is ultimately what's going to give us a computational advantage but we do need to be a bit careful as to whether we lose any physics when we do this. 
> Advanced: Use [`state_degeneracy(N,j)`](https://qutip.org/docs/4.4/apidoc/functions.html?highlight=m_degeneracy#qutip.piqs.state_degeneracy) to calculate the degeneracy of each state. In general there are some subtleties to consider when ignoring degeneracy which might need to be considered depending on the problem at hand (see last paragraph of [Permutational Invariant Quantum Solver](https://qutip.readthedocs.io/en/qutip-5.0.x/guide/guide-piqs.html)). For now, we don't need to worry about this so we will put a pin in this advanced topic and return to it in a later tutorial.


Hopefully you've got a better understanding of these angular momentum numbers. Now we need to link it back to the number of TLS $N$ and the number of delocalised excitations $n_+$.
<!-- #endregion -->

### Dicke states


Consider 4 TLS with a single delocalised excitation, how can we write this in our new angular momentum description with $j,m$?

$m$ is actually very closely related to the number of excitations $n_{+}$, it's:

$$m= \frac{1}{2}\left(n_{+} - n_{-} \right) = \frac{1}{2}\left(n_{+} - (N - n_{+}) \right) = n_+ - N/2$$

For 4 TLS with a single delocalised excitation, we'd have $m = 1 - \frac{4}{2} = -1$. What about $j$?

As we saw earlier, there are several $j$'s for a given $m$. In general, the specific $j$, $m$ combinations come from adding up the angular momentum for many single TLS (with $j=1/2$) like vectors of the same length but different (quantised) orientations. The details are somewhat tedious - often involving [formidable lookup tables](http://pdg.lbl.gov/2019/reviews/rpp2019-rev-clebsch-gordan-coefs.pdf). Luckily for us, QuTiP, has some convenient functions (that are somewhat hidden inside of [`qutip.cy.piqs`](https://github.com/qutip/qutip/blob/85632bc66fdcd45be51e1c280ea7577f04761a67/qutip/cy/piqs.pyx)) to help us.
- [`j_vals(N)`](https://github.com/qutip/qutip/blob/85632bc66fdcd45be51e1c280ea7577f04761a67/qutip/cy/piqs.pyx#L130) - tells us the different $j$ values for $N$ TLS.
- [`m_vals(j)`](https://github.com/qutip/qutip/blob/85632bc66fdcd45be51e1c280ea7577f04761a67/qutip/cy/piqs.pyx#L147) tells us the $m$ values for a given $j$





```python
j_vals(4) # Gives the different j's for 4 TLS
```

```python
m_vals(2) # Gives us m values for j=2
```

```python
m_vals(1) # Gives us m values for j=1
```

```python
m_vals(0) # Gives us m values for j=0
```

We can see that there is an $m=-1$ for $j=2$ and $j=1$. Which one should we pick?


By far the most significant $j$ is the largest $j_{\max} = N/2$. The largest $j$ corresponds to what's called a `Dicke state`. 

A Dicke state is a symmetric state, which means if you swap any of the TLS around, the state remains unchanged. For example, consider a single excitation in 4 TLS. The Dicke state looks like:

$\Psi_0 = \frac{1}{\sqrt{4}}\left(| 0, +, -, -, - \rangle + | 0, -, +, -, - \rangle + | 0, -, -, +, - \rangle + | 0, -, -, -, + \rangle \right)$

Notice that if you swap any two TLS, the state looks the same.

The reason why $j_{\max}$ is most significant is because of the acceleration properties that these Dicke states offer; something people often describe as superradiance and supertransfer. We're going to see these in action in the next sections.

Before we get there, we need to take a short detour into angular momentum conservation.


### Conservation of angular momentum

Angular momentum is conserved in our model and so we have a choice which $j$ value we want to run our simulation with. Once we set the system up with this $j$ it will keep that same $j$. This gives us an additional computational advantage above what we've already got from using the angular momentum description.

Instead of needing to keep track of the $N^2$ different angular momentum states, we only need to keep track of the $2j+1$ different $m$ states that correspond to the $j$ we picked. The worst case scenario is the Dicke states which use $j_{\max} = N/2$. For simulating Dicke states we need to keep track of $N+1$ states. This is an incredible improvement - going from an exponential scaling with number of TLS to linear.

You might think it would be a pain to extract only the states that correspond to a particular $j$, but once again QuTiP has got our back  with [`jmat(j)`](https://qutip.readthedocs.io/en/qutip-5.0.x/apidoc/functions.html#qutip.core.operators.jmat) - it does 2 things:
- Automatically gives us operators in the angular momentum basis, i.e. $|j,m \rangle$
- Returns only parts of the operators that act on the $j$ we pick

Let's look at an example:

```python
N = 2 # 2 TLS
Jx, Jy, Jz = jmat(N/2) # j=j_max = N/2 means only Dicke states allowed
Jz
```

<!-- #region -->
We've got a 3x3 matrix because there are only 3 states corresponding to $j=1$, they are $m=1,0,-1$ . The matrix also signposts to us QuTip's convention for labeling the angular momentum states.


$$
|1,1> = \begin{bmatrix}
 1   \\
 0   \\
 0   \\
 \end{bmatrix}, 
|1,0> = \begin{bmatrix}
 0   \\
 1   \\
0   \\
\end{bmatrix}, 
|1,-1> = \begin{bmatrix}
 0   \\
 0   \\
1   \\
\end{bmatrix}
$$

Largest $m$ at the top of the state vector, smallest $m$ at the bottom.

Ok, we're ready to explore the suped up version of quantum mechanics. 🚀
<!-- #endregion -->

## 8.3 - Superradiance


Let's reconsider the case of spontaneous emission. We saw in [tutorial 3](https://github.com/project-ida/two-state-quantum-systems/blob/master/03-a-two-state-system-in-a-quantised-field.ipynb) that such emission from a TLS can be understood as the result of coupling to a quantised field. The stronger the coupling, the faster the emission as seen by the increased Rabi frequency.

For emission from many TLS, we'd expect the rate to depend on the number of TLS that are excited. We might argue, for example, that the rate of emission is simply the sum of the rates of the individual TLS. In other words, we'd expect a factor of $N$ speed-up for $N$ TLS that are excited.

Let's simulate it and check. We're going to need to make a some modifications to our simulation code and also how we measure "emission rates".


First, we need to adapt `make_operators` function to use `j = jmax` (i.e. we'll initialise with a Dicke state) and we'll also enumerate the states in `nm_list` in terms of number of TLS that are excited $n_+$ instead of using $m$. We can do this because:

$$m = n_+ - N/2$$

Since $m = j, j-1,...-j$ and $N/2 = j$ for Dicke states, then 

$$n_+ = m+j = 2j, 2j-1, 2j-2, ..., 3,2,1,0$$



```python
def make_operators(max_bosons=2, parity=0, num_TLS=1):

    jmax = num_TLS/2              # max j gives us Dicke states
    
    J     = jmat(jmax)
    Jx    = tensor(qeye(max_bosons+1), J[0])                                     # tensorised Jx operator
    Jz    = tensor(qeye(max_bosons+1), J[2])                                     # tensorised Jx operator
    a     = tensor(destroy(max_bosons+1), qeye(J[0].dims[0][0]))                 # tensorised boson destruction operator

    two_state     = Jz                                 # two state system energy operator   Jz
    bosons        = (a.dag()*a+0.5)                    # boson energy operator              𝑎†𝑎+1/2
    number        = a.dag()*a                          # boson number operator              𝑎†𝑎
    interaction   = 2*(a.dag() + a) * Jx                # interaction energy operator        2(𝑎†+𝑎)Jx  
    
    P = (1j*np.pi*(number + Jz + num_TLS/2)).expm()    # parity operator 
    
    # map from QuTiP number states to |n,n_+> states
    possible_ns = range(0, max_bosons+1)
    possible_ms = range(int(2*jmax), -1, -1)
    nm_list = [(n,m) for (n,m) in product(possible_ns, possible_ms)]

    
    if (parity==1) | (parity==-1):
        p               = np.where(P.diag()==parity)[0]
    else:
        p               = np.where(P.diag()==P.diag())[0]
        
    two_state       = two_state.extract_states(p)
    bosons          = bosons.extract_states(p)
    number          = number.extract_states(p)
    interaction     = interaction.extract_states(p)
    nm_list        = [nm_list[i] for i in p]
  
    return two_state, bosons, interaction, number, nm_list
```

Next, we need to figure out how we're going to measure emission rates.


### Emission rates

So far in this tutorial series, we've been using the Rabi frequency $\Omega$ as a measure of the speed of quantum processes like spontaneous emission and excitation transfer. It's a useful metric for systems that exhibit a single Rabi frequency. However, it doesn't formally give us an emission/transfer/transition "rate". 

In quantum mechanics, the transition rate $\left(\Gamma\right)$ from a particular state $\Psi$ is defined by the rate of change of the probability to be in that state $P_{\Psi}$. In other words:

$$\Gamma = \frac{d P_\Psi}{dt}$$

$\Gamma$ essentially tells us how much probability has accumulated per unit time for a transition to occur.

Sometimes it can be more helpful to look at rate of change of the expectation values of certain operators, e.g. the boson number operator $a^{\dagger}a$. Defining the rate by:

$$\Gamma = \frac{d \langle a^{\dagger}a \rangle}{dt}$$

gives us a more direct measure of boson emission rates.

In either case, the challenge for us is that we've seen in previous tutorials that these rates approach zero for short times because $P \sim \cos^2\left(... t \right)$ and $\langle a^{\dagger}a\rangle \sim \sin^2\left(... t  \right)$ initially. This makes comparing rates for different numbers of TLS a numerical nightmare because we get uncomfortably close to doing $\frac{0}{0}$ 😬.

A solution is to create an analytical model of the dynamics at short times scales. Specifically, we can do a taylor expansion up to $t^2$:

$$\langle a^{\dagger}a\rangle = a + bt + ct^2$$

and because we expect $b=0$ (because $\Gamma \rightarrow 0$ as $t \rightarrow 0$), we get the following rate:

$$\frac{d \langle a^{\dagger}a \rangle}{dt} = 2ct$$

With this expression, comparing rates between different scenarios (1 and 2) works out to be the ratio of the expansion coefficients:

$$\frac{\Gamma_1}{\Gamma_2} = \frac{c_1}{c_2}$$

no nasty divide by zero problem 🙌.

We can us SciPy's [`curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) to extract the expansion coefficients we need.

Let's do it!


### Fully excited system

```python
# a + bt + ct^2. 
# a=0 because number of bosons 0 to start
# b=0 because (from experience) rate is 0 to start 
def model_func(t, c):
    return c*t**2
```

```python
DeltaE = 1 # We match the boson energy to the TLS energy to make sure we'll get emission like behaviour
omega = 1
U = 0.001 # Coupling is 10x lower than the last tutorial because we are upping the number of TLS
```

```python
%%time 
# %%time must be at the top of the cell and by itself. It tells you the "Wall time" (how long the cell took to run) at the end of the output (should be about 2 min).

Ns = [1,2,4,8,16,32,64,128] # number of TLS we want to simulation
times = np.linspace(0,  2000, 2000)
rate = [] # For storing emission rates

for i, N in enumerate(Ns):
    if N==1:
        # For N=1 the parity is opposite to the other N's
        two_state, bosons, interaction, number, nm_list = make_operators(max_bosons=N+1, parity=-1, num_TLS=N)
    else:
        two_state, bosons, interaction, number, nm_list = make_operators(max_bosons=N+1, parity=1, num_TLS=N)

    bra_labels, ket_labels = make_braket_labels(nm_list)
    
    H = DeltaE*two_state + omega*bosons + U*interaction

    psi0_ind = nm_list.index((0,N))  # Field in vacuum state (0) with N excitations (N)
    psi0 = basis(len(nm_list), psi0_ind)


    # We can use use QuTips sesolve here because of the shorter
    # simulation time. Sometimes sesolve is still fastser than
    # our custom solver because of unknown optimisations made
    # made the QuTip/Numpy teams.
    # Note progress bar because the last simulation will take about 5 min
    result = sesolve(H, psi0, times, [number], progress_bar=True)

    # For fitting, we'll find when the first boson is emitted, then re-simulate 
    # up to that point so that we can get a better resolution over the shorter time periods.
    # We do this because we expect timescales to shorten as we increase TLS number
    if N==1:
        # N=1 is special because number of bosons never crosses 1 so we need to use find_peaks just like in the last tutorial
        peaks, _ = find_peaks(result.expect[0], prominence=0.05)
        peak_times = times[peaks]
        time_to_emit_one = peak_times[0]
    else: 
        # Approximate time when the expected bosons reaches 1. We look for when expected bosons crosses 1.
        crossing_one_index = np.where(np.diff((result.expect[0] > 1).astype(int)))[0][0]
        time_to_emit_one = times[crossing_one_index]

    times_fit = np.linspace(0,  time_to_emit_one/10, 100)
    result_fit = sesolve(H, psi0, times_fit, [number])

    fit, covariance = curve_fit(model_func, times_fit, result_fit.expect[0],p0=[0.01],maxfev=500)

    rate.append(2*fit[0])

    plt.plot(times, result.expect[0], label="Expected bosons")
    plt.plot(times,model_func(times,*fit),label="Fit")
    plt.xlabel("Time")
    plt.ylim([0,result.expect[0].max()*1.1])
    plt.legend(loc="right")
    plt.title(f"{H_latex} (Fig. {i+1})  \n $\Delta E={DeltaE}$, $\omega={omega}$, $U={U}$, N={N} \n $\Psi_0 =$ {ket_labels[psi0_ind]}")
    plt.show();
```

Although superficially our model fit doesn't look great in e.g. Fig. 8, you'll notice that if we plot for short time scales over which the model was fitted it's fine.

```python
plt.plot(times_fit, result_fit.expect[0], label="Expected bosons")
plt.plot(times_fit,model_func(times_fit,*fit),label="Fit")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title(f"{H_latex} (Fig. 9)  \n $\Delta E={DeltaE}$, $\omega={omega}$, $U={U}$, N={N} \n $\Psi_0 =$ {ket_labels[psi0_ind]}")
plt.show();
```

We'll store the "base" rate of emission $\Gamma_{1E}$ for the case of a single TLS so that we might reference it later

```python
gamma_1E = rate[0]
gamma_1E
```

Let's now see how the emission rate varies with with number of TLS $N$. 

```python
plt.plot(Ns,rate/gamma_1E,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised emission rate ($\Gamma/\Gamma_1$)");
plt.title("Emission from N delocalised excitations (Fig. 10)");
```

Fig. 9 shows us exactly what we expected - a nice linear relationship.

Let's not celebrate too soon though.

If we apply our intuition to the case of a single delocalised excitation spread across $N$ TLS, then we'd expect the emission rates to be independent of $N$. This is however not the case. 

Let's see.


### Single delocalised excitation

```python
%%time 
# %%time must be at the top of the cell and by itself. It tells you the "Wall time" (how long the cell took to run) at the end of the output (should be about 2 min).

Ns = [2,4,8,16,32,64,128] # number of TLS we want to simulation
times = np.linspace(0, 2000, 1000)
rate = [] # For storing emission rates

for i, N in enumerate(Ns):

    two_state, bosons, interaction, number, nm_list = make_operators(max_bosons=2, parity=-1, num_TLS=N)

    bra_labels, ket_labels = make_braket_labels(nm_list)
    
    H = DeltaE*two_state + omega*bosons + U*interaction

    psi0_ind = nm_list.index((0,1))  # Field in vacuum state (0) with 1 excitations (1)
    psi0 = basis(len(nm_list), psi0_ind)

    # We can use use QuTips sesolve here because of the shorter
    # simulation time. Sometimes sesolve is still fastser than
    # our custom solver because of unknown optimisations made
    # made the QuTip/Numpy teams.
    result = sesolve(H, psi0, times, [number])

    # For fitting, we'll find when the first boson is emitted, then re-simulate 
    # up to that point so that we can get a better resolution over the shorter time periods.
    # We do this because we expect timescales to shorten as we increase TLS number
    
    # Because we only have a single excitation the number of bosons never crosses 1 
    # so we need to use find_peaks just like in the last tutorial
    peaks, _ = find_peaks(result.expect[0], prominence=0.05)
    peak_times = times[peaks]
    time_to_emit_one = peak_times[0]

    times_fit = np.linspace(0,  time_to_emit_one/10, 100)
    result_fit = sesolve(H, psi0, times_fit, [number])

    fit, covariance = curve_fit(model_func, times_fit[0:100], result_fit.expect[0][0:100],p0=[0.01],maxfev=500)

    rate.append(2*fit[0])


    plt.plot(times, result.expect[0], label="Expected bosons")
    plt.plot(times,model_func(times,*fit),label="Fit")
    plt.xlabel("Time")
    plt.ylim([0,result.expect[0].max()*1.1])
    plt.legend(loc="right")
    plt.title(f"{H_latex} (Fig. {i+11})  \n $\Delta E={DeltaE}$, $\omega={omega}$, $U={U}$, N={N} \n $\Psi_0 =$ {ket_labels[psi0_ind]}")
    plt.show();
```

```python
plt.plot(Ns,rate/gamma_1E,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised emission rate ($\Gamma/\Gamma_1$)");
plt.title("Emission from 1 delocalised excitation (Fig. 18)");
```

Fig. 18 shows us that we get the same enhancement of emission rates whether we have single excitation or $N$ of them 🤔.

Stranger still is the case where half of the TLS are excited.



### Half excited system

```python
%%time 
# %%time must be at the top of the cell and by itself. It tells you the "Wall time" (how long the cell took to run) at the end of the output (should be about 2 min).

Ns = [4,8,16,32,64,128] # number of TLS we want to simulation
times = np.linspace(0,  2000, 2000)
rate = [] # For storing emission rates

for i, N in enumerate(Ns):

    two_state, bosons, interaction, number, nm_list = make_operators(max_bosons=int(N/2)+1, parity=1, num_TLS=N)

    bra_labels, ket_labels = make_braket_labels(nm_list)
    
    H = DeltaE*two_state + omega*bosons + U*interaction

    psi0_ind = nm_list.index((0,int(N/2)))  # Field in vacuum state (0) with N/2 excitations (int(N/2))
    psi0 = basis(len(nm_list), psi0_ind)


    # We can use use QuTips sesolve here because of the shorter
    # simulation time. Sometimes sesolve is still fastser than
    # our custom solver because of unknown optimisations made
    # made the QuTip/Numpy teams.
    # Note progress bar because the last simulation will take about 5 min
    result = sesolve(H, psi0, times, [number],progress_bar=True)

    # For fitting, we'll find when 1 boson is emitted and re-sim8ulate up to that point
    # so that we can get a better resolution over the shorter time periods. We
    # do this because we expect timescales to shorten as we increase TLS number

    peaks, _ = find_peaks(result.expect[0], prominence=0.05)
    peak_times = times[peaks]
    time_to_peak = peak_times[0]

    # For fitting, we'll find when the first boson is emitted, then re-simulate 
    # up to that pointso that we can get a better resolution over the shorter time periods.
    # We do this because we expect timescales to shorten as we increase TLS number
    crossing_one_index = np.where(np.diff((result.expect[0] > 1).astype(int)))[0][0]
    time_to_emit_one = times[crossing_one_index]

    times_fit = np.linspace(0,  time_to_emit_one/10, 100)
    result_fit = sesolve(H, psi0, times_fit, [number])

    fit, covariance = curve_fit(model_func, times_fit, result_fit.expect[0],p0=[0.01],maxfev=500)

    rate.append(2*fit[0])


    plt.plot(times, result.expect[0], label="Expected bosons")
    plt.plot(times,model_func(times,*fit),label="Fit")
    plt.xlabel("Time")
    plt.ylim([0,result.expect[0].max()*1.1])
    plt.legend(loc="right")
    plt.title(f"{H_latex} (Fig. {i+19})  \n $\Delta E={DeltaE}$, $\omega={omega}$, $U={U}$, N={N} \n $\Psi_0 =$ {ket_labels[psi0_ind]}")
    plt.show();
```

```python
plt.plot(Ns,rate/gamma_1E,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised emission rate ($\Gamma/\Gamma_1$)");
plt.title("Emission from $N/2$ delocalised excitations (Fig. 25)");
```

Fig. 25 shows us that, with $N/2$ excitations, the emission rate is even greater than when all of the TLS are excited 🤯!

Let's quantify this by doing a linear regression of $\log(\Gamma)$ with  $\log(N)$

```python
print("slope = ", linregress(np.log10(Ns), np.log10(rate)).slope)
```

We can see that $\Gamma \sim N^2$ when $n_+ = N/2$. 

This kind of counter intuitive emission rate enhancement was discovered by Dicke in his 1956 paper [Coherence in Spontaneous Radiation Processes](https://journals.aps.org/pr/abstract/10.1103/PhysRev.93.99) where he coined the turn "superradiance".

In general, Dicke found that when $n_+$ excitations are delocalised across $N$ TLS, the emission rate $\Gamma$ is enhanced over the single TLS emission rate $\Gamma_{1E}$ by:

$$\frac{\Gamma}{\Gamma_{1E}} = n_+\left(N - n_+ +1\right)$$

where we can see that $\Gamma$ is largest when half the TLS are excited giving a rate of $\Gamma/\Gamma_{1E} = \frac{N}{2}\left(\frac{N}{2}+1\right)$.

How can we understand this?


### Understanding superradiance

Superradiance might at first seem counterintuitive, but we can understand it from one of the most fundamental principles of quantum mechanics [according to Richard Feynman](https://www.feynmanlectures.caltech.edu/III_01.html#Ch1-S7) which reads:

```
"When an event can occur in several alternative ways, the probability amplitude for the event is the sum of the probability amplitudes for each way considered separately. There is interference"
```

Let's take the example of 2 excitations amongst 4 TLS. The initial delocalised Dicke state looks like:

$\Psi_i = \frac{1}{\sqrt{6}}\left(| 0, +, +, -, - \rangle + | 0, +, -, +, - \rangle + | 0, +, -, -, + \rangle + | 0, -, +, +, - \rangle + | 0, -, +, -, + \rangle + | 0, -, -, +, + \rangle \right)$

We can see there are 6 different configurations for the TLS. Each of the 2 excitations in those 6 configurations could transition from $+$ to a $-$ with a release of a single boson. That means each of the 6 configurations has 2 emission paths that it could go in order to reach one of 4 configurations in final the state:

$\Psi_f = \frac{1}{\sqrt{4}}\left(| 0, +, -, -, - \rangle + | 0, -, +, -, - \rangle + | 0, -, -, +, - \rangle + | 0, -, -, -, + \rangle \right)$

The total number of emission paths is therefore $6\times 2 = 12$. Each of these paths contributes the same to the overall emission amplitude because the Dicke state is constructed with $+$ between each of the configurations that make up the state. This creates what's called "constructive interference" where the effects of each path add up to a larger effect. 

To get the numbers right, we must remember that our states are normalised. The 6 configurations in our starting state means dividing the amplitude by $\sqrt{6}$. The 4 configurations in the final state means dividing the amplitude by $\sqrt{4}$. So the overall amplitude enhancement factor is:

$$\frac{6\times 2}{\sqrt{6}\sqrt{4}} = \sqrt{6}$$

and so the probability enhancement factor (which is proportional to the emission rate) is the square of this, i.e. 6. This is consistent with Dicke's formula.

To derive the general Dicke formula, we just have to do this counting and normalising for the general case:

$$\frac{\Gamma}{\Gamma_{1E}} = \left(\frac{^N C_{n_+} n_+}{\sqrt{^N C_{n_+}}\sqrt{^N C_{{n_+}-1}}}\right)^2 = n_+\left(N-n_++1\right)$$

Now we can really understand just how important those "$+$'s" are that make up the Dicke state. As soon as you allow any "$-$" you reduce the emission rates. Take for example the case of 2 TLS with a single delocalised excitation. If instead of a Dicke state

$\Psi = \frac{1}{\sqrt{2}}\left(| 0, +, -\rangle + | 0, -, + \rangle \right)$

we instead use 

$\Psi = \frac{1}{\sqrt{2}}\left(| 0, +, -\rangle - | 0, -, + \rangle \right)$

then we get no emission at all because we get complete destructive interference of the two paths. Such states are often referred to as "dark states".

Now that we've seen what acceleration factors are possible for spontaneous emission, we might expect to find something similar in the realm of excitation transfer.


## 8.4 - Supertransfer


Just like with superradiance, we're going to work with Dicke states to allow us to conveniently describe and simulate delocalised excitations.

To describe delocalised excitations transferring from one "place" to another, we need to break-up our overall system into 2 parts - system A and system B. Each system can in principle have its own number of TLS ($N_A$, $N_B$) and its own number of excitations ($n_{+A}$, $n_{+B}$). 

The general Hamiltonian for this AB situation is described by:

$$H =  \Delta E_A J_{N_Az}^{(A)} + \Delta E_B J_{N_Bz}^{(B)} + \hbar\omega\left(a^{\dagger}a +\frac{1}{2}\right) + U_A\left( a^{\dagger} + a \right)2J_{N_Ax}^{(A)} + U_B\left( a^{\dagger} + a \right)2J_{N_Bx}^{(B)}$$

It's a bit full on so we won't investigate this Hamiltonian in all its generality today. We'll focus on the case where:
- System A and B have the same transition energy: $ \Delta E_A =  \Delta E_B =  \Delta E$
- System A and B couple to the boson field in the same way: $U_A = U_B = U$
- System A and B consist of the same number ot TLS: $N_A = N_B = N$

```python
H_latex_AB = "$H = \Delta E (J_{Nz}^{(A)}+J_{Nz}^{(B)}) + \hbar\omega(a^{{\dagger}}a +1/2) + U( a^{{\dagger}} + a )2(J_{Nx}^{(A)} + J_{Nx}^{(B)})$ "
```

The process of constructing the additional operators is similar to when we added the quantised field operators back in [tutorial 03](https://github.com/project-ida/two-state-quantum-systems/blob/master/03-a-two-state-system-in-a-quantised-field.ipynb) - we create tensor products of the different operators to make sure they only act on the relevant parts of the state.

```python
def make_operators_AB(max_bosons=2, parity=0, num_TLS_A=1, num_TLS_B=1):

    jmax_A = num_TLS_A/2              # max j gives us Dicke states
    jmax_B = num_TLS_B/2              # max j gives us Dicke states
    
    J_A     = jmat(jmax_A)
    J_B     = jmat(jmax_B)
    Jx_A    = tensor(qeye(max_bosons+1), J_A[0], qeye(J_B[0].dims[0][0]))                                     # tensorised JxA operator
    Jz_A    = tensor(qeye(max_bosons+1), J_A[2], qeye(J_B[0].dims[0][0]))                                     # tensorised JzA operator
    Jx_B    = tensor(qeye(max_bosons+1), qeye(J_A[0].dims[0][0]), J_B[0])                                     # tensorised JxB operator
    Jz_B    = tensor(qeye(max_bosons+1), qeye(J_A[0].dims[0][0]), J_B[2])                                     # tensorised JzB operator
    a       = tensor(destroy(max_bosons+1), qeye(J_A[0].dims[0][0]), qeye(J_B[0].dims[0][0]))                 # tensorised boson destruction operator

    two_state_A     = Jz_A                                 # two state system energy operator   JzA
    two_state_B     = Jz_B                                 # two state system energy operator   JzB
    bosons        = (a.dag()*a+0.5)                       # boson energy operator              𝑎†𝑎+1/2
    number        = a.dag()*a                             # boson number operator              𝑎†𝑎
    interaction_A  = 2*(a.dag() + a) * Jx_A                # interaction energy operator        2(𝑎†+𝑎)JxA 
    interaction_B  = 2*(a.dag() + a) * Jx_B                # interaction energy operator        2(𝑎†+𝑎)JxB
    
    P = (1j*np.pi*(number + Jz_A + Jz_B + (num_TLS_A + num_TLS_B)/2)).expm()    # parity operator 
    
    # map from QuTiP number states to |n,n_+A,n_+B> states
    possible_ns = range(0, max_bosons+1)
    possible_ms_A = range(int(2*jmax_A), -1, -1)
    possible_ms_B = range(int(2*jmax_B), -1, -1)
    nmm_list = [(n,m1,m2) for (n,m1,m2) in product(possible_ns, possible_ms_A, possible_ms_B)]

    
    if (parity==1) | (parity==-1):
        p               = np.where(P.diag()==parity)[0]
    else:
        p               = np.where(P.diag()==P.diag())[0]
        
    two_state_A       = two_state_A.extract_states(p)
    two_state_B       = two_state_B.extract_states(p)
    bosons          = bosons.extract_states(p)
    number          = number.extract_states(p)
    interaction_A     = interaction_A.extract_states(p)
    interaction_B     = interaction_B.extract_states(p)
    nmm_list        = [nmm_list[i] for i in p]
  
    return two_state_A, two_state_B, bosons, interaction_A, interaction_B, number, nmm_list
```

<!-- #region -->
We're aaaalmost ready to rock and roll, but we need a couple of extra bits.

Firstly, let's create an operator that helps us count the number of excitations in system A or B (a bit like our `number` operator for bosons). We can create such an operator from our `two_state` operators because they are just $J_z$ which is just a measure of the $m$ number which in turn is related to the excitation number via $n_+ = m + N/2$ for Dicke states. In the language of expectation values $\langle ... \rangle$:


$\langle \text{two\_state}\rangle = \langle J_z\rangle = \langle m\rangle = \langle n_+ \rangle - N/2$

Next, calculating expectation values. You may recall in the last tutorial that when we simulated excitation transfer we used our custom simulate function from [tutorial 05](https://github.com/project-ida/two-state-quantum-systems/blob/master/05-excitation-transfer.ipynb) because it was quicker over of the long simulation times typically required of excitation transfer. We're going to do the same here. This means we'll need our own way to calculate the expectation values (sesolve did this for us before). Although QuTiP does have the [`expect`](https://qutip.readthedocs.io/en/qutip-5.0.x/apidoc/functions.html#qutip.core.expect.expect) function, it turns out that we need to create a `Qobj` for every time step in order to use this function and that can be very slow. We will instead directly calculate the expectation value using matrix multiplication, i.e.

$<H> = \psi^{\dagger}H\psi = \psi^{\dagger} @ (H @\psi) $

Where @ is the matrix multiplication operator and $\dagger$ in this context means taking the complex conjugate.

Let's automate this process for all time steps using a function.
<!-- #endregion -->

```python
# "states" will be the output of Psi from our custom "simulate" function
def expectation(operator, states):
    operator_matrix = operator.full()
    operator_expect = np.zeros(states.shape[1], dtype=complex)
    for i in range(0,shape(states)[1]):
        e = np.conj(states[:,i])@ (operator_matrix @ states[:,i])
        operator_expect[i] = e
    return operator_expect
```

Let's remember now to adjust the $\Delta E \neq 1$ to make sure we don't get energy transfer to the boson field.

```python
DeltaE = 2.5 # Mismatch between boson energy and the TLS energy to make sure we avoid emission
omega = 1
U = 0.001 # Coupling is 10x lower than the last tutorial because we are upping the number of TLS
```

### Fully excited system A, de-excited system B

```python
%%time 
# %%time must be at the top of the cell and by itself. It tells you the "Wall time" (how long the cell took to run) at the end of the output (should be about 2 min).

Ns = [1,2,4,8,16,32,64] # number of TLS we want to simulation
times = np.linspace(0,  5000000, 1000)
rate = [] # For storing emission rates

for i, N in enumerate(Ns):
    if N==1:
        # For N=1 the parity is opposite to the other N's
        two_state_A, two_state_B, bosons, interaction_A, interaction_B, number, nmm_list = make_operators_AB(max_bosons=2, parity=-1, num_TLS_A=N, num_TLS_B=N)
    else:
        two_state_A, two_state_B, bosons, interaction_A, interaction_B, number, nmm_list = make_operators_AB(max_bosons=2, parity=1, num_TLS_A=N, num_TLS_B=N)


    bra_labels, ket_labels = make_braket_labels(nmm_list)    
    
    H = DeltaE*two_state_A + DeltaE*two_state_B + omega*bosons + U*interaction_A + U*interaction_B

    # Field in vacuum state (0) with N excitations in A (N) and 0 excitations in B (0)
    psi0_ind = nmm_list.index((0,N,0))  
    psi0 = basis(len(nmm_list), psi0_ind)


    # We are using custom simulate function from last tutorial because it's going to be quicker
    # in this case because of the long simulation times
    P, psi, evals, ekets = simulate(H, psi0, times)
    num_A = expectation(two_state_A + N/2, psi)
    num_B = expectation(two_state_B + N/2, psi)

    # For fitting, we'll find when the first excitaton is transferred to B, then re-simulate 
    # up to that point so that we can get a better resolution over the shorter time periods.
    # We do this because we expect timescales to shorten as we increase TLS number
    if N==1:
        # N=1 is special because excitations never crosses 1 so we need to use find_peaks just like in the last tutorial
        peaks, _ = find_peaks(num_B, prominence=0.05)
        peak_times = times[peaks]
        time_to_transfer_one = peak_times[0]
    else: 
        # Approximate time when the expected excitations in B reaches 1.
        crossing_one_index = np.where(np.diff((num_B > 1).astype(int)))[0][0]
        time_to_transfer_one = times[crossing_one_index]

    times_fit = np.linspace(0,  time_to_transfer_one/10, 100)
    P_fit, psi_fit, *_ = simulate(H, psi0, times_fit, evals, ekets)
    num_B_fit = expectation(two_state_B + N/2, psi_fit)

    fit, covariance = curve_fit(model_func, times_fit, num_B_fit,p0=[0.01],maxfev=500)

    rate.append(2*fit[0])


    plt.plot(times, num_A, label="Expected A excitations")
    plt.plot(times, num_B, label="Expected B excitations")
    plt.plot(times,model_func(times,*fit),label="Quadratic fit")
    plt.xlabel("Time")
    plt.ylim([0,num_A.max()*1.1])
    plt.legend(loc="right")
    plt.title(f"{H_latex_AB} \n $\Delta E={DeltaE}$, $\omega={omega}$, $U={U}$, N={N} \n $\Psi_0 =$ {ket_labels[psi0_ind]}     (Fig. {i+26})")
    plt.show();
```

We'll once again store the "base" rate of excitation transfer $\Gamma_{1T}$ for the case of a single TLS in each system A and B so that we might reference it later

```python
gamma_1T = rate[0]
gamma_1T
```

```python
plt.plot(Ns,rate/gamma_1T,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised transfer rate ($\Gamma/\Gamma_1$)");
plt.title("Transfer of $N$ delocalised excitations from A to B (Fig. 33)");
```

```python
print("slope = ", linregress(np.log10(Ns), np.log10(rate)).slope)
```

Fig. 33 confirms the observation from the last tutorial that the excitation transfer has a more favourable scaling with the number of TLS than spontaneous emission - $N^2$ vs $N$ for the fully excited case.
> Note that in the last tutorial we observed $N$ vs $\sqrt{N}$ because we were looking at the Rabi frequency instead of the emission/transfer rates.

We can think of this $N^2$ scaling in terms of pathways just like with spontaneous emission. When an excitation moves from the fully excited system A to de-excited system B, it leaves behind a "hole" in system A which can be in one of $N$ places and moves to system B where it can be one of $N$ places.

Let's see if we get the same scaling $N^2$ with a single excitation in A - mirroring what we found with spontaneous emission.


### Single delocalised excitation in system A, de-excited system B

```python
%%time 
# %%time must be at the top of the cell and by itself. It tells you the "Wall time" (how long the cell took to run) at the end of the output (should be about 2 min).

Ns = [2,4,8,16,32,64] # number of TLS we want to simulation
times = np.linspace(0,  5000000, 1000)
rate = [] # For storing emission rates

for i, N in enumerate(Ns):

    two_state_A, two_state_B, bosons, interaction_A, interaction_B, number, nmm_list = make_operators_AB(max_bosons=2, parity=-1, num_TLS_A=N, num_TLS_B=N)

    bra_labels, ket_labels = make_braket_labels(nmm_list)    
    
    H = DeltaE*two_state_A + DeltaE*two_state_B + omega*bosons + U*interaction_A + U*interaction_B

    # Field in vacuum state (0) with 1 excitation in A (1) and 0 excitations in B (0)
    psi0_ind = nmm_list.index((0,1,0))  
    psi0 = basis(len(nmm_list), psi0_ind)


    # We are using custom simulate function from last tutorial because it's going to be quicker
    # in this case because of the long simulation times
    P, psi, evals, ekets = simulate(H, psi0, times)
    num_A = expectation(two_state_A + N/2, psi)
    num_B = expectation(two_state_B + N/2, psi)

    # For fitting, we'll find when the first excitaton is transferred to B, then re-simulate 
    # up to that point so that we can get a better resolution over the shorter time periods.
    # We do this because we expect timescales to shorten as we increase TLS number

    # Because we only have a single excitation the number of bosons never crosses 1 
    # so we need to use find_peaks just like in the last tutorial
    peaks, _ = find_peaks(num_B, prominence=0.05)
    peak_times = times[peaks]
    time_to_transfer_one = peak_times[0]


    times_fit = np.linspace(0,  time_to_transfer_one/10, 100)
    P_fit, psi_fit, *_ = simulate(H, psi0, times_fit, evals, ekets)
    num_B_fit = expectation(two_state_B + N/2, psi_fit)

    fit, covariance = curve_fit(model_func, times_fit, num_B_fit,p0=[0.01],maxfev=500)

    rate.append(2*fit[0])


    plt.plot(times, num_A, label="Expected A excitations")
    plt.plot(times, num_B, label="Expected B excitations")
    plt.plot(times,model_func(times,*fit),label="Quadratic fit")
    plt.xlabel("Time")
    plt.ylim([0,num_A.max()*1.1])
    plt.legend(loc="right")
    plt.title(f"{H_latex_AB} \n $\Delta E={DeltaE}$, $\omega={omega}$, $U={U}$, N={N} \n $\Psi_0 =$ {ket_labels[psi0_ind]}   (Fig. {i+34})")
    plt.show();
```

```python
plt.plot(Ns,rate/gamma_1T,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised transfer rate ($\Gamma/\Gamma_1$)");
plt.title("Transfer of $N$ delocalised excitations from A to B (Fig. 40)");
```

```python
print("slope = ", linregress(np.log10(Ns), np.log10(rate)).slope)
```

Fig. 40 confirms that we have the same $N^2$ scaling for excitation transfer even when there is only a single excitation in system A.

Thinking again in terms of pathways. The single excitation in system A can be in one of $N$ places and the excitation in each of of those "configurations" can move to one of $N$ "empty" places in the de-excited system B.

The natural next step is to wonder what happens if we play the $N/2$ game. In other words, if we "half-excite" system A, will we find some kind of "supertransfer". 

I think you know what the answer is 😉 but let's see it.


### Half excited system A, de-excited system B

```python
%%time 
# %%time must be at the top of the cell and by itself. It tells you the "Wall time" (how long the cell took to run) at the end of the output (should be about 2 min).

Ns = [2,4,8,16,32,64] # number of TLS we want to simulation
times = np.linspace(0,  5000000, 1000)
rate = [] # For storing emission rates

for i, N in enumerate(Ns):
    if N==2:
        # For N=2 the parity is opposite to the other N's
        two_state_A, two_state_B, bosons, interaction_A, interaction_B, number, nmm_list = make_operators_AB(max_bosons=2, parity=-1, num_TLS_A=N, num_TLS_B=N)
    else:
        two_state_A, two_state_B, bosons, interaction_A, interaction_B, number, nmm_list = make_operators_AB(max_bosons=2, parity=1, num_TLS_A=N, num_TLS_B=N)

    bra_labels, ket_labels = make_braket_labels(nmm_list)    
    
    H = DeltaE*two_state_A + DeltaE*two_state_B + omega*bosons + U*interaction_A + U*interaction_B

    # Field in vacuum state (0) with N/2 excitations in A (int(N/2)) and 0 excitations in B (0)
    psi0_ind = nmm_list.index((0,int(N/2),0))  
    psi0 = basis(len(nmm_list), psi0_ind)


    # We are using custom simulate function from last tutorial because it's going to be quicker
    # in this case because of the long simulation times
    P, psi, evals, ekets = simulate(H, psi0, times)
    num_A = expectation(two_state_A + N/2, psi)
    num_B = expectation(two_state_B + N/2, psi)

    # For fitting, we'll find when the first excitaton is transferred to B, then re-simulate 
    # up to that point so that we can get a better resolution over the shorter time periods.
    # We do this because we expect timescales to shorten as we increase TLS number
    if N==2:
        # N=1 is special because excitations never crosses 1 so we need to use find_peaks just like in the last tutorial
        peaks, _ = find_peaks(num_B, prominence=0.05)
        peak_times = times[peaks]
        time_to_transfer_one = peak_times[0]
    else: 
        # Approximate time when the expected excitations in B reaches 1.
        crossing_one_index = np.where(np.diff((num_B > 1).astype(int)))[0][0]
        time_to_transfer_one = times[crossing_one_index]

    times_fit = np.linspace(0,  time_to_transfer_one/10, 100)
    P_fit, psi_fit, *_ = simulate(H, psi0, times_fit, evals, ekets)
    num_B_fit = expectation(two_state_B + N/2, psi_fit)

    fit, covariance = curve_fit(model_func, times_fit, num_B_fit,p0=[0.01],maxfev=500)

    rate.append(2*fit[0])


    plt.plot(times, num_A, label="Expected A excitations")
    plt.plot(times, num_B, label="Expected B excitations")
    plt.plot(times,model_func(times,*fit),label="Quadratic fit")
    plt.xlabel("Time")
    plt.ylim([0,num_B.max()*1.1])
    plt.legend(loc="right")
    plt.title(f"{H_latex_AB} \n $\Delta E={DeltaE}$, $\omega={omega}$, $U={U}$, N={N} \n $\Psi_0 =$ {ket_labels[psi0_ind]}   (Fig. {i+41})")
    plt.show();
```

```python
plt.plot(Ns,rate/gamma_1T,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised transfer rate ($\Gamma/\Gamma_1$)");
plt.title("Transfer of $N/2$ delocalised excitations from A to B (Fig. 47)");
```

Performing the usual regression of the log of the rate gives:

```python
print("slope = ", linregress(np.log10(Ns[:]), np.log10(rate[:])).slope)
```

and if we look at the trend for higher $N$ we see something that appears to approach $N^3$ 🤯.

```python
print("slope = ", linregress(np.log10(Ns[2:]), np.log10(rate[2:])).slope)
```

Super transfer indeed!

And guess what, we can understand this scaling by enumerating the paths once again. It's more fiddly, but we can do it for the general case of different number of TLS and excitations in system A and B. Let's do it 💪:
1. There are $^{N_{A}}C_{n_{+A}}$ different configurations for the $n_{+A}$ excitations in the $N_A$ TLS of system A
2. In each of those configurations any of the $n_{+A}$ excitatons can move from A to B
3. There are $^{N_{B}}C_{n_{+B}}$ different configurations for the $n_{+B}$ excitations in the $N_B$ TLS of system B
4. In each of those configurations there are $N_B - n_{+B}$ "holes" that can accept an excitation

The total number of paths is therefore:

$$\left[^{N_{A}}C_{n_{+A}} n_{+A}\right]\left[^{N_{B}} C_{n_{+B}} \left(N_B - n_{+B}\right)\right]$$

Once we've normalised the initial and final states and squared it all to get the rate, we arrive at:

$$\frac{\Gamma}{\Gamma_{1T}} = \left(\frac{\left[^{N_{A}}C_{n_{+A}} n_{+A}\right]\left[^{N_{B}} C_{n_{+B}} \left(N_B - n_{+B}\right)\right]}{\sqrt{^{N_{A}}C_{n_{+A}}\,^{N_{B}}C_{n_{+B}}}\sqrt{^{N_{A}}C_{n_{+A}-1}\,^{N_{B}}C_{n_{+B}+1}}}\right)^2 = n_{+A}\left(N_A-n_{+A}+1\right)\left(n_{+B}+1\right)\left(N_B - n_{+B}\right)$$

I know, it's a lot, but if you grind through all the factorials involved in those [combinations](https://en.wikipedia.org/wiki/Combination) I promise you'll get the same answer.

Ok, so let's put our numbers in:
- $N_A = N_B = N$
- $n_{+A} = N/2$
- $n_{+B} = 0$

$$\frac{\Gamma}{\Gamma_{1T}} =n_{+A}\left(N_A-n_{+A}+1\right)\left(n_{+B}+1\right)\left(N_B - n_{+B}\right) = \frac{N}{2}\left(\frac{N}{2}+1\right)N$$

And so we can see that we get a lovely $N^3$ scaling as we increase the number of TLS. You can also recover the $N^2$ scaling we found for the single and fully excited system A scenarios.

This supertransfer was first proposed by Strȩk in their 1977 paper [Cooperative energy transfer](https://www.sciencedirect.com/science/article/abs/pii/0375960177904273?via%3Dihub), but more recent work explicitly demostrating the $N^3$ dependence can be found in the 2010 work of Lloyd on [Symmetry-enhanced supertransfer of delocalized quantum states](https://iopscience.iop.org/article/10.1088/1367-2630/12/7/075020).


## 8.5 - Superradiance vs Supertransfer

<!-- #region -->
We finish with a reflection on some speculation we made in the last tutorial. There we noted how excitation transfer appeared to have a more favourable scaling with the number of TLS compared with spontaneous emission. We wondered about the possibility that the usually very slow excitation transfer rate $\Gamma_T$ could out-compete the spontaneous emission rate $\Gamma_E$. 

We can  quantify this a bit more now. We can ask when is $\Gamma_T > \Gamma_E$? Roughly we can say:


$$
\begin{align}
\Gamma_T &> \Gamma_E  \\
\Gamma_{1T}N^3 &> \Gamma_{1E}N^2  \\
N^2\left(N\Gamma_{1T} - \Gamma_{1E}\right) &> 0  \\
N &> \frac{\Gamma_{1E}}{\Gamma_{1T}}  \\
\end{align}
$$
<!-- #endregion -->

A true comparison between the "base" emission and transfer rates requires us to know these rates for the same $\Delta E$. Unfortunately, we've used $\Delta E = 1$ for spontaneous emission and  $\Delta E = 2.5$ for excitation transfer so we can't make an accurate comparison. 

You may recall that the reason we've been using $\Delta E = 2.5$ for excitation transfer is to suppress spontaneous emission. With $\Delta E = 1$ we'd never observe a "base" excitation transfer rate because spontaneous emission would always dominate.

So, what can we do?

We can start by just pretending that the excitation transfer rate is independent of $\Delta E$. We don't anticipate this to be the case, but it can be instructive to do this pretending sometimes. Now we can just plug in our numbers:

```python
gamma_1E/gamma_1T
```

We'd need to have $N > 6.9 \times 10^6$ for excitation transfer to dominate. This is well outside of what we can simulate but not outside of what we might find in e.g. a solid system that usually contains on the order of $10^{22}$ particles in every $\text{cm}^3$. 

Food for thought 🤔.


## Next up...

A lot of what we've done today has laid the foundations for us to explore emission and transfer rates in many TLS. We've also seen how delocalising our excitations gives us huge speed ups in quantum processes opening up a way to observe otherwise very slow processes like excitation transfer.

Whether excitation transfer can ever truly compete with spontaneous emission remains to be seen. We are however motivated to dig deeper into how the rates depend on the other parameters in our system - specifically $\Delta E$. We'll dig into this next time.

Until then. 👋
