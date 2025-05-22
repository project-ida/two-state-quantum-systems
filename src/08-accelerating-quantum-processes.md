---
jupyter:
  jupytext:
    formats: ipynb,src//md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="7d9c70f1-360d-4efc-a8c0-f764a82f919e" -->
<a href="https://colab.research.google.com/github/project-ida/two-state-quantum-systems/blob/master/08-accelerating-quantum-processes.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/two-state-quantum-systems/blob/master/08-accelerating-quantum-processes.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="0baaef7c-39d8-4b66-a511-1de212d8b1eb" -->
# 8 - Accelerating quantum processes
<!-- #endregion -->

<!-- #region id="279c670c-412c-42a6-abe1-4ca1d212c18c" -->
Last time, we saw hints of being able to accelerate quantum processes like spontaneous emission and excitation transfer by "delocalising" excitations across many TLS.

In this tutorial, we'll see just how much speed-up we can get by developing a description of many TLS (called "Dicke states") that allows us to set up delocalised excitations easily and also helps us avoid getting stuck in a computational bottleneck.

We're covering a lot of ground today so we've split everything up into the following sections:

1. Recap
2. Angular momentum
3. Superradiance
4. Supertransfer
5. Superradiance vs Supertransfer
<!-- #endregion -->

```python id="8a7bd1e4-c140-4072-8752-73533676605f"
# RUN THIS IF YOU ARE USING GOOGLE COLAB
import sys
import os
!pip install qutip==4.7.6
!git clone https://github.com/project-ida/two-state-quantum-systems.git
sys.path.insert(0,'/content/two-state-quantum-systems')
os.chdir('/content/two-state-quantum-systems')
```

```python id="52309eca-119a-44d9-a81f-38ff557f0935"
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
from libs.helper_08_tutorial import *
```

<!-- #region id="5a9ee6d4-86d4-406f-95e7-6826ba9f63dc" -->
## 8.1 -  Recap
<!-- #endregion -->

<!-- #region id="e97aaffb-0d62-47bc-8af6-e9491404113c" -->
So far, we've described a combined TLS & quantum field using a notation like, e.g. $|0,+,-,-,- \rangle$. The first number (in this case $0$) tells us the number of bosons present (this is often referred to as a [Fock state](https://en.wikipedia.org/wiki/Fock_state)), and the $\pm$ tell us what state each of the $N$ TLS are in ($+$ excited, $-$ ground). This is a complete description in the sense that every configuration of the system can be described as a mixture of these states. For example, a single excitation delocalised across 4 TLS with no bosons can be described by:

$\Psi_0 \sim | 0, +, -, -, - \rangle + | 0, -, +, -, - \rangle + | 0, -, -, +, - \rangle + | 0, -, -, -, + \rangle $


The issue with this description is that, for a specific number of bosons, there are $2^N$ possibilities for the state of the TLS and that means it becomes infeasible to simulate more than about 10 TLS.

Because delocalised excitations are of most interest to us today, we don't actually need a lot of the detail that the complete description holds. Superficially, we'd be quite happy with a simpler description like $| n, n_+ \rangle$ - where $n$ is the boson number and $n_+$ is the number of excitations. There would then only be $N+1$  possibilities for the state of the TLS in this case - this is much more favourable from a computational perspective.

Let's see if we can make this simpler description rigorous enough to help us with simulating many TLS.
<!-- #endregion -->

<!-- #region id="03ff06a0-e943-45f3-88d0-d814afb5b903" -->
## 8.2 - Angular momentum $J$
<!-- #endregion -->

<!-- #region id="9a948e99-bf54-41c0-8765-b467b59c6321" -->
Creating a description of delocalised excitations is not quite as simple as $| n, n_+ \rangle$. For example, the following delocalised states $\Psi_1$ and $\Psi_2$ contain the same number of delocalised excitations but they're different:

$\Psi_1 \sim | 0, +, - \rangle + | 0, -, + \rangle $
<!-- #endregion -->

```python id="d213ed1e-84f9-4bd3-9c1f-b260d369153b" outputId="405dfece-4a75-4030-f549-c8f7f109b46f"
psi_1 = basis([2,2], [0,1]) + basis([2,2], [1,0])
psi_1 = psi_1.unit()
psi_1
```

<!-- #region id="258f64ea-9ecb-481e-830e-834c67684b23" -->
$\Psi_2 \sim | 0, +, - \rangle - | 0, -, + \rangle $
<!-- #endregion -->

```python id="6b5d15df-9f54-4fe9-ab32-d313e63553b3" outputId="59fa1c0d-1322-4372-95ff-fc14b1f781b3"
psi_2 = basis([2,2], [0,1]) - basis([2,2], [1,0])
psi_2 = psi_2.unit()
psi_2
```

<!-- #region id="e041a014-12bb-4d90-8ea7-2d13ad63a7cc" -->
What makes these states physically different is related to what we normally think of as their angular momentum. We're therefore going to need to add some angular momentum information to our states.

Ultimately, our states are going to need to look like:

$\Psi \sim | n, j, m \rangle$

Where:
- $n$ - boson number = $0,1,2,3,...$
- $j$ - [total angular momentum quantum number](https://en.wikipedia.org/wiki/Total_angular_momentum_quantum_number) = $0,\frac{1}{2},1,\frac{3}{2},2,\frac{5}{2}, 3, ...$
- $m$ - [magnetic quantum number](https://en.wikipedia.org/wiki/Magnetic_quantum_number) = $j, (j-1), (j-2), ..., -(j-2), -(j-1), -j$

It's not obvious where all those numbers come from and how they can be mapped to e.g. 4 TLS with a single delocalised excitation? So, let's figure that out.
<!-- #endregion -->

<!-- #region id="fc789d6c-320d-41df-9b5f-1c54057979f0" -->
### Angular momentum numbers

Although we don't explicitly have a description of angular momentum in our TLS, you may recall from [tutorial 2](https://github.com/project-ida/two-state-quantum-systems/blob/master/02-perturbing-a-two-state-system.ipynb) that our system is mathematically equivalent to spin 1/2 particles which do have angular momentum. We can also see this explicitly in the language we've been using to describe our Hamiltonian:

$$H =  \Delta E J_{Nz} + \hbar\omega\left(a^{\dagger}a +\frac{1}{2}\right) + U\left( a^{\dagger} + a \right)2J_{Nx}$$
<!-- #endregion -->

```python id="699bf71b-0d02-4c70-b542-6f8ee063f83d"
H_latex = "$H = \Delta E J_{Nz} + \hbar\omega(a^{{\dagger}}a +1/2) + U( a^{{\dagger}} + a )2J_{Nx}$ "
```

<!-- #region id="228c1ba7-c48a-4677-9478-2730935b1672" -->
where:

- The [total angular momentum operators](https://www2.ph.ed.ac.uk/~ldeldebb/docs/QM/lect15.pdf) ($J$) for $N$ TLS:

$$J_{Nx} = \overset{N}{\underset{n=1}{\Sigma}} S_{n x} \,\,\,\,\,\, J_{Ny} = \overset{N}{\underset{n=1}{\Sigma}} S_{n y} \,\,\,\,\,\, J_{Nz} = \overset{N}{\underset{n=1}{\Sigma}} S_{n z}$$

- The spin operators ($S$) for a [spin 1/2 particle](https://en.wikipedia.org/wiki/Spin-%C2%BD#Observables):

$$
S_x = \frac{1}{2}\sigma_x \,\,\,\,\,\, S_y = \frac{1}{2}\sigma_y \,\,\,\,\,\, S_z = \frac{1}{2}\sigma_z
$$
<!-- #endregion -->

<!-- #region id="5901820c-737b-4388-bef4-6f1ce37dd705" -->
We'll continue to talk about angular momentum here to keep as sense of familiarity. Eventually we'll move to a more abstract way of thinking about $J$, but that can wait.

So far, we've seen that the x and z "components" of the total angular momentum operator $J_x$ and $J_z$ are used in the Hamiltonian. I use quote marks around "component" because this is vector language which is not obviously applicable to operators. It turns out, however, that we can (in some sense) treat the angular momentum operator as a vector (see [spinors](https://en.wikipedia.org/wiki/Spinors_in_three_dimensions)). We can create the squared "magnitude" of the total angular momentum operator ($J^2$) much like we would a vector by summing of the squares of the components.

Let's see this for the case of 2 TLS.
<!-- #endregion -->

```python id="bfafcbb4-d231-4744-bcff-23e87f4e402d"
Jx, Jy, Jz = jspin(2, basis="uncoupled")
J2 = Jx*Jx + Jy*Jy + Jz*Jz
```

<!-- #region id="e71b7c4a-a71e-40e6-9af7-63dbc338fc42" -->
What does this operator tell us about how we might go about differentiating between states like $\Psi_1$ and $\Psi_2$ that have the same amount of delocalised excitation?

If $\Psi_1$ and $\Psi_2$ are eigenstates of $J^2$ (i.e. $J^2 \Psi = \lambda \Psi$) then those states have specific, well defined, angular momentum that's characterised by the constant (eigenvalue) $\lambda$. That constant could then be used to label our states.
<!-- #endregion -->

```python id="6723fcee-3451-4b14-8f17-2f51e8b34ca0" outputId="19a0b410-7eb4-47f9-fcfd-cc474bd1369a"
J2*psi_1
```

```python id="4882aeba-34e8-4731-a491-491f0d15a1f0" outputId="bf742243-0a75-44c7-dd8d-46501357617e"
J2*psi_2
```

<!-- #region id="26ecd0b5-63d9-42db-be4f-e16cb466ba2f" -->
We can therefore see that:
- $J^2 \Psi_1 = 2 \Psi_1 \implies$  $\lambda = 2$
- $J^2 \Psi_2 = 0 \Psi_2 \implies$ $\lambda = 0$
<!-- #endregion -->

<!-- #region id="36348b45-9973-4a20-914c-1268692ef62c" -->
Although not immediately obvious, these eigenvalues of $J^2$ [always have the form $j(j+1)$](https://www.feynmanlectures.caltech.edu/II_34.html#Ch34-S7), where $j$ is either an integer or half integer.
<!-- #endregion -->

```python id="e561edd4-2af3-41aa-8de5-c470fed5ff17"
evalsJ, eketsJ = J2.eigenstates()
```

```python id="8f15fa31-c1fa-4d65-a6aa-3241912be2e3" outputId="5b69eb22-aab7-4368-e29b-14d0d3ef6531"
evalsJ
```

<!-- #region id="832932a6-cb40-4af7-8798-f6841038d894" -->
By just looking at these eigenvalues of $J^2$, we know there must be:
- 1 state with $j=0$ (we've seen that already with $\Psi_2$)
- 3 states with $j=1$ (we've seen one of those with $\Psi_1$)
<!-- #endregion -->

<!-- #region id="e0d89852-6f75-4a29-8f25-1ce12fff63b4" -->
You might wonder how we're able to have 3 states with the same angular momentum number $j$? They have different $J_z$ - what is known as the "magnetic quantum number" often given the label $m$.

In quantum mechanics, it has been found experimentally that [angular momentum is quantised](https://www.feynmanlectures.caltech.edu/II_34.html#Ch34-S7) in the sense that when its z component is measured it can only take values $m\hbar$ where $m = j, (j-1), (j-2), ..., -(j-2), -(j-1), -j$. We can see this explicitly by looking at the eigenvalues of $J_z$.
<!-- #endregion -->

```python id="c05c2859-cbd6-48f7-886b-2bf79c2fe9d5"
evalsM, eketsM = Jz.eigenstates()
```

```python id="557c81ef-d176-45e6-b1f7-818a2aafe174" outputId="7d8e6b34-9ac6-4259-b479-ea21284529b7"
evalsM
```

<!-- #region id="592e1cf1-d0cb-4134-9daa-37d51083b036" -->
For $N=2$, the TLS can therefore be described in terms of angular momentum by giving 2 numbers $|j,m\rangle$:
- $| 0, 0 \rangle$ - this is $\Psi_1$
- $|1, -1 \rangle$
- $|1, 0 \rangle$ - this is $\Psi_2$
- $|1, 1\rangle$

This is actually a complete description because for 2 TLS there are only 4 states, $| -, - \rangle$, $| +, - \rangle$, $| -, + \rangle$, $| +, + \rangle$. This is not the case for $N>2$.  In general, we lose the ability to describe every state uniquely when we use this angular momentum description. In other words, there can be many states with the same $j,m$ values (degenerate states). When you enumerate all the states in the angular momentum description, there are $\sim N^2$ possibilities compared to the $2^N$ we've been working with up to now. This is ultimately what's going to give us a computational advantage but we do need to be a bit careful as to whether we lose any physics when we do this.
> Advanced: Use [`state_degeneracy(N,j)`](https://qutip.org/docs/4.4/apidoc/functions.html?highlight=m_degeneracy#qutip.piqs.state_degeneracy) to calculate the degeneracy of each state. In general there are some subtleties to consider when ignoring degeneracy which might need to be considered depending on the problem at hand (see last paragraph of [Permutational Invariant Quantum Solver](https://qutip.readthedocs.io/en/qutip-5.0.x/guide/guide-piqs.html)). For now, we don't need to worry about this so we will put a pin in this advanced topic and return to it in a later tutorial.


Hopefully you've got a better understanding of these angular momentum numbers. Now we need to link it back to the number of TLS $N$ and the number of delocalised excitations $n_+$.
<!-- #endregion -->

<!-- #region id="5e6592fa-a1ad-43f9-a047-6d072594cc4b" -->
### Dicke states
<!-- #endregion -->

<!-- #region id="2b2fd219-b3a2-4865-9c5d-8b229076a5fc" -->
Consider 4 TLS with a single delocalised excitation, how can we write this in our new angular momentum description with $j,m$?

$m$ is actually very closely related to the number of excitations $n_{+}$, it's:

$$m= \frac{1}{2}\left(n_{+} - n_{-} \right) = \frac{1}{2}\left(n_{+} - (N - n_{+}) \right) = n_+ - N/2$$

For 4 TLS with a single delocalised excitation, we'd have $m = 1 - \frac{4}{2} = -1$. What about $j$?

As we saw earlier, there are several $j$'s for a given $m$. In general, the specific $j$, $m$ combinations come from adding up the angular momentum for many single TLS (with $j=1/2$) like vectors of the same length but different (quantised) orientations. The details are somewhat tedious - often involving [formidable lookup tables](http://pdg.lbl.gov/2019/reviews/rpp2019-rev-clebsch-gordan-coefs.pdf). Luckily for us, QuTiP, has some convenient functions (that are somewhat hidden inside of [`qutip.cy.piqs`](https://github.com/qutip/qutip/blob/85632bc66fdcd45be51e1c280ea7577f04761a67/qutip/cy/piqs.pyx)) to help us.
- [`j_vals(N)`](https://github.com/qutip/qutip/blob/85632bc66fdcd45be51e1c280ea7577f04761a67/qutip/cy/piqs.pyx#L130) - tells us the different $j$ values for $N$ TLS.
- [`m_vals(j)`](https://github.com/qutip/qutip/blob/85632bc66fdcd45be51e1c280ea7577f04761a67/qutip/cy/piqs.pyx#L147) tells us the $m$ values for a given $j$




<!-- #endregion -->

```python id="9f73460c-83f2-4f60-84bb-ce2af04c634c" outputId="f191ec7f-4bba-4a76-bc18-278ef898c339"
j_vals(4) # Gives the different j's for 4 TLS
```

```python id="73fcd9e1-c2b8-40a6-805f-a9ec12fa8c41" outputId="8a51accf-6178-4d70-93ad-2813d35e47e3"
m_vals(2) # Gives us m values for j=2
```

```python id="a58475ee-9aaa-443c-b16f-ce1dfbdbed67" outputId="92b9f54b-e315-4674-d476-c55a09220463"
m_vals(1) # Gives us m values for j=1
```

```python id="0726244a-3dff-4cd0-837e-c8acbcde519a" outputId="3b464370-c3d5-45c6-8fb5-317037026017"
m_vals(0) # Gives us m values for j=0
```

<!-- #region id="73b40f0e-242f-4aca-99fb-7e83458b1f81" -->
We can see that there is an $m=-1$ for $j=2$ and $j=1$. Which one should we pick?
<!-- #endregion -->

<!-- #region id="fec217d1-b1db-4ecc-9511-42c3a5fc89d8" -->
By far the most significant $j$ is the largest $j_{\max} = N/2$. The largest $j$ corresponds to what's called a `Dicke state`.

A Dicke state is a symmetric state, which means if you swap any of the TLS around, the state remains unchanged. For example, consider a single excitation in 4 TLS. The Dicke state looks like:

$\Psi_0 = \frac{1}{\sqrt{4}}\left(| 0, +, -, -, - \rangle + | 0, -, +, -, - \rangle + | 0, -, -, +, - \rangle + | 0, -, -, -, + \rangle \right)$

Notice that if you swap any two TLS, the state looks the same.

The reason why $j_{\max}$ is most significant is because of the acceleration properties that these Dicke states offer; something people often describe as superradiance and supertransfer. We're going to see these in action in the next sections.

Before we get there, we need to take a short detour into angular momentum conservation.
<!-- #endregion -->

<!-- #region id="3a82e8a5-77f1-44cc-8cec-200a1f6e975d" -->
### Conservation of angular momentum

Angular momentum is conserved in our model and so we have a choice which $j$ value we want to run our simulation with. Once we set the system up with this $j$ it will keep that same $j$. This gives us an additional computational advantage above what we've already got from using the angular momentum description.

Instead of needing to keep track of the $N^2$ different angular momentum states, we only need to keep track of the $2j+1$ different $m$ states that correspond to the $j$ we picked. The worst case scenario is the Dicke states which use $j_{\max} = N/2$. For simulating Dicke states we need to keep track of $N+1$ states. This is an incredible improvement - going from an exponential scaling with number of TLS to linear.

You might think it would be a pain to extract only the states that correspond to a particular $j$, but once again QuTiP has got our back  with [`jmat(j)`](https://qutip.readthedocs.io/en/qutip-5.0.x/apidoc/functions.html#qutip.core.operators.jmat) - it does 2 things:
- Automatically gives us operators in the angular momentum basis, i.e. $|j,m \rangle$
- Returns only parts of the operators that act on the $j$ we pick

Let's look at an example:
<!-- #endregion -->

```python id="a923ba16-a3b2-4a80-8427-c145b83d488f" outputId="d5f994a5-37f5-4417-82e3-ec8c429fe4e4"
N = 2 # 2 TLS
Jx, Jy, Jz = jmat(N/2) # j=j_max = N/2 means only Dicke states allowed
Jz
```

<!-- #region id="ad7f71b2-79c6-4b4f-9fed-2a2152daaf96" -->
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

<!-- #region id="682347a3-61ba-4748-b694-c1fcdd223340" -->
## 8.3 - Superradiance
<!-- #endregion -->

<!-- #region id="0a1918e0-5097-4ff4-92b5-1dcca9a1eea8" -->
Let's reconsider the case of spontaneous emission. We saw in [tutorial 3](https://github.com/project-ida/two-state-quantum-systems/blob/master/03-a-two-state-system-in-a-quantised-field.ipynb) that such emission from a TLS can be understood as the result of coupling to a quantised field. The stronger the coupling, the faster the emission as seen by the increased Rabi frequency.

For emission from many TLS, we'd expect the rate to depend on the number of TLS that are excited. We might argue, for example, that the rate of emission is simply the sum of the rates of the individual TLS. In other words, we'd expect a factor of $N$ speed-up for $N$ TLS that are excited.

Let's simulate it and check. We're going to need to make a some modifications to our simulation code and also how we measure "emission rates".
<!-- #endregion -->

<!-- #region id="7cb0c0df-5171-465b-824e-9f9e2670a01a" -->
First, we need to adapt `make_operators` function to use `j = jmax` (i.e. we'll initialise with a Dicke state) and we'll also enumerate the states in `nm_list` in terms of number of TLS that are excited $n_+$ instead of using $m$. We can do this because:

$$m = n_+ - N/2$$

Since $m = j, j-1,...-j$ and $N/2 = j$ for Dicke states, then

$$n_+ = m+j = 2j, 2j-1, 2j-2, ..., 3,2,1,0$$


<!-- #endregion -->

```python id="f4ae9bf8-01cb-45e0-b1af-f5d519d594f3"
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

<!-- #region id="15dcd563-1f2f-4942-a1ae-93aecd970e5f" -->
Next, we need to figure out how we're going to measure emission rates.
<!-- #endregion -->

<!-- #region id="21bb73ae-b174-42f4-b6c8-ee3ed370c001" -->
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
<!-- #endregion -->

<!-- #region id="4c6aab5e-2863-4fda-bc0f-099b77a5d4b9" -->
### Fully excited system
<!-- #endregion -->

```python id="5acf92eb-b8eb-4627-8192-5a81d9549ea6"
# a + bt + ct^2.
# a=0 because number of bosons 0 to start
# b=0 because (from experience) rate is 0 to start
def model_func(t, c):
    return c*t**2
```

```python id="3fa0d24c-f522-4c46-bc61-f4bfd6b68449"
DeltaE = 1 # We match the boson energy to the TLS energy to make sure we'll get emission like behaviour
omega = 1
U = 0.001 # Coupling is 10x lower than the last tutorial because we are upping the number of TLS
```

```python id="a7e6f4ce-19af-4c51-93bb-6668a782f439" outputId="73510504-757a-4912-e580-9e0399d0c757"
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

<!-- #region id="c0e63fa6-55bc-4bcf-802b-107d8a9ce3a4" -->
Although superficially our model fit doesn't look great in e.g. Fig. 8, you'll notice that if we plot for short time scales over which the model was fitted it's fine.
<!-- #endregion -->

```python id="ec872d73-f27e-4909-b040-c7e53d6d46ac" outputId="66c89d86-615e-4b94-b422-93bfe323b44c"
plt.plot(times_fit, result_fit.expect[0], label="Expected bosons")
plt.plot(times_fit,model_func(times_fit,*fit),label="Fit")
plt.xlabel("Time")
plt.legend(loc="right")
plt.title(f"{H_latex} (Fig. 9)  \n $\Delta E={DeltaE}$, $\omega={omega}$, $U={U}$, N={N} \n $\Psi_0 =$ {ket_labels[psi0_ind]}")
plt.show();
```

<!-- #region id="1de6dcb2-9928-4f75-8f3e-bbca07ab439b" -->
We'll store the "base" rate of emission $\Gamma_{1E}$ for the case of a single TLS so that we might reference it later
<!-- #endregion -->

```python id="0d8acc5e-6c34-4c4b-bfbe-1ff3d822bf7d" outputId="9dc6ef3f-9272-4aa3-9dda-7364a594c18f"
gamma_1E = rate[0]
gamma_1E
```

<!-- #region id="d1091a88-34c0-4277-b9c4-ac8267e868ef" -->
Let's now see how the emission rate varies with with number of TLS $N$.
<!-- #endregion -->

```python id="221e486e-cb76-4825-8c69-4e4a48bd70d9" outputId="9d5ad18a-7908-4159-ef4a-182f1d008fe7"
plt.plot(Ns,rate/gamma_1E,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised emission rate ($\Gamma/\Gamma_1$)");
plt.title("Emission from N delocalised excitations (Fig. 10)");
```

<!-- #region id="59e303d5-0336-40ff-80ec-0504ad9c67fd" -->
Fig. 9 shows us exactly what we expected - a nice linear relationship.

Let's not celebrate too soon though.

If we apply our intuition to the case of a single delocalised excitation spread across $N$ TLS, then we'd expect the emission rates to be independent of $N$. This is however not the case.

Let's see.
<!-- #endregion -->

<!-- #region id="18e6e973-ef98-4eed-8893-348f88803137" -->
### Single delocalised excitation
<!-- #endregion -->

```python id="33a3639a-4188-44ef-9104-848a48389bc6" outputId="eda20b7c-a938-4c0a-c783-fc1614a547cb"
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

```python id="01ca011f-eb4d-4976-be1f-6b67beaee4bd" outputId="315fb5de-bc62-4026-f444-ef997e46b32d"
plt.plot(Ns,rate/gamma_1E,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised emission rate ($\Gamma/\Gamma_1$)");
plt.title("Emission from 1 delocalised excitation (Fig. 18)");
```

<!-- #region id="80f4b041-2adf-4f97-a7e1-c092b01809ae" -->
Fig. 18 shows us that we get the same enhancement of emission rates whether we have single excitation or $N$ of them 🤔.

Stranger still is the case where half of the TLS are excited.

<!-- #endregion -->

<!-- #region id="0230ba33-660f-4f94-9458-a0e29c11970a" -->
### Half excited system
<!-- #endregion -->

```python id="5757e17d-0452-44f6-8fd3-c6960c2d4312" outputId="863693b4-8aec-43f4-b729-633c3605a95d"
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

```python id="a2c501e0-fa56-470d-be01-f40cbda91299" outputId="3bac1726-33fb-494f-8fbb-b9406ddb3985"
plt.plot(Ns,rate/gamma_1E,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised emission rate ($\Gamma/\Gamma_1$)");
plt.title("Emission from $N/2$ delocalised excitations (Fig. 25)");
```

<!-- #region id="3532b72b-f03d-40c2-b98a-617e42d8ff5a" -->
Fig. 25 shows us that, with $N/2$ excitations, the emission rate is even greater than when all of the TLS are excited 🤯!

Let's quantify this by doing a linear regression of $\log(\Gamma)$ with  $\log(N)$
<!-- #endregion -->

```python id="d623d622-26ca-46a7-bb90-91ed40c4560e" outputId="23204ce6-6b51-4225-e922-3971ce9da2cc"
print("slope = ", linregress(np.log10(Ns), np.log10(rate)).slope)
```

<!-- #region id="182e6239-fc24-4f83-aea3-a2a8c6cfc62a" -->
We can see that $\Gamma \sim N^2$ when $n_+ = N/2$.

This kind of counter intuitive emission rate enhancement was discovered by Dicke in his 1956 paper [Coherence in Spontaneous Radiation Processes](https://journals.aps.org/pr/abstract/10.1103/PhysRev.93.99) where he coined the turn "superradiance".

In general, Dicke found that when $n_+$ excitations are delocalised across $N$ TLS, the emission rate $\Gamma$ is enhanced over the single TLS emission rate $\Gamma_{1E}$ by:

$$\frac{\Gamma}{\Gamma_{1E}} = n_+\left(N - n_+ +1\right)$$

where we can see that $\Gamma$ is largest when half the TLS are excited giving a rate of $\Gamma/\Gamma_{1E} = \frac{N}{2}\left(\frac{N}{2}+1\right)$.

How can we understand this?
<!-- #endregion -->

<!-- #region id="f5646bba-cc52-4554-9c4c-574c4bd3e8d8" -->
### Understanding superradiance

Superradiance might at first seem counterintuitive, but we can understand it from one of the most fundamental principles of quantum mechanics [according to Richard Feynman](https://www.feynmanlectures.caltech.edu/III_01.html#Ch1-S7) which reads:

```
"When an event can occur in several alternative ways, the probability amplitude for the event is the sum of the probability amplitudes for each way considered separately. There is interference"
```

Let's take the example of 2 excitations amongst 4 TLS. The initial delocalised Dicke state looks like:

$\Psi_i = \frac{1}{\sqrt{6}}\left(| 0, +, +, -, - \rangle + | 0, +, -, +, - \rangle + | 0, +, -, -, + \rangle + | 0, -, +, +, - \rangle + | 0, -, +, -, + \rangle + | 0, -, -, +, + \rangle \right)$

We can see there are 6 different configurations for the TLS. Each of the 2 excitations in those 6 configurations could transition from $+$ to a $-$ with a release of a single boson. That means each of the 6 configurations has 2 emission paths that it could go in order to reach one of 4 configurations in final the state:

$\Psi_f = \frac{1}{\sqrt{4}}\left(| 1, +, -, -, - \rangle + | 1, -, +, -, - \rangle + | 1, -, -, +, - \rangle + | 1, -, -, -, + \rangle \right)$

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
<!-- #endregion -->

<!-- #region id="de93e3e8-4603-4ee1-b0eb-9a34bba46b2e" -->
## 8.4 - Supertransfer
<!-- #endregion -->

<!-- #region id="4262d7e8-55b2-48f0-82be-f0a2f42d764c" -->
Just like with superradiance, we're going to work with Dicke states to allow us to conveniently describe and simulate delocalised excitations.

To describe delocalised excitations transferring from one "place" to another, we need to break-up our overall system into 2 parts - system A and system B. Each system can in principle have its own number of TLS ($N_A$, $N_B$) and its own number of excitations ($n_{+A}$, $n_{+B}$).

The general Hamiltonian for this AB situation is described by:

$$H =  \Delta E_A J_{N_Az}^{(A)} + \Delta E_B J_{N_Bz}^{(B)} + \hbar\omega\left(a^{\dagger}a +\frac{1}{2}\right) + U_A\left( a^{\dagger} + a \right)2J_{N_Ax}^{(A)} + U_B\left( a^{\dagger} + a \right)2J_{N_Bx}^{(B)}$$

It's a bit full on so we won't investigate this Hamiltonian in all its generality today. We'll focus on the case where:
- System A and B have the same transition energy: $ \Delta E_A =  \Delta E_B =  \Delta E$
- System A and B couple to the boson field in the same way: $U_A = U_B = U$
- System A and B consist of the same number ot TLS: $N_A = N_B = N$
<!-- #endregion -->

```python id="603fd39d-8c2a-44c0-b9f4-fffc8b802fd0"
H_latex_AB = "$H = \Delta E (J_{Nz}^{(A)}+J_{Nz}^{(B)}) + \hbar\omega(a^{{\dagger}}a +1/2) + U( a^{{\dagger}} + a )2(J_{Nx}^{(A)} + J_{Nx}^{(B)})$ "
```

<!-- #region id="d8052fa2-9efc-421e-9b76-5c14f48fe43b" -->
The process of constructing the additional operators is similar to when we added the quantised field operators back in [tutorial 03](https://github.com/project-ida/two-state-quantum-systems/blob/master/03-a-two-state-system-in-a-quantised-field.ipynb) - we create tensor products of the different operators to make sure they only act on the relevant parts of the state.
<!-- #endregion -->

```python id="d5e7fdac-26b4-466d-8642-ad4715d45199"
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

<!-- #region id="7319a502-bed9-4f3d-9018-bdff8ea7ef30" -->
We're aaaalmost ready to rock and roll, but we need a couple of extra bits.

Firstly, let's create an operator that helps us count the number of excitations in system A or B (a bit like our `number` operator for bosons). We can create such an operator from our `two_state` operators because they are just $J_z$ which is just a measure of the $m$ number which in turn is related to the excitation number via $n_+ = m + N/2$ for Dicke states. In the language of expectation values $\langle ... \rangle$:


$\langle \text{two\_state}\rangle = \langle J_z\rangle = \langle m\rangle = \langle n_+ \rangle - N/2$

Next, calculating expectation values. You may recall in the last tutorial that when we simulated excitation transfer we used our custom simulate function from [tutorial 05](https://github.com/project-ida/two-state-quantum-systems/blob/master/05-excitation-transfer.ipynb) because it was quicker over of the long simulation times typically required of excitation transfer. We're going to do the same here. This means we'll need our own way to calculate the expectation values (sesolve did this for us before). Although QuTiP does have the [`expect`](https://qutip.readthedocs.io/en/qutip-5.0.x/apidoc/functions.html#qutip.core.expect.expect) function, it turns out that we need to create a `Qobj` for every time step in order to use this function and that can be very slow. We will instead directly calculate the expectation value using matrix multiplication, i.e.

$<H> = \psi^{\dagger}H\psi = \psi^{\dagger} @ (H @\psi) $

Where @ is the matrix multiplication operator and $\dagger$ in this context means taking the complex conjugate.

Let's automate this process for all time steps using a function.
<!-- #endregion -->

```python id="2fb83355-ba21-46d0-833f-0239c552e8e8"
# "states" will be the output of Psi from our custom "simulate" function
def expectation(operator, states):
    operator_matrix = operator.full()
    operator_expect = np.zeros(states.shape[1], dtype=complex)
    for i in range(0,shape(states)[1]):
        e = np.conj(states[:,i])@ (operator_matrix @ states[:,i])
        operator_expect[i] = e
    return operator_expect
```

<!-- #region id="44a9913f-521a-4315-b03f-706bc70aa8cf" -->
Let's remember now to adjust the $\Delta E \neq 1$ to make sure we don't get energy transfer to the boson field.
<!-- #endregion -->

```python id="8435b8f5-3891-4284-bfb9-06f58473498f"
DeltaE = 2.5 # Mismatch between boson energy and the TLS energy to make sure we avoid emission
omega = 1
U = 0.001 # Coupling is 10x lower than the last tutorial because we are upping the number of TLS
```

<!-- #region id="33911c81-f8c8-4383-b786-c3f02e9003cb" -->
### Fully excited system A, de-excited system B
<!-- #endregion -->

```python id="54bbc040-cac6-4596-ad32-c4c634cbb6be" outputId="96347e3f-8aa6-48e9-ee91-b2bcda4b4128"
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

<!-- #region id="45a7779d-1138-4bc1-8951-3b8afffaa019" -->
We'll once again store the "base" rate of excitation transfer $\Gamma_{1T}$ for the case of a single TLS in each system A and B so that we might reference it later
<!-- #endregion -->

```python id="79bbefe5-a514-4ab6-ae08-f097518ebf80" outputId="a284533c-f00d-4d95-d70f-e6e9478f8d47"
gamma_1T = rate[0]
gamma_1T
```

```python id="6da6cf8e-a278-4415-b293-68c05aef4b86" outputId="90f9aa51-b366-4f50-dacb-91b9ebe3df7f"
plt.plot(Ns,rate/gamma_1T,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised transfer rate ($\Gamma/\Gamma_1$)");
plt.title("Transfer of $N$ delocalised excitations from A to B (Fig. 33)");
```

```python id="a27e72db-f72f-4d65-9b00-0e8a14ea459b" outputId="49e37bc5-968a-408f-b778-54290230385b"
print("slope = ", linregress(np.log10(Ns), np.log10(rate)).slope)
```

<!-- #region id="5fc40817-934a-413a-83e3-a4cd0de12fdb" -->
Fig. 33 confirms the observation from the last tutorial that the excitation transfer has a more favourable scaling with the number of TLS than spontaneous emission - $N^2$ vs $N$ for the fully excited case.
> Note that in the last tutorial we observed $N$ vs $\sqrt{N}$ because we were looking at the Rabi frequency instead of the emission/transfer rates.

We can think of this $N^2$ scaling in terms of pathways just like with spontaneous emission. When an excitation moves from the fully excited system A to de-excited system B, it leaves behind a "hole" in system A which can be in one of $N$ places and moves to system B where it can be one of $N$ places.

Let's see if we get the same scaling $N^2$ with a single excitation in A - mirroring what we found with spontaneous emission.
<!-- #endregion -->

<!-- #region id="e67cde3b-f2d6-4e5a-af13-d74af8a35275" -->
### Single delocalised excitation in system A, de-excited system B
<!-- #endregion -->

```python id="bf1c37eb-e17a-4903-b56d-502b1833a05e" outputId="41384490-1800-4be9-8392-4bbcdbd4da41"
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

```python id="c15cde38-7d29-4870-a847-2a2fe2953536" outputId="7b54413f-4ea9-4298-edcf-d0b500e7e3b1"
plt.plot(Ns,rate/gamma_1T,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised transfer rate ($\Gamma/\Gamma_1$)");
plt.title("Transfer of a single delocalised excitation from A to B (Fig. 40)");
```

```python id="5b190c23-ada8-447e-90c4-301e2477b498" outputId="84b56c7a-8e15-4f4e-e690-0a59819db287"
print("slope = ", linregress(np.log10(Ns), np.log10(rate)).slope)
```

<!-- #region id="5dc2e40c-d6a3-4a32-a2c6-262837816754" -->
Fig. 40 confirms that we have the same $N^2$ scaling for excitation transfer even when there is only a single excitation in system A.

Thinking again in terms of pathways. The single excitation in system A can be in one of $N$ places and the excitation in each of of those "configurations" can move to one of $N$ "empty" places in the de-excited system B.

The natural next step is to wonder what happens if we play the $N/2$ game. In other words, if we "half-excite" system A, will we find some kind of "supertransfer".

I think you know what the answer is 😉 but let's see it.
<!-- #endregion -->

<!-- #region id="4c0ce86d-3e83-460e-aa86-0eb4527fee68" -->
### Half excited system A, de-excited system B
<!-- #endregion -->

```python id="095fa52c-8af2-436f-8b43-d6f58b1cdded" outputId="b6f5ed12-f3e1-43c4-b454-e706031e0139"
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

```python id="400823ba-6df9-4ff6-a753-b4884bdf95fc" outputId="1ab0cd3e-109b-4cec-fff0-b42428b5b2a9"
plt.plot(Ns,rate/gamma_1T,"-o")
plt.xlabel("Number of TLS (N)")
plt.ylabel("Normalised transfer rate ($\Gamma/\Gamma_1$)");
plt.title("Transfer of $N/2$ delocalised excitations from A to B (Fig. 47)");
```

<!-- #region id="d471a472-6a6b-4683-aa46-b7b2db20d0fb" -->
Performing the usual regression of the log of the rate gives:
<!-- #endregion -->

```python id="46e88695-c208-43d7-b8aa-f034b0a10826" outputId="3c23350e-54dc-44ed-fe23-629ce4de1696"
print("slope = ", linregress(np.log10(Ns[:]), np.log10(rate[:])).slope)
```

<!-- #region id="b4ea423b-1200-40ab-8216-62ab86f26ab3" -->
and if we look at the trend for higher $N$ we see something that appears to approach $N^3$ 🤯.
<!-- #endregion -->

```python id="9c27706e-8c26-4416-a9f1-2f14c0493738" outputId="947c8e18-df4d-40b0-d7e7-7bc169873798"
print("slope = ", linregress(np.log10(Ns[2:]), np.log10(rate[2:])).slope)
```

<!-- #region id="ce58c6f7-c99e-4b3a-98e5-d5059149d3ed" -->
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
<!-- #endregion -->

<!-- #region id="e5f11dc3-467b-40d6-9c37-6b5157f78597" -->
## 8.5 - Superradiance vs Supertransfer
<!-- #endregion -->

<!-- #region id="f52e788e-2235-49c8-bafe-159398d8ab7b" -->
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

<!-- #region id="1fff3e84-8b14-44d7-9b36-6b7f88a539fd" -->
A true comparison between the "base" emission and transfer rates requires us to know these rates for the same $\Delta E$. Unfortunately, we've used $\Delta E = 1$ for spontaneous emission and  $\Delta E = 2.5$ for excitation transfer so we can't make an accurate comparison.

You may recall that the reason we've been using $\Delta E = 2.5$ for excitation transfer is to suppress spontaneous emission. With $\Delta E = 1$ we'd never observe a "base" excitation transfer rate because spontaneous emission would always dominate.

So, what can we do?

We can start by just pretending that the excitation transfer rate is independent of $\Delta E$. We don't anticipate this to be the case, but it can be instructive to do this pretending sometimes. Now we can just plug in our numbers:
<!-- #endregion -->

```python id="6f371002-5bbc-451b-bcfe-1ef480572abe" outputId="253707af-df78-487f-935e-197f8fe6e487"
gamma_1E/gamma_1T
```

<!-- #region id="95430763-1a4d-4972-a1d2-0309ef3f98e0" -->
We'd need to have $N > 6.9 \times 10^6$ for excitation transfer to dominate. This is well outside of what we can simulate but not outside of what we might find in e.g. a solid system that usually contains on the order of $10^{22}$ particles in every $\text{cm}^3$.

Food for thought 🤔.
<!-- #endregion -->

<!-- #region id="fccf0665-9cca-423f-bfe4-7315bd944b98" -->
## Next up...

A lot of what we've done today has laid the foundations for us to explore emission and transfer rates in many TLS. We've also seen how delocalising our excitations gives us huge speed ups in quantum processes opening up a way to observe otherwise very slow processes like excitation transfer.

Whether excitation transfer can ever truly compete with spontaneous emission remains to be seen. We are however motivated to dig deeper into how the rates depend on the other parameters in our system - specifically $\Delta E$. We'll dig into this next time.

Until then. 👋
<!-- #endregion -->
