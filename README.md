# two-state-quantum-systems

This repo is dedicated to exploring the dynamics of [two state quantum systems](https://en.wikipedia.org/wiki/Two-state_quantum_system). There will be a particular focus on the interaction of such systems with bosonic fields as exemplified in the simplified [Jaynes–Cummings model](https://en.wikipedia.org/wiki/Jaynes%E2%80%93Cummings_model). We will gradually build up to working with more advanced models such as the [spin-boson](http://dx.doi.org/10.1088/0953-4075/41/3/035601) and [Dicke](http://dx.doi.org/10.1002/qute.201800043) models.

## Why two state systems?

The most common two state system discussed in introductory quantum mechanics courses is the spin of an electron. It was discovered experimentally, in [1922 by Stern and Gerlach](https://www.feynmanlectures.caltech.edu/II_35.html#Ch35-S2), that the spin of an electron along any direction is quantised, taking values of either $+\hbar/2$ or $-\hbar/2$. Although interesting in its own right, the mathematical description of spin has much broader applications - **any two state problem can be translated into an equivalent spinning electron problem**.

It turns out that the dynamics of some complicated systems can be approximately described by considering only two states. For example:

- [The ammonia molecule](https://www.feynmanlectures.caltech.edu/III_08.html#Ch8-S6) and the associated [ammonia maser](https://www.feynmanlectures.caltech.edu/III_09.html)
- [Molecular bonding](https://www.feynmanlectures.caltech.edu/III_10.html#Ch10-S1)
- [Nuclear forces](https://www.feynmanlectures.caltech.edu/III_10.html#Ch10-S2)
- [Photon polarization](https://www.feynmanlectures.caltech.edu/III_11.html#Ch11-S4)
- The decay of [strange particles](https://www.feynmanlectures.caltech.edu/III_11.html#Ch11-S5)
- ... etc

It is therefore worthwhile to develop a strong intuition about how two state systems behave.

## Tutorials

The tutorials below are aimed at someone whose knowledge of quantum mechanics is at least at an advanced undergraduate level. The spirit of the tutorials is mainly towards demonstrating interesting quantum phenomena using the latest computational tools. To this end, we make significant use of an open source python software called [QuTiP](http://qutip.org/) - huge thanks goes to their developers 🙏.

- [01 - An isolated two state system](01-an-isolated-two-state-system.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/01-an-isolated-two-state-system.ipynb" target="_parent"> <img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" height="22"/></a>
- [02 - Perturbing a two state system](02-perturbing-a-two-state-system.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/02-perturbing-a-two-state-system.ipynb" target="_parent"> <img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" height="22"/></a>
- [03 - A two state system in a quantised field](03-a-two-state-system-in-a-quantised-field.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/03-a-two-state-system-in-a-quantised-field.ipynb" target="_parent"> <img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" height="22"/></a>
- [04 - Spin-boson model](04-spin-boson-model.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/04-spin-boson-model.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" height="22"/></a>
- [05 - Excitation transfer](05-excitation-transfer.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/05-excitation-transfer.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" height="22"/></a>
- [06 - Excitation transfer revisited](06-excitation-transfer-revisited.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/06-excitation-transfer-revisited.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" height="22"/></a>
- [07 - Many two state systems](07-many-two-state-systems.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/07-many-two-state-systems.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" height="22"/></a>
- [08 - Accelerating quantum processes](08-accelerating-quantum-processes.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/08-accelerating-quantum-processes.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" height="22"/></a>
- [09 - Destructive interference](09-destructive-interference.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/09-destructive-interference.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" height="22"/></a>
- [10 - Deep strong coupling ](10-deep-strong-coupling.ipynb)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.jupyter.org/github/project-ida/two-state-quantum-systems/blob/master/10-deep-strong-coupling.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" height="22"/></a>

## Intellectual Inspiration
These tutorials have been inspired by the work of Prof. Peter Hagelstein of MIT and his proposition of resonance energy transfer (RET) between atomic nuclei.

## Financial support
We would not have been able to continue creating these tutorials without the generous support of grants from the [Anthropocene Institute](https://anthropoceneinstitute.com/). 
