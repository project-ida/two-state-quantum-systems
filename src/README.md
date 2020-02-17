# Jupyter notebook source files

This folder contains what we call `source files` for the Jupyter notebooks in the home directory of this repo.

## What are notebook source files?

Notebook meta data and outputs are removed and the remaining content is converted into markdown format - this is then a `source file`.

## Why do we need notebook source files?
Without some kind of source file, version control of notebooks is extremely difficult.

## How are these source files created?

The files are automatically generated using [Jupytext](https://github.com/mwouts/jupytext) in conjunction with Jupyter Lab.

Jupytext has a nice [online demo](https://mybinder.org/v2/gh/mwouts/jupytext/master?urlpath=lab/tree/demo/get_started.ipynb) to help you get familiar with the [Jupytext commands](https://github.com/mwouts/jupytext#jupytext-commands-in-jupyterlab).



## How do I create source files locally?

A simple `pip install jupytext` should be all you need to activate the Jupytext commands in your local Jupyter lab. If this doesn't work then go to `settings`, `enable extension manager`. From there you can search for and install the Jupytext plugin manually.

If you follow the instructions in the [online demo](https://mybinder.org/v2/gh/mwouts/jupytext/master?urlpath=lab/tree/demo/get_started.ipynb)  then you will generate some nice source files. The final step that's required to place the source files inside a subdirectory called `src` is to open up the notebook as a raw text file, navigate to the bottom and modify the jupytext part of the meta data to look like this:

```
"jupytext": {
   "formats": "ipynb,src//md"
  }
```

I'm sorry for this last part - it a bit annoying, but currently unavoidable.
