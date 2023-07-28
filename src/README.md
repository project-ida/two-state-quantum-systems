# Jupyter notebook source files

This folder contains what we call `source files` for the Jupyter notebooks in the home directory of this repo.

## What are notebook source files?

Notebook meta data and outputs are removed and the remaining content is converted into markdown format - this is then a `source file`.

## Why do we need notebook source files?
Without some kind of source file, version control of notebooks is extremely difficult.





## How do I create / update source files?

We use [Jupytext](https://github.com/mwouts/jupytext) to pair `.ipynb` files with `.md` files. Once the pairing is set-up once, you'll never have to worry about it again - the two files will be kept in sync.

You'll need to have `jupytext` installed and also the jupytext jupyter lab extension. A simple `pip install jupytext` should sort out both requirements.

To check this has worked, open the command palette in jupyter lab via `View -> Activate Command Palette` and type `jupytext`. If you see references to e.g. "Pair Notebook..." then all is good. If this doesn't work, go to `Settings -> Enable Extension Manager`. From there, you can search for and install the Jupytext plugin manually via `View -> Extension Manager`.

To pair a notebook with a source file and make sure that source file ends up in the `src` folder:
- Open a new notebook
- Open the command palette
- Type `Pair Notebook` and select `Pair Notebook with Markdown` option
- Close the notebook
- Open the notebook in a plain text editor
- Navigate to the bottom and modify the jupytext part of the meta data to look like this:
  ```
  "jupytext": {
    "formats": "ipynb,src//md"
    }
  ```
- Open the notebook in Jupyter lab and hit the save button - a new source file should appear in the `src` folder
- Delete any additional source files that might have been accidentally created in the root directory
- Commit your changes to the repo