# Building the Jupyter Book V1

Reference: https://jupyterbook.org/v1/start/build.html

## Installation

Install Jupyter Book using `uv`:

```bash
uv add "jupyter-book<2"
```

## Build Instructions

### Initialize the book structure

```bash
jupyter-book create ./report/
```

### Build the book

```bash
jupyter-book build ./report/
```

## Viewing the Book

Open the generated HTML file in your browser:

```
./report/_build/html/index.html
```

