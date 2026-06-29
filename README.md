# PolymerCanon — BigSMILES Canonicalization

This project provides a canonicalization algorithm for BigSMILES strings, which are used to represent polymer structures.

This code accompanies the paper:

> **A Canonical Text Representation for Polymers via BigSMILES and Tree Automata**
> https://pubs.acs.org/doi/10.1021/acs.jcim.5c02784

---

## Requirements

- **Python 3.10.8**

---

## Installation

### 1. Clone or download the repository

### 2. Create a virtual environment with Python 3.10

**Mac/Linux:**
```bash
python3.10 -m venv venv
```

**Windows:**
```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install the dependencies

```bash
pip install -r polymercanon/requirements.txt
```

---

## Usage

### Canonicalizing a single BigSMILES string

```python
from polymercanon.canon_tools import canonicalize_bigsmiles

bigsmiles = "CCO{[>][<]CCO[>][<]}CCO"
canonical = canonicalize_bigsmiles(bigsmiles=bigsmiles, output_folder="Output", plot=False)
print(canonical)
```

### Running the built-in test set

The file `polymercanon/run.py` contains a set of validation examples. To run them:

```bash
cd polymercanon
python run.py
```

Results will be saved in the `Validation/Tests/` folder, including:
- One subfolder per test case with intermediate automaton plots
- A `Results.xlsx` file with the input BigSMILES and its canonical form

To add or change test cases, edit the `validation_set` list in `run.py`.

### Parameters of `canonicalize_bigsmiles`

| Parameter | Type | Description |
|---|---|---|
| `bigsmiles` | `str` | Input BigSMILES string |
| `output_folder` | `str` | Folder where output files will be saved |
| `plot` | `bool` | If `True`, saves plots of intermediate automata |

---

## Project Structure

```
FINAL CODE ACTUALLY USED IN PAPER/
├── polymercanon/
│   ├── run.py                    # Entry point: runs the validation test set
│   ├── canon_tools.py            # Main canonicalization functions
│   ├── string_conversion.py      # Converts automata back to BigSMILES
│   ├── tree_automata.py          # Tree automaton data structures
│   ├── sobj_string.py            # Stochastic object string utilities
│   ├── polymersearch/            # Graph construction and search tools
│   ├── parser/                   # BigSMILES parser
│   └── requirements.txt          # Python dependencies
└── README.md
```
