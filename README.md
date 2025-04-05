# Fundamentals of Image Analysis

This repository contains materials and examples for the course **Fundamentals of Image Analysis** (_Czech: Základy analýzy obrazu_) taught at **VŠB – Technical University of Ostrava**. Each lecture has its own dedicated folder.

## Structure

```
.
├── lec1/
├── lec2/
├── lec3/
├── lec4/
├── lec5/
├── lec6/
├── lec7/
├── lec8/
└── requirements.txt
```

- `lec1` to `lec8` – folders with code and materials for individual lectures
- `requirements.txt` – list of required Python packages
- `venv/` – virtual environment (can be ignored when cloning)

## Usage

1. Clone the repository:

```bash
git clone https://github.com/MAKVSB/ZAO.git ZAO
cd ZAO
```

2. Create and activate the virtual environment (if not already created):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```
