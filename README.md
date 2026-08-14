# Installation

### 1. Prerequisites
Make sure you have these installed on your system:
- **Python**: Version 3.10 or higher.
- **Rust**: A functional Rust toolchain (`cargo` / `rustc`). You can install it via [rustup](https://rustup.rs/).
- **GAP**: If compiling SPIRAL from source, make sure the GAP engine is available.

### 2. Python Environment Setup
Install the required packages. I highly recommend using a virtual environment. 

```bash
# Create and activate a virtual environment
python3 -m venv spindle_env
source spindle_env/bin/activate

# Install required Python dependencies
pip install "qiskit<2.5" "numpy<2" qiskit-ibm-runtime pyscf rustworkx ffsim maturin pyyaml qiskit-addon-sqd qc-pyci
```

### 3. Compiling SPIRAL
This program uses the GAP-based SPIRAL engine and its Rust bindings (`py-spiral-quantum`). **Both must be compiled for the program to run.**

First, compile the core SPIRAL engine:
```bash
cd spiral-software
mkdir build
cd build
cmake ..
make install
cd ../..
```

Now compile the Rust bindings. If you're using a virtual environment (as recommended):
```bash
cd py-spiral-quantum
maturin develop --release
```

If you're not using a virtual environment:
```bash
cd py-spiral-quantum
maturin build --release
pip install target/wheels/py_spiral_quantum-*.whl --user
```

### 4. IBM Credentials
To access IBM's QPUs, you'll need to set up your IBM Quantum credentials. Authenticate your Qiskit Runtime Service locally, or create a `.env` file at the project's root with the following variables (recommended):
```env
API_TOKEN=<your_api_token>
INSTANCE_CRN=<your_instance_crn>
```

---

# Usage

### Generating the Molecule Files (FCIDUMPs)

Before you use Spindle, you need to add molecules to the `FCIDUMP/` folder, which will be empty at first. The `fcidump.py` script, which is provided, will generate ~30 sample molecules ranging from 2 to 90 orbitals. This will help you get started.

```bash
python fcidump.py
```

### Adding Your Own Molecules

If your molecules are already in FCIDUMP format, you can simply drag them into the `FCIDUMP/` folder.

Otherwise, you'll need to generate the FCIDUMPs first:

1. Search up a molecule on PubChem: https://pubchem.ncbi.nlm.nih.gov/compound
2. Scroll down to the "3D Conformer" section.
3. Click "Download Coordinates" and save the SDF file.
4. Convert it to .xyz using [Open Babel](https://openbabel.org/) or an online converter.
5. Drag the .xyz file into the `xyz/` folder.
6. Go to `fcidump.py` and add your molecule to the molecules list.
8. Following the provided examples, set the molecule's `geom` field to the .xyz file.
9. Add additional fields (charge, spin, basis, etc.) if needed.
9. Execute the script.

### Running with Default Settings
Spindle is configured purely through YAML files, so you can easily reuse and share your setups. If you run `main.py` without specifying a config file, Spindle will generate a fresh `config/default.yaml` file, populate it with the default values, and run the pipeline using those.

```bash
python main.py FCIDUMP/my_molecule.txt
```

### Running with Custom Settings
In the `config/` directory, you can also create your own custom settings by copying and pasting `default.yaml` and editing the values. If you're not sure what a particular setting does, you can find descriptions for all of the settings in `config.py`.

You only need to define the settings you want to change; everything else will safely fall back to its default value. However, I recommend copying all of the fields in `default.yaml`, as the default values may change in future iterations of this software.

Pass the config file as the second argument:
```bash
python main.py FCIDUMP/my_molecule.txt config/my_run.yaml
```

### Injecting Custom Circuits

Instead of automatically generating the LUCJ ansatz from the FCIDUMP, you can also use a pre-made circuit. The repository has an empty `circuits/` folder, where you can place your .qpy or .qasm files (Spindle supports both).

```bash
python main.py FCIDUMP/my_molecule.txt [optional config] circuits/my_circuit.qpy
```

> [!NOTE]
> Spindle recognizes FCIDUMPs, config files, and custom circuits by file extension, so the order of the arguments doesn't matter.