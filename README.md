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
pip install qiskit qiskit-ibm-runtime pyscf rustworkx ffsim maturin pyyaml
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

Spindle is configured purely through YAML config files, so you can easily reuse and share your exact experimental setups.

### Running with Default Settings
If you run `main.py` without specifying a config file, Spindle will automatically generate a fresh `config/default.yaml` file populated with the default values:

```bash
python main.py FCIDUMP/my_molecule.txt
```

### Running with Custom Settings
In the `config/` directory, you can also create your own custom settings by copying and pasting `default.yaml` and editing the values. If you're not sure what a particular setting does, you can find descriptions for all of the settings in `config.py`.

You only need to define the settings you want to change, as everything else will safely fall back to its default value. However, it's highly recommended to copy all of the fields in `default.yaml`, as the default values may change in future iterations of this software.

Pass the config file as the second argument:
```bash
python main.py FCIDUMP/my_molecule.txt config/my_run.yaml
```

`FCIDUMP/` contains example molecules to run, but feel free to add your own!