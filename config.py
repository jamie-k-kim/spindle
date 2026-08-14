from dataclasses import dataclass
from typing import Optional

@dataclass
class RunConfig:
    # Routing
    qpu_threshold: int = 40             # Largest size to run on the QPU. Split anything bigger.
    classical_threshold: int = 20       # If a fragment is this small, solve it classically.
    fci_threshold: int = 10             # If a fragment is this small, solve classically with exact FCI.
    
    # EWF
    ewf_bath_type: str = "mp2"          # Method used to surround and protect the fragments.
    ewf_truncation: float = 1e-5        # How much surrounding noise to ignore when splitting.

    # SPIRAL (disabled for custom circuits)
    spiral_chunk_size: int = 1000       # How many gates to pack into a single bundle so SPIRAL doesn't crash.
    spiral_max_gates: int = 100000      # The absolute max number of gates allowed before we give up.

    # Auto-tuning
    num_layouts: int = 50               # How many different ways to try placing the circuit onto the QPU.
    tuning_top_candidates: int = 3      # How many layouts survive the fast heuristic filter to enter simulation tie-breaker.
    tuning_sim_threshold: int = 20      # Max qubits for simulation tie-breaker. Larger circuits skip simulation.
    beta: float = 0.001                 # Helps the program pick the most accurate QPU layout.
    min_count: int = 1                  # Throw away quantum results that occur less than this.

    # QPU
    transpile_optimization_level: int = 1  # Qiskit optimization level for ISA mapping (0-3).
    use_spiral: bool = True             # Enables SPIRAL for circuit optimization.
    use_spanning_tree: bool = True      # Spanning tree Givens elimination (works only if connectivity is square).
    real_qpu: bool = False              # Use an actual IBM QPU instead of a simulator.
    backend: Optional[str] = None       # Specific name of the IBM QPU (e.g. ibm_brisbane).
    shots: int = 100000                 # How many times to repeat the quantum experiment.
    connectivity: str = "heavy-hex"     # How the qubits are wired together ("heavy-hex" or "square").
    lucj_reps: int = 2                  # How many layers of the ansatz to run (more layers = more accurate but more noise).
    lucj_tol: float = 1e-5              # Precision limit for the quantum math.
    lucj_optimize: bool = False         # Let the program guess better initial settings for the circuit.
    lucj_method: str = "L-BFGS-B"       # Math algorithm used to find those better initial settings.
    
    # TrimSQD
    trim_fraction: float = 0.10         # What percentage of bad quantum results to throw away each round.
    trim_stop_tol: float = 1e-4         # Stop throwing away results when the energy stops changing by this amount.

    # Reference / Grading
    reference_threshold: int = 30       # Calculate the answer classically if the molecule is smaller than this, just for grading / reference.
    reference_fci_threshold: int = 10   # Calculate the FCI (perfect answer) if the molecule is smaller than this, otherwise use CCSD.

def generate_default_yaml(path: str):
    import yaml
    import os
    import dataclasses
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    default_config = RunConfig()
    
    with open(path, "w") as f:
        yaml.dump(dataclasses.asdict(default_config), f, default_flow_style=False, sort_keys=False)
    print(f"     [Info] Auto-generated default config at {path}")

def load_config(path: str) -> RunConfig:
    import yaml
    import os
    
    if path == "config/default.yaml":
        generate_default_yaml(path)
    elif not os.path.exists(path):
        print(f"     [Warning] Config file {path} not found. Using defaults.")
        return RunConfig()
            
    with open(path, "r") as f:
        try:
            custom_settings = yaml.safe_load(f)
            if not custom_settings:
                custom_settings = {}
        except Exception as e:
            print(f"     [Error] Failed to parse {path}: {e}. Using defaults.")
            return RunConfig()
            
    default_config = RunConfig()
    
    # Only override fields that actually exist in the dataclass
    for key, value in custom_settings.items():
        if hasattr(default_config, key):
            setattr(default_config, key, value)
        else:
            print(f"     [Warning] Unknown config key '{key}' found in {path}. Ignoring.")
            
    return default_config
