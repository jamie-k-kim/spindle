# These are some helper functions for tasks that I find myself running often.

import os
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_aer import AerSimulator
from dotenv import load_dotenv


def connect_backend(
    use_real_qpu=False,
    token=None,
    use_dotenv_token=True,
    instance=None,
    use_dotenv_instance=True,
    specific_backend=None,
    min_qubits=0,
    noiseless=False,
    method=None,
    use_gpu=False,
):
    """
    Most to least resource intensive:
    superop
    unitary
    density_matrix
    statevector
    extended_stabilizer
    matrix_product_state
    stabilizer
    """

    if use_real_qpu:
        
        CRN = None
        if instance:
            CRN = instance;
        elif use_dotenv_instance:
            load_dotenv()
            CRN = os.getenv("INSTANCE_CRN")
        # Else leave instance blank

        TOKEN = None
        if token:
            TOKEN = token
        elif use_dotenv_token:
            load_dotenv()
            TOKEN = os.getenv("API_TOKEN")
        # Else use saved token

        # Strip any literal single/double quotes or whitespace left by dotenv parser
        if TOKEN:
            TOKEN = TOKEN.strip().strip("'\"")
        if CRN:
            CRN = CRN.strip().strip("'\"")

        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN, instance=CRN)

        if specific_backend:
            backend = service.backend(specific_backend)
        else:
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)

        print(f"Deploying to physical QPU: {backend.name}")

    else:
        backend = AerSimulator.from_backend(FakeBrisbane())
        print("Fake IBM Brisbane")
        if noiseless:
            backend.set_options(noise_model=None)
            print("Noiseless")
        else:
            print("Noisy")

        if method:
            backend.set_options(method=method)
            print (f"Method: {method}")

        if use_gpu:
            backend.set_options(device="GPU")
            print("Using GPU")

    return backend


def run_circuit(transpiled_circuit, backend, shots):
    sampler = Sampler(mode=backend)
    job = sampler.run([transpiled_circuit], shots=shots)
    return job.result()


def get_counts(result):
    # If it is a standard Qiskit/Aer Result object (non-sampler), use get_counts() directly
    if hasattr(result, "get_counts"):
        try:
            return result.get_counts()
        except Exception:
            pass

    # If it is a PrimitiveResult (Qiskit Runtime SamplerV2 result)
    try:
        # Get the first PubResult
        pub_res = result[0]
        pub_data = pub_res.data
        
        # Qiskit DataBin contains dynamically named fields for each classical register.
        # We find the first field that contains count data.
        fields = getattr(pub_data, "_fields", None) or [name for name in dir(pub_data) if not name.startswith('_')]
        for field in fields:
            reg_data = getattr(pub_data, field, None)
            if reg_data is not None and hasattr(reg_data, "get_counts"):
                return reg_data.get_counts()
    except Exception:
        pass

    raise TypeError(f"Could not extract counts from result object of type: {type(result)}")