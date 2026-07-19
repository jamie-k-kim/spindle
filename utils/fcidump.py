import os
from pyscf import gto, scf, mcscf, tools

molecules = {
    'H2': {
        'geom': 'H 0 0 0; H 0 0 0.741',
    },
    'LiH': {
        'geom': 'Li 0 0 0; H 0 0 1.595',
    },
    'BeH2': {
        'geom': 'Be 0 0 0; H 0 0 1.326; H 0 0 -1.326',
    },
    'H2O': {
        'geom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
    },
    'H2O_4o4e': {
        'geom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
        'cas': (4, 4)
    },
    'CO': {
        'geom': 'C 0 0 0; O 0 0 1.128',
    }
}

basis_set = 'sto-3g' 

# Ensure the FCIDUMP output folder exists relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_dir = os.path.join(project_dir, 'FCIDUMP')
os.makedirs(output_dir, exist_ok=True)

for name, info in molecules.items():
    print(f'--- Processing {name} ---')
    
    mol = gto.M(
        atom=info['geom'],
        basis=basis_set,
        symmetry=True,
        spin=0,
        charge=0
    )
    
    mf = scf.RHF(mol)
    mf.kernel()
    
    filename = f'fci_dump_{name}.txt'
    # Fallback name for output file in the main folder format
    if name == 'H2O_4o4e':
        filename = 'H2O_4o4e.txt'
    elif name == 'H2':
        filename = 'H2_2o2e.txt'
    elif name == 'LiH':
        filename = 'LiH_6o4e.txt'
    elif name == 'BeH2':
        filename = 'BeH2_7o6e.txt'
    elif name == 'H2O':
        filename = 'H2O_7o10e.txt'
    elif name == 'CO':
        filename = 'CO_10o14e.txt'
        
    filepath = os.path.join(output_dir, filename)
    
    if 'cas' in info:
        ncas, nelec = info['cas']
        print(f'Running CASSCF({ncas}, {nelec}) active space...')
        mc = mcscf.CASSCF(mf, ncas, nelec)
        mc.kernel()
        tools.fcidump.from_mcscf(mc, filepath, tol=1e-12)
    else:
        tools.fcidump.from_scf(mf, filepath, tol=1e-12)
        
    print(f'Success: FCIDUMP saved to {filepath}\n')
