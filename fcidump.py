import os
from pyscf import gto, scf, mcscf, tools

basis_set = 'sto-3g' # Default

# Ordered from least to most orbitals
molecules = {
    'H2_2o2e': {
        'geom': 'xyz/H2.xyz',
    },
    'HeH_2o2e': {
        'geom': 'xyz/HeH.xyz',
        'charge': 1
    },
    'LiH_6o4e': {
        'geom': 'xyz/LiH.xyz',
    },
    'OH_6o9e': {
        'geom': 'xyz/OH.xyz',
        'spin': 1
    },
    'HF_6o10e': {
        'geom': 'xyz/HF.xyz',
    },
    'BeH2_7o6e': {
        'geom': 'xyz/BeH2.xyz',
    },
    'water_7o10e': {
        'geom': 'xyz/water.xyz',
    },
    'BH3_8o8e': {
        'geom': 'xyz/BH3.xyz',
    },
    'ammonia_8o10e': {
        'geom': 'xyz/ammonia.xyz',
    },
    'methane_9o10e': {
        'geom': 'xyz/methane.xyz',
    },
    'CO_10o14e': {
        'geom': 'xyz/CO.xyz',
    },
    'N2_10o14e': {
        'geom': 'xyz/N2.xyz',
    },
    'NO_10o15e': {
        'geom': 'xyz/NO.xyz',
        'spin': 1
    },
    'O2_10o16e': {
        'geom': 'xyz/O2.xyz',
        'spin': 2
    },
    'F2_10o18e': {
        'geom': 'xyz/F2.xyz',
    },
    'HCN_11o14e': {
        'geom': 'xyz/HCN.xyz',
    },
    'acetylene_12o14e': {
        'geom': 'xyz/acetylene.xyz',
    },
    'formaldehyde_12o16e': {
        'geom': 'xyz/formaldehyde.xyz',
    },
    'H2O2_12o18e': {
        'geom': 'xyz/H2O2.xyz',
    },
    'ethylene_14o16e': {
        'geom': 'xyz/ethylene.xyz',
    },
    'methanol_14o18e': {
        'geom': 'xyz/methanol.xyz',
    },
    'hydrazine_14o18e': {
        'geom': 'xyz/hydrazine.xyz',
    },
    'methylamine_15o18e': {
        'geom': 'xyz/methylamine.xyz',
    },
    'O3_15o24e': {
        'geom': 'xyz/O3.xyz',
    },
    'ethanol_21o26e': {
        'geom': 'xyz/ethanol.xyz',
    },
    'N2_28o14e': {
        'geom': 'xyz/N2.xyz',
        'basis': 'cc-pVDZ' # Larger basis to get 28 orbitals
    },
    'benzene_36o42e': {
        'geom': 'xyz/benzene.xyz',
    },
    'ethylene_48o16e': {
        'geom': 'xyz/ethylene.xyz',
        'basis': 'cc-pVDZ' # Larger basis to get 48 orbitals
    },
    'acetaminophen_64o80e': {
        'geom': 'xyz/acetaminophen.xyz',
    },
    'vitamin_c_68o92e': {
        'geom': 'xyz/vitamin_c.xyz',
    },
    'aspirin_73o94e': {
        'geom': 'xyz/aspirin.xyz',
    },
    'caffeine_80o102e': {
        'geom': 'xyz/caffeine.xyz',
    },
    'tryptophan_87o108e': {
        'geom': 'xyz/tryptophan.xyz',
    },
}

# Make sure the FCIDUMP folder exists relative to the script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'FCIDUMP')
os.makedirs(output_dir, exist_ok=True)

for name, info in molecules.items():
    print(f'Generating {name}...')
    
    spin = info.get('spin', 0)
    mol = gto.M(
        atom=info['geom'],
        basis=info.get('basis', basis_set),
        symmetry=True,
        spin=spin,
        charge=info.get('charge', 0)
    )
    
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
        
    mf.kernel()
    
    filename = f'{name}.txt'
        
    filepath = os.path.join(output_dir, filename)
    
    if 'cas' in info:
        ncas, nelec = info['cas']
        print(f'Running CASSCF({ncas}, {nelec}) active space...')
        mc = mcscf.CASSCF(mf, ncas, nelec)
        mc.kernel()
        tools.fcidump.from_mcscf(mc, filepath, tol=1e-12)
    else:
        tools.fcidump.from_scf(mf, filepath, tol=1e-12)
        
    print(f'FCIDUMP saved to {filepath}\n')
