import json 
from pymatgen.core.periodic_table import Element 
from unique_elements import sorted_elements

atom_features = {}

for symbol in sorted_elements:
    try:
        elem = Element(symbol)
        atomic_number  = elem.Z
        eneg = elem.X if elem.X is not None else 0.0
        radius = elem._atomic_radius if elem._atomic_radius is not None else 0.0

        valence = elem.full_electronic_structure[-1][2]

        feature_vectors = [atomic_number,eneg,radius,valence]
        atom_features[str(atomic_number)]=feature_vectors

    except Exception as e:
        print(f"Error for {symbol}:{e}")

with open('atom_init.json','w') as f:
    json.dump(atom_features, f, indent=4)

print("atom_init.json file created")
