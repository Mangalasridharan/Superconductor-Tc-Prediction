import os
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element

all_elements = set()
file_directory=""

# Iterate through all .cif files in the directory
for file in os.listdir(file_directory):
    if file.endswith(".cif"):
        try:
            structure = Structure.from_file(os.path.join(file_directory, file))
            # Extract elements from possibly disordered/mixed sites
            for site in structure:
                elements_in_site = [el.symbol for el in site.species.elements]
                all_elements.update(elements_in_site)
        except Exception as e:
            print(f"Error reading {file}: {e}")


# Convert to list and sort in periodic table order
sorted_elements = sorted(all_elements, key=lambda x: Element(x).Z)

# Output results
print("Total unique elements found:", len(sorted_elements))
print("Sorted elements in periodic table order:")
print(sorted_elements)

