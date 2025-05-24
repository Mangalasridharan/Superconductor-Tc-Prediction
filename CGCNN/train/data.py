from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]                  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)          # Append atom features
        batch_nbr_fea.append(nbr_fea)            # Append neighbor bond features
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)              # Append the target value for this crystal
        batch_cif_ids.append(cif_id)             # Append the CIF ID
        base_idx += n_i                          # Update the base index for next crystal
    return (torch.cat(batch_atom_fea, dim=0),    # Shape: (N, atom_fea_len)
            torch.cat(batch_nbr_fea, dim=0),     # Shape: (N, M, nbr_fea_len)
            torch.cat(batch_nbr_fea_idx, dim=0), # Shape: (N, M)
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids                            # List of LongTensors (N0,),Shape: (N0, 1), List of CIF IDs

class AtomInitializer(object):
    """
    Base class for initializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)  # Store the allowed atom types (as a set of atomic numbers)
        self._embedding = {}               # Dictionary mapping atom type to its embedding vector
    
    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types  # Ensure the requested atom is in the known set
        return self._embedding[atom_type]    # Return the feature vector for the atom
    
    def load_state_dict(self, state_dict):
        self._embedding = state_dict                    # Load an external dictionary of atom embeddings
        self.atom_types = set(self._embedding.keys())   # Update the set of atom types
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}    # Create a reverse dictionary: idx → atom_type
        
    def state_dict(self):
        return self._embedding  # Return current embeddings (like for saving checkpoint)
    
    def decode(self, idx):
        if not hasattr(self, '_decodedict'):  # Lazy construction of decode map if not yet defined
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]  # Return atom type from index
    
class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)  # Load the JSON file into a dictionary
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}  # Convert string keys to integer atomic numbers
        atom_types = set(elem_embedding.keys())  # Get the set of atomic numbers in the file
        super(AtomCustomJSONInitializer, self).__init__(atom_types)  # Call the base class constructor
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)  # Store each feature vector as a NumPy array


class CIFData(Dataset):
    """
    PyTorch Dataset for loading and processing CIF crystal structure files
    along with target properties and custom edge features.

    Edge features: [distance, electronegativity_difference, sum_of_atomic_radii]
    """

    def __init__(self, root_dir, max_num_nbr=20, radius=15, dmin=0, step=0.2,
                 random_seed=123):
        # Root directory of the dataset containing .cif files, id_prop.csv, and atom_init.json
        self.root_dir = root_dir
        
        # Maximum number of neighbors per atom
        self.max_num_nbr = max_num_nbr

        # Radius to search for neighboring atoms
        self.radius = radius

        # Check that root directory exists
        assert os.path.exists(root_dir), 'root_dir does not exist!'

        # Load target property CSV: each row contains [cif_id, property_value]
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]  # List of [id, target] rows

        # Shuffle dataset for randomness (useful during training)
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        # Load atom initialization file (contains EN, radius, etc. per atomic number)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)

    def __len__(self):
        # Return number of CIF entries in dataset
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + '.cif'))

        atom_fea = np.vstack([
            self.ari.get_atom_fea(crystal[i].species.elements[0].number)
            for i in range(len(crystal))
        ])
        atom_fea = torch.Tensor(atom_fea)

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea, crystal_atom_idx = [], [], []

        for i, nbr in enumerate(all_nbrs):
            nbr_i_idx, nbr_i_fea = [], []

            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    f'{cif_id} did not find enough neighbors. Consider increasing radius.')

                pad_length = self.max_num_nbr - len(nbr)

                for _, dist, j,*rest in nbr:
                    fea_i = self.ari.get_atom_fea(crystal[i].species.elements[0].number)
                    fea_j = self.ari.get_atom_fea(crystal[j].species.elements[0].number)

                    en_i = fea_i[1]
                    en_j = fea_j[1]
                    r_i = fea_i[2]
                    r_j = fea_j[2]

                    nbr_i_fea.append([dist, abs(en_i - en_j), r_i + r_j])
                    nbr_i_idx.append(j)

                for _ in range(pad_length):
                    nbr_i_fea.append([self.radius + 1., 0.0, 0.0])
                    nbr_i_idx.append(0)

            else:
                for entry in nbr[:self.max_num_nbr]:
                    try:
                        dist = entry[1]
                        j = entry[2]
                    except Exception as e:
                        print(f"Bad entry: {entry}")
                        raise e

                    fea_i = self.ari.get_atom_fea(crystal[i].species.elements[0].number)
                    fea_j = self.ari.get_atom_fea(crystal[j].species.elements[0].number)

                    en_i = fea_i[1]
                    en_j = fea_j[1]
                    r_i = fea_i[2]
                    r_j = fea_j[2]

                    nbr_i_fea.append([dist, abs(en_i - en_j), r_i + r_j])
                    nbr_i_idx.append(j)

            nbr_fea.append(nbr_i_fea)
            nbr_fea_idx.append(nbr_i_idx)
            crystal_atom_idx.extend([i] * len(nbr_i_idx))

        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        nbr_fea = torch.Tensor(nbr_fea)
        target = torch.tensor([float(target)],dtype=torch.float)  # just a float tensor
        crystal_atom_idx = torch.LongTensor(crystal_atom_idx)

        return (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), target, cif_id
