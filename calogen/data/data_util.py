from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from calogen.core.caloch_eval import HighLevelFeatures as HLF
from calogen.core.xml_handler import XMLHandler

# ═══════════════════════════════════════════════════════════════════════
#  Truth format metadata
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TruthFormat:
    """Records the truth HDF5 format so generated output can mirror it.

    Two known formats and their representations:

    ====================  ===================  ========================
    Trait                 CaloChallenge        LEMURS (Par04SiW)
    ====================  ===================  ========================
    energy_key            ``incident_energies`` ``incident_energy``
    energy_is_1d          False (N, 1)         True (N,)
    showers_grid_shape    None (flat)          (9, 16, 45)  (R, Phi, Z)
    ====================  ===================  ========================
    """

    energy_key: str = "incident_energies"
    energy_is_1d: bool = False
    showers_grid_shape: Optional[Tuple[int, ...]] = None

    @staticmethod
    def detect_from_file(file_path: str) -> "TruthFormat":
        """Auto-detect truth format by inspecting an HDF5 file.

        Detection strategy:
        - If ``"incident_energy"`` (singular) exists → LEMURS format
        - Otherwise → CaloChallenge format
        """
        with h5py.File(file_path, "r") as handle:
            if "incident_energy" in handle:
                showers_ds = handle["showers"]
                ndim = showers_ds.ndim
                if ndim > 2:
                    # LEMURS: (N, R, Phi, Z) — key the grid shape from dims 1+
                    grid_shape = tuple(int(s) for s in showers_ds.shape[1:])
                else:
                    grid_shape = None
                return TruthFormat(
                    energy_key="incident_energy",
                    energy_is_1d=True,
                    showers_grid_shape=grid_shape,
                )

        # CaloChallenge or similar flat format
        return TruthFormat()

    def unflatten_showers(self, flat: np.ndarray) -> np.ndarray:
        """Reshape flat showers (N, total_cells) to truth grid shape.

        For LEMURS, this is the inverse of ``_transpose_and_flatten``:
        ``flat → (N, Z, R, Phi) → transpose → (N, R, Phi, Z)``.
        """
        if self.showers_grid_shape is None:
            return flat
        R, Phi, Z = self.showers_grid_shape
        return flat.reshape(flat.shape[0], Z, R, Phi).transpose(0, 2, 3, 1)

    def flatten_showers(self, showers: np.ndarray) -> np.ndarray:
        """Reshape truth showers (N, R, Phi, Z) to flat (N, Z*R*Phi).

        No-op for already-flat data (``showers_grid_shape is None``) or 2-D
        input.  Exact inverse of :meth:`unflatten_showers` and the same
        convention as ``calogen.core.lemurs._transpose_and_flatten``.
        """
        if self.showers_grid_shape is None or showers.ndim <= 2:
            return showers
        axes = (0, showers.ndim - 1, *range(1, showers.ndim - 1))  # (0, 3, 1, 2)
        return np.transpose(showers, axes).reshape(showers.shape[0], -1)

    def format_energy(self, energy: np.ndarray) -> np.ndarray:
        """Ensure energy has the correct shape for this format."""
        if self.energy_is_1d:
            return energy.reshape(-1)
        return energy.reshape(-1, 1)


# ═══════════════════════════════════════════════════════════════════════

def load_data_calo(filename, layer_boundaries, energy=None):
    data = {}
    data_file = h5py.File(filename, 'r')
    if energy is not None:
        energy_mask = data_file["incident_energies"][:] == energy
        data["energy"] = data_file["incident_energies"][:][energy_mask].reshape(-1, 1) / 1.e3
        for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
            data[f"layer_{layer_index}"] = data_file["showers"][..., layer_start:layer_end][energy_mask.flatten()] / 1.e3
    else:
        data["energy"] = data_file["incident_energies"][:] / 1.e3
        #data["energy"] = data_file["incident_energies"][:] / 1.e3
        for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
            data[f"layer_{layer_index}"] = data_file["showers"][..., layer_start:layer_end] / 1.e3

    data_file.close()
    
    return data

def load_data(data_file, particle_type,  xml_filename, threshold=1e-5, energy=None, indices: np.array=None):
    """Loads the data for a dataset 1 from the calo challenge"""
    
    # Create a XML_handler to extract the layer boundaries. (Geometric setup is stored in the XML file)
    xml_handler = XMLHandler(particle_name=particle_type, 
    filename=xml_filename)
    
    layer_boundaries = np.unique(xml_handler.GetBinEdges())

    # Prepare a container for the loaded data
    data = {}

    # Load and store the data. Make sure to slice according to the layers.
    # Also normalize to 100 GeV (The scale of the original data is MeV)
    should_close = False
    if isinstance(data_file, str):
        data_file = h5py.File(data_file, 'r')
        should_close = True
    #data["energy"] = data_file["incident_energies"][:] / 1.e3
    if energy is not None:
        energy_mask = data_file["incident_energies"][:] == energy
        data["energy"] = data_file["incident_energies"][:][energy_mask].reshape(-1, 1) / 1.e3
        # print(energy_mask.shape)
        for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
            data[f"layer_{layer_index}"] = data_file["showers"][..., layer_start:layer_end][energy_mask.flatten()] / 1.e3
    else:
        if indices is not None:
            sorted_indices = np.sort(indices)
            data["energy"] = data_file["incident_energies"][sorted_indices].reshape(-1, 1) / 1.e3
            for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
                data[f"layer_{layer_index}"] = data_file["showers"][sorted_indices][..., layer_start:layer_end]/ 1.e3
            # print(data[f"layer_{layer_index}"].shape)
        else:
            data["energy"] = data_file["incident_energies"][:].reshape(-1, 1) / 1.e3
            for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
                data[f"layer_{layer_index}"] = data_file["showers"][..., layer_start:layer_end] / 1.e3
            # print(data[f"layer_{layer_index}"].shape)
    if should_close:
        data_file.close()
    
    return data, layer_boundaries

def get_energy_and_sorted_layers(data):
    """returns the energy and the sorted layers from the data dict"""
    
    # Get the incident energies
    energy = data["energy"]

    # Get the number of layers layers from the keys of the data array
    number_of_layers = len(data)-1
    
    # Create a container for the layers
    layers = []

    # Append the layers such that they are sorted.
    for layer_index in range(number_of_layers):
        layer = f"layer_{layer_index}"
        
        layers.append(data[layer])
        
            
    return energy, layers


def save_data(data, filename):
    """Saves the data with the same format as dataset 1 from the calo challenge"""
    
    # extract the needed data
    incident_energies, layers = get_energy_and_sorted_layers(data)
    
    # renormalize the energies
    incident_energies *= 1.e3
    
    # concatenate the layers and renormalize them, too           
    showers = np.concatenate(layers, axis=1) * 1.e3
            
    save_file = h5py.File(filename, 'w')
    save_file.create_dataset('incident_energies', data=incident_energies)
    save_file.create_dataset('showers', data=showers)
    save_file.close()            
 
def save_data_with_format(
    data: dict,
    filename: str,
    truth_format: TruthFormat,
    dataset_name: str = "showers",
):
    """Save postprocessed generator output matching the truth data format.

    Like :func:`save_data` but respects the original HDF5 schema —
    correct energy key, energy dimensionality, and shower grid shape.

    Parameters
    ----------
    data: dict
        Postprocessed dict with ``"energy"`` (GeV) and ``"layer_*"`` keys.
    filename: str
        Output HDF5 path.
    truth_format: TruthFormat
        Schema metadata describing how the truth file is structured.
    dataset_name: str
        HDF5 dataset name for showers (default ``"showers"``).
    """
    energy, layers = get_energy_and_sorted_layers(data)

    # Convert to MeV
    incident_energy_mev = energy * 1.e3
    showers_mev = np.concatenate(layers, axis=1) * 1.e3

    # Apply truth format
    if truth_format.showers_grid_shape is not None:
        showers_mev = truth_format.unflatten_showers(showers_mev)

    energy_out = truth_format.format_energy(incident_energy_mev)

    with h5py.File(filename, "w") as handle:
        handle.create_dataset(truth_format.energy_key, data=energy_out)
        handle.create_dataset(dataset_name, data=showers_mev)

_NOISE_STEPS = 2 ** 24  # 24-bit uniform grid, the precision of a float32 draw


def add_noise(
    x: torch.Tensor | np.ndarray,
    noise_width: float = 1.0e-6,
    *,
    copy: bool = True,
    draw_dtype: np.dtype | type | None = None,
):
    """Add uniform noise ``U[0, noise_width)`` elementwise.

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Input to perturb.  The return type follows the input type.
    noise_width : float
        Upper bound of the uniform dose.  ``<= 0`` returns ``x`` untouched.
    copy : bool
        Copy ``x`` before adding (default).  ``copy=False`` adds in place and
        returns the same object -- only pass it when the caller owns ``x``.
    draw_dtype : numpy dtype, optional
        Draw the noise directly on a float32 grid (24 bits) instead of the
        legacy float64 stream.  ~35% faster for float32 input, identical
        distribution, but a *different RNG stream*: seeds no longer reproduce
        the legacy draws.  ``None`` (default) keeps the float64 draw.
    """
    if noise_width <= 0.0:
        return x

    if isinstance(x, np.ndarray):
        if copy:
            x = np.copy(x)
        if draw_dtype is None:
            noise = np.random.rand(*x.shape)          # float64, legacy stream
            noise *= noise_width
        else:
            # One uint32 per element instead of one float64 (two uint32):
            # halves the RNG work, then scale in the target precision.
            counts = np.random.randint(0, _NOISE_STEPS, x.shape, dtype=np.uint32)
            noise = counts.astype(np.float32, copy=False)
            noise *= np.float32(noise_width) * np.float32(1.0 / _NOISE_STEPS)
        x += noise
        return x

    # torch tensors stay tensors (the previous np.copy detour returned an array)
    x = x.clone() if copy else x
    return x + torch.rand_like(x) * noise_width

def get_energy_dims(x, c, layer_boundaries, eps=1.e-10):
    """Appends the extra dimensions and the layer energies to the conditions
    The layer energies will always be the last #layers entries, the extra dims will
    be the #layers entries directly after the first entry - the incident energy.
    Inbetween additional features might be appended as further conditions"""
    
    x = np.copy(x)
    c = np.copy(c)
    
    #add_noise = np.random.rand(*x.shape)*1.0e-6
    #x += add_noise

    layer_energies = []

    for layer_start, layer_end in zip(layer_boundaries[:-1], layer_boundaries[1:]):
        
        # Compute total energy of current layer
        layer_energy = np.sum(x[..., layer_start:layer_end], axis=1, keepdims=True)
        
        # Normalize current layer
        x[..., layer_start:layer_end] = x[..., layer_start:layer_end] / (layer_energy + eps)
        
        # Store its energy for later
        layer_energies.append(layer_energy)
        
    layer_energies_np = np.array(layer_energies).T[0]

    # Compute the generalized extra dimensions
    extra_dims = [np.sum(layer_energies_np, axis=1, keepdims=True) / c]

    for layer_index in range(len(layer_boundaries)-2):
        extra_dim = layer_energies_np[..., [layer_index]] / (np.sum(layer_energies_np[..., layer_index:], axis=1, keepdims=True) + eps)
        extra_dims.append(extra_dim)
        
    # Collect all the conditions
    #all_conditions = [c] + extra_dims
    #c = np.concatenate(all_conditions, axis=1)
    extra_dims = np.concatenate(extra_dims, axis=1)

    #extra_dims[:,0] /= 3.5

    return c, extra_dims

def get_energies(x, c, layer_boundaries):
    egs = []
    for layer_start, layer_end in zip(layer_boundaries[:-1], layer_boundaries[1:]):
        layer_energy = np.sum(x[..., layer_start:layer_end], axis=1, keepdims=True)
        egs.append(layer_energy)
    egs = np.array(egs).reshape(len(layer_boundaries)-1, -1)
    egs /= (egs.sum(0)+1.0e-10)
    e_tc = (np.sum(x, axis=1, keepdims=True)/c).T
    return np.insert(egs, 0, e_tc, axis=0).T

def preprocess_wEinc(data, layer_boundaries, eps=1.0e-10):
    "Transform a là CERN VAE"
    energy, layers = get_energy_and_sorted_layers(data)

    x = np.concatenate(layers, axis=1)
    c = energy

    x = normalize_layers_Einc(x, c, layer_boundaries)
    print("shape of inputs:")
    print(x.shape, c.shape)
    return x, c

def preprocess_wenergy(data, layer_boundaries, eps=1.0e-10, rew=1.0):
    "Transform a là CERN VAE"
    energy, layers = get_energy_and_sorted_layers(data)

    x = np.concatenate(layers, axis=1)
    c = energy

    en_scaled = get_energies(x, c, layer_boundaries)

    x = normalize_layers(x, c, layer_boundaries)
    x = np.concatenate((x, en_scaled), axis=1)

    #reweight
    x = x**rew

    print("shape of inputs:")
    print(x.shape, c.shape)
    return x, c

def preprocess(data, layer_boundaries, eps=1.e-10, u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1e10, verbose=False):
    """Transforms the dict 'data' into the ndarray 'x'. Furthermore, the events
    are masked and the extra dims are appended to the incident energies"""
    energy, layers = get_energy_and_sorted_layers(data)

    # Concatenate the layers
    x = np.concatenate(layers, axis=1)
        
    # Remove all no-interaction events (only 0.7%)
    binary_mask = np.sum(x, axis=1) >= 0
    
    c = energy
    c, extra_dims = get_energy_dims(x, c, layer_boundaries, eps)

    binary_mask &= extra_dims[:,0] < u0up_cut
    binary_mask &= extra_dims[:,0] >= u0low_cut
    
    binary_mask &= ((x < dep_cut).prod(-1) != 0)
    
    if verbose:
        print(f"cut on zero energy dep.: #", (np.sum(x, axis=1)>=0).sum())
        print(f"cut on u0 upper {u0up_cut}: #", (extra_dims[:,0] < u0up_cut).sum())
        print(f"cut on u0 lower {u0low_cut}: #", (extra_dims[:,0] >= u0low_cut).sum())
        print(f"dep cut {dep_cut}: #", (x<dep_cut).prod(-1).sum())

    x = x[binary_mask]
    c = c[binary_mask]
    extra_dims = extra_dims[binary_mask]
    if verbose:
        print("final shape of dataset: ", x.shape)

    x = normalize_layers(x, c, layer_boundaries)

    x = np.concatenate((x, extra_dims), axis=1)
   
    #reweight
    #x = x**rew

    return x, c

def postprocess_wEinc(x, c, layer_boundaries, quantiles, threshold=1.0e-4):
    "Reverse the effect of preprocess_wenergy"
    assert len(x) == len(c)
    assert len(x.shape) == 2

    x = np.copy(x)
    c = np.copy(c)

    print("shape of output:")
    print(x.shape, c.shape)
    x[x<quantiles] = 0.0
    x = unnormalize_layers_wEinc(x, c, layer_boundaries)
    
    data = {}
    data["energy"] = c[..., [0]]
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        data[f"layer_{layer_index}"] = x[..., layer_start:layer_end]

    return data


def postprocess_wenergy(x, c, layer_boundaries, quantiles, threshold=1.0e-4, rew=1.0):
    "Reverse the effect of preprocess_wenergy"
    assert len(x) == len(c)
    assert len(x.shape) == 2

    x = np.copy(x)
    c = np.copy(c)

    print("shape of output:")
    print(x.shape, c.shape)
    x[x<quantiles] = 0.0
    x = x**(1/rew)
    x = unnormalize_layers_wenergies(x, c, layer_boundaries)
    

    data = {}
    data["energy"] = c[..., [0]]
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        data[f"layer_{layer_index}"] = x[..., layer_start:layer_end]

    return data

def postprocess(x, c, layer_boundaries, quantiles, threshold=1e-4, rew=1.0):
    """Reverses the effect of the preprocess funtion"""
    
    # Input sanity checks
    assert len(x) == len(c)
    assert len(x.shape) == 2
    assert len(x.shape) == 2
    
    # Makes sure, that the original set is not modified inplace
    x = np.copy(x)
    c = np.copy(c)

    # Set all energies smaller than a threshold to 0. Also prevents negative energies that might occur due to the alpha parameter in
    # the logit preprocessing
    # TODO: Pipe to params
    #x[x < threshold] = 0.
    x[x < quantiles] = 0.0
    
    #reweight
    #x = x**(1/rew)

    x = unnormalize_layers(x, c, layer_boundaries)

    # Create a new dict 'data' for the output
    data = {}
    data["energy"] = c[..., [0]]
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        data[f"layer_{layer_index}"] = x[..., layer_start:layer_end]

    return data

def normalize_layers(x, c, layer_boundaries, eps=1.e-10):
    """Normalizes each layer by its energy"""
    
    # Prevent inplace operations
    x = np.copy(x)
    c = np.copy(c)
    
    # Get the number of layers
    number_of_layers = len(layer_boundaries) - 1

    # Split up the conditions
    incident_energy = c[..., [0]]
    extra_dims = c[..., 1:number_of_layers+1]
    
    # Use the exact layer energies for numerical stability
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        x[..., layer_start:layer_end] = x[..., layer_start:layer_end] / ( np.sum(x[..., layer_start:layer_end], axis=1, keepdims=True) + eps)
        
    return x

def normalize_layers_Einc(x, c, layer_boundaries, eps=1.0e-10):
    
    # Prevent inplace operations
    x = np.copy(x)
    c = np.copy(c)
    
    # Get the number of layers
    number_of_layers = len(layer_boundaries) - 1

    # Split up the conditions
    incident_energy = c[..., [0]]
   
    x /= incident_energy
        
    return x

def unnormalize_layers_wenergies(x, c, layer_boundaries, eps=1.0e-10):
    output = np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64)

    number_of_layers = len(layer_boundaries)-1

    incident_energy = c[..., [0]]
    extra_dims = x[..., -(number_of_layers+1):]
    
    x = x[:, :(-number_of_layers+1)]

    layer_energies = []
    en_tot = np.multiply(incident_energy.flatten(), extra_dims[:,0])
    for i in range(extra_dims.shape[-1]-1):
        ens = extra_dims[:, i+1]*en_tot
        layer_energies.append(ens)

    layer_energies = np.vstack(layer_energies).T
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        output[..., layer_start:layer_end] = x[..., layer_start:layer_end] * layer_energies[..., [layer_index]]  / \
                                             (np.sum(x[..., layer_start:layer_end], axis=1, keepdims=True) + eps)
 
    return output

def unnormalize_layers_wEinc(x, c, layer_boundaries, eps=1.e-10):
    """Reverses the effect of the normalize_layers function"""
    
    # Here we should not use clone, since it might result
    # in a memory leak, if this functions is used on tensors
    # with gradients. Instead, we use a different output tensor
    # to prevent inplace operations.
    output = np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64)
    
    # Get the number of layers
    number_of_layers = len(layer_boundaries) - 1

    # Split up the conditions
    incident_energy = c[..., [0]]
    
    # Normalize each layer and multiply it with its original energy
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        output[..., layer_start:layer_end] = x[..., layer_start:layer_end] * incident_energy
    return output


def unnormalize_layers(x, c, layer_boundaries, eps=1.e-10):
    """Reverses the effect of the normalize_layers function"""
    
    # Here we should not use clone, since it might result
    # in a memory leak, if this functions is used on tensors
    # with gradients. Instead, we use a different output tensor
    # to prevent inplace operations.
    output = np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64)
    
    # Get the number of layers
    number_of_layers = len(layer_boundaries) - 1

    # Split up the conditions
    incident_energy = c[..., [0]]
    extra_dims = x[..., -number_of_layers:]
    extra_dims[:, (-number_of_layers+1):] = np.clip(extra_dims[:, (-number_of_layers+1):], a_min=0., a_max=1.)   #clipping 
    x = x[:, :-number_of_layers]
    
    layer_energies = []
    en_tot = np.multiply(incident_energy.flatten(), extra_dims[:,0])
    cum_sum = np.zeros_like(en_tot, dtype=np.float64)
    for i in range(extra_dims.shape[-1]-1):
        ens = (en_tot - cum_sum)*extra_dims[:,i+1]
        layer_energies.append(ens)
        cum_sum += ens

    layer_energies.append((en_tot - cum_sum))
    layer_energies = np.vstack(layer_energies).T
    # Normalize each layer and multiply it with its original energy
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        output[..., layer_start:layer_end] = x[..., layer_start:layer_end] * layer_energies[..., [layer_index]]  / \
                                             (np.sum(x[..., layer_start:layer_end], axis=1, keepdims=True) + eps)
    return output

def save_hlf(hlf, filename):
    """ Saves high-level features class to file """
    print("Saving file with high-level features.")
    #filename = os.path.splitext(os.path.basename(ref_name))[0] + '.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(hlf, file)
    print("Saving file with high-level features DONE.")

def get_hlf(shower, particle_type, layer_boundaries, threshold=1.e-4):
    "returns a hlf class needed for plotting"
    #x = x.cpu()
    #c = c.cpu()
    
    hlf = HLF.HighLevelFeatures(particle_type,
                                f"/remote/gpu06/favaro/calo_inn/datasets/calo_challenge/binning_dataset_1_{particle_type}s.xml")
    
    # like in save function:
    # extract the needed data
    incident_energies, layers = get_energy_and_sorted_layers(shower)
    
    # renormalize the energies
    #incident_energies = shower['incident_energies']
    incident_energies *= 1.e3
    
    # concatenate the layers and renormalize them, too 
    #showers = shower['showers']
    showers = np.concatenate(layers, axis=1) * 1.e3
    
    
    hlf.CalculateFeatures(showers)
    hlf.Einc = incident_energies
    hlf.showers = showers
    
    return hlf

def save_hlf(hlf, filename):
    """ Saves high-level features class to file """
    print("Saving file with high-level features.")
    #filename = os.path.splitext(os.path.basename(ref_name))[0] + '.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(hlf, file)
    print("Saving file with high-level features DONE.")

def generate_Einc_ds1(energy=None, sample_multiplier=1000):
    ret = np.logspace(8, 18, 11, base=2)
    ret = np.tile(ret, 10)
    ret = np.array([
        *ret,
        *np.tile(2.0 ** 19, 5),
        *np.tile(2.0 ** 20, 3),
        *np.tile(2.0 ** 21, 2),
        *np.tile(2.0 ** 22, 1),
    ])
    ret = np.tile(ret, sample_multiplier)
    if energy is not None:
        ret = ret[ret == energy]
    np.random.shuffle(ret)
    return ret


def _build_data_dict(
    showers: np.ndarray,
    energies: np.ndarray,
    layer_boundaries: np.ndarray,
) -> dict:
    """Build data dict from raw arrays, matching legacy load_data."""
    E_SCALE = 1.e3  # Python float → float64, matches legacy
    data = {}
    data["energy"] = energies.reshape(-1, 1) / E_SCALE
    for li, (ls, le) in enumerate(
        zip(layer_boundaries[:-1], layer_boundaries[1:])
    ):
        data[f"layer_{li}"] = showers[..., ls:le] / E_SCALE
    return data