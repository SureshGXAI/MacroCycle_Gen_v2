"""
Shared chemistry helpers used by train.py, generate.py, evaluate.py, and
optimize.py — kept in one place so representation handling (SMILES vs
SELFIES) and property/SA-score computation aren't duplicated four times.
"""
import os
import urllib.request

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

_SASCORER_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
_sascorer_module = None  # lazy-loaded singleton


def decode_sequence(seq, representation):
    """
    Convert a model-generated raw sequence string back into a SMILES string.
    For 'smiles' representation this is the identity function; for 'selfies'
    it invokes the SELFIES decoder (which, notably, can only fail if `seq`
    itself is not well-formed SELFIES — a *valid* SELFIES string always
    decodes to a valid molecule by construction).
    """
    if not seq:
        return None
    if representation == "smiles":
        return seq
    elif representation == "selfies":
        import selfies as sf
        try:
            return sf.decoder(seq)
        except Exception:
            return None
    else:
        raise ValueError(f"Unknown representation: {representation}")


def mol_from_sequence(seq, representation):
    smi = decode_sequence(seq, representation)
    if not smi:
        return None
    return Chem.MolFromSmiles(smi)


def _download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def get_sascorer():
    """
    Lazily download + import RDKit's standard contrib Synthetic Accessibility
    scorer (Ertl & Schuffenhauer, J. Cheminform 2009). This script isn't
    bundled with the `rdkit` pip package, so it's fetched once from the
    RDKit GitHub repo and cached locally. Requires internet on first call
    only; returns None (with a printed warning) if unavailable, so callers
    should treat SA score as optional/best-effort.
    """
    global _sascorer_module
    if _sascorer_module is not None:
        return _sascorer_module

    py_path = os.path.join(_SASCORER_CACHE_DIR, "sascorer.py")
    data_path = os.path.join(_SASCORER_CACHE_DIR, "fpscores.pkl.gz")
    base = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score"

    try:
        if not os.path.exists(py_path):
            _download(f"{base}/sascorer.py", py_path)
        if not os.path.exists(data_path):
            _download(f"{base}/fpscores.pkl.gz", data_path)

        import sys
        if _SASCORER_CACHE_DIR not in sys.path:
            sys.path.insert(0, _SASCORER_CACHE_DIR)
        # sascorer.py loads fpscores.pkl.gz relative to its own __file__, which
        # works automatically since we downloaded both into the same directory.
        import sascorer  # noqa: E402
        _sascorer_module = sascorer
        return sascorer
    except Exception as e:
        print(f"[chem_utils] Could not load SA scorer ({e}). "
              f"SA score will be reported as None. Requires internet access "
              f"on first call to download from the RDKit GitHub repo.")
        return None


def sa_score(mol):
    """Synthetic accessibility score, ~1 (easy) to ~10 (hard). None if unavailable."""
    scorer = get_sascorer()
    if scorer is None or mol is None:
        return None
    try:
        return scorer.calculateScore(mol)
    except Exception:
        return None


def compute_properties(mol):
    """Full property dict for a valid RDKit Mol, matching the dataset's own columns
    where possible (MW/LogP/HBA/HBD/PSA/RotB) plus QED, SA score, and ring stats."""
    if mol is None:
        return {}
    props = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "PSA": Descriptors.TPSA(mol),
        "RotB": Descriptors.NumRotatableBonds(mol),
        "QED": QED.qed(mol),
        "SA": sa_score(mol),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
    }
    return props


# ---------------------------------------------------------------------------
# Structural helpers used by metrics.py (fingerprints, scaffolds, ring size).
# ---------------------------------------------------------------------------
_morgan_gen = None


def _get_morgan_generator(radius=2, n_bits=2048):
    """
    Morgan/ECFP4 fingerprint generator. Newer RDKit deprecates
    AllChem.GetMorganFingerprintAsBitVect in favour of rdFingerprintGenerator;
    try the new API first and fall back so this works on either version.
    """
    global _morgan_gen
    if _morgan_gen is not None:
        return _morgan_gen
    try:
        from rdkit.Chem import rdFingerprintGenerator
        _morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    except Exception:
        _morgan_gen = None  # signals "use the legacy path"
    return _morgan_gen


def morgan_fp(mol, radius=2, n_bits=2048):
    """ECFP4 bit vector for one Mol, or None."""
    if mol is None:
        return None
    gen = _get_morgan_generator(radius, n_bits)
    if gen is not None:
        return gen.GetFingerprint(mol)
    from rdkit.Chem import AllChem
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def morgan_fps(smiles_list, radius=2, n_bits=2048):
    """List of ECFP4 bit vectors, silently skipping unparseable SMILES."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        fp = morgan_fp(mol, radius, n_bits)
        if fp is not None:
            fps.append(fp)
    return fps


def morgan_fp_array(smiles_list, radius=2, n_bits=2048):
    """
    (N, n_bits) numpy array of ECFP4 bits for sklearn, plus the indices of the
    SMILES that actually parsed (so caller labels stay aligned with rows).
    """
    import numpy as np
    from rdkit import DataStructs

    rows, keep = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if smi else None
        fp = morgan_fp(mol, radius, n_bits)
        if fp is None:
            continue
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
        keep.append(i)
    if not rows:
        return np.zeros((0, n_bits), dtype=np.int8), []
    return np.vstack(rows), keep


def murcko_scaffold(smiles):
    """Bemis-Murcko scaffold SMILES, or None."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        return None


def max_ring_size(smiles):
    """
    Size of the largest ring, used for the macrocycle check (>=12 atoms).

    Caveat: RDKit's default ring perception returns the SSSR, in which a large
    ring fused to smaller ones can be decomposed into the smaller ones, so this
    can UNDERCOUNT for bridged/fused systems. That's why metrics.py compares the
    generated macrocycle rate against the same measurement on the real training
    set rather than against a nominal 100%.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return float("nan")
    rings = mol.GetRingInfo().AtomRings()
    return max((len(r) for r in rings), default=0)


def brics_fragments(smiles):
    """BRICS fragment SMILES for one molecule (empty list on failure)."""
    from rdkit.Chem import BRICS
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return []
    try:
        return list(BRICS.BRICSDecompose(mol))
    except Exception:
        return []


def validity_uniqueness_novelty(sequences, representation, train_smiles_set):
    """
    Core generative-model metrics, representation-aware.
    Returns (valid_smiles_list, validity_rate, uniqueness_rate, novelty_rate).
    """
    valid_smiles = []
    for seq in sequences:
        mol = mol_from_sequence(seq, representation)
        if mol is not None:
            valid_smiles.append(Chem.MolToSmiles(mol))

    n = len(sequences)
    validity = len(valid_smiles) / n if n else 0.0
    uniqueness = len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0
    novel = [s for s in valid_smiles if s not in train_smiles_set]
    novelty = len(novel) / len(valid_smiles) if valid_smiles else 0.0
    return valid_smiles, validity, uniqueness, novelty
