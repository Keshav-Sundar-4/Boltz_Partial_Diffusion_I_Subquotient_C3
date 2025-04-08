import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional
import math
import random
import yaml

import click
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.parse.yaml import get_radius
from boltz.data.parse.yaml import get_symmetry_type

from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1
from boltz.data.module import symmetry_awareness as symmetry


CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = (
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
)



@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 0.8
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True



@rank_zero_only
def download(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(MODEL_URL, str(model))  # noqa: S310


def check_inputs(
    data: Path,
    outdir: Path,
    override: bool = False,
) -> list[Path]:
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        filtered_data = []
        for d in data:
            if d.suffix in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                filtered_data.append(d)
            elif d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

        data = filtered_data
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()}

    # Remove them from the input data
    if existing and not override:
        data = [d for d in data if d.stem not in existing]
        num_skipped = len(existing) - len(data)
        msg = (
            f"Found some existing predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing predictions, will override."
        click.echo(msg)

    return data


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.

    """
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))

def create_atoms_from_pdb(pdb_structure) -> "np.ndarray":
    """
    Create a structured and contiguous atom array from the parsed PDB structure,
    preserving unique atom names so the mmCIF writer won't treat them as duplicates.

    Parameters
    ----------
    pdb_structure : dict
        Dictionary returned by parse_pdb(), containing keys:
          - "atoms": a dict with arrays for 'serial', 'name', 'resname', 'chain_id',
                     'res_seq', 'coords' (shape (N,3)), and 'element'.
          - "chains": a dict mapping chain_id -> list of atom indices.
          - "residues": a dict mapping (chain_id, res_seq) -> list of atom indices.

    Returns
    -------
    np.ndarray
        A NumPy structured array with the following dtype:
          [
              ('name',    np.dtype("4i1")),
              ('element', np.dtype("i1")),
              ('charge',  np.dtype("i1")),
              ('coords',  np.dtype("3f4")),
              ('conformer', np.dtype("3f4")),
              ('is_present', np.dtype("?")),
              ('chirality', np.dtype("i1")),
          ]
        This ensures each atom has a unique 'name' field (ASCII-coded), which mmCIF
        interprets as the atom_id, preventing duplicates.
    """
    import numpy as np

    # Helper: convert a string to a fixed-length ASCII array of shape (length,)
    def to_ascii_array(s: str, length: int) -> np.ndarray:
        arr = np.zeros(length, dtype=np.int8)
        for i, c in enumerate(s[:length]):
            arr[i] = ord(c)
        return arr

    atoms_dict = pdb_structure["atoms"]
    N = len(atoms_dict["serial"])

    # Prepare the final structured dtype:
    #   - 'name' is 4 ASCII bytes
    #   - 'element' is 1 byte, so we store only the first character
    #   - others remain zero or default
    atom_dtype = np.dtype([
        ('name',      np.dtype("4i1")),
        ('element',   np.dtype("i1")),    # single character
        ('charge',    np.dtype("i1")),
        ('coords',    np.dtype("3f4")),
        ('conformer', np.dtype("3f4")),
        ('is_present', np.dtype("?")),
        ('chirality', np.dtype("i1")),
    ])

    # Allocate the structured array
    arr = np.empty(N, dtype=atom_dtype)

    # Convert and store coords
    coords = np.array(atoms_dict["coords"], dtype=np.float32)

    # Fill fields
    for i in range(N):
        # Convert 'name' to a 4-byte ASCII array
        atom_name = atoms_dict["name"][i]
        arr["name"][i] = to_ascii_array(atom_name, 4)

        # Convert 'element' to single ASCII byte (first character)
        element_str = atoms_dict["element"][i]
        if len(element_str) > 0:
            arr["element"][i] = ord(element_str[0])
        else:
            arr["element"][i] = 0  # unknown => 0

    arr["charge"] = 0
    arr["coords"] = coords
    arr["conformer"] = coords.copy()
    arr["is_present"] = True
    arr["chirality"] = 0

    # Force contiguous memory for each field.
    new_arr = np.empty(arr.shape, dtype=atom_dtype)
    for field in arr.dtype.names:
        new_arr[field] = np.ascontiguousarray(arr[field])
    new_arr = new_arr.copy(order="C")

    return new_arr



def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 4096,
    use_msa_server: bool = False,
) -> tuple[Optional["BoltzProcessedInput"], float, float, str, int, dict]:
    """Process the input data and output directory."""
    click.echo("Processing input data.")
    existing_records = None

    monomer_molecular_weight = 0.0
    radius = 0.0
    symmetry_type = "NA"
    n = 0
    chain_symmetry_groups = {}

    aa_weights = {
        'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
        'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
        'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
        'M': 131.04049, 'F': 147.06841, 'P': 97.05276, 'S': 87.03203,
        'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841
    }

    manifest_path = out_dir / "processed" / "manifest.json"
    if manifest_path.exists():
        click.echo(f"Found a manifest file at output directory: {out_dir}")
        manifest: Manifest = Manifest.load(manifest_path)
        input_ids = [d.stem for d in data]
        existing_records, processed_ids = zip(
            *[
                (record, record.id)
                for record in manifest.records
                if record.id in input_ids
            ]
        )
        if isinstance(existing_records, tuple):
            existing_records = list(existing_records)
        missing = len(input_ids) - len(processed_ids)
        if not missing:
            click.echo("All examples in data are processed. Updating the manifest")
            updated_manifest = Manifest(existing_records)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")
            if existing_records:
                first_record_path = [d for d in data if d.stem == existing_records[0].id][0]
                if first_record_path.suffix in (".yml", ".yaml"):
                    with first_record_path.open("r") as file:
                        yaml_data = yaml.safe_load(file)
                    radius = get_radius(yaml_data)
                    symmetry_type = get_symmetry_type(yaml_data)
                else:
                    radius = 0.0
                    symmetry_type = "NA"
                first_record = existing_records[0]
                n = len(first_record.chains)
                if symmetry_type != "NA":
                    chain_symmetry_groups = {f'{symmetry_type} Symmetry Test': [list(range(n))]}
                else:
                    chain_symmetry_groups = {}
                monomer_molecular_weight = 0
                mw_string = ""
                if 'sequences' in yaml_data:
                    for seq_data in yaml_data['sequences']:
                        if seq_data['protein']['id'] == 'A':
                            mw_string = seq_data['protein']['sequence']
                            break
                if mw_string:
                    monomer_molecular_weight = sum(aa_weights.get(aa, 0) for aa in mw_string)
                else:
                    print("Warning: Sequence for protein A not found in existing record.")
                return None, monomer_molecular_weight, radius, symmetry_type, n, chain_symmetry_groups
            else:
                return None, 0.0, 0.0, "NA", 0, {}

        click.echo(f"{missing} missing ids. Preprocessing these ids")
        missing_ids = list(set(input_ids).difference(set(processed_ids)))
        data = [d for d in data if d.stem in missing_ids]
        assert len(data) == len(missing_ids)

    msa_dir = out_dir / "msa"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)

    if existing_records is not None:
        click.echo(f"Found {len(existing_records)} records. Adding them to records")

    records: list[Record] = existing_records if existing_records is not None else []
    for path in tqdm(data):
        try:
            if path.suffix in (".yml", ".yaml"):
                target, yaml_data = parse_yaml(path, ccd)

                # Calculate Monomer Molecular Weight (Chain A)
                mw_string = ""
                for seq_data in yaml_data['sequences']:
                    if seq_data['protein']['id'] == 'A':
                        mw_string = seq_data['protein']['sequence']
                        break
                if mw_string:
                    monomer_molecular_weight = sum(aa_weights.get(aa, 0) for aa in mw_string)
                else:
                    print("Warning: Sequence for protein A not found.")

                radius = get_radius(yaml_data)
                symmetry_type = get_symmetry_type(yaml_data)
                n = len(target.record.chains)
                chain_symmetry_groups = {f'{symmetry_type} Symmetry Test': [list(range(n))]} if symmetry_type != "NA" else {}

            elif path.is_dir():
                msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {path.suffix}, "
                    "please provide a .yaml file."
                )
                raise RuntimeError(msg)

            # (The remainder of the function remains unchanged, including MSA processing.)
            prot_id = const.chain_type_ids["PROTEIN"]
            to_generate = {}
            for chain in target.record.chains:
                if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                    entity_id = chain.entity_id
                    msa_id = f"{target.record.id}_{entity_id}"
                    to_generate[msa_id] = target.sequences[entity_id]
                    chain.msa_id = msa_dir / f"{msa_id}.csv"
                elif chain.msa_id == 0:
                    chain.msa_id = -1

            if to_generate and not use_msa_server:
                msg = "Missing MSA's in input and --use_msa_server flag not set."
                raise RuntimeError(msg)

            if to_generate:
                msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
                click.echo(msg)
                compute_msa(
                    data=to_generate,
                    target_id=target.record.id,
                    msa_dir=msa_dir,
                    msa_server_url=msa_server_url,
                    msa_pairing_strategy=msa_pairing_strategy,
                )

            msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
            msa_id_map = {}
            for msa_idx, msa_id in enumerate(msas):
                msa_path = Path(msa_id)
                if not msa_path.exists():
                    msg = f"MSA file {msa_path} not found."
                    raise FileNotFoundError(msg)
                processed = processed_msa_dir / f"{target.record.id}_{msa_idx}.npz"
                msa_id_map[msa_id] = f"{target.record.id}_{msa_idx}"
                if not processed.exists():
                    if msa_path.suffix == ".a3m":
                        msa = parse_a3m(
                            msa_path,
                            taxonomy=None,
                            max_seqs=max_msa_seqs,
                        )
                    elif msa_path.suffix == ".csv":
                        msa = parse_csv(msa_path, max_seqs=max_msa_seqs)
                    else:
                        msg = f"MSA file {msa_path} not supported, only a3m or csv."
                        raise RuntimeError(msg)
                    msa.dump(processed)

            for c in target.record.chains:
                if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                    c.msa_id = msa_id_map[c.msa_id]

            records.append(target.record)
            struct_path = structure_dir / f"{target.record.id}.npz"
            target.structure.dump(struct_path)

        except Exception as e:
            if len(data) > 1:
                print(f"Failed to process {path}. Skipping. Error: {e}.")
            else:
                raise e

    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")
    processed_input = BoltzProcessedInput(manifest, structure_dir, msa_dir)
    return processed_input, monomer_molecular_weight, radius, symmetry_type, n, chain_symmetry_groups



@click.group()
def cli() -> None:
    """Boltz1."""
    return


@click.group()
def cli() -> None:
    """Boltz1."""
    return


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--pdb",  # Changed to option
    required=False,
    type=click.Path(exists=True),
    help="Path to an optional PDB file for initial coordinates.",
)
@click.option(
    "--mmcif",  # Added mmcif_file option
    required=False,
    type=click.Path(exists=True),
    help="Path to an optional mmCIF file for initial coordinates.",
)
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help="The directory where to download the data and model. Default is ~/.boltz.",
    default="~/.boltz",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--step_scale",
    type=float,
    help="The step size is related to the temperature at which the diffusion process samples the distribution. The lower the higher the diversity among samples (recommended between 1 and 2). Default is 1.638.",
    default=1.638,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help="Pairing strategy to use. Used only if --use_msa_server is set. Options are 'greedy' and 'complete'",
    default="greedy",
)

def predict(
    data: str,
    pdb: Optional[str],  # Keep this
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float = 1.638,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    mmcif: Optional[str] = None,  # Add this for mmCIF
) -> None:
    """Run predictions with Boltz-1."""
    import numpy as np

    # ... (rest of the function setup remains the same) ...
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache)

    # Validate inputs
    data = check_inputs(data, out_dir, override)
    if not data:
        click.echo("No predictions to run, exiting.")
        return

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy()
        if len(data) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions."
            )
            raise ValueError(msg)

    msg = f"Running predictions for {len(data)} structure"
    msg += "s" if len(data) > 1 else ""
    click.echo(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    processed, molecular_weight, radius, symmetry_type, n1, chain_symmetry_groups = process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )


    if symmetry_type.startswith("C") or symmetry_type.startswith("D"):
        n = int(symmetry_type.split("_")[1])
    elif symmetry_type.startswith("T"):
        n = 12  # Set n to 12 for T symmetry (Tetrahedral)
    elif symmetry_type.startswith("O"):
        n = 24  # Set n to 24 for O symmetry (Octahedral)
    elif symmetry_type.startswith("I"):
        n = 60  # Set n to 60 for I symmetry (Icosahedral)
    else:
        n = 0  # Default value if symmetry type is not recognized

    if radius == 0:
        radius = symmetry.calculate_radius_from_mw(molecular_weight, n, symmetry_type)

    print("Molecular weight is: ", molecular_weight)
    print("Radius is: ", radius)
    print("Symmetry Type is ", symmetry_type)
    print("N is ", n)
    print("n1 is ", n1)
    print("Chain_symmetry_groups is", chain_symmetry_groups)

    # Load processed data
    processed_dir = out_dir / "processed"


    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )

    # Load model
    if checkpoint is None:
        checkpoint = cache / "boltz1_conf.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale
    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        symmetry_type=symmetry_type,
        chain_symmetry_groups=chain_symmetry_groups,
        diffusion_process_args=asdict(diffusion_params),
        radius=radius,
        ema=False,
    )
    model_module.eval()


    # Handle PDB *OR* mmCIF, but not both.
    if pdb and mmcif:
        raise ValueError("Cannot specify both --pdb_file and --mmcif_file.")

    if pdb:
        pdb = Path(pdb)
        if pdb.exists():
            pdb_struct = parse_pdb(str(pdb))
            pdb_atoms = create_atoms_from_pdb(pdb_struct)
            input_coords = torch.tensor(pdb_atoms["coords"], dtype=torch.float32)
            input_coords = input_coords.unsqueeze(0)
            model_module.input_coords = input_coords  # Attach to the model

    elif mmcif:  # Handle mmCIF separately
        mmcif_file_path = Path(mmcif)
        if mmcif_file_path.exists():
            mmcif_struct = parse_mmcif(str(mmcif_file_path))
            if mmcif_struct is not None:
                mmcif_atoms = create_atoms_from_mmcif(mmcif_struct)
                input_coords = torch.tensor(mmcif_atoms["coords"], dtype=torch.float32)
                input_coords = input_coords.unsqueeze(0)
                model_module.input_coords = input_coords

    # (rest of the function remains the same)

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
    )


    from pytorch_lightning import Trainer
    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )


    # Compute predictions
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )




def parse_pdb(pdb_path: str) -> dict:
    """
    Parse a PDB file and return a dictionary with atom information,
    preserving the original atom names and element columns as found in
    the file (i.e. no mmCIF normalization is applied).

    The returned dictionary has two keys:
      - "atoms": a dict with NumPy arrays for fields:
           "serial", "name", "resname", "chain", "resseq", "icode",
           "coords" (an array of shape (N, 3)), "occupancy", "temp_factor",
           "element", and "charge".
      - "residues": a dictionary mapping each residue (by (chain, resseq, icode))
           to a list of atom indices.
      - "chains": a dictionary mapping each chain to a list of atom indices.
    """
    import numpy as np

    atoms = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Parse fields exactly by column positions:
                try:
                    serial = int(line[6:11])
                except ValueError:
                    continue  # skip lines that don't parse correctly
                # Preserve the original atom name (columns 13-16)
                name = line[12:16]
                # Alternate location (column 17)
                altloc = line[16] if line[16] != " " else ""
                resname = line[17:20]
                chain = line[21]
                try:
                    resseq = int(line[22:26])
                except ValueError:
                    resseq = 0
                icode = line[26] if line[26] != " " else ""
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                occ_str = line[54:60].strip()
                occupancy = float(occ_str) if occ_str != "" else 0.0
                temp_str = line[60:66].strip()
                temp_factor = float(temp_str) if temp_str != "" else 0.0
                element = line[76:78].strip()  # element field as provided in the file
                charge = line[78:80].strip()

                atom = {
                    "serial": serial,
                    "name": name,
                    "altloc": altloc,
                    "resname": resname,
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                    "x": x,
                    "y": y,
                    "z": z,
                    "occupancy": occupancy,
                    "temp_factor": temp_factor,
                    "element": element,
                    "charge": charge
                }
                atoms.append(atom)

    # Build NumPy arrays for each field.
    serials = np.array([atom["serial"] for atom in atoms], dtype=np.int32)
    names = np.array([atom["name"] for atom in atoms])
    resnames = np.array([atom["resname"] for atom in atoms])
    chains = np.array([atom["chain"] for atom in atoms])
    resseqs = np.array([atom["resseq"] for atom in atoms], dtype=np.int32)
    icode_arr = np.array([atom["icode"] for atom in atoms])
    coords = np.array([[atom["x"], atom["y"], atom["z"]] for atom in atoms], dtype=np.float32)
    occupancies = np.array([atom["occupancy"] for atom in atoms], dtype=np.float32)
    temp_factors = np.array([atom["temp_factor"] for atom in atoms], dtype=np.float32)
    elements = np.array([atom["element"] for atom in atoms])
    charges = np.array([atom["charge"] for atom in atoms])

    # Build residue and chain indices.
    residue_dict = {}
    chain_dict = {}
    for i, atom in enumerate(atoms):
        res_key = (atom["chain"], atom["resseq"], atom["icode"])
        residue_dict.setdefault(res_key, []).append(i)
        chain_dict.setdefault(atom["chain"], []).append(i)

    return {
        "atoms": {
            "serial": serials,
            "name": names,
            "resname": resnames,
            "chain": chains,
            "resseq": resseqs,
            "icode": icode_arr,
            "coords": coords,
            "occupancy": occupancies,
            "temp_factor": temp_factors,
            "element": elements,
            "charge": charges,
        },
        "residues": residue_dict,
        "chains": chain_dict
    }


def create_atoms_from_pdb(pdb_structure) -> "np.ndarray":
    """
    Create a structured and contiguous atom array from the parsed PDB structure,
    preserving unique atom names so the mmCIF writer won't treat them as duplicates.
    Modified to use a fixed-length string for the atom name.
    
    Parameters
    ----------
    pdb_structure : dict
        Dictionary returned by parse_pdb(), containing keys such as "atoms" and "resseq".
    
    Returns
    -------
    np.ndarray
        A NumPy structured array with fields:
          - 'name': fixed-length string of 4 characters,
          - 'element': 1-byte integer (storing the ASCII code of the first character),
          - 'charge', 'coords', 'conformer', 'is_present', 'chirality',
          - 'chain': a 1-character string,
          - 'res_seq': an integer residue sequence.
    """
    import numpy as np

    atoms_dict = pdb_structure["atoms"]
    N = len(atoms_dict["serial"])

    # Use a fixed-length Unicode string for the atom name (4 characters).
    atom_dtype = np.dtype([
        ('name',      'U4'),         # atom name as a 4-character string
        ('element',   np.dtype("i1")), # store first character of element as integer
        ('charge',    np.dtype("i1")),
        ('coords',    np.dtype("3f4")),
        ('conformer', np.dtype("3f4")),
        ('is_present',np.dtype("?")),
        ('chirality', np.dtype("i1")),
        ('chain',     'U1'),         # chain identifier (1-character string)
        ('res_seq',   np.int32),     # residue sequence number
    ])

    arr = np.empty(N, dtype=atom_dtype)
    coords = np.array(atoms_dict["coords"], dtype=np.float32)

    for i in range(N):
        # Get the atom name and ensure it is exactly 4 characters (pad if necessary)
        atom_name = atoms_dict["name"][i]
        fixed_name = (atom_name + "    ")[:4]  # pad with spaces then slice
        arr["name"][i] = fixed_name

        element_str = atoms_dict["element"][i]
        if len(element_str) > 0:
            arr["element"][i] = ord(element_str[0])
        else:
            arr["element"][i] = 0  # unknown element

        # Set chain id and residue sequence number
        arr["chain"][i] = atoms_dict["chain"][i].strip() if atoms_dict["chain"][i] else " "
        arr["res_seq"][i] = atoms_dict["resseq"][i]

    # Fill other fields
    arr["charge"] = 0
    arr["coords"] = coords
    arr["conformer"] = coords.copy()
    arr["is_present"] = True
    arr["chirality"] = 0

    # Ensure contiguous memory for each field
    new_arr = np.empty(arr.shape, dtype=atom_dtype)
    for field in arr.dtype.names:
        new_arr[field] = np.ascontiguousarray(arr[field])
    new_arr = new_arr.copy(order="C")
    return new_arr


def parse_mmcif(mmcif_path: str) -> Optional[dict]:
    """
    Parse an mmCIF file and return a dictionary with atom information.
    Handles missing values in essential fields gracefully.  Returns None
    if critical parsing errors occur.  This version does NOT use external
    dependencies (like mmcif-pdbx). It directly parses the ATOM records.
    """
    import numpy as np

    try:
        with open(mmcif_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error opening or reading mmCIF file: {e}")
        return None

    atoms = []
    for line in lines:
        if line.startswith("ATOM"):
            try:
                # Split the line by spaces, handling multiple spaces correctly.
                parts = line.split()

                # Basic validation:  mmCIF ATOM records should have at least 11 fields
                if len(parts) < 11:
                  continue

                serial = int(parts[1])
                name = parts[2]
                altloc = parts[4] if parts[4] != "." else ""
                resname = parts[5]
                chain = parts[6] if parts[6] != "." else "A" #default chain id of A
                
                # Handle the case where resseq might be '?'
                try:
                    resseq = int(parts[8])
                except ValueError:
                    resseq = -1  # Assign a default value, e.g., -1 for missing resseq

                
                icode =  ""  # Not directly in your example, can be extracted, but often empty
                x = float(parts[10])
                y = float(parts[11])
                z = float(parts[12])
                #you may have to modify these indexes if more values are present

                # Handle optional occupancy and temp_factor (if present, adjust indices)
                occupancy = 1.0  # Default
                temp_factor = 0.0 # Default
                element = ""
                charge = ""
                
                if len(parts) >= 14:
                  try:
                      occupancy = float(parts[13])
                  except (ValueError, IndexError):
                      pass  # Keep default if parsing fails.
                if len(parts) >= 15:
                    try:
                        temp_factor = float(parts[14])
                    except (ValueError, IndexError):
                        pass
                if len(parts) >= 17:
                    element = parts[16]

                if len(parts) >= 18:
                    charge = parts[17]


                atom = {
                    "serial": serial,
                    "name": name,
                    "altloc": altloc,
                    "resname": resname,
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                    "x": x,
                    "y": y,
                    "z": z,
                    "occupancy": occupancy,
                    "temp_factor": temp_factor,
                    "element": element,
                    "charge": charge
                }
                atoms.append(atom)

            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse ATOM line in mmCIF: {line.strip()} - {e}")
                continue  # Skip to the next line

    if not atoms:
        print(f"Warning: No ATOM records found in mmCIF file: {mmcif_path}")
        return None

    # Convert to NumPy arrays (similar to the previous version, but from our parsed data).
    atom_data = {}
    atom_data["serial"] = np.array([atom["serial"] for atom in atoms], dtype=np.int32)
    atom_data["name"] = np.array([atom["name"] for atom in atoms])
    atom_data["altloc"] = np.array([atom["altloc"] for atom in atoms])
    atom_data["resname"] = np.array([atom["resname"] for atom in atoms])
    atom_data["chain"] = np.array([atom["chain"] for atom in atoms])
    atom_data["resseq"] = np.array([atom["resseq"] for atom in atoms], dtype=np.int32)
    atom_data["icode"] = np.array([atom["icode"] for atom in atoms])
    atom_data["x"] = np.array([atom["x"] for atom in atoms], dtype=np.float32)
    atom_data["y"] = np.array([atom["y"] for atom in atoms], dtype=np.float32)
    atom_data["z"] = np.array([atom["z"] for atom in atoms], dtype=np.float32)
    atom_data["occupancy"] = np.array([atom["occupancy"] for atom in atoms], dtype=np.float32)
    atom_data["temp_factor"] = np.array([atom["temp_factor"] for atom in atoms], dtype=np.float32)
    atom_data["element"] = np.array([atom["element"] for atom in atoms])
    atom_data["charge"] = np.array([atom["charge"] for atom in atoms])

    coords = np.column_stack([atom_data["x"], atom_data["y"], atom_data["z"]]).astype(np.float32)
    atom_data["coords"] = coords

    # Build residue and chain indices.
    residue_dict = {}
    chain_dict = {}
    for i, atom in enumerate(atoms):
        res_key = (atom["chain"], atom["resseq"], atom["icode"])  # Include icode
        residue_dict.setdefault(res_key, []).append(i)
        chain_dict.setdefault(atom["chain"], []).append(i)

    return {
        "atoms": atom_data,
        "residues": residue_dict,
        "chains": chain_dict,
    }




def create_atoms_from_mmcif(mmcif_structure: dict) -> "np.ndarray":
    """
    Create a structured atom array from the parsed mmCIF structure.
    This function is similar to create_atoms_from_pdb, but adapted for mmCIF.
    """
    import numpy as np

    atoms_dict = mmcif_structure["atoms"]
    N = len(atoms_dict["serial"])

    atom_dtype = np.dtype([
        ('name',      'U4'),  # atom name (label_atom_id)
        ('element',   np.dtype("i1")), # element (type_symbol)
        ('charge',    np.dtype("i1")),
        ('coords',    np.dtype("3f4")),
        ('conformer', np.dtype("3f4")),
        ('is_present',np.dtype("?")),
        ('chirality', np.dtype("i1")),
        ('chain',     'U1'),   # chain (label_asym_id)
        ('res_seq',   np.int32),  # residue sequence (label_seq_id)
    ])

    arr = np.empty(N, dtype=atom_dtype)
    coords = np.array(atoms_dict["coords"], dtype=np.float32)

    for i in range(N):
        atom_name = atoms_dict["name"][i]
        arr["name"][i] = (atom_name + "    ")[:4]

        element_str = atoms_dict["element"][i]
        if len(element_str) > 0:
            arr["element"][i] = ord(element_str[0])
        else:
            arr["element"][i] = 0  # unknown

        arr["chain"][i] = atoms_dict["chain"][i].strip()
        arr["res_seq"][i] = atoms_dict["resseq"][i]

    arr["charge"] = 0
    arr["coords"] = coords
    arr["conformer"] = coords.copy()
    arr["is_present"] = True
    arr["chirality"] = 0

    new_arr = np.empty(arr.shape, dtype=atom_dtype)
    for field in arr.dtype.names:
        new_arr[field] = np.ascontiguousarray(arr[field])
    new_arr = new_arr.copy(order="C")
    return new_arr




if __name__ == "__main__":
    cli()
