#!/usr/bin/env python3
"""
This script post-processes an mmCIF structure file. It specifically uses
**Chain 'A'** from the first model as the reference.
It first generates a C3 symmetric trimer (the ASU or "triangle") using this
Chain 'A'. Then, it takes this trimer ASU and generates the full complex
by applying additional symmetry rotations specified by the --group argument.
Any other chains in the input file are ignored.

It supports symmetry groups for the second step:
  - Cyclic: C_n (e.g. C_6) -> N ops
  - Dihedral: D_n (e.g. D_12) -> 2N ops
  - Tetrahedral: T -> 12 ops
  - Octahedral: O -> 24 ops
  - Icosahedral: I (uses I/C3 pseudoquotient) -> 20 ops

Final complex has 3 * (order of group) chains.
"""

import argparse
import numpy as np
import math
import sys
import copy
from itertools import product

# Biopython modules
from Bio.PDB import MMCIFParser, MMCIFIO, Structure, Model, Chain
from Bio.PDB.PDBExceptions import PDBConstructionException  # Import specific exception

# Constants
TAU = 0.5 * (1 + math.sqrt(5))  # Golden ratio needed for Icosahedral generators
C3_GENERATION_ANGLE_1 = 2 * math.pi / 3  # 120 degrees
C3_GENERATION_ANGLE_2 = 4 * math.pi / 3  # 240 degrees

# Character set for chain IDs (A-Z, a-z)
CHAIN_ID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAIN_ID_BASE = len(CHAIN_ID_CHARS)  # Should be 52

# -------------------------------
# Symmetry Group Functions (Original T, O)
# -------------------------------
ROT_DICT = {
    "O": [
        # ... O matrices ...
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
    ],
    "T": [
        # ... T matrices ...
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    ],
}

# -------------------------------
# Matrix Generation Functions
# -------------------------------

def _build_rot_matrix(angle: float, axis: str = 'z') -> np.ndarray:
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    if axis.lower() == 'z':
        return np.array([[cos_a, -sin_a, 0.0],
                         [sin_a, cos_a, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    # ... other axes ...
    elif axis.lower() == 'y':
        return np.array([[cos_a, 0.0, sin_a],
                         [0.0, 1.0, 0.0],
                         [-sin_a, 0.0, cos_a]], dtype=np.float64)
    elif axis.lower() == 'x':
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, cos_a, -sin_a],
                         [0.0, sin_a, cos_a]], dtype=np.float64)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

def calculate_cyclic_symmetry_matrices(n_rotations: int) -> list:
    return [_build_rot_matrix(2 * math.pi * i / n_rotations, 'z') for i in range(n_rotations)]

def calculate_dihedral_symmetry_matrices(n_rotations: int) -> list:
    # ... unchanged ...
    rotation_matrices = calculate_cyclic_symmetry_matrices(n_rotations)
    reflection_matrix = _build_rot_matrix(math.pi, 'x')
    reflected_matrices = [reflection_matrix @ rm for rm in rotation_matrices]
    rotation_matrices.extend(reflected_matrices)
    return rotation_matrices

def _calculate_full_icosahedral_matrices(tree_depth: int = 5) -> list:
    # ... unchanged ...
    generator_1 = np.array([[-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
    generator_2 = np.array([[0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]], dtype=np.float64)
    generator_3 = np.array([[0.5, -0.5 * TAU, 0.5 / TAU],
                            [0.5 * TAU, 0.5 / TAU, -0.5],
                            [0.5 / TAU, 0.5, 0.5 * TAU]], dtype=np.float64)
    generators = [generator_1, generator_2, generator_3]
    matrices = set()
    current_set = [np.eye(3, dtype=np.float64)]
    matrices.add(tuple(np.eye(3).flatten()))
    for _ in range(tree_depth):
        new_products = set()
        temp_current_list = [np.array(m).reshape(3, 3) for m in matrices]
        for mat in temp_current_list:
            for gen in generators:
                product_mat = np.round(mat @ gen, decimals=6)
                new_products.add(tuple(product_mat.flatten()))
        prev_len = len(matrices)
        matrices.update(new_products)
        if len(matrices) == prev_len and len(matrices) >= 60:
            break
        if len(matrices) >= 60:
            matrices = set(list(matrices)[:60])
            break
    mats = [np.array(m).reshape(3, 3) for m in matrices]
    if len(mats) != 60:
        print(f"[Warning] Expected 60 I matrices, got {len(mats)}.", file=sys.stderr)
    mats.sort(key=lambda m: tuple(np.round(m, 5).flatten()))  # Sort based on rounded values
    return mats

def calculate_pseudoquotient_matrices(G_group_name: str, G_div_group_name: str) -> list:
    # ... unchanged ...
    if not G_div_group_name.startswith("C_"):
        raise ValueError("G_div must be cyclic")
    n_div = int(G_div_group_name.split("_")[1])
    if G_group_name != "I" or n_div != 3:
        raise NotImplementedError(f"Only I/C_3 supported")
    print(f"[Info] Calculating pseudoquotient {G_group_name}/{G_div_group_name}...")
    full_group_matrices = np.array(_calculate_full_icosahedral_matrices())
    G_div_matrices = np.array(calculate_cyclic_symmetry_matrices(n_div))
    target_trace = 2 * math.cos(2 * math.pi / n_div) + 1
    traces = np.trace(full_group_matrices, axis1=1, axis2=2)
    idxs = np.where(np.abs(traces - target_trace) < 1e-6)[0]
    if idxs.size == 0:
        raise RuntimeError(f"No element found matching C_{n_div} trace")
    idx_equal_theta = idxs[0]
    try:
        L_div, V_div = np.linalg.eig(G_div_matrices[1])
        L, V = np.linalg.eig(full_group_matrices[idx_equal_theta])
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Eig failed: {e}")
    try:
        if abs(np.linalg.det(V)) < 1e-8:
            raise np.linalg.LinAlgError("Singular matrix V")
        W = np.real(V_div @ np.linalg.inv(V))
        if not np.allclose(W @ W.T, np.eye(3), atol=1e-5):
            print("[Warning] W not orthogonal. Using pseudo-inverse.", file=sys.stderr)
            try:
                W_inv = np.linalg.pinv(W)
            except np.linalg.LinAlgError:
                print("[Error] pinv(W) failed.", file=sys.stderr)
                W_inv = W.T
        else:
            W_inv = W.T
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Could not compute W: {e}")
    N = full_group_matrices.shape[0]
    transformed_full_group = np.matmul(np.matmul(W, full_group_matrices), W_inv)
    unique_representatives = []
    processed_indices = set()
    for i in range(N):
        if i in processed_indices:
            continue
        mat_i_transformed = transformed_full_group[i]
        unique_representatives.append(mat_i_transformed)
        processed_indices.add(i)
        for k in range(1, n_div):
            target_mat = np.matmul(mat_i_transformed, G_div_matrices[k])
            min_dist = float('inf')
            best_match_idx = -1
            for j in range(N):
                if j not in processed_indices:
                    dist = np.linalg.norm(transformed_full_group[j] - target_mat)
                    if dist < 1e-4 and dist < min_dist:
                        min_dist = dist
                        best_match_idx = j
            if best_match_idx != -1:
                processed_indices.add(best_match_idx)
    quotient_matrices = np.array(unique_representatives)
    expected_size = N // n_div
    if quotient_matrices.shape[0] != expected_size:
        print(f"[Warning] Expected {expected_size} pseudoquotient matrices, found {quotient_matrices.shape[0]}. Clustering?", file=sys.stderr)
    quotient_matrices = quotient_matrices[np.lexsort(np.round(quotient_matrices.reshape(-1, 9), 5).T)]
    return list(quotient_matrices)

def get_target_symmetry_matrices(symmetry_group: str) -> list:
    # ... unchanged logic ... (Returns 20 matrices for I/C3, includes identity for others) ...
    print(f"[Info] Getting matrices for final complex symmetry: {symmetry_group}")
    mats = []
    is_pseudoquotient_case = False
    if symmetry_group.startswith("C_"):
        try:
            n_rotations = int(symmetry_group[2:])
            mats = calculate_cyclic_symmetry_matrices(n_rotations)
        except ValueError as e:
            raise ValueError(f"Invalid n in C_n: {symmetry_group} ({e})")
    elif symmetry_group.startswith("D_"):
        try:
            n_rotations = int(symmetry_group[2:])
            mats = calculate_dihedral_symmetry_matrices(n_rotations)
        except ValueError as e:
            raise ValueError(f"Invalid n in D_n: {symmetry_group} ({e})")
    elif symmetry_group == "I":
        mats = calculate_pseudoquotient_matrices("I", "C_3")
        is_pseudoquotient_case = True
        if len(mats) != 20:
            print(f"[Error] Expected 20 I/C3 matrices, got {len(mats)}.", file=sys.stderr)
        else:
            print(f"[Info] Using the {len(mats)} matrices from I/C3 pseudoquotient.")
        return mats  # Return directly for I/C3
    elif symmetry_group in ("T", "O"):
        mats = [np.array(mat, dtype=np.float64) for mat in ROT_DICT[symmetry_group]]
    else:
        raise ValueError(f"Unsupported group: {symmetry_group}")
    identity = np.eye(3, dtype=np.float64)
    identity_found_idx = -1
    for idx, mat in enumerate(mats):
        if np.allclose(mat, identity, atol=1e-6):
            identity_found_idx = idx
            break
    if identity_found_idx > 0:
        mats.insert(0, mats.pop(identity_found_idx))
    elif identity_found_idx == -1:
        print(f"[Info] Adding identity matrix to start for group {symmetry_group}.")
        mats.insert(0, identity)
    print(f"[Info] Using {len(mats)} matrices for {symmetry_group} (identity first)")
    return mats

# -------------------------------
# Helper Functions for Applying Rotations
# -------------------------------
def apply_rotation_to_chain(chain: Chain, rot_matrix: np.ndarray) -> None:
    # ... unchanged ...
    if np.allclose(rot_matrix, np.eye(3), atol=1e-6):
        return
    for residue in chain:
        for atom in residue:
            old_coord = atom.get_coord()
            new_coord = np.dot(rot_matrix, old_coord)
            atom.set_coord(new_coord)

# --- NEW generate_chain_ids ---
def generate_chain_ids(n: int) -> list:
    """
    Generate a list of n unique chain IDs.
    Uses A-Z, a-z first, then two-character IDs AA, AB, ..., Az, BA, ...
    Handles up to 52 + 52*52 = 2756 chains.
    """
    chain_ids = []
    if n <= 0:
        return chain_ids

    num_single = CHAIN_ID_BASE
    num_double = CHAIN_ID_BASE * CHAIN_ID_BASE

    for i in range(n):
        if i < num_single:
            # Single character ID (A-Z, a-z)
            chain_ids.append(CHAIN_ID_CHARS[i])
        elif i < num_single + num_double:
            # Double character ID (AA, AB, ..., zz)
            adj_i = i - num_single  # Adjust index for the double-char range
            idx1 = adj_i // CHAIN_ID_BASE  # Index for the first character
            idx2 = adj_i % CHAIN_ID_BASE   # Index for the second character
            chain_ids.append(CHAIN_ID_CHARS[idx1] + CHAIN_ID_CHARS[idx2])
        else:
            # Fallback for extremely large numbers (or extend logic for 3 chars etc.)
            print(f"[Warning] Requested {n} chains, exceeding 2-character limit ({num_single+num_double}). Using prefix fallback.", file=sys.stderr)
            prefix = (i // (num_single + num_double)) + 1
            fallback_idx = i % (num_single + num_double)
            # Re-use the single/double logic for the fallback index
            if fallback_idx < num_single:
                chain_ids.append(f"{prefix}{CHAIN_ID_CHARS[fallback_idx]}")
            else:
                adj_i = fallback_idx - num_single
                idx1 = adj_i // CHAIN_ID_BASE
                idx2 = adj_i % CHAIN_ID_BASE
                chain_ids.append(f"{prefix}{CHAIN_ID_CHARS[idx1]}{CHAIN_ID_CHARS[idx2]}")
            # This fallback might still create PDB spec issues if prefix+chars > allowed ID length

    if len(set(chain_ids)) != n:
        print("[Error] Generated chain IDs are not unique! Check generation logic.", file=sys.stderr)
        # Potentially exit or raise an error here

    return chain_ids
# --- End NEW generate_chain_ids ---

# -------------------------------
# Core Processing Logic
# -------------------------------
def create_trimer_asu(input_chain_A: Chain) -> list:
    # ... unchanged ...
    print(f"[Info] Creating C3 trimer ASU from input chain '{input_chain_A.id}'...")
    rot_120 = _build_rot_matrix(C3_GENERATION_ANGLE_1, 'z')
    rot_240 = _build_rot_matrix(C3_GENERATION_ANGLE_2, 'z')
    chain_A = copy.deepcopy(input_chain_A)
    chain_B = copy.deepcopy(input_chain_A)
    chain_C = copy.deepcopy(input_chain_A)
    apply_rotation_to_chain(chain_B, rot_120)
    apply_rotation_to_chain(chain_C, rot_240)
    # Use temporary IDs that are unlikely to conflict with the final chain IDs.
    chain_A.id = "asu_1"
    chain_B.id = "asu_2"
    chain_C.id = "asu_3"
    print("[Info] Trimer ASU created (temporarily named asu_1, asu_2, asu_3).")
    return [chain_A, chain_B, chain_C]

def process_mmcif_with_symmetry(input_cif_path: str, target_symmetry_matrices: list, output_cif_path: str) -> None:
    # ... unchanged ... (Error handling for add already included conceptually) ...
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("input_structure", input_cif_path)
    except Exception as e:
        print(f"Error reading {input_cif_path}: {e}", file=sys.stderr)
        sys.exit(1)
    if len(structure) == 0:
        print(f"Error: No models in {input_cif_path}.", file=sys.stderr)
        sys.exit(1)
    model_orig = structure[0]
    input_chain_A = None
    available_chains = []
    for chain in model_orig.get_chains():
        available_chains.append(chain.id)
        if chain.id == 'A':
            input_chain_A = chain
    if input_chain_A is None:
        print(f"Error: Chain 'A' not found in {input_cif_path}. Found: {available_chains}", file=sys.stderr)
        sys.exit(1)
    print(f"[Info] Found Chain 'A'. Ignored: {[c for c in available_chains if c != 'A']}.")
    trimer_asu = create_trimer_asu(input_chain_A)
    new_structure = Structure.Structure("symmetrized_complex")
    new_model = Model.Model(0)
    new_structure.add(new_model)
    num_chains_in_asu = len(trimer_asu)
    num_target_ops = len(target_symmetry_matrices)
    total_chains_needed = num_chains_in_asu * num_target_ops
    final_chain_ids = generate_chain_ids(total_chains_needed)
    print(f"[Info] Applying {num_target_ops} ops to {num_chains_in_asu}-chain ASU.")
    print(f"[Info] Generating {total_chains_needed} total chains.")
    chain_id_counter = 0
    print("[Info] Generating final symmetric complex...")
    try:
        # Import a fresh Chain constructor to create new chain objects.
        from Bio.PDB.Chain import Chain as NewChain
        for i, target_rot_matrix in enumerate(target_symmetry_matrices):
            for asu_chain_template in trimer_asu:
                if chain_id_counter >= total_chains_needed:
                    print("[Error] Chain ID counter exceeded.", file=sys.stderr)
                    break
                # Make a deep copy of the ASU chain and apply the rotation.
                temp_chain = copy.deepcopy(asu_chain_template)
                apply_rotation_to_chain(temp_chain, target_rot_matrix)
                # Create a new chain object with the desired new ID and transfer all residues.
                new_chain = NewChain(final_chain_ids[chain_id_counter])
                for residue in temp_chain.get_list():
                    new_chain.add(residue)
                new_model.add(new_chain)
                chain_id_counter += 1
            if chain_id_counter >= total_chains_needed and i < num_target_ops - 1:
                print("[Error] Ran out of chain IDs.", file=sys.stderr)
                break
    except PDBConstructionException as e:
        print(f"\n[FATAL ERROR] Failed to add chain to model: {e}", file=sys.stderr)
        print(f"This likely means a duplicate chain ID ('{new_chain.id}' at index {chain_id_counter}) was generated.", file=sys.stderr)
        print("Please check the 'generate_chain_ids' function logic.", file=sys.stderr)
        sys.exit(1)  # Exit after the specific error
    except Exception as e:  # Catch other potential errors during generation
        print(f"\n[FATAL ERROR] An unexpected error occurred during chain generation: {e}", file=sys.stderr)
        sys.exit(1)

    io = MMCIFIO()
    io.set_structure(new_structure)
    try:
        io.save(output_cif_path)
        print(f"Successfully wrote {chain_id_counter} chains for {args.group} complex to {output_cif_path}")
    except Exception as e:
        print(f"Error writing {output_cif_path}: {e}", file=sys.stderr)
        sys.exit(1)

# -------------------------------
# Main Execution
# -------------------------------
def main():
    # ... unchanged ...
    global args
    parser = argparse.ArgumentParser(
        description="Generate full symmetric complex from Chain 'A' by first creating a C3 trimer ASU, then applying target symmetry (--group)."
    )
    parser.add_argument("input_file", help="Path to input mmCIF (.cif). Uses Chain 'A' only.")
    parser.add_argument("output_file", help="Path to output mmCIF (.cif).")
    parser.add_argument("--group", required=True, help="Target symmetry group (C_n, D_n, T, O, I). 'I' uses I/C3 quotient.")
    args = parser.parse_args()
    if not args.input_file.lower().endswith(".cif"):
        print("Error: Input file must be .cif", file=sys.stderr)
        sys.exit(1)
    if not args.output_file.lower().endswith(".cif"):
        print("Error: Output file must be .cif", file=sys.stderr)
        sys.exit(1)
    try:
        target_symmetry_matrices = get_target_symmetry_matrices(args.group)
    except (ValueError, NotImplementedError, RuntimeError) as e:
        print(f"Error generating matrices for '{args.group}': {e}", file=sys.stderr)
        sys.exit(1)
    if not target_symmetry_matrices:
        print(f"Error: No matrices for '{args.group}'.", file=sys.stderr)
        sys.exit(1)
    process_mmcif_with_symmetry(args.input_file, target_symmetry_matrices, args.output_file)

if __name__ == "__main__":
    main()
