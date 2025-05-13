

"""
Layers for euclidean symmetry group operations.
Includes standard point groups and Icosahedral Subquotient C3 logic.
Based on original code and SUBQUOTIENT modifications.
"""

import itertools
import math
from itertools import product
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
# from scipy.optimize import linear_sum_assignment # Not used in the final selection
# from scipy.spatial.transform import Rotation # Not used in the final selection
# from tqdm import tqdm # Not used

# Removed chroma import as it's likely external / specific to original SUBQUOTIENT context
# from chroma.layers.structure import backbone

TAU = 0.5 * (1 + math.sqrt(5))

# Keep ROT_DICT for T and O groups
ROT_DICT = {
    "O": [
        # ... (original O matrices) ...
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
        # ... (original T matrices) ...
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

# Cache for generated point groups to avoid redundant computation
_point_group_cache = {}

def get_point_group(group: str, n_subunits: int = None) -> torch.Tensor:
    """
    Get representation of group elements as torch.Tensor.
    For group 'I', returns the 20 I/C3 pseudo-quotient matrices.

    Args:
        group (str): Group name, e.g., "C_3", "D_6", "T", "O", "I".
        n_subunits (int, optional): Number of subunits, primarily used for C_n and D_n if not in name. Ignored for T, O, I.

    Returns:
        torch.Tensor: Rotation matrices for the queried point group.
                      For 'I', returns the 20 I/C3 matrices.
    """
    cache_key = group
    if group.startswith("C_") or group.startswith("D_"):
        try:
             n = int(group.split("_")[1])
             cache_key = f"{group.split('_')[0]}_{n}" # Normalize key
        except (IndexError, ValueError):
             if n_subunits is None:
                  raise ValueError(f"Order 'n' must be specified for group {group} either in name (e.g., C_3) or via n_subunits.")
             n = n_subunits
             cache_key = f"{group.split('_')[0]}_{n}"


    if cache_key in _point_group_cache:
        return _point_group_cache[cache_key]

    if group.startswith("C"):
        n = int(cache_key.split("_")[1])
        G = get_Cn_groups(n)
    elif group.startswith("D"):
        n = int(cache_key.split("_")[1])
        G = get_Dn_groups(n)
    elif group == "I":
        # *** MODIFICATION: Return I/C3 pseudo-quotient matrices ***
        G = get_pseudoquotient('I', 'C_3')
        print(f"DEBUG: Pseudoquotient shape: {G.shape}")
    elif group == "O" or group == "T":
        if group not in ROT_DICT:
             raise ValueError(f"Precomputed matrices for {group} not found in ROT_DICT.")
        G = torch.Tensor(np.array(ROT_DICT[group]))
    else:
        raise ValueError(f"Symmetry group '{group}' is not recognized or supported.")

    _point_group_cache[cache_key] = G
    return G


def get_Cn_groups(n: int) -> torch.Tensor:
    """get rotation matrices for Cyclic groups C_n"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("Order n for C_n must be a positive integer.")
    G = []
    for ri in range(n):
        angle = ri * 2.0 * math.pi / n
        cos_phi = math.cos(angle)
        sin_phi = math.sin(angle)
        # Use higher precision and round at the end if necessary
        g = np.array(
            [[cos_phi, -sin_phi, 0.0], [sin_phi, cos_phi, 0.0], [0.0, 0.0, 1.0]]
        )
        G.append(g)
    # Rounding can introduce small errors, consider using without rounding if precision is critical
    return torch.tensor(np.round(np.array(G), decimals=8), dtype=torch.float32)


def get_Dn_groups(n: int) -> torch.Tensor:
    """get rotation matrices for Dihedral groups D_n"""
    if not isinstance(n, int) or n < 1: # D1 is C2 technically, but allow n=1
        raise ValueError("Order n for D_n must be a positive integer.")

    # Get C_n rotations first
    Cn_rots = get_Cn_groups(n) # Shape (n, 3, 3)

    # Define the C2 rotation axis (typically x-axis)
    # This C2 flips the z-axis and y-axis
    c2_flip = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32)

    # Generate the other n elements by composing C_n elements with the flip
    # g_dn = c2_flip @ g_cn
    Dn_other_rots = torch.einsum('ij,njk->nik', c2_flip, Cn_rots)

    # Combine C_n rotations and the new rotations
    G = torch.cat([Cn_rots, Dn_other_rots], dim=0) # Shape (2n, 3, 3)

    # Remove duplicates which can occur for small n (e.g., D1, D2)
    G_unique = torch.unique(torch.round(G, decimals=6), dim=0)

    return G_unique


def get_I_rotations(tree_depth: int = 5) -> torch.Tensor:
    """
    Generate the 60 rotation matrices for the Icosahedral group (I)
    using generators and group multiplication.
    """
    # Check cache first
    cache_key = "I_full_60"
    if cache_key in _point_group_cache:
        return _point_group_cache[cache_key]

    # Generators for the rotational subgroup of the icosahedral group (A5)
    # Using standard generators that produce rotations only
    # Generator 1: 180-degree rotation around an axis through edge centers
    g1 = torch.tensor([[-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, 1.0]], dtype=torch.float32)
    # Generator 2: 120-degree rotation around an axis through vertex opposite vertex
    g2 = torch.tensor([[0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]], dtype=torch.float32)
    # Generator 3: Another rotation (can be derived, but often included)
    # This generator corresponds to a rotation by 2*pi/5 around an axis through face centers
    phi = TAU # Golden ratio conjugate is 1/TAU
    g3 = 0.5 * torch.tensor([[TAU - 1, -TAU, 1.0],
                             [TAU, 1.0, TAU - 1],
                             [-1.0, TAU - 1, TAU]], dtype=torch.float32)
    # Corrected g3 based on common representations (ensure determinant is +1)
    # Let's use the one from the original SUBQUOTIENT code:
    g3_alt = torch.tensor([
            [0.5, -0.5 * TAU, 0.5 / TAU],
            [0.5 * TAU, 0.5 / TAU, -0.5],
            [0.5 / TAU, 0.5, 0.5 * TAU]], dtype=torch.float32)
    # Verify determinant is +1
    # print(f"Det g1={torch.linalg.det(g1)}, Det g2={torch.linalg.det(g2)}, Det g3_alt={torch.linalg.det(g3_alt)}")

    generators = [g1, g2, g3_alt]
    identity = torch.eye(3, dtype=torch.float32)

    # Use a breadth-first search or iterative approach to generate elements
    sym_ops = {tuple(identity.flatten().tolist())} # Store flattened tuples in a set for uniqueness
    queue = [identity]
    max_ops = 60
    generated_matrices = [identity]

    processed_strings = {tuple(identity.flatten().tolist())}

    current_queue = [identity]
    all_ops_matrices = [identity]

    # Iteratively multiply current ops by generators
    # Limit iterations to prevent infinite loop in case of issues
    for _ in range(tree_depth + 2): # More iterations might be needed
        next_queue = []
        added_new = False
        for op in current_queue:
            for gen in generators:
                new_op = torch.matmul(op, gen)
                new_op_rounded = torch.round(new_op, decimals=6)
                new_op_tuple = tuple(new_op_rounded.flatten().tolist())

                if new_op_tuple not in processed_strings:
                    processed_strings.add(new_op_tuple)
                    next_queue.append(new_op) # Add the non-rounded version
                    all_ops_matrices.append(new_op)
                    added_new = True
                    if len(all_ops_matrices) >= max_ops:
                        break
            if len(all_ops_matrices) >= max_ops:
                break
        current_queue = next_queue
        if not added_new or len(all_ops_matrices) >= max_ops:
            break

    # Stack and find unique based on rounded values, but return original precision
    if not all_ops_matrices: # Should not happen if identity is added
         return torch.eye(3, dtype=torch.float32).unsqueeze(0)

    unique_matrices = torch.stack(all_ops_matrices, dim=0)
    # Further ensure uniqueness after stacking
    unique_matrices_rounded = torch.round(unique_matrices, decimals=6)
    _, unique_indices = np.unique(unique_matrices_rounded.numpy(), axis=0, return_index=True)

    # Sort indices to maintain some consistency if needed, though order isn't guaranteed
    unique_indices = np.sort(unique_indices)
    final_ops = unique_matrices[unique_indices]

    if final_ops.shape[0] != max_ops:
         print(f"Warning: Expected {max_ops} unique Icosahedral rotations, but found {final_ops.shape[0]}. Check generators or tree depth.")

    _point_group_cache[cache_key] = final_ops
    return final_ops

def get_pseudoquotient(
    G_name: str,
    G_div_name: str
) -> torch.Tensor:
    """
    Finds the set of rotation matrices representing the pseudo-quotient G / G_div,
    returned in the ORIGINAL coordinate frame.

    Internally computes the transformation to a canonical frame (e.g., C3-aligned),
    identifies coset representatives in that frame, and then transforms these
    representatives back to the original frame before returning.

    Mimics the logic from the reference symmetry_postprocess.py script for
    calculating representatives, but adds the transformation back step.
    """
    if not (G_name == 'I' and G_div_name == 'C_3'):
        raise NotImplementedError(f"Pseudoquotient calculation implemented only for I / C_3.")

    # Use a cache key specific to the *original frame* output
    cache_key = "I_sub_C3_original_frame"
    if cache_key in _point_group_cache:
        print(f"DEBUG: Using cached I/C3 matrices (original frame).")
        return _point_group_cache[cache_key]

    # --- Use float64 for internal numerical stability ---
    internal_dtype = torch.float64
    # --- Define final output dtype ---
    final_dtype = torch.float32

    print(f"DEBUG: Calculating I/C3 matrices for original frame...")

    # 1. Get Groups
    # Assuming get_I_rotations and get_Cn_groups return tensors on some default device
    # Make sure to move them to a consistent device if needed, e.g., cpu
    G = get_I_rotations()      # Full group (60, 3, 3)
    G_div = get_Cn_groups(3)   # Subgroup (3, 3, 3)
    device = G.device          # Use device from loaded groups

    G = G.to(device=device, dtype=internal_dtype)
    G_div = G_div.to(device=device, dtype=internal_dtype)

    N = G.shape[0]      # 60
    n_div = G_div.shape[0] # 3

    # 2. Find Alignment Rotation (W) and its Inverse (W_inv)
    target_trace = 2 * math.cos(2 * math.pi / n_div) + 1 # Should be 0.0 for C3
    traces = torch.einsum('bii->b', G)
    trace_tolerance = 1e-5
    indices_equal_theta = torch.where(torch.abs(traces - target_trace) < trace_tolerance)[0]

    if len(indices_equal_theta) == 0:
        print(f"ERROR: No element found in G matching C3 trace ({target_trace:.3f} +/- {trace_tolerance}). Traces found: {traces}")
        raise RuntimeError(f"Cannot find C3 element in Icosahedral group by trace.")
    idx_equal_theta = indices_equal_theta[0].item()
    g_theta = G[idx_equal_theta]        # A C3 rotation from the I group
    g_div_gen = G_div[1]                # The canonical 120-degree C3 rotation

    print(f"DEBUG: Using element G[{idx_equal_theta}] (trace {traces[idx_equal_theta]:.5f}) to align with canonical C3.")

    try:
        # Compute eigenvectors/values for alignment
        L_div, V_div = torch.linalg.eig(g_div_gen)
        L, V = torch.linalg.eig(g_theta)

        # V contains eigenvectors of g_theta as columns. V maps from g_theta's eigenbasis to standard basis.
        # V_div contains eigenvectors of g_div_gen as columns. V_div maps from g_div_gen's eigenbasis to standard basis.
        # We want W such that W @ g_theta @ W_inv = g_div_gen (approximately)
        # This means W aligns g_theta's frame TO g_div_gen's frame.
        # W should map V -> V_div. W = V_div @ V_inv
        V_inv = torch.linalg.pinv(V) # Use pseudo-inverse for stability
        W = torch.matmul(V_div, V_inv)

        # W might be complex, but should be approximately real for rotation alignment
        if torch.max(torch.abs(W.imag)) > trace_tolerance:
             print(f"Warning: Alignment matrix W has significant imaginary component: {torch.max(torch.abs(W.imag)):.2e}. Using real part.")
        W = W.real # Take real part

        # Calculate W_inv. Prefer transpose if W is orthogonal, otherwise use pinv.
        identity_3x3 = torch.eye(3, device=device, dtype=internal_dtype)
        if torch.allclose(torch.matmul(W, W.T), identity_3x3, atol=1e-4):
             W_inv = W.T
             print("DEBUG: Alignment matrix W appears orthogonal. Using transpose for inverse.")
        else:
             print(f"[Warning] W is not orthogonal (max diff from I: {torch.max(torch.abs(torch.matmul(W, W.T) - identity_3x3)):.2e}). Using pseudo-inverse for W_inv.")
             try:
                 W_inv = torch.linalg.pinv(W)
                 # Verify W_inv @ W ~= I
                 if not torch.allclose(torch.matmul(W_inv, W), identity_3x3, atol=1e-4):
                     print("[Warning] Pseudo-inverse W_inv may be inaccurate.")
             except torch.linalg.LinAlgError as e_pinv:
                 print(f"ERROR: torch.linalg.pinv(W) failed: {e_pinv}. Falling back to W.T")
                 W_inv = W.T # Fallback, might be inaccurate

    except torch.linalg.LinAlgError as e:
        print(f"ERROR: torch.linalg computation failed during alignment: {e}")
        raise RuntimeError(f"Could not compute alignment matrix W: {e}")

    # 3. Transform G to the Canonical (C3-aligned) Frame
    # G_transformed = W @ G @ W_inv
    # Perform calculation g'_i = W @ g_i @ W_inv for all g_i in G
    G_transformed = torch.einsum('ij,gjk,kl->gil', W, G, W_inv)
    print(f"DEBUG: Successfully transformed {G_transformed.shape[0]} matrices to canonical frame.")

    # 4. Identify Coset Representatives in the Transformed Space
    representatives_transformed = [] # Store the chosen representatives (in transformed space)
    processed_indices = set()
    distance_tolerance = 1e-4 # Tolerance for matching matrices

    print("DEBUG: Identifying cosets in transformed space...")
    num_cosets_found = 0
    for i in range(N): # Iterate through indices 0 to 59
        if i in processed_indices:
            continue

        # Found a new representative in the transformed space
        num_cosets_found += 1
        rep_i_transformed = G_transformed[i]
        representatives_transformed.append(rep_i_transformed)
        processed_indices.add(i)

        # Find and mark other elements in the same coset (g'_i * g_div_k)
        for k in range(1, n_div): # Look for elements related by G_div[1], G_div[2]
            # Target matrix in transformed space: rep_i_transformed @ G_div[k]
            # G_div[k] are the canonical C3 rotations
            target_mat = torch.matmul(rep_i_transformed, G_div[k])

            min_dist = torch.inf
            best_match_idx = -1

            # Search for the best match among *unprocessed* transformed matrices
            for j in range(N):
                if j not in processed_indices:
                    # Compare G_transformed[j] with target_mat
                    dist = torch.linalg.norm(G_transformed[j] - target_mat)
                    if dist < distance_tolerance and dist < min_dist:
                        min_dist = dist
                        best_match_idx = j

            if best_match_idx != -1:
                processed_indices.add(best_match_idx)
            else:
                # If match not found below tolerance, check if the closest one is very close anyway
                min_dist_overall = torch.inf
                for j in range(N):
                     if j not in processed_indices:
                         dist = torch.linalg.norm(G_transformed[j] - target_mat)
                         min_dist_overall = min(min_dist_overall, dist)
                print(f"Warning: Could not find exact match (tol={distance_tolerance:.1e}) for representative index {i} with G_div[{k}]. Closest distance found: {min_dist_overall:.2e}")


    print(f"DEBUG: Coset identification loop finished. Found {len(representatives_transformed)} representatives (transformed space).")
    print(f"DEBUG: Total indices processed: {len(processed_indices)} (should be 60)")

    if not representatives_transformed:
        print("ERROR: No representatives found!")
        # Return empty tensor with correct shape and final dtype
        return torch.empty((0, 3, 3), device=device, dtype=final_dtype)

    # Stack the representatives found in the transformed space
    G_quot_transformed = torch.stack(representatives_transformed, dim=0) # Shape (expected 20, 3, 3)

    expected_size = N // n_div # 20
    if G_quot_transformed.shape[0] != expected_size:
        print(f"ERROR: Expected {expected_size} pseudoquotient matrices in transformed space, found {G_quot_transformed.shape[0]}. Aborting.")
        # Return empty tensor or raise error? Let's return empty.
        return torch.empty((0, 3, 3), device=device, dtype=final_dtype)
    else:
        print(f"DEBUG: Found correct number ({G_quot_transformed.shape[0]}) of representatives in transformed space.")

    # 5. Transform Representatives BACK to the Original Coordinate Frame
    # G_quot_original = W_inv @ G_quot_transformed @ W
    print("DEBUG: Transforming representatives back to original coordinate frame...")
    G_quot_original = torch.einsum('ij,gjk,kl->gil', W_inv, G_quot_transformed, W)

    # 6. Sort and Finalize
    # Lexicographical sort (optional, but helps consistency)
    try:
        G_quot_np = G_quot_original.cpu().numpy()
        sort_key = np.round(G_quot_np.reshape(G_quot_np.shape[0], -1), decimals=5)
        sorted_indices = np.lexsort(sort_key.T)
        G_quot_final_original = G_quot_original[sorted_indices]
        print("DEBUG: Sorted representatives lexicographically.")
    except Exception as sort_err:
        print(f"Warning: Sorting failed ({sort_err}). Returning unsorted representatives.")
        G_quot_final_original = G_quot_original

    # Convert to the final desired dtype (e.g., float32)
    G_quot_final_original = G_quot_final_original.to(dtype=final_dtype)

    # Check for identity in the *original* frame
    identity_orig = torch.eye(3, device=device, dtype=final_dtype)
    has_identity = any(torch.allclose(rep, identity_orig, atol=1e-4) for rep in G_quot_final_original)
    if not has_identity and G_quot_final_original.shape[0] > 0:
        print(f"Warning: Identity matrix not found among the final {G_quot_final_original.shape[0]} representatives (in original frame).")
    elif has_identity:
        print("DEBUG: Identity matrix is present in the final representatives (original frame).")


    # Cache the result (original frame matrices)
    _point_group_cache[cache_key] = G_quot_final_original
    print(f"DEBUG: Successfully calculated and cached {G_quot_final_original.shape[0]} I/C3 matrices for original frame.")

    return G_quot_final_original




def calculate_radius_from_mw(molecular_weight: float, n: int, symmetry_type: str) -> float:
    """
    Calculates the estimated radius of the COMPLEX for COM regularization based on
    molecular weight of the MONOMER and symmetry type.

    Args:
        molecular_weight: Molecular weight of a SINGLE MONOMER (in Daltons).
        n: Symmetry order (e.g., number of subunits in Cn or Dn).
        symmetry_type: Type of symmetry (e.g., "C_n", "D_n", "T", "O", "I").

    Returns:
        The calculated approximate radius of the complex (distance from origin to monomer COM).
    """
    # Basic check
    if molecular_weight <= 0:
        raise ValueError("Molecular weight must be positive.")

    # Estimate monomer radius (approximation using empirical formula for globular proteins)
    # R = (3 * V / (4 * pi))^(1/3) where V = MW * partial_specific_volume / Avogadro
    # Simplified: R ~ 0.66 * MW^(1/3) Angstrom (from Erickson 2009, Biol Proced Online)
    # Using a factor closer to 0.7-0.8 might be better for radius of gyration. Let's use ~0.73.
    # r_monomer = 0.73 * (molecular_weight ** (1.0/3.0)) # Radius in Angstroms
    # The original code had a divisor of 1.3, let's use that for consistency:
    r_monomer = (0.66 * (molecular_weight ** (1.0/3.0))) / 1.3

    if r_monomer <= 0: # Should not happen if MW > 0
        return 0.0

    radius = 0.0
    sym_base = symmetry_type.split('_')[0]

    if sym_base == "C":
        if n < 2: return r_monomer # Treat C1 as monomer radius
        if n == 2:
            radius = r_monomer # Monomer COM is at monomer radius
        else:
            # Radius of the ring circumscribing the monomer COMs
            radius = r_monomer / math.sin(math.pi / n)

    elif sym_base == "D":
        if n < 1: raise ValueError("D symmetry requires n >= 1")
        if n == 1: # D1 is C2
            radius = r_monomer
        else:
            # Dihedral: two rings stacked or interleaved
            # Calculate radius of one ring
            r_ring = r_monomer / math.sin(math.pi / n) if n > 1 else 0 # Treat D1 ring radius as 0? No, use C2 logic.
            # Distance from center to COM is hypotenuse of triangle: (r_ring, r_monomer)
            # Assumes layers are offset vertically by r_monomer from the xy-plane
            radius = math.sqrt(r_ring**2 + r_monomer**2)

    elif symmetry_type == "T": # 12 subunits
        # Approximation based on geometry - distance from center to vertex COM
        # This depends on how T symmetry is constructed. Assuming vertices of tetrahedron are centers...
        # A rough estimate might involve relating to cube vertices?
        # Using the provided formula from original code:
        radius = r_monomer * (1 + math.sqrt(6)/2) # Is this COM radius or extent? Assume COM.

    elif symmetry_type == "O": # 24 subunits
        # Similar estimation challenge. Using provided formula:
        radius = r_monomer * (math.sqrt(2) + 1)

    elif symmetry_type == "I": # 60 subunits
        # Using provided formula:
        radius = r_monomer * ((1 + math.sqrt(5)) / 2 * math.sqrt(2*(5+math.sqrt(5))/5) + 1)

    else:
        print(f"Warning: Radius calculation not implemented for symmetry type {symmetry_type}. Returning monomer radius.")
        radius = r_monomer

    return radius


def get_atom_indices_for_chain(chain_id: int, feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Finds all atom indices associated with a given chain ID (asym_id).

    Args:
        chain_id (int): The integer asymmetric chain ID.
        feats (dict): The feature dictionary containing 'asym_id' (potentially B, N_res)
                      and 'atom_to_token' (potentially B, N_atom, N_res).

    Returns:
        torch.Tensor: 1D tensor containing the atom indices for the specified chain.
                      Handles potential batch dimensions. Returns empty tensor if chain not found.
                      Returns indices relative to the flattened atom dimension.
    """
    if "asym_id" not in feats or "atom_to_token" not in feats:
        raise KeyError("Features dictionary must contain 'asym_id' and 'atom_to_token'.")

    asym_ids = feats["asym_id"]         # Shape (..., N_res,) - Ellipsis handles optional batch dim
    atom_to_token = feats["atom_to_token"] # Shape (..., N_atom, N_res)

    # Find residues belonging to the target chain
    # chain_residue_mask will have shape (..., N_res,)
    chain_residue_mask = (asym_ids == chain_id)

    if not torch.any(chain_residue_mask):
        # print(f"Warning: Chain ID {chain_id} not found in feats['asym_id'].")
        return torch.tensor([], dtype=torch.long, device=asym_ids.device)

    # Ensure tensors are suitable for bmm if batch dim exists
    # Add batch dim if missing for consistent processing
    batched_a2t = atom_to_token if atom_to_token.ndim == 3 else atom_to_token.unsqueeze(0)
    batched_mask = chain_residue_mask if chain_residue_mask.ndim == 2 else chain_residue_mask.unsqueeze(0)
    # batched_a2t shape: (B, N_atom, N_res)
    # batched_mask shape: (B, N_res)

    # --- FIXED LINE using Batch Matrix Multiply ---
    # Calculate which atoms map to *any* residue in the chain mask
    # (B, N_atom, N_res) @ (B, N_res, 1) -> (B, N_atom, 1)
    # Result is > 0 if an atom maps to at least one residue in the chain
    chain_atom_mask = (torch.bmm(batched_a2t.float(), batched_mask.unsqueeze(-1).float()) > 0).squeeze(-1)
    # chain_atom_mask shape: (B, N_atom)

    # Since the function is expected to return indices for a single structure (implicitly batch 0),
    # we take the mask for the first batch element.
    # If multiple batches were intended, the function signature/usage would need adjustment.
    chain_atom_mask_single = chain_atom_mask[0] # Shape (N_atom,)

    # Get the indices where the atom mask is true
    chain_indices = torch.nonzero(chain_atom_mask_single, as_tuple=False).squeeze(-1) # Shape (N_chain_atoms,)

    return chain_indices


def get_subunit_atom_indices(
    symmetry_type: str,
    chain_symmetry_groups: Optional[Dict[str, List[List[int]]]], # e.g., {"C_3": [[0, 1, 2]]}
    feats: dict[str, torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Gets the atom indices for each subunit based on symmetry type and chain groups.
    For 'I' symmetry, specifically looks for chains A-F (mapped to asym_id 0-5).

    Args:
        symmetry_type (str): The type of symmetry (e.g., "C_3", "T", "O", "I").
        chain_symmetry_groups (dict, optional): A dictionary defining chain groups.
                                               Expected format: {sym_type: [[id1, id2,..], [id_a, id_b,..]]}
                                               where ids are integer asym_ids.
        feats (dict): The feature dictionary containing 'asym_id'.
        device: The torch device.

    Returns:
        A list of tensors, where each tensor contains the atom indices for a subunit.
        The order matters, especially for 'I' symmetry (expects A, B, C, D, E, F).
    """

    subunits = []

    # --- Special Handling for 'I' symmetry (I/C3) ---
    if symmetry_type == 'I':
        # Assume chains A, B, C, D, E, F correspond to asym_id 0, 1, 2, 3, 4, 5
        # This mapping convention needs to be ensured during data preparation.
        target_chain_ids = list(range(6)) # 0=A, 1=B, 2=C, 3=D, 4=E, 5=F
        chain_labels = ['A', 'B', 'C', 'D', 'E', 'F'] # For warnings

        found_ids = set(feats["asym_id"].unique().tolist())

        for i, chain_id in enumerate(target_chain_ids):
            if chain_id not in found_ids:
                 print(f"Warning: Expected chain '{chain_labels[i]}' (asym_id {chain_id}) for 'I' symmetry, but it was not found in input features.")
                 # Add an empty tensor to maintain order? Or raise error? Let's add empty for now.
                 subunits.append(torch.tensor([], dtype=torch.long, device=device))
            else:
                 atom_indices = get_atom_indices_for_chain(chain_id, feats)
                 if atom_indices.numel() == 0:
                     print(f"Warning: Chain '{chain_labels[i]}' (asym_id {chain_id}) found, but has no associated atoms according to atom_to_token map.")
                 subunits.append(atom_indices.to(device)) # Ensure device
        # Potentially add remaining chains if more than 6 are present?
        # For I/C3, we strictly need A-F for the dynamic rotation finding.
        # If other chains exist, how should they be handled? Ignore for now.
        return subunits # Return only the first 6 (potentially empty)


    # --- Handling based on chain_symmetry_groups (for C_n, D_n etc.) ---
    elif chain_symmetry_groups and symmetry_type in chain_symmetry_groups:
        groups = chain_symmetry_groups[symmetry_type]
        if groups:
             # Use the first defined group for this symmetry type
             chain_ids_in_group = groups[0]
             for chain_id in chain_ids_in_group:
                 atom_indices = get_atom_indices_for_chain(chain_id, feats)
                 subunits.append(atom_indices.to(device)) # Ensure device
             if subunits: # If we found subunits via groups, return them
                 return subunits
        # Fall through to default if group definition was empty or didn't yield atoms

    # --- Default Behavior: Treat each unique asym_id as a subunit ---
    # This is a fallback if no specific logic matches or if chain_symmetry_groups is not provided/applicable.
    # The order will depend on torch.unique.
    print(f"Info: Using default subunit identification for symmetry '{symmetry_type}'. Treating each unique asym_id as a subunit.")
    unique_asym_ids = torch.unique(feats["asym_id"])
    unique_asym_ids = torch.sort(unique_asym_ids).values # Sort for consistency

    for asym_id in unique_asym_ids:
        atom_indices = get_atom_indices_for_chain(asym_id.item(), feats)
        subunits.append(atom_indices.to(device)) # Ensure device

    # Filter out potentially empty subunits if any chain ID had no atoms
    subunits = [s for s in subunits if s.numel() > 0]

    return subunits


def reorder_point_group(
    G: torch.Tensor,
    identity_matrix: torch.Tensor,
    group_name: str, # Keep for potential future use, not strictly needed now
    reference_point: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Reorders rotation matrices (excluding identity) based on the distance
    of a transformed reference point from the identity-transformed point.

    Args:
        G (torch.Tensor): Set of rotation matrices (N, 3, 3), potentially including identity.
        identity_matrix (torch.Tensor): The identity matrix (3, 3).
        group_name (str): Name of the group (for context).
        reference_point (torch.Tensor, optional): A point (3,) to transform.
                                                  Defaults to [100, 0, 0].

    Returns:
        torch.Tensor: Reordered rotation matrices (N-1, 3, 3), excluding identity.
    """
    device = G.device
    dtype = G.dtype
    N = G.shape[0]

    if reference_point is None:
        reference_point = torch.tensor([100.0, 0.0, 0.0], device=device, dtype=dtype)
    else:
        reference_point = reference_point.to(device, dtype)

    if reference_point.shape != (3,):
        raise ValueError("reference_point must be a 1D tensor of shape (3,)")

    identity_matrix = identity_matrix.to(device, dtype)

    # Transform the reference point by all matrices in G
    # transformed_points = torch.einsum('nij,j->ni', G, reference_point) # Correct: (N, 3, 3) @ (3,) -> (N, 3)
    transformed_points = torch.matmul(G, reference_point) # Simpler syntax

    # Identify the location corresponding to the identity matrix
    identity_loc = torch.matmul(identity_matrix, reference_point)

    # Calculate squared distances from the identity location
    distances_sq = torch.sum((transformed_points - identity_loc)**2, dim=-1) # Shape (N,)

    # Get the sorted indices based on distance
    sorted_indices = torch.argsort(distances_sq)

    # Filter out the index corresponding to the identity matrix
    # We compare the matrices themselves for robustness against numerical precision
    is_identity_mask = torch.all(torch.isclose(G, identity_matrix.unsqueeze(0), atol=1e-5), dim=(1, 2)) # Shape (N,)

    # Find the index of the identity matrix in the original G
    identity_original_idx = torch.nonzero(is_identity_mask, as_tuple=False)

    if len(identity_original_idx) == 0:
        print(f"Warning: Identity matrix not found in the provided group G for {group_name}. Reordering might be incorrect.")
        # Proceed assuming the closest point (index 0 in sorted) might be identity, or return all?
        # Let's assume the first sorted index corresponds to identity if it wasn't found explicitly
        identity_idx_in_sorted = 0 # Fallback assumption
    elif len(identity_original_idx) > 1:
         print(f"Warning: Multiple identity matrices found in group G for {group_name}. Using the first.")
         identity_original_idx = identity_original_idx[0]
         # Find where this original index appears in the sorted list
         identity_idx_in_sorted = (sorted_indices == identity_original_idx.item()).nonzero(as_tuple=False).item()
    else:
        # Find where the unique identity index appears in the sorted list
        identity_idx_in_sorted = (sorted_indices == identity_original_idx.item()).nonzero(as_tuple=False).item()


    # Create the final order by removing the identity index from the sorted list
    final_order_indices = [idx.item() for i, idx in enumerate(sorted_indices) if i != identity_idx_in_sorted]

    # Reorder G using the final indices
    G_reordered = G[final_order_indices] # Shape (N-1, 3, 3)

    return G_reordered


# --- NEW FUNCTION ---
def get_pseudoquotient_operators_transformed_numpy_style(
    G_name: str = 'I',
    G_div_name: str = 'C_3',
    device: torch.device = torch.device('cpu'), # Device for the final output tensor
    dtype: torch.dtype = torch.float64 # Dtype for the final output tensor
) -> torch.Tensor:
    """
    Calculates the I/C3 pseudo-quotient representatives *in the transformed
    (C3-aligned) frame* using NumPy linalg for numerical consistency with
    symmetry_postprocess.py.

    Args:
        G_name (str): Parent group name (must be 'I').
        G_div_name (str): Subgroup name (must be 'C_3').
        device (torch.device): The desired device for the output tensor.
        dtype (torch.dtype): The desired dtype for the output tensor.

    Returns:
        torch.Tensor: A tensor of shape (20, 3, 3) containing the 20
                      transformed pseudo-quotient operators (g'_i), sorted
                      lexicographically, on the specified device and dtype.
                      Returns empty tensor on failure.
    """
    if not (G_name == 'I' and G_div_name == 'C_3'):
        raise NotImplementedError(f"Pseudoquotient calculation implemented only for I / C_3.")

    # Use a cache key specific to this NumPy-style calculation output
    cache_key = f"I_sub_C3_transformed_numpy_style_{dtype}"
    if cache_key in _point_group_cache:
        print(f"DEBUG: Using cached transformed I/C3 operators (NumPy style).")
        # Ensure cached tensor is returned on the correct device
        return _point_group_cache[cache_key].to(device)

    print(f"DEBUG: Calculating transformed I/C3 operators (NumPy style)...")
    internal_np_dtype = np.float64 # Use float64 for NumPy calculations

    # 1. Get Groups as NumPy arrays using float64
    try:
        G_np = _calculate_full_icosahedral_matrices_numpy().astype(internal_np_dtype)

        # Get C3 group using NumPy directly
        n_div = 3
        G_div_np = []
        for ri in range(n_div):
            angle = ri * 2.0 * math.pi / n_div
            cos_phi = math.cos(angle)
            sin_phi = math.sin(angle)
            g = np.array(
                [[cos_phi, -sin_phi, 0.0], [sin_phi, cos_phi, 0.0], [0.0, 0.0, 1.0]],
                dtype=internal_np_dtype
            )
            G_div_np.append(g)
        G_div_np = np.array(G_div_np)

    except Exception as e:
        print(f"ERROR: Failed to get base groups as NumPy arrays: {e}")
        return torch.empty((0, 3, 3), device=device, dtype=dtype)


    N = G_np.shape[0]      # 60
    n_div = G_div_np.shape[0] # 3

    # 2. Find Alignment Rotation (W) and Inverse (W_inv) using NumPy
    target_trace = 2 * np.cos(2 * np.pi / n_div) + 1 # Should be 0.0 for C3
    traces = np.trace(G_np, axis1=1, axis2=2)
    trace_tolerance = 1e-5 # Match tolerance
    indices_equal_theta = np.where(np.abs(traces - target_trace) < trace_tolerance)[0]

    if indices_equal_theta.size == 0:
        print(f"ERROR: No element found in G_np matching C3 trace ({target_trace:.3f} +/- {trace_tolerance}). Traces found: {traces}")
        raise RuntimeError(f"Cannot find C3 element in Icosahedral group by trace.")
    idx_equal_theta = indices_equal_theta[0]
    g_theta = G_np[idx_equal_theta]        # A C3 rotation from the I group
    g_div_gen = G_div_np[1]                # The canonical 120-degree C3 rotation

    print(f"DEBUG: Using element G_np[{idx_equal_theta}] (trace {traces[idx_equal_theta]:.5f}) to align with canonical C3.")

    try:
        # Compute eigenvectors/values for alignment using NumPy
        L_div_np, V_div_np = np.linalg.eig(g_div_gen)
        L_np, V_np = np.linalg.eig(g_theta)

        # Calculate W = V_div @ V_inv using NumPy pinv
        V_inv_np = np.linalg.pinv(V_np)
        W_np = np.matmul(V_div_np, V_inv_np)

        # Check and handle complex part
        if np.max(np.abs(W_np.imag)) > trace_tolerance:
             print(f"Warning: Alignment matrix W_np has significant imaginary component: {np.max(np.abs(W_np.imag)):.2e}. Using real part.")
        W_np = W_np.real # Take real part

        # Calculate W_inv_np using NumPy pinv or transpose
        identity_3x3_np = np.eye(3, dtype=internal_np_dtype)
        if np.allclose(np.matmul(W_np, W_np.T), identity_3x3_np, atol=1e-4): # Match tolerance
             W_inv_np = W_np.T
             # print("DEBUG: Alignment matrix W_np appears orthogonal. Using transpose for inverse.")
        else:
             print(f"[Warning] W_np is not orthogonal (max diff from I: {np.max(np.abs(np.matmul(W_np, W_np.T) - identity_3x3_np)):.2e}). Using pseudo-inverse for W_inv_np.")
             try:
                 W_inv_np = np.linalg.pinv(W_np)
                 if not np.allclose(np.matmul(W_inv_np, W_np), identity_3x3_np, atol=1e-4):
                     print("[Warning] Pseudo-inverse W_inv_np may be inaccurate.")
             except np.linalg.LinAlgError as e_pinv:
                 print(f"ERROR: np.linalg.pinv(W_np) failed: {e_pinv}. Falling back to W_np.T")
                 W_inv_np = W_np.T # Fallback

    except np.linalg.LinAlgError as e:
        print(f"ERROR: NumPy linalg computation failed during alignment: {e}")
        raise RuntimeError(f"Could not compute alignment matrix W_np: {e}")

    # 3. Transform G to the Canonical (C3-aligned) Frame using NumPy matmul
    # G_transformed = W @ G @ W_inv
    G_transformed_np = np.einsum('ij,gjk,kl->gil', W_np, G_np, W_inv_np, optimize=True)
    # print(f"DEBUG: Successfully transformed {G_transformed_np.shape[0]} matrices to canonical frame (NumPy).")

    # 4. Identify Coset Representatives in the Transformed Space using NumPy
    representatives_transformed_np = []
    processed_indices = set()
    distance_tolerance = 1e-4 # Match tolerance

    # print("DEBUG: Identifying cosets in transformed space (NumPy)...")
    num_cosets_found = 0
    for i in range(N): # Iterate through indices 0 to 59
        if i in processed_indices:
            continue

        num_cosets_found += 1
        rep_i_transformed = G_transformed_np[i]
        representatives_transformed_np.append(rep_i_transformed)
        processed_indices.add(i)

        for k in range(1, n_div): # Look for elements related by G_div_np[1], G_div_np[2]
            target_mat = np.matmul(rep_i_transformed, G_div_np[k])
            min_dist = np.inf
            best_match_idx = -1

            for j in range(N):
                if j not in processed_indices:
                    dist = np.linalg.norm(G_transformed_np[j] - target_mat)
                    if dist < distance_tolerance and dist < min_dist:
                        min_dist = dist
                        best_match_idx = j

            if best_match_idx != -1:
                processed_indices.add(best_match_idx)
            # else: # Optional warning if match not found (already present in original code)
            #     min_dist_overall = np.inf
            #     # ... (code to find minimum distance if needed for debugging) ...
            #     print(f"Warning: Could not find exact match (tol={distance_tolerance:.1e}) for rep idx {i} with G_div_np[{k}]. Closest dist: {min_dist_overall:.2e}")


    # print(f"DEBUG: Coset identification loop finished (NumPy). Found {len(representatives_transformed_np)} representatives.")

    if not representatives_transformed_np:
        print("ERROR: No representatives found (NumPy calculation)!")
        return torch.empty((0, 3, 3), device=device, dtype=dtype)

    # Stack the representatives found in the transformed space
    G_quot_transformed_np = np.array(representatives_transformed_np) # Shape (expected 20, 3, 3)

    expected_size = N // n_div # 20
    if G_quot_transformed_np.shape[0] != expected_size:
        print(f"ERROR: Expected {expected_size} pseudoquotient matrices, found {G_quot_transformed_np.shape[0]} (NumPy calculation).")
        return torch.empty((0, 3, 3), device=device, dtype=dtype)
    # else:
        # print(f"DEBUG: Found correct number ({G_quot_transformed_np.shape[0]}) of transformed representatives (NumPy).")

    # 5. Sort the *transformed* representatives lexicographically using NumPy
    try:
        sort_key_np = np.round(G_quot_transformed_np.reshape(expected_size, -1), decimals=5)
        sorted_indices = np.lexsort(sort_key_np.T)
        G_quot_transformed_sorted_np = G_quot_transformed_np[sorted_indices]
        # print("DEBUG: Sorted transformed representatives lexicographically (NumPy).")
    except Exception as sort_err:
        print(f"Warning: Sorting failed ({sort_err}). Using unsorted transformed representatives.")
        G_quot_transformed_sorted_np = G_quot_transformed_np

    # 6. Convert the final NumPy array to a PyTorch Tensor with desired dtype and device
    final_ops_tensor = torch.from_numpy(G_quot_transformed_sorted_np).to(device=device, dtype=dtype)

    # Cache the result before returning
    _point_group_cache[cache_key] = final_ops_tensor
    print(f"DEBUG: Successfully calculated and cached {final_ops_tensor.shape[0]} transformed I/C3 operators (NumPy style).")

    return final_ops_tensor


def _calculate_full_icosahedral_matrices_numpy(tree_depth: int = 5) -> np.ndarray:
    """
    Generates the 60 rotation matrices for the Icosahedral group (I)
    using generators and group multiplication, returning NumPy array.
    (Essentially the NumPy logic from symmetry_postprocess.py)
    """
    # Check cache first (optional, can rely on get_I_rotations cache)
    # cache_key = "I_full_60_numpy"
    # if cache_key in _point_group_cache:
    #     return _point_group_cache[cache_key]

    TAU_np = 0.5 * (1 + np.sqrt(5))
    generator_1 = np.array([[-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
    generator_2 = np.array([[0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]], dtype=np.float64)
    generator_3 = np.array([[0.5, -0.5 * TAU_np, 0.5 / TAU_np],
                            [0.5 * TAU_np, 0.5 / TAU_np, -0.5],
                            [0.5 / TAU_np, 0.5, 0.5 * TAU_np]], dtype=np.float64)
    generators = [generator_1, generator_2, generator_3]
    matrices = set()
    identity_np = np.eye(3, dtype=np.float64)
    matrices.add(tuple(identity_np.flatten()))

    current_set_matrices = [identity_np]

    for _ in range(tree_depth + 2): # Match iterations
        new_products_set = set()
        next_set_matrices = []
        num_before = len(matrices)

        for mat in current_set_matrices:
            for gen in generators:
                product_mat = np.round(mat @ gen, decimals=6) # Use numpy matmul
                product_tuple = tuple(product_mat.flatten())
                if product_tuple not in matrices:
                    matrices.add(product_tuple)
                    new_products_set.add(product_tuple)
                    # Store the matrix itself for the next iteration
                    next_set_matrices.append(product_mat)

        current_set_matrices.extend(next_set_matrices) # Add newly generated matrices
        if len(matrices) == num_before or len(matrices) >= 60:
            break

    # Convert unique tuples back to matrices
    mats_list = [np.array(m).reshape(3, 3) for m in matrices]
    final_ops = np.array(mats_list)

    if final_ops.shape[0] != 60:
         print(f"[Warning symmetry_awareness] Expected 60 I matrices, got {final_ops.shape[0]}.")

    # Sort lexicographically based on rounded values
    sort_key = np.round(final_ops.reshape(final_ops.shape[0], -1), decimals=5)
    sorted_indices = np.lexsort(sort_key.T)
    final_ops_sorted = final_ops[sorted_indices]

    # _point_group_cache[cache_key] = final_ops_sorted # Optional cache
    return final_ops_sorted


# --- Functions removed as they seem unused based on the request ---
# def subsample(...)
# def symmetrize_XCS(...)
# def get_radius_ratio(...)
# def calculate_ring_radius(...)
