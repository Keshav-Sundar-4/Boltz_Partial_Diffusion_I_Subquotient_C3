#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import random
from collections import defaultdict # Import defaultdict

# Mapping from three-letter codes to one-letter codes for standard amino acids.
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def parse_atom_line(line):
    """
    Parse a PDB ATOM line into its components.
    """
    record = line[0:6].strip()
    try:
        serial = int(line[6:11])
    except ValueError:
        serial = 0 # Handle potential non-integer serial numbers if needed
    name = line[12:16].strip()
    altLoc = line[16]
    resName = line[17:20].strip()
    chainID = line[21]
    try:
        resSeq = int(line[22:26])
    except ValueError:
        resSeq = 0 # Handle potential non-integer residue sequence numbers
    iCode = line[26].strip()
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        x, y, z = 0.0, 0.0, 0.0 # Handle potential non-float coordinates
    occupancy = line[54:60].strip()
    tempFactor = line[60:66].strip()
    element = line[76:78].strip()
    charge = line[78:80].strip()
    return {
        'record': record,
        'serial': serial,
        'name': name,
        'altLoc': altLoc,
        'resName': resName,
        'chainID': chainID,
        'resSeq': resSeq,
        'iCode': iCode,
        'x': x,
        'y': y,
        'z': z,
        'occupancy': occupancy,
        'tempFactor': tempFactor,
        'element': element,
        'charge': charge,
        'original_line': line # Store original line for potential formatting issues
    }

def format_atom_line(atom):
    """
    Format an atom dictionary back into a PDB-formatted ATOM line.
    Tries to preserve original spacing as much as possible, but updates key fields.
    """
    # Use default values if parsing failed or values are empty
    occ = 1.00
    if atom['occupancy']:
        try:
            occ = float(atom['occupancy'])
        except ValueError:
            pass # Keep default
    temp = 0.00
    if atom['tempFactor']:
        try:
            temp = float(atom['tempFactor'])
        except ValueError:
             pass # Keep default

    # Basic PDB format string
    line_format = "{record:<6}{serial:>5} {name:<4}{altLoc}{resName:>3} {chainID}{resSeq:>4}{iCode:<1}   {x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{temp:>6.2f}          {element:>2}{charge:>2}"

    # Pad name correctly: If len(name) < 4, pad with spaces on the right. If atom name has 4 chars (e.g., ' ZN '), it needs alignment.
    atom_name_padded = atom['name']
    if len(atom_name_padded) < 4:
        atom_name_padded = (atom_name_padded + "   ")[:4] # Left align within 4 chars
    elif len(atom_name_padded) == 4 and not atom_name_padded.startswith(" "): # Common case like ' CA '
         atom_name_padded = " " + atom_name_padded[:3] # Assume standard PDB format of space + 3 letters


    return line_format.format(
        record=atom['record'],
        serial=atom['serial'] if atom['serial'] is not None else 0,
        name=atom_name_padded,
        altLoc=atom['altLoc'] if atom['altLoc'] else ' ',
        resName=atom['resName'],
        chainID=atom['chainID'], # This will be the *new* chain ID
        resSeq=atom['resSeq'] if atom['resSeq'] is not None else 0,
        iCode=atom['iCode'] if atom['iCode'] else ' ',
        x=atom['x'],
        y=atom['y'],
        z=atom['z'],
        occ=occ,
        temp=temp,
        element=atom['element'] if atom['element'] else '  ',
        charge=atom['charge'] if atom['charge'] else '  '
    )


def extract_sequence(chain_atoms):
    """
    Extract the protein sequence from a chain based on unique residues.
    """
    residues = {}
    # Use a set to track unique residue identifiers (seq + icode) to maintain order
    seen_res_keys = set()
    ordered_res_keys = []
    for atom in chain_atoms:
        # Only consider standard amino acid residues for sequence
        if atom['resName'] in THREE_TO_ONE:
            res_key = (atom['resSeq'], atom['iCode'])
            if res_key not in seen_res_keys:
                seen_res_keys.add(res_key)
                ordered_res_keys.append(res_key)
                residues[res_key] = atom['resName']

    # Sort keys primarily by residue sequence number, secondarily by insertion order (captured by list)
    # This handles cases where resSeq might not be strictly consecutive but order matters
    sorted_keys = sorted(ordered_res_keys, key=lambda x: x[0])

    seq = ""
    for key in sorted_keys:
        three_letter = residues[key].upper()
        one_letter = THREE_TO_ONE.get(three_letter, 'X') # Use 'X' for unknown/non-standard
        seq += one_letter
    return seq

def find_smallest_equilateral_triangle(ref, candidate_ids, chain_com):
    """
    For a given reference chain id and candidate set (other chain IDs),
    iterate over all pairs to form triangles with the reference.
    For each triangle, compute:
       - The three pairwise distances.
       - Perimeter = d1 + d2 + d3.
       - Error = max(d1, d2, d3) - min(d1, d2, d3).
    We then choose the triangle using lexicographic ordering on (perimeter, error),
    which prioritizes the smallest (by overall size) nearly equilateral triangle.
    Returns the triangle (as a tuple of three chain IDs) and the associated score tuple.
    """
    best_triangle = None
    best_score = None  # (perimeter, error)
    candidates = list(candidate_ids)
    if len(candidates) < 2:
        return None, None # Not enough candidates to form a triangle

    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            cid1 = candidates[i]
            cid2 = candidates[j]
            # Ensure all chain IDs exist in chain_com before calculating distance
            if ref not in chain_com or cid1 not in chain_com or cid2 not in chain_com:
                continue # Skip if any COM is missing

            d1 = np.linalg.norm(chain_com[ref] - chain_com[cid1])
            d2 = np.linalg.norm(chain_com[ref] - chain_com[cid2])
            d3 = np.linalg.norm(chain_com[cid1] - chain_com[cid2])

            perimeter = d1 + d2 + d3
            error = max(d1, d2, d3) - min(d1, d2, d3)
            score = (perimeter, error)

            if best_score is None or score < best_score:
                best_score = score
                best_triangle = (ref, cid1, cid2) # Keep order: ref, then the pair

    return best_triangle, best_score

def main():
    parser = argparse.ArgumentParser(
        description="Process an icosahedral protein PDB for partial diffusion setup using two smallest neighboring equilateral triangles with specific reordering."
    )
    parser.add_argument("pdb_file", help="Input PDB file")
    args = parser.parse_args()

    # --- Read and parse the PDB file ---
    try:
        with open(args.pdb_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        sys.exit(f"Error reading file: {e}")

    atoms_by_chain = defaultdict(list)
    all_atoms = []
    original_chain_ids = set()

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"): # Include HETATM in case they are part of chains
            atom = parse_atom_line(line)
            if atom['serial'] is not None: # Basic check for valid parse
                 atoms_by_chain[atom['chainID']].append(atom)
                 all_atoms.append(atom)
                 original_chain_ids.add(atom['chainID'])

    if not all_atoms:
        sys.exit("No ATOM/HETATM lines found or parsed successfully in the provided PDB file.")
    print(f"Found {len(original_chain_ids)} unique original chain IDs: {sorted(list(original_chain_ids))}")


    # --- Step 1: Re-center the structure (rigid body translation) ---
    if not all_atoms:
         sys.exit("Cannot re-center: No atoms found.")
    coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in all_atoms])
    overall_com = np.mean(coords, axis=0)
    # Apply translation to all atoms
    for atom in all_atoms:
        atom['x'] -= overall_com[0]
        atom['y'] -= overall_com[1]
        atom['z'] -= overall_com[2]
    # Update the coordinates in atoms_by_chain as well, since all_atoms holds references
    print(f"Structure re-centered around COM: {overall_com}")


    # --- Step 2: Compute each chain's COM (using updated coordinates) ---
    chain_com = {}
    valid_chains = {} # Store only chains with valid COM
    for cid, atom_list in atoms_by_chain.items():
        if not atom_list: continue # Skip empty chains
        coords = np.array([[a['x'], a['y'], a['z']] for a in atom_list])
        if coords.size > 0:
            chain_com[cid] = np.mean(coords, axis=0)
            valid_chains[cid] = atom_list # Keep track of atoms for valid chains
        else:
            print(f"Warning: Chain {cid} has no coordinates for COM calculation. Skipping.")

    all_valid_chain_ids = list(valid_chains.keys())
    if len(all_valid_chain_ids) < 6:
        sys.exit(f"Not enough chains with valid COMs to form two triangles (need at least 6, found {len(all_valid_chain_ids)}).")

    # --- Step 3: Form the first triangle ---
    # Try finding the overall best triangle first
    global_best_triangle = None
    global_best_score = None

    for ref_candidate in all_valid_chain_ids:
        candidate_pool = set(all_valid_chain_ids) - {ref_candidate}
        current_triangle, current_score = find_smallest_equilateral_triangle(ref_candidate, candidate_pool, chain_com)
        if current_triangle:
             if global_best_score is None or current_score < global_best_score:
                  global_best_score = current_score
                  global_best_triangle = current_triangle

    if global_best_triangle is None:
        sys.exit("Could not form any valid first equilateral triangle.")

    triangle1 = global_best_triangle
    score1 = global_best_score
    print(f"Selected First triangle (chain IDs): {triangle1} with score (perimeter, error): {score1}")


    # --- Step 4: Form the second triangle ---
    candidate_pool2 = set(all_valid_chain_ids) - set(triangle1)
    if len(candidate_pool2) < 3:
        sys.exit(f"Not enough remaining chains ({len(candidate_pool2)}) to form the second triangle.")

    # Find the best starting chain for the second triangle (closest to centroid of first)
    triangle1_coords = np.array([chain_com[cid] for cid in triangle1])
    centroid1 = np.mean(triangle1_coords, axis=0)

    best_ref2 = None
    min_dist_to_centroid = float('inf')

    for cid in candidate_pool2:
        dist = np.linalg.norm(chain_com[cid] - centroid1)
        if dist < min_dist_to_centroid:
            min_dist_to_centroid = dist
            best_ref2 = cid

    if best_ref2 is None:
         sys.exit("Could not find a suitable reference chain for the second triangle.")

    # Now find the best triangle using this reference chain
    candidate_pool_for_triangle2 = candidate_pool2 - {best_ref2}
    triangle2, score2 = find_smallest_equilateral_triangle(best_ref2, candidate_pool_for_triangle2, chain_com)

    if triangle2 is None:
        sys.exit("Could not form the second equilateral triangle.")
    print(f"Selected Second triangle (chain IDs): {triangle2} with score (perimeter, error): {score2}")


    # --- Step 5: Reorder chains based on COM distances ---
    print("\n--- Reordering Chains ---")
    t1_list = list(triangle1) # Convert to lists for modification
    t2_list = list(triangle2)

    # Step A: Find closest inter-triangle pair
    min_dist = float('inf')
    closest_pair = (None, None)
    print("Calculating inter-triangle distances:")
    for c1 in t1_list:
        for c2 in t2_list:
            dist = np.linalg.norm(chain_com[c1] - chain_com[c2])
            print(f"  Dist({c1}, {c2}) = {dist:.3f}")
            if dist < min_dist:
                min_dist = dist
                closest_pair = (c1, c2)

    print(f"Closest pair: {closest_pair} with distance {min_dist:.3f}")
    closest_c1, closest_c2 = closest_pair

    # Step B: Reorder based on closest pair (make them index 0)
    idx1 = t1_list.index(closest_c1)
    t1_list[0], t1_list[idx1] = t1_list[idx1], t1_list[0] # Swap closest_c1 to position 0
    idx2 = t2_list.index(closest_c2)
    t2_list[0], t2_list[idx2] = t2_list[idx2], t2_list[0] # Swap closest_c2 to position 0
    print(f"After swapping closest to front: T1={t1_list}, T2={t2_list}")

    # Step C & D: Reorder T2 based on distance to T1[1] (new chain 2)
    chain_2 = t1_list[1]
    chain_5 = t2_list[1]
    chain_6 = t2_list[2]

    dist_2_5 = np.linalg.norm(chain_com[chain_2] - chain_com[chain_5])
    dist_2_6 = np.linalg.norm(chain_com[chain_2] - chain_com[chain_6])
    print(f"Dist({chain_2}, {chain_5}) = {dist_2_5:.3f}")
    print(f"Dist({chain_2}, {chain_6}) = {dist_2_6:.3f}")

    if dist_2_6 > dist_2_5:
        # Swap chain_5 and chain_6 so chain_5 is closer to chain_2
        t2_list[1], t2_list[2] = t2_list[2], t2_list[1]
        print(f"Swapped T2[1] and T2[2] as {chain_6} was further from {chain_2}.")
        print(f"Final T2 order: {t2_list}")
    else:
        print(f"No swap needed for T2[1] and T2[2] as {chain_5} is closer to/equidistant from {chain_2}.")

    # Step E: Define the final ordered list of original chain IDs
    selected_chains_original_ids = t1_list + t2_list
    print(f"Final ordered list of original chain IDs: {selected_chains_original_ids}")

    # --- Step 6: Assign new chain IDs and prepare for output ---
    new_chain_ids_map = {} # Map from original ID to new ID (A-F)
    final_output_atoms = []
    target_new_ids = ['A', 'B', 'C', 'D', 'E', 'F']

    for i, original_cid in enumerate(selected_chains_original_ids):
        new_id = target_new_ids[i]
        new_chain_ids_map[original_cid] = new_id
        print(f"  Mapping Original Chain {original_cid} -> New Chain {new_id}")
        # Collect atoms for this chain and update their chainID
        for atom in valid_chains[original_cid]:
            atom['chainID'] = new_id # Update the chain ID in the atom dictionary
            final_output_atoms.append(atom)

    # Ensure atom serial numbers are unique and sequential in the output
    for i, atom in enumerate(final_output_atoms):
        atom['serial'] = i + 1

    # --- Step 7: Write out the new PDB file ---
    out_pdb_filename = args.pdb_file.replace(".pdb", "_partial_diffusion_setup.pdb")
    try:
        with open(out_pdb_filename, "w") as out_f:
            for atom in final_output_atoms:
                 out_f.write(format_atom_line(atom) + "\n")
            out_f.write("TER\n") # Add TER card between chains if needed, simplified here
            out_f.write("END\n")
    except Exception as e:
        sys.exit(f"Error writing output PDB file '{out_pdb_filename}': {e}")


    # --- Step 8: Compute the radius ---
    # Use the COM distance of the *first* chain in the *final* ordering (Chain A).
    # Ensure chain_com contains the re-centered COM.
    first_chain_original_id = selected_chains_original_ids[0]
    if first_chain_original_id in chain_com:
        radius = np.linalg.norm(chain_com[first_chain_original_id])
        print(f"\nCalculated radius based on COM distance of Chain A ({first_chain_original_id}): {radius:.3f}")
    else:
        print(f"Warning: Could not find COM for chain {first_chain_original_id}. Using radius 0.0.")
        radius = 0.0


    # --- Step 9: Extract sequences for the selected chains in the *final* order ---
    sequences_yaml = ""
    for original_cid in selected_chains_original_ids:
        new_protein_id = new_chain_ids_map[original_cid] # Get the A-F ID
        # Extract sequence using the atoms associated with the *original* chain ID
        # Ensure we use the atom list from 'valid_chains' which contains the original data
        seq = extract_sequence(valid_chains[original_cid])
        sequences_yaml += f"  - protein:\n      id: {new_protein_id}\n      sequence: {seq}\n      msa: empty\n"


    # --- Step 10: Write out the YAML file ---
    yaml_content = f"""version: 1
radius: {radius:.3f}
symmetry: C3 # Assuming C3 symmetry based on triangular setup - adjust if needed

sequences:
{sequences_yaml}"""
    out_yaml_filename = args.pdb_file.replace(".pdb", ".yaml")
    try:
        with open(out_yaml_filename, "w") as yaml_f:
            yaml_f.write(yaml_content)
    except Exception as e:
        sys.exit(f"Error writing YAML file '{out_yaml_filename}': {e}")

    print(f"\nOutput PDB file written to: {out_pdb_filename}")
    print(f"Output YAML file written to: {out_yaml_filename}")

if __name__ == "__main__":
    main()
