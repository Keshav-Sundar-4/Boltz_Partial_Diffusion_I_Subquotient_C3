#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import random

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
    serial = int(line[6:11])
    name = line[12:16].strip()
    altLoc = line[16]
    resName = line[17:20].strip()
    chainID = line[21]
    resSeq = int(line[22:26])
    iCode = line[26].strip()
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
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
        'charge': charge
    }

def format_atom_line(atom):
    """
    Format an atom dictionary back into a PDB-formatted ATOM line.
    """
    occ = float(atom['occupancy']) if atom['occupancy'] else 1.00
    temp = float(atom['tempFactor']) if atom['tempFactor'] else 0.00
    return f"{atom['record']:<6}{atom['serial']:>5} {atom['name']:<4}{atom['altLoc']}{atom['resName']:>3} {atom['chainID']}" \
           f"{atom['resSeq']:>4}{atom['iCode']:<1}   {atom['x']:>8.3f}{atom['y']:>8.3f}{atom['z']:>8.3f}" \
           f"{occ:>6.2f}{temp:>6.2f}          {atom['element']:>2}{atom['charge']:>2}"

def extract_sequence(chain_atoms):
    """
    Extract the protein sequence from a chain based on unique residues.
    """
    residues = {}
    for atom in chain_atoms:
        res_key = (atom['resSeq'], atom['iCode'])
        if res_key not in residues:
            residues[res_key] = atom['resName']
    sorted_keys = sorted(residues.keys(), key=lambda x: (x[0], x[1]))
    seq = ""
    for key in sorted_keys:
        three_letter = residues[key].upper()
        one_letter = THREE_TO_ONE.get(three_letter, 'X')
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
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            cid1 = candidates[i]
            cid2 = candidates[j]
            d1 = np.linalg.norm(chain_com[ref] - chain_com[cid1])
            d2 = np.linalg.norm(chain_com[ref] - chain_com[cid2])
            d3 = np.linalg.norm(chain_com[cid1] - chain_com[cid2])
            perimeter = d1 + d2 + d3
            error = max(d1, d2, d3) - min(d1, d2, d3)
            score = (perimeter, error)
            if best_score is None or score < best_score:
                best_score = score
                best_triangle = (ref, cid1, cid2)
    return best_triangle, best_score

def main():
    parser = argparse.ArgumentParser(
        description="Process an icosahedral protein PDB for partial diffusion setup using two smallest neighboring equilateral triangles."
    )
    parser.add_argument("pdb_file", help="Input PDB file")
    args = parser.parse_args()

    # --- Read and parse the PDB file ---
    try:
        with open(args.pdb_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        sys.exit(f"Error reading file: {e}")
    atoms = []
    for line in lines:
        if line.startswith("ATOM"):
            atoms.append(parse_atom_line(line))
    if not atoms:
        sys.exit("No ATOM lines found in the provided PDB file.")

    # --- Step 1: Re-center the structure (rigid body translation) ---
    total = np.array([0.0, 0.0, 0.0])
    for atom in atoms:
        total += np.array([atom['x'], atom['y'], atom['z']])
    overall_com = total / len(atoms)
    for atom in atoms:
        atom['x'] -= overall_com[0]
        atom['y'] -= overall_com[1]
        atom['z'] -= overall_com[2]

    # --- Step 2: Group atoms by chain and compute each chain's COM ---
    chains = {}
    for atom in atoms:
        cid = atom['chainID']
        chains.setdefault(cid, []).append(atom)
    if len(chains) < 3:
        sys.exit("Not enough chains to form a triangle.")
    chain_com = {}
    for cid, atom_list in chains.items():
        coords = np.array([[a['x'], a['y'], a['z']] for a in atom_list])
        chain_com[cid] = np.mean(coords, axis=0)
    all_chain_ids = list(chains.keys())
    if len(all_chain_ids) < 6:
        sys.exit("Not enough chains to form two triangles (need at least 6).")

    # --- Step 3: Form the first triangle ---
    # Pick one COM at random.
    random.seed()
    ref1 = random.choice(all_chain_ids)
    candidate_pool1 = set(all_chain_ids) - {ref1}
    triangle1, score1 = find_smallest_equilateral_triangle(ref1, candidate_pool1, chain_com)
    if triangle1 is None:
        sys.exit("Could not form the first equilateral triangle.")
    print(f"First triangle (chain IDs): {triangle1} with score (perimeter, error): {score1}")

    # --- Step 4: Form the second triangle ---
    # From COMs not in the first triangle, choose the one whose COM is closest to the centroid of the first triangle.
    triangle1_coords = np.array([chain_com[cid] for cid in triangle1])
    centroid1 = np.mean(triangle1_coords, axis=0)
    candidate_pool2 = set(all_chain_ids) - set(triangle1)
    best_candidate = None
    best_distance = None
    for cid in candidate_pool2:
        d = np.linalg.norm(chain_com[cid] - centroid1)
        if best_distance is None or d < best_distance:
            best_distance = d
            best_candidate = cid
    if best_candidate is None:
        sys.exit("Could not find a candidate for the second triangle.")
    candidate_pool_for_triangle2 = candidate_pool2 - {best_candidate}
    triangle2, score2 = find_smallest_equilateral_triangle(best_candidate, candidate_pool_for_triangle2, chain_com)
    if triangle2 is None:
        sys.exit("Could not form the second equilateral triangle.")
    print(f"Second triangle (chain IDs): {triangle2} with score (perimeter, error): {score2}")

    # --- Step 5: Combine the two triangles (preserving order) ---
    selected_chains = list(triangle1) + list(triangle2)
    if len(set(selected_chains)) != 6:
        sys.exit("The selected chains do not form 6 unique subunits.")
    print(f"Selected chains for output (ordered): {selected_chains}")

    # --- Step 5.1: Reassign chain names based on the triangle order ---
    # First triangle gets A, B, C; second triangle gets D, E, F.
    new_chain_ids = {}
    for i, cid in enumerate(selected_chains):
        new_chain_ids[cid] = chr(65 + i)  # A, B, C, D, E, F
    print("Reassigned chain IDs based on triangle order:")
    for old, new in new_chain_ids.items():
        print(f"  Original chain {old} -> New chain {new}")

    # --- Step 6: Write out the new PDB file with atoms from the 6 selected chains ---
    out_pdb_filename = args.pdb_file.replace(".pdb", "_partial_diffusion_setup.pdb")
    try:
        with open(out_pdb_filename, "w") as out_f:
            for atom in atoms:
                if atom['chainID'] in new_chain_ids:
                    # update atom's chainID to the new value
                    atom['chainID'] = new_chain_ids[atom['chainID']]
                    out_f.write(format_atom_line(atom) + "\n")
            out_f.write("END\n")
    except Exception as e:
        sys.exit(f"Error writing output PDB file: {e}")

    # --- Step 7: Compute the radius ---
    # Use the COM distance of any random chain (from the complete set, not just the selected ones),
    # ensuring it is calculated post re-centering.
    random_chain = random.choice(list(chains.keys()))
    radius = np.linalg.norm(chain_com[random_chain])

    # --- Step 8: Independently extract the sequence for each selected chain ---
    # Follow the order of selected_chains so that the first 3 are from the first triangle and the next 3 from the second.
    sequences_yaml = ""
    for cid in selected_chains:
        protein_id = new_chain_ids[cid]
        seq = extract_sequence(chains[cid])
        sequences_yaml += f"  - protein:\n      id: {protein_id}\n      sequence: {seq}\n      msa: empty\n"

    # --- Step 9: Write out the YAML file ---
    yaml_content = f"""version: 1
radius: {radius:.3f}
symmetry: I

sequences:
{sequences_yaml}"""
    out_yaml_filename = args.pdb_file.replace(".pdb", ".yaml")
    try:
        with open(out_yaml_filename, "w") as yaml_f:
            yaml_f.write(yaml_content)
    except Exception as e:
        sys.exit(f"Error writing YAML file: {e}")

    print(f"Output PDB file written to: {out_pdb_filename}")
    print(f"Output YAML file written to: {out_yaml_filename}")

if __name__ == "__main__":
    main()
