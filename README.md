# Boltz_Partial_Diffusion_I_Subquotient_C3

An important note is that partial_diffusion_setup.py will create 2 triangles. Ensure that it creates [A,B,C] matches up with [D,E,F]. Importantly, when using the I subquotient C3 matrices (size=20), ensure that A maps to D, B maps to E, and D maps to F. The code should be doing this right now, but just verify using a few examples, as it has the potential to break down the line functionality. 


# Install matplotlib
pip install matplotlib

# File Replacements:

Note that these file paths are based on the Filezilla boltz env folder. 

## main.py
Replace: /work/keshavsundar/env/boltz_glycan/lib/python3.10/site-packages/boltz/main.py

## model.py
Replace: /work/keshavsundar/env/boltz_glycan/lib/python3.10/site-packages/boltz/model/model.py

## diffusion.py
Replace: /work/keshavsundar/env/boltz_glycan/lib/python3.10/site-packages/boltz/model/modules/diffusion.py

## symmetry.py
Add: /work/keshavsundar/env/boltz_glycan/lib/python3.10/site-packages/boltz/data/module/symmetry_awareness.py

## yaml.py
Add: /work/keshavsundar/env/boltz_glycan/lib/python3.10/site-packages/boltz/data/parse/yaml.py

## example_boltz_symmetry_test.sh
Use this file as an example. However, importantly, the setup file generates an accurate yaml. So you can simply use this. Be wary, as this will require changes to the shell script, as you need to call a different yaml if you want to automate this. 

## symmetry_postprocess.py
This is a general postprocessing script to create the full subquotient icosahedron. Note that this will only use chain A of the input file to create said icosahedron. The way to run the script is as follows: 'python symmetry_postprocess.py <input_name>.cif <output_name>.cif --group I'

## RMSD calculation
Yang TO DO

# General Info
Something important is that the partial diffusion set up file creates an output pdb file, which will be your input pdb file. Then, the yaml file it outputs can be the input yaml file in one-shot. No changes are required to the yaml file. Additionally, note that you can cange the amount of partial diffusion steps on line 740. Currently the total number of steps is 200, so the default (50) is t=0.25. diffusion.py is the the file.

