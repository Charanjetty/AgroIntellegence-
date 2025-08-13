import numpy as np

try:
    with np.load('croprecommender_mlp.npz') as data:
        print("Keys found in the .npz file:")
        for key in data.keys():
            print(f"- {key}")
except FileNotFoundError:
    print("Error: The file 'croprecommender.mlp.npz' was not found.")