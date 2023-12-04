import numpy as np

# Load the .npz file
data = np.load('eval_results.npz',allow_pickle=True)

# Print the content of the file
print(data.files)
for file in data.files:
    print(f'{file}: {data[file]}')
