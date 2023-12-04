import os
import numpy as np

# Get a list of all npz files in the current directory
npz_files = [file for file in os.listdir() if file.endswith(".npz")]

# Initialize dictionaries to store task counts for each file and overall counts
overall_task_counts = {}
file_task_counts = {}

# Process each npz file
for npz_file in npz_files:
    # Load the data from the npz file
    data = np.load(npz_file, allow_pickle=True)

    # Initialize counts for this file
    file_task_counts[npz_file] = {}

    # Process each key in the file
    for key in data.keys():
        # Get the entries for the current key
        entries = data[key]
        print(entries,'entries')
        entries = entries[:-1]
        # Process each entry in the key
        for task_name in entries:
            task_name = task_name[0]
            print(task_name,'task_name')

            # Update overall counts
            if task_name not in overall_task_counts:
                overall_task_counts[task_name] = 1
            else:
                overall_task_counts[task_name] += 1

            # Update file-specific counts
            if task_name not in file_task_counts[npz_file]:
                file_task_counts[npz_file][task_name] = 1
            else:
                file_task_counts[npz_file][task_name] += 1

# Save the results to a new npz file
# np.savez('eval_results.npz', overall_task_counts=overall_task_counts, file_task_counts=file_task_counts)
# Save the results to a text file
with open('eval_results.txt', 'w') as txt_file:
    # Write overall task counts
    txt_file.write("Overall Task Counts:\n")
    for task_name, count in overall_task_counts.items():
        txt_file.write(f"{task_name}: {count}\n")

    txt_file.write("\n")

    # Write file-specific task counts
    txt_file.write("File-Specific Task Counts:\n")
    for npz_file, counts in file_task_counts.items():
        txt_file.write(f"{npz_file}:\n")
        task_num = 0
        for task_name, count in counts.items():
            txt_file.write(f"  {task_name}: {count}\n")
            task_num += 1
        txt_file.write(f"Total task count: {task_num}\n")
        txt_file.write("\n")