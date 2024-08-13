import os

# Specify the directory containing the files
directory = '/Users/domalberts/Documents/GitHub/hetero_swarm/full_env_verify'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Construct the full file path
    old_file = os.path.join(directory, filename)

    # Skip if it's not a file
    if not os.path.isfile(old_file):
        continue

    # Create the new filename
    new_filename = 'full_env_' + filename
    new_file = os.path.join(directory, new_filename)

    # Rename the file
    os.rename(old_file, new_file)

print("Renaming completed!")