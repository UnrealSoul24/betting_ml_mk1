
import os

filepath = 'src/batch_predict.py'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# We expect line 760 (index 759) to be the return statement
# We expect line 857 (index 856) to be the def _get_season_stats

idx_ret = 760 - 1
idx_def = 857 - 1

print(f"Line {idx_ret+1}: {lines[idx_ret].rstrip()}")
try:
    print(f"Line {idx_def+1}: {lines[idx_def].rstrip()}")
except IndexError:
    print(f"Line {idx_def+1}: <Out of bounds>")

# Scan for the second return statement just in case line numbers shifted slightly
# We look for duplicates of `return {'predictions': sorted_res, 'trixie': trixie}`
target_ret = "        return {'predictions': sorted_res, 'trixie': trixie}"
found_indices = [i for i, line in enumerate(lines) if target_ret.strip() in line]

print(f"Found return statements at indices: {found_indices}")

if len(found_indices) >= 2:
    # We want to keep up to the FIRST return statement (inclusive)
    # And then resume at the LAST helper function definition?
    # Or just cut out the middle.
    
    first_ret_idx = found_indices[0]
    second_ret_idx = found_indices[1]
    
    # We want to keep 0..first_ret_idx
    # And we want to keep (second_ret_idx + some_offset) .. end
    # The duplicate block ends at second_ret_idx.
    # The helper functions start after.
    
    # Check what is after second_ret_idx
    print(f"Lines after second return:")
    print(lines[second_ret_idx+1:second_ret_idx+5])
    
    # It seems there is a blank line and then `def _get_season_stats`.
    # Let's find index of `def _get_season_stats` after first_ret_idx
    
    def_idx = -1
    for i in range(first_ret_idx, len(lines)):
        if "def _get_season_stats" in lines[i]:
            def_idx = i
            break
            
    if def_idx != -1:
        print(f"Found next function at {def_idx}")
        
        # New content
        new_lines = lines[:first_ret_idx+1] + ["\n"] + lines[def_idx:]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        print("Fixed file.")
    else:
        print("Could not find _get_season_stats function.")
else:
    print("Did not find duplicate return statements. Modification might not be needed or logic mismatch.")
