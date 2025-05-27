# read all the files in results/ind_runs and get the names of all the files. 
# Copy all the information from the files to a new file in results/ind_runs_copy

import os
import shutil
import ast
# Get the names of all the files in results/ind_runs
files = os.listdir('./results/indiv_runs')

# Create a new directory to store the copied files
if not os.path.exists('./results/indiv_runs_copy'):
    os.makedirs('./results/indiv_runs_copy')

# Copy all the information from the files to a new file in ./results/indiv_runs_copy
# Copy line by line. If the line starts with 'All data' or 'Signature', then modify them 
new_key_signature = ['dataset_name', 'grounder', 'kge', 'model_name', 'rule_miner','neg', 'e'] 
for file in files:
    # spliit the name by '-', rewrite the name skipping the 2nd and 4th to last element
    new_name = '-'.join(file.split('-')[:6] + file.split('-')[8:] )
    print('new name:', new_name)

    with open('./results/indiv_runs/' + file, 'r') as f:
        with open('./results/indiv_runs_copy/' + new_name, 'w') as f_copy:
            for line in f:
                if line.startswith('All data'): # create a dictionary with the line. 
                    dic = {}
                    for i in line.split(';')[1:]:
                        # print('line element:', i)
                        key, value = i.split(':')
                        dic[key] = value

                        if key == 'keys_signature': # modify it to get the new one
                            # the old key signature is a list of strings
                            old_key_signature = ast.literal_eval(value)
                            dic[key] = new_key_signature

                        if key == 'run_signature': # modify it to get the new one
                            dic_signature = {}
                            old_signature = value
                            # print('old signature:', old_signature)
                            # print('old key signature:', old_key_signature)
                            # get a dic with the old signature
                            for j,element in enumerate(old_signature.split('-')):
                                # print('j',j,'old_key_signature[j]', old_key_signature[j], 'element', element)
                                dic_signature[old_key_signature[j]] = element
                            # write the new signature with only keys from key_signature
                            new_signature = '-'.join([dic_signature[key] for key in new_key_signature])
                            # print('the new signature is:', new_signature)
                            # print('the new key signature is:', new_key_signature)
                        # write the new line by joining the keys and values of the dictionary by a ':', and the elements of the dictionary by a ';'
                    line = 'All data;' + ';'.join([str(key) + ':' + str(value) for key,value in dic.items()]) 
                    print('\nnew line:', line)

                if line.startswith('Signature'):
                    line = 'Signature:' + new_signature + '\n'
                    print('new line:', line)
                f_copy.write(line)  
