import os

path = 'experiments/ranks/ranking/'
path_fidelity = 'experiments/ranks/fidelity/'
files = os.listdir(path)

files_with_reasoner = [f for f in files if 'no_reasoner' not in f and 'fidelity' not in f]
files_no_reasoner = [f for f in files if 'no_reasoner' in f]


for f in files_with_reasoner:
    model_name = f.split('-')[2] 
    grounder = f.split('-')[1]
    metric = f.split('-')[-1]
    no_reasoner = f.replace(model_name, 'no_reasoner').replace(grounder, 'backward_0_1').replace(metric, '')

    found = False
    for f2 in files_no_reasoner:
        if no_reasoner in f2:
            # Read files and store key-value pairs, allowing repeated keys
            with open(path + f, 'r') as file:
                model_data = [line.strip().split(':') for line in file.readlines()]
            
            with open(path + f2, 'r') as file:
                kge_data = [line.strip().split(':') for line in file.readlines()]
            
            # print('model_data:', model_data)
            # print('kge_data:', kge_data)

            # Calculate fidelity
            fidelity_count = 0
            total = 0
            for model_line, kge_line in zip(model_data, kge_data):
                query_model, ranked_query_model = model_line[0], model_line[1]
                query_kge, ranked_query_kge = kge_line[0], kge_line[1]
                # print(f'\nQuery: {query_model}, {query_kge}')
                # print(f'Ranks: {ranked_query_model}, {ranked_query_kge}')
                assert query_model == query_kge, 'Queries do not match'
                if ranked_query_model == ranked_query_kge:
                    fidelity_count += 1
                total += 1

            fidelity = fidelity_count / total if total > 0 else 0
            print(f'Fidelity for {f} is {fidelity:.3f}')

            # Write fidelity to file
            os.makedirs(path_fidelity, exist_ok=True)
            with open(f"{path_fidelity}{f.replace('.txt','')}_fidelity_{round(fidelity, 3)}.txt", 'w') as file:
                file.write(f"{f}: {fidelity:.3f}\n")

            found = True    
            # print(oisdb)

    if not found:
        print(f'No counterpart for {f}, it should be {no_reasoner}')

