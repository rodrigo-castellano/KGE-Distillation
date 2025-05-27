# take all the constants from train.txt for every line, with format predicate(constant1,constant2).
# write the unique constants to domain2constants.txt
dataset = 'kinship'
# dataset = 'kinship_family'
# path = './train.txt'
# ctes_path = './domain2constants.txt'
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
path = current_dir + '/'+dataset+'/train.txt'
ctes_path = current_dir + '/'+dataset+'/domain2constants.txt'
predicates_path = current_dir + '/'+dataset+'/relations.txt'

constants = set()
predicates = set()
with open(path, 'r') as f: 
    for line in open(path, 'r'):
        predicate = line.split('(')[0]
        consant1 = line.split('(')[1].split(',')[0]
        consant2 = line.split('(')[1].split(',')[1].split(')')[0]
        # print(consant1, consant2)
        constants.add(consant1)
        constants.add(consant2)
        predicates.add(predicate)

# if the constants are numbers, order them by value
# if they are strings, order them alphabetically
constants = sorted(constants, key=lambda x: int(x) if x.isdigit() else x)
predicates = sorted(predicates, key=lambda x: int(x) if x.isdigit() else x)
print(len(constants), len(predicates))

with open(ctes_path, 'w') as f:
    f.write('cte ' )
    for cte in constants:
        f.write(cte + ' ')

# with open(predicates_path, 'w') as f:
#     for pred in predicates:
#         f.write(pred + '\n')

# for valid and test, cerate new files with only queries whose predicates and constants are present in the train set
path = current_dir + '/'+dataset+'/valid.txt'
lines_valid = []
with open(path, 'r') as f:  
    for line in open(path, 'r'):
        predicate = line.split('(')[0]
        consant1 = line.split('(')[1].split(',')[0]
        consant2 = line.split('(')[1].split(',')[1].split(')')[0]
        # if the predicate and the constants are in the train set, add the line to the lines
        if predicate in predicates and consant1 in constants and consant2 in constants:
            lines_valid.append(line)

# create a file called valid_new.txt with the lines
with open(current_dir + '/'+dataset+'/valid_new.txt', 'w') as f:
    for line in lines_valid:
        f.write(line)


path = current_dir + '/'+dataset+'/test.txt'
lines_valid = []
with open(path, 'r') as f:  
    for line in open(path, 'r'):
        predicate = line.split('(')[0]
        consant1 = line.split('(')[1].split(',')[0]
        consant2 = line.split('(')[1].split(',')[1].split(')')[0]
        # if the predicate and the constants are in the train set, add the line to the lines
        if predicate in predicates and consant1 in constants and consant2 in constants:
            lines_valid.append(line)

# create a file called test_new.txt with the lines
with open(current_dir + '/'+dataset+'/test_new.txt', 'w') as f:
    for line in lines_valid:
        f.write(line)
            


