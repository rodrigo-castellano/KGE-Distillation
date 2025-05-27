import os

path = 'C:\\Users\\rodri\\Downloads\\PhD_code\\Review_grounders\\keras_ns_grounders\\experiments\\data\\kinship_family'
path_dest = 'C:\\Users\\rodri\\Downloads\\PhD_code\\Review_grounders\\keras_ns_grounders\\experiments\\data\\kinship_family_triplets'
# path_dest = 'C:\\Users\\rodri\\Downloads\\PhD\\Review_grounders\\countries_s1_triplets'
# assert that the paths are not the same
assert path != path_dest, 'The paths are the same'
relations = []
entities = []
facts = []
train = []
valid = []
test = []

if not os.path.exists(path_dest):
    os.makedirs(path_dest)

for set in ['train', 'valid', 'test']:
    # read the train file: set.txt
    with open(os.path.join(path, set+'.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            # split it by '('
            line = line.split('(')
            # get the first element
            relation = line[0]
            # split the second element by )
            line1 = line[1].split(')')[0]
            # split the second element by ','
            entities_list = line1.split(',')
            # get the first element
            entity1 = entities_list[0]
            # get the second element
            entity2 = entities_list[1]
            # append the relation to the list
            relations.append(relation)
            # append the entities to the list
            entities.append(entity1)
            entities.append(entity2)
            print(relation, entity1, entity2)
            # write into train_triplets.txt
            with open(os.path.join(path_dest, set+'.txt'), 'a') as f1:
                f1.write(entity1 + '\t' + relation + '\t' +  entity2 + '\n')

print('entities', entities)
print('relations', relations)


# get the unique entities
# entities = list(set(entities))
unique_entities = []
for item in entities:
    if item not in unique_entities:
        unique_entities.append(item)
# write into entities.txt
with open(os.path.join(path_dest, 'entities.txt'), 'w') as f:
    for entity in unique_entities:
        f.write(entity + '\n')


# get the unique relations
# relations = list(set(relations))
unique_relations = []
for item in relations:
    if item not in unique_relations:
        unique_relations.append(item)
# write into relations.txt
with open(os.path.join(path_dest, 'relations.txt'), 'w') as f:
    for relation in unique_relations:
        f.write(relation + '\n')




        