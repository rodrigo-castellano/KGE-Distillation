import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))

import random
from os.path import join
from typing import Dict, List, Tuple, Union, Iterable
import numpy as np
import tensorflow as tf
import ns_lib as ns
from ns_lib.grounding import *
from model_utils import * 
from ns_lib.logic.commons import Atom, FOL,RuleGroundings
from dataset import KGCDataHandler
import argparse
from ns_lib.grounding.grounder_factory import BuildGrounder
from ns_lib.grounding.backward_chaining_grounder import ApproximateBackwardChainingGrounder

parser = argparse.ArgumentParser(description='Description of your script')  
args = parser.parse_args()


random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


# args.grounder = 'backward_0_1'
args.grounder = 'backward_0_1'
# args.grounder = 'backwardnoprune_2_1'

# args.dataset_name = 'umls'
# args.dataset_name = 'nations'
# args.dataset_name = 'ablation_d2'
# args.dataset_name = 'countries_s3'
# args.dataset_name = 'kinship_family'
args.dataset_name = 'kinship'
# args.dataset_name = 'wn18rr'
# args.dataset_name = 'FB15k237'
# args.dataset_name = 'pharmkg_full'
args.data_path = "experiments/data"
args.facts_file = 'facts.txt'
args.train_file = 'train.txt'  
args.valid_file = 'valid.txt'
args.test_file = 'test.txt'
args.domain_file = 'domain2constants.txt'
args.rules_file = 'rules.txt'

# args.rules_file = 'rules_generated_amie.txt'
# args.rules_file = 'rules_amie_full.txt'
# args.rules_file = 'rules_michelangelo.txt'
# args.rules_file = 'rules_combined.txt'
# args.rules_file = 'rules_uniker.txt'



args.num_negatives = 0
args.test_negatives = 0

args.format = "functional"

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Data Loading
data_handler = KGCDataHandler(
    dataset_name=args.dataset_name,
    base_path=args.data_path,
    format=args.format,
    domain_file= args.domain_file,
    train_file= args.train_file,
    valid_file=args.valid_file,
    test_file= args.test_file,
    fact_file= args.facts_file)


dataset_train = data_handler.get_dataset(split="train",number_negatives=args.num_negatives)
dataset_test = data_handler.get_dataset(split="test", corrupt_mode='TAIL',number_negatives=args.test_negatives)

fol = data_handler.fol 
facts = fol.facts
rules = ns.utils.read_rules(join(args.data_path, args.dataset_name, args.rules_file),args)
queries, labels = dataset_test[0:len(dataset_test)]


type = args.grounder
print('Building Grounder', type, flush=True)

if 'backward' in type:
    # if the count of '_' the name is 2, it means that the parameter 'a' is included. Else there is no parameter a. It goes after the first '_'
    backward_width = None
    if type.count('_') == 2:
        backward_width = int(type[type.index('_')+1]) # take the first character after the first '_'
        backward_depth = int(type[-1])
        type = 'ApproximateBackwardChainingGrounder'
    else:
        backward_depth = int(type[-1])
        type = 'BackwardChainingGrounder'
    prune_incomplete_proofs = False if 'noprune' in args.grounder else True
    print('Grounder: ',args.grounder,'backward_depth:', backward_depth, 'Prune:', prune_incomplete_proofs, 'backward_width:', backward_width)

if type == 'ApproximateBackwardChainingGrounder':
    # Requires Horn Clauses.
    engine = ApproximateBackwardChainingGrounder(
        rules, facts=facts, domains={d.name:d for d in fol.domains},
        domain2adaptive_constants=None,
        pure_adaptive=get_arg(args, 'engine_pure_adaptive', False),
        num_steps=backward_depth,
        max_unknown_fact_count=backward_width,
        max_unknown_fact_count_last_step=backward_width,
        prune_incomplete_proofs=prune_incomplete_proofs,
        max_groundings_per_rule=get_arg(
            args, 'backward_chaining_max_groundings_per_rule', -1),
        force_determinism=False)


# engine = BuildGrounder(args, rules, facts=facts, fol=fol, domain2adaptive_constants=None)

import time
start = time.time()

queries = queries[:]
print('number of queries:',len(queries))

len_groundings = []
n_queries_with_groundings = 0
queries_with_groundings = []
for i,query in enumerate(queries):
    print('i',i,'/',len(queries))
    # print('\nquery:',query)
    facts = sorted(facts)
    ground_formulas = engine.ground(sorted(tuple(facts)),tuple(ns.utils.to_flat(query)),deterministic=True)

    print('num groundings:',len([grounding for rule in ground_formulas for grounding in ground_formulas[rule]]))
    print('current coverage:',round(n_queries_with_groundings/i,3)) if i > 0 else 0
    len_groundings.append(len([grounding for rule in ground_formulas for grounding in ground_formulas[rule]]))
    n_queries_with_groundings += 1 if len_groundings[-1] > 0 else 0
    predicate = query[0][0]
    cte1 = query[0][1]
    cte2 = query[0][2]
    query_str = f"{predicate}({cte1},{cte2})."
    print('query:',query_str)
    queries_with_groundings.append(query_str) if len_groundings[-1] > 0 and query_str not in queries_with_groundings else 0

    # print('ground_formulas:')
    # for rule in ground_formulas:
        # print('Rule:',rule)#,ground_formulas[rule])
        # for grounding in ground_formulas[rule]:
            # print(grounding[0][0],'       ',grounding[1][0], grounding[1][1])

print('num groundings:', sum(len_groundings),'avg number of grounding:',round(np.mean(len_groundings),3), 'std:',round(np.std(len_groundings),3))
print('coverage:',round(n_queries_with_groundings/len(queries),3))

# ground_formulas = engine.ground(tuple(facts),tuple(ns.utils.to_flat(queries)),deterministic=True)
# print('num groundings:',len([grounding for rule in ground_formulas for grounding in ground_formulas[rule]]))

print('Time:',round(time.time()-start,3))

# write the queries into a file with the name of the dataset, the grounder
with open(f'test_{args.dataset_name}_{args.grounder}.txt', 'w') as f:
    for item in queries_with_groundings:
        f.write("%s\n" % item) 