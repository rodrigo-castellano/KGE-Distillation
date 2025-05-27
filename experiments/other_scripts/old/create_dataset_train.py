import copy
import time
from typing import List, Set, Tuple, Dict, Union
from keras_ns.logic.commons import Atom, Domain, Rule
from keras_ns.grounding.engine import Engine
from itertools import chain, product
import shutil
from typing import Dict, List, Tuple, Union, Iterable

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from typing import Dict, List
from keras_ns.logic.commons import Atom, FOL,RuleGroundings
import time
import argparse
import keras_ns as ns
from itertools import product
import numpy as np
from os.path import join
import random
import pickle

from dataset import KGCDataHandler, build_domains
from model import CollectiveModel
from keras.callbacks import CSVLogger
from keras_ns.logic.commons import Atom, Domain, Rule, RuleLoader
from keras_ns.nn.kge import KGEFactory
from keras_ns.utils import MMapModelCheckpoint, KgeLossFactory, read_file_as_lines
from keras_ns.grounding import  AtomIndex, get_atoms, get_atoms_on_groundings, backward_chaining_grounding_one_rule
from keras_ns.utils import get_arg

def PruneIncompleteProofs(rule2groundings: Dict[str, Set[Tuple[Tuple, Tuple]]],
                          rule2proofs:Dict[str, List[Tuple[Tuple, List[Tuple]]]],
                          fact_index: AtomIndex,
                          num_steps: int) ->  Dict[str, Set[Tuple[Tuple, Tuple]]]:
    #for rn,g in rule2groundings.items():
    #    print('RIN', rn, g)
    # Here we will keep all the proves that are complete, i.e. all the atoms
    atom2proved: Dist[Tuple[str, str, str], bool] = {}
    # go through every atom to prove from all queries
    for i in range(num_steps ):
        for rule_name,proofs in rule2proofs.items():
            for query_and_proof in proofs:
                query, proof = query_and_proof[0], query_and_proof[1]
                if query not in atom2proved or not atom2proved[query]:
                    atom2proved[query] = all(
                        [atom2proved.get(a, False) for a in proof])

    # Now atom2proved has all proved atoms. Scan the groundings and keep only
    # the ones that have been proved within num_steps:
    pruned_rule2groundings = {}
    for rule_name,groundings in rule2groundings.items():
        pruned_groundings = []
        for g in groundings:
            head_atoms = g[0]
            # WE CHECK IF ALL THE ATOMS IN THE HEAD ARE PROVED
            if all([(atom2proved.get(a, False) or
                     fact_index._index.get(a, None) is not None)
                    for a in head_atoms]):                
                pruned_groundings.append(g)
        pruned_rule2groundings[rule_name] = set(pruned_groundings)
    return pruned_rule2groundings

def Prune_groundings_per_level(groundings_per_level,
                          rule2proofs:Dict[str, List[Tuple[Tuple, List[Tuple]]]],
                          fact_index: AtomIndex,
                          num_steps: int) ->  Dict[str, Set[Tuple[Tuple, Tuple]]]:
    #for rn,g in groundings_per_level.items():
    #    print('RIN', rn, g)
    # Here we will keep all the proves that are complete, i.e. all the atoms
    atom2proved: Dist[Tuple[str, str, str], bool] = {}
    # go through every atom to prove from all queries
    for i in range(num_steps):
        for rule_name,proofs in rule2proofs.items():
            for query_and_proof in proofs:
                query, proof = query_and_proof[0], query_and_proof[1]
                if query not in atom2proved or not atom2proved[query]:
                    atom2proved[query] = all(
                        [atom2proved.get(a, False) for a in proof])

    # Now atom2proved has all proved atoms. Scan the groundings and keep only
    # the ones that have been proved within num_steps:
    pruned_groundings_per_level = {}
    pruned_groundings = []
    for g in groundings_per_level:
        head_atoms = g[0]
        if all([(atom2proved.get(a, False) or
                    fact_index._index.get(a, None) is not None)
                for a in head_atoms]):                
            pruned_groundings.append(g)
    pruned_groundings_per_level = set(pruned_groundings)
    return pruned_groundings_per_level


# res is a Set of (Tuple_head_groundings, Tuple_body_groundings)
def backward_chaining_grounding_one_rule_with_domains(
    pred_counts,
    atoms_remove,
    groundings_per_level,
    domains: Dict[str, Domain],
    rule: Rule,
    queries: List[Tuple],
    fact_index: AtomIndex,
    max_unknown_fact_count: int,
    res: Set[Tuple[Tuple, Tuple]]=None,
    proofs: Dict[Tuple[Tuple, Tuple], List[Tuple[Tuple, Tuple]]]=None,step=-1, n_steps=-1) -> Union[
        None, Set[Tuple[Tuple, Tuple]]]:
    
    start = time.time()
    # We have a rule like A(x,y) B(y,z) => C(x,z)
    assert len(rule.head) == 1, (
        'Rule is not a Horn clause %s' % str(rule))
    head = rule.head[0]
    build_proofs: bool = (proofs is not None)

    new_ground_atoms = set()
    pred_counts_ordered = sorted(pred_counts, key=pred_counts.get, reverse=True)
    count_groundings = 0
    cont = 0
    lim=10000
    for q in queries:
      cont += 1 
    #   print('\n\n***************q', q,'********************') if cont< lim else None
      if q[0] != head[0]:  # predicates must match.
        continue
      # Get the variable assignments from the head.
      head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}
      for i in range(len(rule.body)):
        body_atom = rule.body[i]
        ground_body_atom = (body_atom[0], ) + tuple(
            [head_ground_vars.get(body_atom[j+1], None)
             for j in range(len(body_atom)-1)])
        # print('\n- i', i,'. ground_body_atom:', ground_body_atom, '. Substitution (by None) of the vars not present in head.') if cont< lim else None
        if all(ground_body_atom[1:]):
            groundings = (ground_body_atom,)
        else:
            groundings = fact_index._index.get(ground_body_atom, [])
            # print(' GROUNDINGS', groundings) if cont< lim else None
        if len(rule.body) == 1:
            # Shortcut, we are done, the clause has no free variables.
            new_ground_atoms.add(((q,), groundings))
            continue
        for ground_atom in groundings:
            # print('     -GROUND ATOM', ground_atom)
            head_body_ground_vars = copy.copy(head_ground_vars)
            head_body_ground_vars.update(
                {v: a for v, a in zip(body_atom[1:], ground_atom[1:])})

            free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
                               if v not in head_body_ground_vars]
            free_vars = [vd[0] for vd in free_var2domain]
            # print('     FREE VARS_SPAN', list(product(*[domains[vd[1]].constants for vd in free_var2domain])))
            for ground_vars in product(
                *[domains[vd[1]].constants for vd in free_var2domain]):
                var2ground = dict(zip(free_vars, ground_vars))
                full_ground_vars = {**head_body_ground_vars, **var2ground}
                # print('     FULL_VARS', full_ground_vars) if cont< lim else None
                accepted: bool = True
                body_grounding = []
                if build_proofs:
                    body_grounding_to_prove = []
                unknown_fact_count: int = 0
                for j in range(len(rule.body)):
                    if i == j:
                        # print('         -j=i')
                        new_ground_atom = ground_atom
                        # by definition as it is coming from the groundings.
                        is_known_fact = True
                    else:
                        body_atom2 = rule.body[j]
                        new_ground_atom = (body_atom2[0], ) + tuple(
                            [full_ground_vars.get(body_atom2[k+1], None)
                             for k in range(len(body_atom2)-1)])
                        if new_ground_atom == q:
                            # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, ' Same atom as query, discard')
                            accepted = False
                            break
                        is_known_fact = (fact_index._index.get(
                            new_ground_atom, None) is not None)

                    assert all(new_ground_atom), (
                        'Unexpected free variables in %s' %
                        str(new_ground_atom))

                    if not is_known_fact and (
                            max_unknown_fact_count < 0 or
                            unknown_fact_count < max_unknown_fact_count):
                        body_grounding.append(new_ground_atom)
                        if build_proofs:
                            body_grounding_to_prove.append(new_ground_atom)
                        unknown_fact_count += 1
                        # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Accepted. We have to prove it')
                    elif is_known_fact:
                        body_grounding.append(new_ground_atom)
                        # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Accepted')
                    else:
                        # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Discard',unknown_fact_count,'/', max_unknown_fact_count)
                        accepted = False
                        break
                if accepted:
                    count_groundings += 1
                    # print('ADDED', q, '->', tuple(body_grounding), 'TO_PROVE',          str(body_grounding_to_prove) if build_proofs else '')
                    # print('     ADDED', q, '->', tuple(body_grounding)) if cont< lim else None
                    new_ground_atoms.add(((q,), tuple(body_grounding)))
                    if build_proofs:
                        proofs.append((q, body_grounding_to_prove))
                    # CHECK IF ALL THE ATOMS ARE KNOWN. IF YES, CHOOSE ONE OF THEM TO REMOVE. CHOOSE ONLY THE ONES THAT ARE HEAD OF RULES
                    if unknown_fact_count == 0 and step != n_steps-1:
                        # print('\nALL ATOMS KNOWN', body_grounding) if cont< lim else None
                        possible_atoms = [atom for atom in body_grounding if atom[0] in pred_counts]
                        # print('POSSIBLE ATOMS', possible_atoms) if cont< lim else None
                        if len(possible_atoms) != 0:
                            # for each predicate, go through all the atoms.if the predicate is the same as the most frequent predicate, remove it
                            for pred in pred_counts_ordered:
                                atom_found = False
                                for atom in possible_atoms: 
                                    # print('     Predicate', pred, 'Atom', atom) if cont< lim else None
                                    # if the predicate is the same as the most frequent predicate, remove it
                                    if atom[0] == pred:
                                        # print('     (fully found. ADDED', q, '->', tuple(body_grounding)) if cont< lim else None
                                        # print('     Predicate', pred, 'Atom', atom,'. Remove') if cont< lim else None
                                        if step not in atoms_remove:
                                            atoms_remove[step] = set()
                                        atoms_remove[step].add(atom)
                                        atom_found = True
                                        break
                                if atom_found:
                                    break
    # print('Number of groundings gone through', count_groundings) if cont< lim else None
    end = time.time()
    res.update(new_ground_atoms) if res is not None else None       

    if step not in groundings_per_level:
        groundings_per_level[step] = set()
    if new_ground_atoms is not None:
        for g in new_ground_atoms:
            groundings_per_level[step].add(g)
    # print('New groundings: ', len(new_ground_atoms)) if len(new_ground_atoms) >0 else None
    # print('Atoms to remove so far:',len(atoms_remove[step])) if step in atoms_remove else None
    # print('Unique groundings per level, level',step,':',len(groundings_per_level[step])) 
    # print('Groundings in res for this rule', len(res)) if res is not None else None
    # Order the groundings in groundings_levels
    # ordered = sorted(groundings_levels[step], key=lambda x: (x[0], x[1][0],x[1][1])) if len(groundings_levels[step]) > 1 else groundings_levels[step]
            
    # THIS IS JUST TO SEE THE UNIQUE GROUNDINGS IN RES
    # unique_groundings_in_res = set()
    # if res_full is not None:
    #     for rule in res_full:
    #         for i,g in enumerate(res_full[rule]):
    #             # print('     Grounding',i, grounding) if cont< lim else None
    #             unique_groundings_in_res.add(g)
    # # I also need to take into acc the ones from this level
    # for g in new_ground_atoms:
    #     unique_groundings_in_res.add(g)
    # # print('Unique groundings in res', len(unique_groundings_in_res)) if res is not None else None

    # AFTER I UPDATE RES, THE NUMBER OF GROUNDINGS IN self.rule2groundings IS GREATER THAN THE NUMBER OF GROUNDINGS IN RES. I NEED TO REMOVE THE DUPLICATES
    if res is None:
        return new_ground_atoms,atoms_remove,groundings_per_level
    else:
        res,atoms_remove,groundings_per_level


class BackwardChainingGrounder(Engine):

    def __init__(self, rules: List[Rule], facts: List[Union[Atom, str, Tuple]],
                 domains: Dict[str, Domain],
                 max_unknown_fact_count: int=1,
                 max_unknown_fact_count_last_step: int=0,
                 num_steps: int=1,
                 prune_incomplete_proofs: bool=True):
        self.max_unknown_fact_count = max_unknown_fact_count
        self.max_unknown_fact_count_last_step = max_unknown_fact_count_last_step
        self.num_steps = num_steps
        self.prune_incomplete_proofs = prune_incomplete_proofs
        self.rules = rules
        self.domains = domains
        self.facts = [a if isinstance(a,Tuple) else a.toTuple()
                      if isinstance(a,Atom) else Atom(s=a).toTuple()
                      for a in facts]
        # self.facts = facts
        for rule in self.rules:
            assert len(rule.head) == 1, (
                '%s is not a Horn clause' % str(rule))
        self._fact_index = AtomIndex(self.facts)
        self.relation2queries = {}
        self.rule2groundings = {}
        self.rule2proofs = {}

    def _init_internals(self, queries: List[Tuple]):
        # this tell us the queries for each relation to analyse.
        self.relation2queries = {}  # reset
        for q in queries:
            if q[0] not in self.relation2queries:
                self.relation2queries[q[0]] = set()
            self.relation2queries[q[0]].add(q)

        # the other are not reset as they are incrementally added.
        for rule in self.rules:
            if rule.name not in self.rule2groundings:
                self.rule2groundings[rule.name] = set()
            if rule.name not in self.rule2proofs:
                self.rule2proofs[rule.name] = []

    # Ground a batch of queries, the result is cached for speed.
    def ground(self,
               pred_counts,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:

        if self.rules is None or len(self.rules) == 0:
            return []
        atoms_remove = {}
        groundings_per_level = {}
        self._init_internals(queries)
        self._rule2processed_queries = {rule.name: set() for rule in self.rules}
        for step in range(self.num_steps):
            # print('STEP NUMBER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', step,'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^','step ',step,'/', self.num_steps, 'known body',step == self.num_steps - 1, )
            for j,rule in enumerate(self.rules):
                # print('\nrule ', rule, ' """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" ')
                # Here we assume to have a Horn clause, fix it.
                queries_per_rule = list(
                    self.relation2queries.get(rule.head[0][0], set()))
                # print('\nqueries_per_rule\n',len(queries_per_rule), queries_per_rule)
                if not queries_per_rule:
                    continue
                backward_chaining_grounding_one_rule_with_domains(
                    pred_counts,
                    atoms_remove,
                    groundings_per_level,
                    self.domains, rule, queries_per_rule, self._fact_index,
                    # max_unknown_fact_count
                    (self.max_unknown_fact_count if step < self.num_steps - 1
                     else self.max_unknown_fact_count_last_step),
                    # Output added here.
                    res=self.rule2groundings[rule.name],
                    # Proofs added here.
                    proofs=(self.rule2proofs[rule.name] if self.prune_incomplete_proofs else None),
                    step=step, n_steps=self.num_steps)
                print('Total  groundings in res after rule',j,'/',len(self.rules),', step',step,sum([len(v) for k, v in self.rule2groundings.items()])) # IS IS MORE THAN THE GROUNDINGS_per_level BECAUSE THERE ARE DUPLICATES
                # Update the list of processed rules.
                self._rule2processed_queries[rule.name].update(queries_per_rule)
            if step == self.num_steps - 1:
                break
            # FROM THE ATOMS TO REMOVE, I SHOULD ONLY KEEP THE ONES THAT ARE PROVED, AND AT MOST 1 ATOM IS MISSING
            # Get the queries for the next iteration.
            new_queries = set()
            for rule in self.rules:
                groundings = self.rule2groundings[rule.name]
                # Get the new queries left to prove, these facts that are not
                # been processed already and that are not known facts.
                new_queries.update(
                    [a for a in get_atoms_on_groundings(groundings)
                     if a not in self._rule2processed_queries[rule.name] and
                     self._fact_index._index.get(a, None) is None])
            # print('\nNEW Q',len(new_queries),'\n', list(new_queries), ' FROM groundings', len(groundings))
            # Here we update the queries to process in the next iteration, we only keep the new ones.
            self._init_internals(list(new_queries))

        print('\nNum groundings',sum([len(v) for k, v in self.rule2groundings.items()]))
        if self.prune_incomplete_proofs:
            self.rule2groundings = PruneIncompleteProofs(self.rule2groundings,
                                                         self.rule2proofs,
                                                         self._fact_index,
                                                         self.num_steps)
            print('Num groundings after pruning',sum([len(v) for k, v in self.rule2groundings.items()]))
        
        for level in range(self.num_steps):
            # if the level is in the keys of groundings_per_level, prune the groundings
            if level in groundings_per_level:
                # print the keys of the groundings_per_level
                print('\nNum groundings in level',level,',',len(groundings_per_level[level]))
                if self.prune_incomplete_proofs:
                    groundings_per_level[level] = Prune_groundings_per_level(groundings_per_level[level],
                                                                self.rule2proofs,
                                                                self._fact_index,
                                                                self.num_steps)
                    print('Num groundings in level',level,', after pruning,',len(groundings_per_level[level]))

        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name: RuleGroundings(
                rule_name, sorted(list(groundings), key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name: RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}
        return ret,atoms_remove,groundings_per_level

 
 

def find_atom_in_groundings(atom, groundings): 
    # this, in case of pruning, checks if the atom is still present in the groundings
    # Only on the body
    for grounding in groundings:
        for body_atom in grounding[1]:
            if body_atom == atom:
                return True
    return False

def find_cutoff(atoms_remove,data_handler,threshold=0.4):
    # from atoms_remove, count how many times each predicate appears, and divide it by the total number of atoms in the facts with that predicate
    pred_counts_to_remove = {}
    for  atom in atoms_remove:
        pred = atom[0]
        if pred not in pred_counts_to_remove:
            pred_counts_to_remove[pred] = 0
        pred_counts_to_remove[pred] += 1

    # calculate the number of times that predicate appears in train_known_facts_set
    pred_counts_facts = {}
    for atom in data_handler.train_known_facts_set:
        pred = atom[0]
        if pred not in pred_counts_facts:
            pred_counts_facts[pred] = 0
        pred_counts_facts[pred] += 1
    
    # calculate the % of atoms to remove out of the total number of atoms in the facts with that predicate
    pred_cutoff = {}
    max_atoms_allowed = {}
    print('Threshold per predicate:')
    for pred in pred_counts_to_remove:
        max_atoms_allowed[pred] = int(threshold*pred_counts_facts[pred])
        pred_cutoff[pred] = pred_counts_to_remove[pred] / pred_counts_facts[pred]
        print('     ',pred, 'percentage of atoms with that predicate in facts: ', np.round(pred_cutoff[pred],3),'. ', pred_counts_to_remove[pred], '/', pred_counts_facts[pred])
    return max_atoms_allowed,pred_cutoff,pred_counts_to_remove,pred_counts_facts

def write_dataset(base_path, dest_path, args, data_handler, new_facts, new_train, ctes_facts, num_steps, prune_backward):
    if not os.path.exists(dest_path): os.mkdir(dest_path)
    facts_file = join(dest_path,'facts.txt')
    train_file = join(dest_path,'train.txt')
    test_file = join(dest_path,'test.txt')
    valid_file = join(dest_path,'valid.txt')
    domain_file = join(dest_path,'domain2constants.txt')

    with open(train_file, 'w') as f:
        for train_fact in new_train:
            f.write(str(train_fact[0])+'('+str(train_fact[1])+','+str(train_fact[2])+').\n')
    if len(new_facts) > 0:
        with open(facts_file, 'w') as f:
            for fact in new_facts:
                f.write(str(fact[0])+'('+str(fact[1])+','+str(fact[2])+').\n')

    # write a new test and valid files, skip the atoms with constants not present in ctes_facts
    with open(test_file, 'w') as f:
        for test_fact in data_handler.test_facts:
            if test_fact[1] in ctes_facts and test_fact[2] in ctes_facts:
                f.write(str(test_fact[0])+'('+str(test_fact[1])+','+str(test_fact[2])+').\n')
    with open(valid_file, 'w') as f:
        for valid_fact in data_handler.valid_facts:
            if valid_fact[1] in ctes_facts and valid_fact[2] in ctes_facts:
                f.write(str(valid_fact[0])+'('+str(valid_fact[1])+','+str(valid_fact[2])+').\n')
    # Write the domain file 
    domain_ctes = data_handler.domain2constants
    # Now I have to selec from domain_ctes only the constants that are present in the new_facts
    domain_ctes_write = {}
    for k,ctes in domain_ctes.items():
        domain_ctes_write[k] = [cte for cte in ctes if cte in ctes_facts]    

    with open(domain_file, 'w') as f:
        for k,ctes in domain_ctes_write.items():
            if k!='default':
                f.write(k+' ')
                [f.write(cte+' ') for cte in ctes] 
                f.write('\n')
    
        # copy the rules file
        rules_file = join(base_path, args.dataset_name, args.rules_file)
        dest_rules_file = join(dest_path,args.rules_file)   
        shutil.copyfile(rules_file, dest_rules_file)



def main(base_path, output_filename, kge_output_filename, log_filename, args):

    print('\nARGS', args,'\n')

    seed = get_arg(args, 'seed_run_i', 0)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
  
    # Data Loading
    data_handler = KGCDataHandler(
        dataset_name=args.dataset_name,
        base_path=base_path,
        format=get_arg(args, 'format', None, True),
        domain_file= args.domain_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        fact_file= args.facts_file)
    
    dataset_train = data_handler.get_dataset(split="train",number_negatives=args.num_negatives)
    dataset_test = data_handler.get_dataset(split="test", corrupt_mode='TAIL',number_negatives=args.test_negatives)
    
    fol = data_handler.fol 

    enable_rules = (args.reasoner_depth > 0 and args.num_rules > 0)
    if enable_rules: 
        rules = ns.utils.read_rules(join(base_path, args.dataset_name, args.rules_file),args)

        if 'backward' in args.grounder:
            print('Using backward chaining with %d steps' % args.reasoner_depth)
            num_steps = int(args.grounder.split('_')[-1])
            prune_backward = True  
            print('Using backward chaining with %d steps' % num_steps)
            engine = BackwardChainingGrounder(rules, facts=list(data_handler.train_known_facts_set),
                                                        domains={d.name:d for d in fol.domains},
                                                        num_steps=num_steps,prune_incomplete_proofs=prune_backward)

    else:
        rules = []
        engine = None
 
    queries, labels = dataset_test[0:len(dataset_test)]
    facts = fol.facts
    rules = engine.rules


    # For every rule, get the predicate in the head. Count the number of times each predicate appears
    pred_head_facts_counts = {}
    for rule in rules:
        pred = rule.head[0][0]
        if pred not in pred_head_facts_counts:
            pred_head_facts_counts[pred] = 0
        pred_head_facts_counts[pred] += 1
    # print the number of times each predicate appears in the head of the rules
    print('pred_head_facts_counts', pred_head_facts_counts)
    pred_counts = sorted(pred_head_facts_counts, key=pred_head_facts_counts.get, reverse=True)

    groundings,atoms_remove_per_level,groundings_per_level= engine.ground(pred_head_facts_counts,tuple(facts),tuple(ns.utils.to_flat(queries)),deterministic=True)

    # filters: 
    # dont remove atoms if they are in the body in the last level
    # dont remove atoms if at any level (except for the first one), there are more than 1 atom to remove in the body of a grounding
    # dont remove atoms if they are in the query of a grounding in the last level 
    # NEED TO BE CAREFUL, BECAUSE IF I REMOVE, EVEN THOUGH IT IS NOT IN THE LAST LEVEL, IT MIGHT BE IN THE LEVEL BEFORE AND BECAUSE OF THAT ATOM, IT CAN MAKE 
    # A TOTAL OF 2 BODY ATOMS MISSING AND THEREFORE THE GROUNDING IS NOT PROVED. What I should do is, for every atom to remove, check if that atom in present in 
    # any grounding that goes up to the last level. If it is, then I should not remove it.


    if num_steps == 2:
        print('take into account atoms from level 0 ') 
        remove_levels =[0]
    elif num_steps == 3:
        print('take into account atoms from level 0 and 1') 
        remove_levels =[0,1]
    else:  
        print('ERROR: num_steps should be 2 or 3 minimum')
    print()


    # Take only atoms from  the levels we are interested in
    # Filter the atoms that are not in the proved groundings.  
    atoms_remove = set()
    print ('ATOMS TO REMOVE:')
    for k,v in atoms_remove_per_level.items():
        if k in remove_levels:
            print('level',k,', number of atoms to remove',len(v))
            count = 0
            for atom in set(v):
                # check if the atom is in the pruned groundings of that level, because otherwise that body atom is not proved, and we have to remove the atom 
                atom_found = find_atom_in_groundings(atom, groundings_per_level[k])
                # print('  Atom', atom,  '     in body of groundings, level',k,'       ',atom_found)
                if atom_found:
                    count += 1
                    atoms_remove.add(atom)
            # print('level',k,', number of atoms to remove taking into account prunning:',count)
            print()


    # The measure to do the cutoff is, for each predicate, the % of atoms out of the number of atoms in facts with that predicate
    # I could also take into account how many times they are in the body of a grounding in the last level, i.e. if I do backward 3, level 3
    # Ex: If in a grounding at level 2 I remove  LocIn(Italy,Europe), then it is a problem if that is later in another grounding at level 3, because if it is not in the body at level 3, it is not proved
    max_atoms_allowed,pred_cutoff,pred_counts_to_remove,pred_counts_facts = find_cutoff(atoms_remove,data_handler,threshold=0.4)

    # from atoms_remove, check how many atoms are in the body of the grounding of the last level 
    # I SHOULD ALSO CHECK IF THEY ARE ANY THE QUERY
    atoms_problem = set()
    if num_steps-1 in groundings_per_level and len(groundings_per_level[num_steps-1])!=0 :
        for grounding in groundings_per_level[num_steps-1]:
            for body_atom in grounding[1]:
                if body_atom in atoms_remove:
                    # print('atom', atom, 'is in the body of a grounding in the last level. Added to problematic atoms')
                    atoms_problem.add(atom)
    # else:
        # print('No groundings in the last level')


    # How many groundings for all the levels except the last one, have more than 1 atoms to remove in the body
    # If it is more than 1, we are not going to the next level because we are not admitting the subsitution   
    for level,groundings_level_i in groundings_per_level.items():
        if level != num_steps-1:
            # print('level', level)
            for grounding in groundings_level_i:
                cuenta = 0 
                atoms_problematic = set()
                # print('body', grounding[1])
                for body_atom in grounding[1]:
                    # print('     body_atom', body_atom)
                    # for atom in atoms_remove:
                    # print('             atom', atom)
                    # if atom == body_atom:
                    if body_atom in atoms_remove:
                        # print('atom', atom, 'is in the body ', body_atom,'------------------------------------------------------------------------')
                        cuenta += 1
                        atoms_problematic.add(atom)
                if cuenta > 1:
                    atoms_problem.update(atoms_problematic)
                    # print('grounding level',level,' has more than 1 atom to removed in the body because of', atoms_problematic,'. Added to problematic atoms')


 


    # hipothetically, check how many groundings there are in the new dataset by not counting groundings with atoms to remove
    # Select the grounding from level 0:
    
    check_list_atoms = list()#set()
    for grounding_0 in groundings_per_level[0]:
        if num_steps == 2:
        # Take the body atoms of the grounding, if any body atom is the head of a grounding in the next level, append it in a list
            body_atoms_0 = grounding_0[1]
            for body_atom_0 in body_atoms_0:
                # if atom is in the head of a grounding in the next level, then we have to remove it
                for grounding_1 in groundings_per_level[1]:
                    if body_atom_0 in grounding_1[0]:
                        # if body_atom_0 in atoms_remove:
                        # add every from grounding_0 and grounding_1 to the set
                        # check_list_atoms.update((atom for atom in grounding_0[1]))
                        # check_list_atoms.add(grounding_0[0]) 
                        # check_list_atoms.update((atom for atom in grounding_1[1]))
                        # check_list_atoms.add(grounding_1[0])
                        # check_list_atoms.append(grounding_0[0])
                        # check_list_atoms.extend(atom for atom in grounding_0[1])
                        # check_list_atoms.append(grounding_1[0])
                        # check_list_atoms.extend(atom for atom in grounding_1[1])
                        print('groudnding 0',grounding_0[0],grounding_0[1])
                        print('groudnding 1',grounding_1[0],grounding_1[1])
                        check_list_atoms.append(grounding_0)
    print('number of groundings to check:',len(check_list_atoms))                           




    print(' atoms found: ',len(check_list_atoms))
    print('Number of atoms to remove:', len(atoms_remove))
    # Exclude problematic atoms from atoms_remove
    atoms_remove -= atoms_problem
    print('problematic atoms:',atoms_problem)
    print('Number of atoms to remove after excluding porblematic:', len(atoms_remove)) 


    total_groundings = set()
    for level,groundings_level_i in groundings_per_level.items():
        for grounding in groundings_level_i:
            total_groundings.add(grounding)
    print('Total number of unique groundings with original dataset:'+ str(len(total_groundings)))
    print('\nNumber of groundings per level with original dataset:\n')
    for level in range(num_steps):
        print('Level {}. Number of groundings: {}\n'.format(level,len(groundings_per_level[level])) if level in groundings_per_level else 'level {}. No groundings in this level\n'.format(level))




    # # Remove atoms randomly if there are more than the max allowed
    # for predicate,cutoff in max_atoms_allowed.items():
    #     # if there are still atoms to remove, do it randomly
    #     if len(atoms_remove) > cutoff:
    #         if atom[0] == predicate:
    #             atoms_remove = random.sample(list(atoms_remove), cutoff)
    # print('Number of atoms to remove after random cutoff:', len(atoms_remove)) 


    new_facts = set(data_handler.known_facts) - set(atoms_remove)
    new_train = set(data_handler.train_facts) - set(atoms_remove)
    ctes_facts = set()
    for facts_set in (new_facts,new_train):
        for (p,a1,a2) in facts_set:
            ctes_facts.update((a1,a2))

    if len(atoms_remove) > 0:
        dest_path = join(base_path, args.dataset_name+'_reason_'+str(num_steps))
        write_dataset(base_path, dest_path, args, data_handler, new_facts, new_train, ctes_facts, num_steps, prune_backward)

        # Print all the info in a .txt file
        output_file = join(dest_path,'info.txt')
        with open(output_file, 'w') as f:
            f.write('Number of atoms removed: {}\n'.format(len(atoms_remove))) 
            f.write('Problematic atoms:\n'+str(atoms_problem)) 
            f.write('\nNumber of times a predicate appears in the head of rules (in train facts): {}\n'.format(pred_head_facts_counts))

            for pred in pred_counts_to_remove:
                f.write(str(pred)+ ', (Fraction of) Atoms from facts to be deleted with this predicate: '+ str(np.round(pred_cutoff[pred],3))+'. '+ str(pred_counts_to_remove[pred])+ '/'+ str(pred_counts_facts[pred])+'\n')

            f.write('Atoms removed:\n') 
            for atom in atoms_remove:
                f.write(str(atom)+'\n')
            
            total_groundings = set()
            for level,groundings_level_i in groundings_per_level.items():
                for grounding in groundings_level_i:
                    total_groundings.add(grounding)
            f.write('Total number of unique groundings with original dataset:'+ str(len(total_groundings)))

            f.write('\nNumber of groundings per level with original dataset:\n')
            for level in range(num_steps):
                f.write('Level {}. Number of groundings: {}\n'.format(level,len(groundings_per_level[level])) if level in groundings_per_level else 'level {}. No groundings in this level\n'.format(level))

    















    
    # something interesting to do is to check, with the new dataset, how many new groudnings we have in the last level

    # Data Loading
    data_handler = KGCDataHandler(
        dataset_name=args.dataset_name+'_reason_'+str(num_steps),
        base_path=base_path,
        format=get_arg(args, 'format', None, True),
        domain_file= args.domain_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        fact_file= args.facts_file)
    
    dataset_test = data_handler.get_dataset(split="test", corrupt_mode='TAIL',number_negatives=args.test_negatives)
    fol = data_handler.fol 

    rules = ns.utils.read_rules(join(base_path, args.dataset_name, args.rules_file),args)

    num_steps = int(args.grounder.split('_')[-1])
    prune_backward = True  
    engine = BackwardChainingGrounder(rules, facts=list(data_handler.train_known_facts_set),
                                                    domains={d.name:d for d in fol.domains},
                                                    num_steps=num_steps,prune_incomplete_proofs=prune_backward)
 
 
    queries, labels = dataset_test[0:len(dataset_test)]
    facts = fol.facts
    rules = engine.rules

    # For every rule, get the predicate in the head. Count the number of times each predicate appears
    pred_head_facts_counts = {}
    for rule in rules:
        pred = rule.head[0][0]
        if pred not in pred_head_facts_counts:
            pred_head_facts_counts[pred] = 0
        pred_head_facts_counts[pred] += 1

    groundings,atoms_remove_per_level,groundings_per_level_new= engine.ground(pred_head_facts_counts,tuple(facts),tuple(ns.utils.to_flat(queries)),deterministic=True)


    # in the output file, print the number of groundings in every level
    with open(output_file, 'a') as f:

        total_groundings = set()
        for level,groundings_level_i in groundings_per_level_new.items():
            for grounding in groundings_level_i:
                total_groundings.add(grounding)
        f.write('Total number of unique groundings with new dataset:'+ str(len(total_groundings)))

        f.write('\nNumber of groundings per level with new dataset:\n')
        for level in range(num_steps):
            f.write('Level {}. Number of groundings: {}\n'.format(level,len(groundings_per_level_new[level])) if level in groundings_per_level_new else 'level {}. No groundings in this level\n'.format(level))


    print('Total number of unique groundings with new dataset:'+ str(len(total_groundings)))

    print('\nNumber of groundings per level with new dataset:\n')
    for level in range(num_steps):
        print('Level {}. Number of groundings: {}\n'.format(level,len(groundings_per_level_new[level])) if level in groundings_per_level_new else 'level {}. No groundings in this level\n'.format(level))