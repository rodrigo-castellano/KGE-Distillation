import os
import tensorflow as tf
import ns_lib as ns
from itertools import product
import numpy as np
from os.path import join
import random
import pickle
from typing import List, Tuple, Dict

from dataset import KGCDataHandler
from model import CollectiveModel
from keras.callbacks import CSVLogger
from ns_lib.logic.commons import Atom, Domain, FOL, Rule, RuleLoader
from ns_lib.grounding.grounder_factory import BuildGrounder
from ns_lib.utils import MMapModelCheckpoint, KgeLossFactory, get_arg
import time
from model_utils import * 
from ns_lib.utils import nested_dict
explain_enabled: bool = False
from collections import Counter

import copy
from itertools import product
import time
from typing import List, Set, Tuple, Dict, Union

from ns_lib.logic.commons import Atom, Domain, Rule, RuleGroundings
from ns_lib.grounding.engine import Engine

# class DomainFullGrounder(Engine):

#     def __init__(self,
#                  rules: List[Rule],
#                  domains:Dict[str, Domain],
#                  domain2adaptive_constants: Dict[str, List[str]]=None,
#                  pure_adaptive: bool=False,
#                  exclude_symmetric: bool=False,
#                  exclude_query: bool=False,
#                  limit: int=None):

#         self.rules = rules
#         # The flat grounder is not query oriented.
#         self.domains = domains
#         self.domain2adaptive_constants = domain2adaptive_constants
#         self.pure_adaptive = pure_adaptive
#         if self.pure_adaptive:
#             assert self.domain2adaptive_constants, (
#                 'Need adaptive variable if in pure adaptive mode.')
#         self.limit = limit
#         self.exclude_symmetric = exclude_symmetric
#         self.exclude_query = exclude_query

#     #@lru_cache
#     def ground(self,
#                facts: List[Tuple],
#                queries: List[Tuple],
#                **kwargs) -> Dict[str, RuleGroundings]:

#         res = {}
#         for rule in self.rules:
#             print('RULE', rule, 'out of', len(self.rules))
#             added = 0
#             groundings = []

#             if self.pure_adaptive:
#                 ground_var_groups = [self.domain2adaptive_constants.get(d, [])
#                                      for d in rule.vars2domain.values()]
#             elif self.domain2adaptive_constants is not None:
#                 ground_var_groups = [self.domains[d].constants +
#                                      self.domain2adaptive_constants.get(d, [])
#                                      for d in rule.vars2domain.values()]
#             else:
#                 ground_var_groups = [self.domains[d].constants
#                                      for d in rule.vars2domain.values()]

#             for ground_vars in product(*ground_var_groups):
#                 var_assignments = {k:v for k,v in zip(
#                     rule.vars2domain.keys(), ground_vars)}

#                 is_good = True
#                 body_atoms = []
#                 for atom in rule.body:
#                     ground_atom = (atom[0], ) + tuple(
#                         [var_assignments.get(atom[j+1], None)
#                          for j in range(len(atom)-1)])
#                     assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
#                     if (self.exclude_symmetric and
#                         ground_atom[1] == ground_atom[2]):
#                         is_good = False
#                         break
#                     if self.exclude_query and ground_atom in queries:
#                         is_good = False
#                         break
#                     body_atoms.append(ground_atom)

#                 head_atoms = []
#                 for atom in rule.head:
#                     ground_atom = (atom[0], ) + tuple(
#                         [var_assignments.get(atom[j+1], atom[j+1])
#                          for j in range(len(atom)-1)])
#                     assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
#                     if self.exclude_symmetric and ground_atom[1] == ground_atom[2]:
#                         is_good = False
#                         break
#                     if self.exclude_query and ground_atom in queries:
#                         is_good = False
#                         break
#                     head_atoms.append(ground_atom)

#                 # Check that nothing has been discarded.
#                 if is_good:
#                     groundings.append((tuple(head_atoms), tuple(body_atoms)))
#                     added += 1
#                     if self.limit is not None and self.limit >= added:
#                         break

#             res[rule.name] = groundings
#             # res[rule.name] = RuleGroundings(rule.name, groundings=groundings)
#         groundings_per_level = {0: res}
#         return res, groundings_per_level


# class AtomIndex():
#     def __init__(self, facts: List[Tuple[str, str, str]]):
#         _index = {}
#         for f in facts:
#             r_idx, s_idx, o_idx = f
#             key = (r_idx, None, None)
#             if key not in _index:
#                 _index[key] = set()
#             _index[key].add(f)
#             key = (r_idx, s_idx, None)
#             if key not in _index:
#                 _index[key] = set()
#             _index[key].add(f)
#             key = (r_idx,  None, o_idx)
#             if key not in _index:
#                 _index[key] = set()
#             _index[key].add(f)
#             key = f
#             _index[key] = set([f])

#         # Store tuple instead of sets.
#         self._index = {k: tuple(v) for k, v in _index.items()}

#     def get_matching_atoms(self,
#                            atom: Tuple[str, str, str]) -> Tuple[Tuple[str, str, str]]:
#         return self._index.get(atom, [])

# def get_atoms(r2g:Dict[str, Set[Tuple[Tuple, Tuple]]]) -> Set[Tuple]:
#     atoms = set()
#     for r, g in r2g.items():
#         for rule_atoms in g:
#             for atom in rule_atoms[0]:
#                 atoms.add(atom)
#             for atom in rule_atoms[1]:
#                 atoms.add(atom)
#     return atoms

# def get_atoms_on_groundings(groundings:Set[Tuple[Tuple, Tuple]]) -> Set[Tuple]:
#     atoms = set()
#     for rule_atoms in groundings:
#         for atom in rule_atoms[0]:  # head
#             atoms.add(atom)
#         for atom in rule_atoms[1]:  # tail
#             atoms.add(atom)
#     return atoms

# # res is a Set of (Tuple_head_groundings, Tuple_body_groundings)
# #@profile
# def approximate_backward_chaining_grounding_one_rule(
#     groundings_per_level,
#     domains: Dict[str, Domain],
#     domain2adaptive_constants: Dict[str, List[str]],
#     pure_adaptive: bool,
#     rule: Rule,
#     queries: List[Tuple],
#     fact_index: AtomIndex,
#     # Max number of unknown facts to expand:
#     # 0 implements the known_body_grounder
#     # -1 or body_size implements pure unrestricted backward_chaining (
#     max_unknown_fact_count: int,
#     res: Set[Tuple[Tuple, Tuple]]=None,
#     # proof is a dictionary:
#     # atom -> list of atoms needed to be proved in the application of the rule.
#     proofs: Dict[Tuple[Tuple, Tuple], List[Tuple[Tuple, Tuple]]]=None,
#     step=-1, n_steps=-1):
#     #start = time.time()
#     # We have a rule like A(x,y) B(y,z) => C(x,z)
#     assert len(rule.head) == 1, (
#         'Rule is not a Horn clause %s' % str(rule))
#     head = rule.head[0]
#     build_proofs: bool = (proofs is not None)

#     new_ground_atoms = set()
    
#     # lim = 1
#     # cont = 0
#     for q in queries:
#     #   groundings_per_query = 0
#     #   cont += 1 
#     #   print('\n\n***************q', q,'********************') if cont< lim else None
#       if q[0] != head[0]:  # predicates must match.
#         continue

#       # Get the variable assignments from the head.
#       head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}

#       for i in range(len(rule.body)):
#         # Ground atom by replacing variables with constants.
#         # The result is the partially ground atom A('Antonio',None)
#         # with None indicating unground variables.
#         body_atom = rule.body[i]
#         ground_body_atom = (body_atom[0], ) + tuple(
#             [head_ground_vars.get(body_atom[j+1], None)
#              for j in range(len(body_atom)-1)])
#         # print('\n- i', i,'. ground_body_atom:', ground_body_atom, '. Substitution (by None) of the vars not present in head.') if cont< lim else None
#         if all(ground_body_atom[1:]):
#             groundings = (ground_body_atom,)
#         else:
#             # Tuple of atoms matching A(Antonio,None) in the facts.
#             # This is the list of ground atoms for the i-th atom in the body.
#             # groundings = fact_index.get_matching_atoms(ground_body_atom)
#             groundings = fact_index._index.get(ground_body_atom, [])
#             # print(' GROUNDINGS', groundings) if cont< lim else None

#         if len(rule.body) == 1:
#             # Shortcut, we are done, the clause has no free variables.
#             # Return the groundings.
#             # print('groundings already done, #all vars are subtituted', groundings) if cont< lim else None
#             # print('ADDED', q, '->', (groundings,)) if cont< lim else None
#             new_ground_atoms.add(((q,), groundings))
#             continue

#         for ground_atom in groundings:
#             # print('     -GROUND ATOM', ground_atom) if cont< lim else None
#             # This loop is only needed to ground at least one atom in the body
#             # of the formula. Otherwise it would be enough to start with the
#             # loop for ground_vars in product(...) but it would often expand
#             # into many nodes. The current solution does not allow to fully
#             # recover a classical backward_chaining, though.
#             head_body_ground_vars = copy.copy(head_ground_vars)
#             head_body_ground_vars.update(
#                 {v: a for v, a in zip(body_atom[1:], ground_atom[1:])})

#             free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
#                                if v not in head_body_ground_vars]
#             free_vars = [vd[0] for vd in free_var2domain]
#             if pure_adaptive:
#                 ground_var_groups = [domain2adaptive_constants.get(vd[1], [])
#                                      for vd in free_var2domain]
#             elif domain2adaptive_constants is not None:
#                 ground_var_groups = [domains[vd[1]].constants +
#                                      domain2adaptive_constants.get(vd[1], [])
#                                      for vd in free_var2domain]
#             else:
#                 ground_var_groups = [domains[vd[1]].constants
#                                      for vd in free_var2domain]

#             # Iterate over the groundings of the free vars.
#             # If no free vars are available, product returns a single empty
#             # tuple, meaning that we still correctly enter in the following
#             # for loop for a single round.
#             # print('     FREE VARS_SPAN', list(product(*[domains[vd[1]].constants for vd in free_var2domain])))
#             for ground_vars in product(*ground_var_groups):
#                 var2ground = dict(zip(free_vars, ground_vars))
#                 full_ground_vars = {**head_body_ground_vars, **var2ground}
#                 # print('     FULL_VARS', full_ground_vars) if cont< lim else None

#                 accepted: bool = True
#                 body_grounding = []
#                 if build_proofs:
#                     body_grounding_to_prove = []
#                 unknown_fact_count: int = 0
#                 for j in range(len(rule.body)):
#                     if i == j:
#                         # print('         -j=i') if cont< lim else None
#                         new_ground_atom = ground_atom
#                         # by definition as it is coming from the groundings.
#                         is_known_fact = True
#                     else:
#                         body_atom2 = rule.body[j]
#                         new_ground_atom = (body_atom2[0], ) + tuple(
#                             [full_ground_vars.get(body_atom2[k+1], None)
#                              for k in range(len(body_atom2)-1)])
#                         if new_ground_atom == q:
#                             # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, ' Same atom as query, discard') if cont< lim else None
#                             accepted = False
#                             break
#                         is_known_fact = (fact_index._index.get(
#                             new_ground_atom, None) is not None)

#                     assert all(new_ground_atom), (
#                         'Unexpected free variables in %s' %
#                         str(new_ground_atom))
#                     if not is_known_fact and (
#                             max_unknown_fact_count < 0 or
#                             unknown_fact_count < max_unknown_fact_count):
#                         body_grounding.append(new_ground_atom)
#                         if build_proofs:
#                             body_grounding_to_prove.append(new_ground_atom)
#                         unknown_fact_count += 1
#                         # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Accepted. We have to prove it') if cont< lim else None
#                     elif is_known_fact:
#                         body_grounding.append(new_ground_atom)
#                         # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Accepted') if cont< lim else None
#                     else:
#                         # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Discard',unknown_fact_count,'/', max_unknown_fact_count) if cont< lim else None
#                         accepted = False
#                         break

#                 if accepted:
#                     # print('     ADDED', q, '->', tuple(body_grounding)) if cont< lim else None
#                     # print('ADDED', q, '->', tuple(body_grounding), 'TO_PROVE',          str(body_grounding_to_prove) if build_proofs else '')
#                     new_ground_atoms.add(((q,), tuple(body_grounding)))
#                     # groundings_per_query +=1
#                     if build_proofs:
#                         proofs.append((q, body_grounding_to_prove))
                    
#     #   print('NUM_GROUNDINGS for the query',q, groundings_per_query) #, 'TIME', end - start)
#     #   groundings_numbers.append(groundings_per_query)
#     #   print('       AVG_NUM_GROUNDINGS', sum(groundings_numbers)/len(groundings_numbers))
#     # print('AVG_NUM_GROUNDINGS', sum(groundings_numbers)/len(groundings_numbers))

#     end = time.time()
#     # print('NUM GROUNDINGS', len(new_ground_atoms),'. TIME', end - start)
#     # print('NEW GROUND ATOMS', new_ground_atoms) if cont< lim else None

#     if step not in groundings_per_level:
#         groundings_per_level[step] = set()
#     if new_ground_atoms is not None:
#         for g in new_ground_atoms:
#             groundings_per_level[step].add(g)

#     if res is None:
#         return new_ground_atoms, groundings_per_level
#     else:
#         res.update(new_ground_atoms), groundings_per_level

# # res is a Set of (Tuple_head_groundings, Tuple_body_groundings)
# #@profile
# def backward_chaining_grounding_one_rule(
#     groundings_per_level,
#     domains: Dict[str, Domain],
#     domain2adaptive_constants: Dict[str, List[str]],
#     pure_adaptive: bool,
#     rule: Rule,
#     queries: List[Tuple],
#     fact_index: AtomIndex,
#     res: Set[Tuple[Tuple, Tuple]]=None,
#     step=-1, n_steps=-1):
#     # We have a rule like A(x,y) B(y,z) => C(x,z)
#     assert len(rule.head) == 1, (
#         'Rule is not a Horn clause %s' % str(rule))
#     head = rule.head[0]

#     new_ground_atoms = set()

#     for q in queries:
#       if q[0] != head[0]:  # predicates must match.
#         continue

#       # Get the variable assignments from the head.
#       head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}

#       free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
#                          if v not in head_ground_vars]
#       free_vars = [vd[0] for vd in free_var2domain]
#       if pure_adaptive:
#           ground_var_groups = [domain2adaptive_constants.get(vd[1], [])
#                                for vd in free_var2domain]
#       elif domain2adaptive_constants is not None:
#           ground_var_groups = [domains[vd[1]].constants +
#                                domain2adaptive_constants.get(vd[1], [])
#                                for vd in free_var2domain]
#       else:
#           ground_var_groups = [domains[vd[1]].constants
#                                for vd in free_var2domain]

#       # Iterate over the groundings of the free vars.
#       # If no free vars are available, product returns a single empty
#       # tuple, meaning that we still correctly enter in the following
#       # for loop for a single round.
#       #print('FREE VARS_SPAN', list(product(*ground_var_groups)))
#       for ground_vars in product(*ground_var_groups):
#           var2ground = dict(zip(free_vars, ground_vars))
#           full_ground_vars = {**head_ground_vars, **var2ground}
#           body_grounding = []
#           for j in range(len(rule.body)):
#               body_atom = rule.body[j]
#               new_ground_atom = (body_atom[0], ) + tuple(
#                   [full_ground_vars.get(body_atom[k+1], None)
#                    for k in range(len(body_atom)-1)])
#               body_grounding.append(new_ground_atom)
#           new_ground_atoms.add(((q,), tuple(body_grounding)))

#     if step not in groundings_per_level:
#         groundings_per_level[step] = set()
#     if new_ground_atoms is not None:
#         for g in new_ground_atoms:
#             groundings_per_level[step].add(g)

#     if res is None:
#         return new_ground_atoms, groundings_per_level
#     else:
#         res.update(new_ground_atoms), groundings_per_level


# #@profile
# def PruneIncompleteProofs(rule2groundings: Dict[str, Set[Tuple[Tuple, Tuple]]],
#                           rule2proofs:Dict[str, List[Tuple[Tuple, List[Tuple]]]],
#                           fact_index: AtomIndex,
#                           num_steps: int) ->  Dict[str, Set[Tuple[Tuple, Tuple]]]:
#     #for rn,g in rule2groundings.items():
#     #    print('RIN', rn, len(g))
#     atom2proved: Dist[Tuple[str, str, str], bool] = {}

#     # This loop iteratively finds the atoms that are already proved.
#     for i in range(num_steps):
#         for rule_name,proofs in rule2proofs.items():
#             for query_and_proof in proofs:
#                 query, proof = query_and_proof[0], query_and_proof[1]
#                 if query not in atom2proved or not atom2proved[query]:
#                     atom2proved[query] = all(
#                         [atom2proved.get(a, False)
#                          # This next check is useless as atoms added in the proofs
#                          # are by definition not proved already in the data.
#                          # or fact_index._index.get(a, None) is not None)
#                          for a in proof])

#     # Now atom2proved has all proved atoms. Scan the groundings and keep only
#     # the ones that have been proved within num_steps:
#     pruned_rule2groundings = {}
#     for rule_name,groundings in rule2groundings.items():
#         pruned_groundings = []
#         for g in groundings:
#             head_atoms = g[0]
#             # WE CHECK IF ALL THE ATOMS IN THE HEAD ARE PROVED
#             # all elements in the grounding are either in the training data
#             # or they are provable using the rules,
#             if all([(atom2proved.get(a, False) or
#                      fact_index._index.get(a, None) is not None)
#                     for a in head_atoms]):
#                 pruned_groundings.append(g)
#         pruned_rule2groundings[rule_name] = set(pruned_groundings)
#     #for rn,g in pruned_rule2groundings.items():
#     #    print('ROUT', rn, len(g))
#     return pruned_rule2groundings

# def Prune_groundings_per_level(groundings_per_level,
#                           rule2proofs:Dict[str, List[Tuple[Tuple, List[Tuple]]]],
#                           fact_index: AtomIndex,
#                           num_steps: int) ->  Dict[str, Set[Tuple[Tuple, Tuple]]]:
#     #for rn,g in groundings_per_level.items():
#     #    print('RIN', rn, g)
#     # Here we will keep all the proves that are complete, i.e. all the atoms
#     atom2proved: Dist[Tuple[str, str, str], bool] = {}
#     # go through every atom to prove from all queries
#     for i in range(num_steps):
#         for rule_name,proofs in rule2proofs.items():
#             for query_and_proof in proofs:
#                 query, proof = query_and_proof[0], query_and_proof[1]
#                 if query not in atom2proved or not atom2proved[query]:
#                     atom2proved[query] = all(
#                         [atom2proved.get(a, False) for a in proof])

#     # Now atom2proved has all proved atoms. Scan the groundings and keep only
#     # the ones that have been proved within num_steps:
#     pruned_groundings_per_level = {}
#     pruned_groundings = []
#     for g in groundings_per_level:
#         head_atoms = g[0]
#         if all([(atom2proved.get(a, False) or
#                     fact_index._index.get(a, None) is not None)
#                 for a in head_atoms]):                
#             pruned_groundings.append(g)
#     pruned_groundings_per_level = set(pruned_groundings)
#     return pruned_groundings_per_level


# class ApproximateBackwardChainingGrounder(Engine):

#     def __init__(self, rules: List[Rule], facts: List[Union[Atom, str, Tuple]],
#                  domains: Dict[str, Domain],
#                  domain2adaptive_constants: Dict[str, List[str]]=None,
#                  pure_adaptive: bool=False,
#                  max_unknown_fact_count: int=1,
#                  max_unknown_fact_count_last_step: int=1,
#                  num_steps: int=1,
#                  max_groundings_per_rule: int=-1,  # to speedup the computation.
#                  # Whether the groundings should be accumulated across calls.
#                  accumulate_groundings: bool=False,
#                  prune_incomplete_proofs: bool=True):
#         self.max_unknown_fact_count = max_unknown_fact_count
#         self.max_unknown_fact_count_last_step = max_unknown_fact_count_last_step
#         self.num_steps = num_steps
#         self.accumulate_groundings = accumulate_groundings
#         self.prune_incomplete_proofs = prune_incomplete_proofs
#         self.max_groundings_per_rule = max_groundings_per_rule
#         self.rules = rules
#         self.domains = domains
#         self.domain2adaptive_constants = domain2adaptive_constants
#         self.pure_adaptive = pure_adaptive
#         self.facts = [a if isinstance(a,Tuple) else a.toTuple()
#                       if isinstance(a,Atom) else Atom(s=a).toTuple()
#                       for a in facts]
#         # self.facts = facts
#         for rule in self.rules:
#             assert len(rule.head) == 1, (
#                 '%s is not a Horn clause' % str(rule))
#         self._fact_index = AtomIndex(self.facts)
#         self.relation2queries = {}
#         self.rule2groundings = {}
#         self.rule2proofs = {}

#     def _init_internals(self, queries: List[Tuple], clean: bool):
#         # this tell us the queries for each relation to analyse.
#         self.relation2queries = {}  # reset
#         for q in queries:
#             if q[0] not in self.relation2queries:
#                 self.relation2queries[q[0]] = set()
#             self.relation2queries[q[0]].add(q)

#         # If clean=False, groundings are incrementally added.
#         for rule in self.rules:
#             if clean or rule.name not in self.rule2groundings:
#                 self.rule2groundings[rule.name] = set()
#             # if clean or rule.name not in self.rule2proofs:
#                 self.rule2proofs[rule.name] = []

#     # Ground a batch of queries, the result is cached for speed.
#     #@profile
#     def ground(self,
#                facts: List[Tuple],
#                queries: List[Tuple],
#                **kwargs):

#         if self.rules is None or len(self.rules) == 0:
#             return []

#         # When accumulating groundings, we keep a single large set of
#         # groundings that are reused over all batches.
#         groundings_per_level = {}
#         self._init_internals(queries, clean=(not self.accumulate_groundings))
#         # order also the relation2queries
#         # for k,v in self.relation2queries.items():
#         #     self.relation2queries[k] = sorted(list(v), key=lambda x: (x[0], x[1:])) if len(v) < 50 else v
#         # print('\nAtoms to process per query. self.relation2queries\n',self.relation2queries)
#         # Keeps track of the queris already processed for this rule.
#         self._rule2processed_queries = {rule.name: set() for rule in self.rules}
#         # groundings_numbers = []
#         for step in range(self.num_steps):
#             # print('STEP NUMBER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', step,'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^','step ',step,'/', self.num_steps, 'known body',step == self.num_steps - 1, )
#             for j,rule in enumerate(self.rules):
#                 # print('\nrule ', rule, ' """"""""""""""""""""""""""""""""""""""""" """""""""""""""""""""""""" ')
#                 # Here we assume to have a Horn clause, fix it.
#                 queries_per_rule = list(
#                     self.relation2queries.get(rule.head[0][0], set()))
#                 # print('\nqueries_per_rule\n',len(queries_per_rule), queries_per_rule)
#                 if not queries_per_rule:
#                     continue
#                 approximate_backward_chaining_grounding_one_rule(
#                     groundings_per_level,
#                     self.domains,
#                     self.domain2adaptive_constants,
#                     self.pure_adaptive,
#                     rule, queries_per_rule, self._fact_index,
#                     # max_unknown_fact_count
#                     (self.max_unknown_fact_count if step < self.num_steps - 1
#                      else self.max_unknown_fact_count_last_step),
#                     # Output added here.
#                     res=self.rule2groundings[rule.name],
#                     # Proofs added here.
#                     proofs=(self.rule2proofs[rule.name]
#                             if self.prune_incomplete_proofs else None),
#                     # groundings_numbers=groundings_numbers
#                     step=step, n_steps=self.num_steps
#                     )
#                 # Update the list of processed rules.
#                 self._rule2processed_queries[rule.name].update(queries_per_rule)
#                 print('Total  groundings in res after rule',j,'/',len(self.rules),', step',step,sum([len(v) for k, v in self.rule2groundings.items()])) # IS IS MORE THAN THE GROUNDINGS_per_level BECAUSE THERE ARE DUPLICATES

 
#             if step == self.num_steps - 1:
#                 break

#             # Get the queries for the next iteration.
#             new_queries = set()
#             for rule in self.rules:
#                 groundings = self.rule2groundings[rule.name]
#                 # Get the new queries left to prove, these facts that are not
#                 # been processed already and that are not known facts.
#                 new_queries.update(
#                     [a for a in get_atoms_on_groundings(groundings)
#                      if a not in self._rule2processed_queries[rule.name] and
#                      self._fact_index._index.get(a, None) is None])
#             # print('\nNEW Q',len(new_queries),'\n', list(new_queries), ' FROM groundings', len(groundings))
#             # Here we update the queries to process in the next iteration, we only keep the new ones.
#             self._init_internals(list(new_queries), clean=False)

#         print('Num groundings',sum([len(v) for k, v in self.rule2groundings.items()]))
#         if self.prune_incomplete_proofs:
#             # check all the groundings with at least 1 atom missing, to see if they are proved (all atoms present in the facts)
#             # print('\nstarting PruneIncompleteProofs')
#             self.rule2groundings = PruneIncompleteProofs(self.rule2groundings,
#                                                          self.rule2proofs,
#                                                          self._fact_index,
#                                                          self.num_steps)
#             print('Num groundings after pruning',sum([len(v) for k, v in self.rule2groundings.items()]))

#         for level in range(self.num_steps):
#             # if the level is in the keys of groundings_per_level, prune the groundings
#             if level in groundings_per_level:
#                 # print the keys of the groundings_per_level
#                 print('\nNum groundings in level',level,',',len(groundings_per_level[level]))
#                 if self.prune_incomplete_proofs:
#                     groundings_per_level[level] = Prune_groundings_per_level(groundings_per_level[level],
#                                                                 self.rule2proofs,
#                                                                 self._fact_index,
#                                                                 self.num_steps)
#                     print('Num groundings in level',level,', after pruning,',len(groundings_per_level[level]))

#         # Create a dict with all the relevant info, i.e., number of groundings per rule and the groundings per level.
        

#         # print('\nFinal groundings\n')
#         # This should be done after sorting the groundings to ensure the output
#         # to be deterministic.
#         if self.max_groundings_per_rule > 0:
#             self.rule2groundings = {rule_name:set(list(groundings)[:self.max_groundings_per_rule])
#                                     for rule_name,groundings in self.rule2groundings.items()}

#         if 'deterministic' in kwargs and kwargs['deterministic']:
#             ret = {rule_name: RuleGroundings(
#                 rule_name, sorted(list(groundings), key=lambda x : x.__repr__()))
#                    for rule_name,groundings in self.rule2groundings.items()}
#         else:
#             ret = {rule_name: RuleGroundings(rule_name, list(groundings))
#                    for rule_name,groundings in self.rule2groundings.items()}

#         return self.rule2groundings, groundings_per_level


# class BackwardChainingGrounder(Engine):

#     def __init__(self, rules: List[Rule],
#                  facts: List[Union[Atom, str, Tuple]],
#                  domains: Dict[str, Domain],
#                  domain2adaptive_constants: Dict[str, List[str]]=None,
#                  pure_adaptive: bool=False,
#                  num_steps: int=1,
#                  # Whether the groundings should be accumulated across calls.
#                  accumulate_groundings: bool=False):
#         self.num_steps = num_steps
#         self.accumulate_groundings = accumulate_groundings
#         self.rules = rules
#         self.domains = domains
#         self.domain2adaptive_constants = domain2adaptive_constants
#         self.pure_adaptive = pure_adaptive
#         self.facts = [a if isinstance(a,Tuple) else a.toTuple()
#                       if isinstance(a,Atom) else Atom(s=a).toTuple()
#                       for a in facts]
#         # self.facts = facts
#         for rule in self.rules:
#             assert len(rule.head) == 1, (
#                 '%s is not a Horn clause' % str(rule))
#         self._fact_index = AtomIndex(self.facts)
#         self.relation2queries = {}
#         self.rule2groundings = {}
#         self.rule2proofs = {}

#     def _init_internals(self, queries: List[Tuple], clean: bool):
#         self.relation2queries = {}  # reset
#         for q in queries:
#             if q[0] not in self.relation2queries:
#                 self.relation2queries[q[0]] = set()
#             self.relation2queries[q[0]].add(q)

#         # If clean=False, groundings are incrementally added.
#         for rule in self.rules:
#             if clean or rule.name not in self.rule2groundings:
#                 self.rule2groundings[rule.name] = set()
#                 self.rule2proofs[rule.name] = []

#     # Ground a batch of queries, the result is cached for speed.
#     #@profile
#     def ground(self,
#                facts: List[Tuple],
#                queries: List[Tuple],
#                **kwargs) :

#         if self.rules is None or len(self.rules) == 0:
#             return []

#         # When accumulating groundings, we keep a single large set of
#         # groundings that are reused over all batches.
#         groundings_per_level = {}
#         self._init_internals(queries, clean=(not self.accumulate_groundings))
#         # Keeps track of the queris already processed for this rule.
#         self._rule2processed_queries = {rule.name: set() for rule in self.rules}
#         for step in range(self.num_steps):
#             # print('STEP', step)
#             for j,rule in enumerate(self.rules):
#                 # Here we assume to have a Horn clause, fix it.
#                 queries_per_rule = list(
#                     self.relation2queries.get(rule.head[0][0], set()))
#                 if not queries_per_rule:
#                     continue
#                 backward_chaining_grounding_one_rule(
#                     groundings_per_level,
#                     self.domains,
#                     self.domain2adaptive_constants,
#                     self.pure_adaptive,
#                     rule, queries_per_rule, self._fact_index,
#                     # Output added here.
#                     res=self.rule2groundings[rule.name],
#                     step=step, n_steps=self.num_steps)
#                 # Update the list of processed rules.
#                 self._rule2processed_queries[rule.name].update(queries_per_rule)
#                 print('Total  groundings in res after rule',j,'/',len(self.rules),', step',step,sum([len(v) for k, v in self.rule2groundings.items()])) # IS IS MORE THAN THE GROUNDINGS_per_level BECAUSE THERE ARE DUPLICATES

#             if step == self.num_steps - 1:
#                 break

#             # Get the queries for the next iteration.
#             new_queries = set()
#             for rule in self.rules:
#                 groundings = self.rule2groundings[rule.name]
#                 # Get the new queries left to prove, these facts that are not
#                 # been processed already and that are not known facts.
#                 new_queries.update(
#                     [a for a in get_atoms_on_groundings(groundings)
#                      if a not in self._rule2processed_queries[rule.name] and
#                      self._fact_index._index.get(a, None) is None])
#             # print(step, 'NEW Q', list(new_queries)[:10], 'FROM', len(groundings))
#             self._init_internals(list(new_queries), clean=False)

#         for level in range(self.num_steps):
#             if level in groundings_per_level:
#                 print('\nNum groundings in level',level,',',len(groundings_per_level[level]))

#         # if 'deterministic' in kwargs and kwargs['deterministic']:
#         #     ret = {rule_name: RuleGroundings(
#         #         rule_name, sorted(list(groundings), key=lambda x : x.__repr__()))
#         #            for rule_name,groundings in self.rule2groundings.items()}
#         # else:
#         #     ret = {rule_name: RuleGroundings(rule_name, list(groundings))
#         #            for rule_name,groundings in self.rule2groundings.items()}

#         return self.rule2groundings, groundings_per_level



# def BuildGrounder(args, rules: List[Rule], facts: List[Tuple], fol: FOL,
#                   domain2adaptive_constants: Dict[str, List[str]]):
#     type = args.grounder
#     print('Building Grounder:', type, flush=True)

#     if 'backward' in type:
#         # if the count of '_' the name is 2, it means that the parameter 'a' is included. Else there is no parameter a. It goes after the first '_'
#         backward_width = None
#         if type.count('_') == 2:
#             backward_width = int(type[type.index('_')+1]) # take the first character after the first '_'
#             backward_depth = int(type[-1])
#             type = 'ApproximateBackwardChainingGrounder'
#         else:
#             backward_depth = int(type[-1])
#             type = 'BackwardChainingGrounder'

#         prune_incomplete_proofs = False #if (backward_width is None or backward_width == 0) else True
#         print('Grounder: ',args.grounder,'backward_depth:', backward_depth, 'Prune:', prune_incomplete_proofs, 'backward_width:', backward_width)

#     if type == 'ApproximateBackwardChainingGrounder':
#         # Requires Horn Clauses.
#         return ApproximateBackwardChainingGrounder(
#             rules, facts=facts, domains={d.name:d for d in fol.domains},
#             domain2adaptive_constants=domain2adaptive_constants,
#             pure_adaptive=get_arg(args, 'engine_pure_adaptive', False),
#             num_steps=backward_depth,
#             max_unknown_fact_count=backward_width,
#             max_unknown_fact_count_last_step=backward_width,
#             prune_incomplete_proofs=prune_incomplete_proofs,
#             max_groundings_per_rule=get_arg(
#                 args, 'backward_chaining_max_groundings_per_rule', -1),
#             )

#     elif type == 'BackwardChainingGrounder':
#         # Requires Horn Clauses.
#         return BackwardChainingGrounder(
#             rules, facts=facts,
#             domains={d.name:d for d in fol.domains},
#             domain2adaptive_constants=domain2adaptive_constants,
#             pure_adaptive=get_arg(args, 'engine_pure_adaptive', False),
#             num_steps=backward_depth)

#     elif type == 'full':
#         return DomainFullGrounder(
#             rules, domains={d.name:d for d in fol.domains},
#             domain2adaptive_constants=domain2adaptive_constants)
#     else:
#         assert False, 'Unknown grounder %s' % type

#     return None







def main(data_path,args):

    ''' 
    INFO TO WRITE IN TXT:
        For train, eval and test:  
        - Number of facts/queries
        - Time to create the data generator
        - Number of groundings
        - Number of groundings per level
        - Number of groundings per rule
        - Number of heads grounded
        - plots of the empirical distribution of the number of groundings per head (also cumulative)

    '''

    print('\nARGS', args,'\n')
    seed = get_arg(args, 'seed_run_i', 0)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    ragged = get_arg(args, 'ragged', None, True)
    start_train = time.time()

    # DATASET PREPARATION
    data_handler = KGCDataHandler(
        dataset_name=args.dataset_name,
        base_path=data_path,
        format=get_arg(args, 'format', None, True),
        domain_file= args.domain_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        fact_file= args.facts_file)
    

    fol = data_handler.fol
    domain2adaptive_constants: Dict[str, List[str]] = None

    # DEFINING RULES AND GROUNDING ENGINE
    rules = ns.utils.read_rules(join(data_path, args.dataset_name, args.rules_file),args)
    facts = list(data_handler.train_known_facts_set)
    engine = BuildGrounder(args, rules, facts, fol, domain2adaptive_constants)



    file = 'grounding_'+str(args.dataset_name) + '_' + str(args.grounder) + '.txt'
    folder = './experiments/grounding_info/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder+'plots/', exist_ok=True)

    # with open(folder+file, 'a') as f:
    #     f.write('\n\n\nDataset : '+str(args.dataset_name))
    #     f.write('\nGrounder : '+str(args.grounder)+'\n\n\n')
        
    # for set_data in ['test', 'valid', 'train']:
    for set_data in ['test']:

        # DATA GENERATORS
        if set_data == 'train': 
            dataset = data_handler.get_dataset(split="train",number_negatives=args.num_negatives)
        elif set_data == 'valid':
            dataset = data_handler.get_dataset(split="valid",number_negatives=args.valid_negatives, corrupt_mode=args.corrupt_mode)
        elif set_data == 'test':
            # dataset = data_handler.get_dataset(split="test",  number_negatives=args.test_negatives,  corrupt_mode=args.corrupt_mode)
            dataset = data_handler.get_dataset(split="test",  number_negatives=2)

        queries, labels  = dataset[0:len(dataset)]
        # take a random batch of 1000 queries
        # queries = random.sample(queries, 1000)
        # positive_queries = queries
        positive_queries = [q[0] for q in queries]
        positive_queries = list(set(positive_queries))

        groundings = {}
        # groundings_level = {}

        start = time.time()
        print('number of facts',len(facts))
        # all_groundings,groundings_level = engine.ground(tuple(facts),tuple(ns.utils.to_flat(queries)),deterministic=True)
        ret = engine.ground(tuple(facts),tuple(ns.utils.to_flat(queries)),deterministic=True)
        end = time.time()
        time_ground = np.round(end - start,2)
        print('time to ground',time_ground)
        # convert ret to all_groundings
        all_groundings = {}
        for rule_name,RuleGroundings in ret.items():
            all_groundings[rule_name] = RuleGroundings.groundings

        print('000')
        from collections import defaultdict
        num_groundings_per_head = defaultdict(int)
        for rule_name,groundings in all_groundings.items():
            for g in groundings:
                head = g[0][0]
                if head in positive_queries: # WE ARE FILTERING ONLY GROUNDINGS OF POSITIVE QUERIES, BUT THERE ARE MANY OTHER GROUNDINGS
                    num_groundings_per_head[head] += 1
                    # print('head',head)


        print('111')
        n_heads = len(num_groundings_per_head)

        # Count of unique values (histogram). The head of the counter is the number of groundings, the value is the number of heads with that number of groundings.
        num_groundings_per_head = Counter(num_groundings_per_head.values())
        # sort the counter
        num_groundings_per_head = dict(sorted(num_groundings_per_head.items(), key=lambda item: item[0]))
        print('222')
        n_groundings = sum([len(v) for k, v in all_groundings.items()])
        n_groundings_per_rule = {k: len(v) for k, v in all_groundings.items()}
        # n_groundings_level = {k: len(v) for k, v in groundings_level.items()}

        # print the results
        print('set : ',set_data)
        print('number_of_queries : ',len(positive_queries))
        print('time_to_ground : ',time_ground)
        print('n_groundings : ',n_groundings)
        # print('n_groundings_per_level : ',n_groundings_level)
        print('n_groundings_per_rule : ',n_groundings_per_rule)
        print('n_heads_grounded : ',n_heads)
        print('ratio of grounded queries : ',round(n_heads/len(positive_queries),3))

        # Write the results in a txt file
        with open(folder+file, 'w') as f:
            f.write('dataset : '+str(args.dataset_name)+'\n')
            f.write('grounder : '+str(args.grounder)+'\n\n')
            f.write('set : '+set_data+'\n')
            f.write('number_of_queries : '+str(len(positive_queries))+'\n')
            f.write('time_to_ground : '+str(time_ground)+'\n')
            f.write('n_groundings : '+str(n_groundings)+'\n')
            # f.write('n_groundings_per_level : '+str(n_groundings_level)+'\n')
            f.write('n_groundings_per_rule : '+str(n_groundings_per_rule)+'\n')
            f.write('n_heads_grounded : '+str(n_heads)+'\n')
            f.write('ratio of grounded queries : '+str(round(n_heads/len(positive_queries),3))+'\n')
            f.write('\n\n')
    
        
        # save the plot of the distribution to the file:   str(args.dataset_name) + ' - ' + str(args.grounder) + ' - ' + set_data + '.png'
        title = str(args.dataset_name) + ' - ' + str(args.grounder) + ' - ' + set_data
        plt.bar(num_groundings_per_head.keys(), num_groundings_per_head.values())
        plt.xlabel('Number of groundings')
        plt.ylabel('Number of heads')
        plt.title('Number of groundings per head. '+title, wrap=True)
        plt.savefig(folder  +'plots/' + title + '.png')
        plt.close()

        # plot the cumulative distribution
        plt.bar(num_groundings_per_head.keys(), np.cumsum(list(num_groundings_per_head.values())))
        plt.xlabel('Number of groundings')
        plt.ylabel('Cumulative number of heads')
        plt.title('Cumulative number of heads per number of groundings. '+title, wrap=True)
        plt.savefig(folder + 'plots/' + title + ' - cumulative.png')
        plt.close()

    return None



 