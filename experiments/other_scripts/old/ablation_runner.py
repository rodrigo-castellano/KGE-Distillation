import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(True)
print("GPUs used: ", gpus)
# tf.config.run_functions_eagerly(True)
import copy
from itertools import product
import shutil as sh
import ns_lib as ns
from ns_lib.utils import NSParser
import numpy as np
import ast
import argparse
from model_utils import * 

from ablation_count_groundings import main

if __name__ == '__main__':

    log_folder :str = "./experiments/runs/"
    ckpt_folder = "./../checkpoints/"
    data_path :str = "experiments/data"
    DATASET_NAME = ['test_small'] #['ablation_d1','ablation_d2','ablation_d3']# ['kinship_family','pharmkg_small','pharmkg_full','wn18rr'] # 'countries_s1','countries_s2','countries_s3',
    GROUNDER = ['backward_0_1', 'backward_0_2','backward_0_3','backward_1_1','backward_1_2','backward_1_3','backward_2_1','backward_2_2','backward_2_3'] #, 'backward_noprune_1_1', 'backward_1_2', 'backward_noprune_1_2', 'backward_1_3', 'backward_noprune_1_3', 'full']
    MODEL_NAME =  ['sbr']  
    RULE_MINER = ['amie','None'] 
    SEED = [[0]]
    NEG_PER_SIDE = [0]


    # This is in case I want to take inputs from the command line

    parser = argparse.ArgumentParser(description='Description of your script')  
    parser.add_argument("--d", default = None, help="dataset",nargs='+')
    parser.add_argument("--m", default = None, help="model",nargs='+')
    parser.add_argument("--g", default = None, help="grounder",nargs='+')
    parser.add_argument("--s", default = None, help="seed")

    args = parser.parse_args()
    if args.s is not None:
        SEED = [ast.literal_eval(args.s) ]
        assert isinstance(SEED, list) 
    if args.m is not None:
        MODEL_NAME = args.m
    if args.d is not None:
        DATASET_NAME = args.d
    if args.g is not None:
        GROUNDER = args.g

    del args.s
    del args.m
    del args.d  
    del args.g
    print('Running experiments for the following parameters:','DATASET_NAME:',DATASET_NAME,'GROUNDER:',GROUNDER,'MODEL_NAME:',MODEL_NAME,'SEED:',SEED)
    
    all_args = []
    for dataset_name,grounder, model_name, rule_miner, seed, neg, in product(
            DATASET_NAME,GROUNDER, MODEL_NAME, RULE_MINER, SEED, NEG_PER_SIDE,):  

        if not os.path.exists(os.path.join(data_path, dataset_name)):
            print('skipping, dataset not existing', dataset_name)
            continue

        # Define sets for quick membership testing
        light_datasets_full = {'countries_s1', 'countries_s2', 'pharmkg_small'}
        heavy_datasets_domainbody_relationentity = {'countries_s3', 'wn18rr', 'pharmkg_full', 'FB15k237', 'kinship_family'}

        # Discern the datasets for which the grounders full, domainbody, and relationentity are too heavy to run
        if model_name == 'no_reasoner' and grounder not in {'backward_1_1'}:
            continue

        if dataset_name == 'countries_s3' and (grounder == 'full'):
            continue
        elif (dataset_name == 'kinship_family' or dataset_name == 'wn18rr' or dataset_name == 'FB15k237') and (grounder == 'full'
                     or grounder == 'backward_1_3' or grounder == 'backward_2_1' or grounder == 'backward_2_2' or grounder == 'backward_2_3'):
            continue

        args.dataset_name = dataset_name
        args.grounder = grounder
        args.rule_miner = rule_miner 
        args.seed = seed
        args.ragged = True
        args.format = "functional"

        if (dataset_name == 'pharmkg_full' or dataset_name == 'wn18rr' or dataset_name == 'FB15k237'): # For heavy datasets, run only one seed
            args.seed = [0]
        args.facts_file = 'facts.txt'
        args.train_file = 'train.txt'  
        args.valid_file = 'valid.txt'
        args.test_file = 'test.txt'
        args.domain_file = 'domain2constants.txt'
        args.rules_file = 'rules.txt'

        # Select the rules file
        rules_files = {'amie': 'rules_amie.txt', 'ncrl': 'rules_ncrl.txt', 'None': 'rules.txt'}
        args.rules_file = rules_files.get(rule_miner) or ValueError(f'Rule miner not recognized for {dataset_name}')
        if not os.path.exists(os.path.join(data_path, dataset_name, args.rules_file)):
            continue

        # Data params
        args.corrupt_mode = 'TAIL' if ('countries' in dataset_name or 'ablation' in dataset_name or 'test' in dataset_name) else 'HEAD_AND_TAIL'
        args.num_negatives = neg  
        args.valid_negatives = 0
        args.test_negatives = 500 if dataset_name=='FB15k237' else None # all possible negatives

        all_args.append(copy.deepcopy(args))

    def main_wrapper(args): 
        main(data_path,args)

    for args in all_args:
        main_wrapper(args)
