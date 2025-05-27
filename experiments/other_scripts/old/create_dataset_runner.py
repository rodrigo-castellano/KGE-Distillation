import sys
sys.path.append('C:\\Users\\rodri\\Downloads\\PhD\\Review_grounders\\keras_ns_grounders')
sys.path.append('/home/castellanoontiv/keras_ns_grounders')
sys.path.append('/media/users/castellanoontiv/keras_ns_grounders/')
import os
import copy
import os
from itertools import product
from create_dataset_train import main
import shutil as sh
import keras_ns as ns
from keras_ns.utils import NSParser
import time
import datetime
import numpy as np
import ast
import tensorflow as tf



if __name__ == '__main__':

    base_path :str = "data"
    epochs: int = 100
    assert epochs > 0
    DATASET_NAME =  ['countries_s3'] # ['pharmkg_small','kinship_family','pharmkg_full']  #
    MODIFIED_DATASET = [False]# True]
    GROUNDER = ['backward_2']#,'backward_3']    
    KGE = ['complex']   
    MODEL_NAME =  ['sbr',]  
    RULE_MINER = ['amie','None'] 
    E = [100] 
    DEPTH = [1]
    SEED = [[0,1,2,3,4]]
    NEG_PER_SIDE = [1]
    WEIGHT_LOSS = [.5]  
    DROPOUT = [0.0]
    R = [0.0]
    RR = [0.0]
    LR = [0.01]
    NUM_RULES = [1] 
    VALID_SIZE = [None]


    all_args = []

    for dataset_name,modified_dataset, grounder, kge, model_name, rule_miner, e, dp, seed, neg, w_loss,  dropout, r, lr, nr, rr in product(
            DATASET_NAME,MODIFIED_DATASET, GROUNDER, KGE, MODEL_NAME, RULE_MINER, E, DEPTH, SEED, NEG_PER_SIDE, WEIGHT_LOSS, DROPOUT, R,
            LR, NUM_RULES, RR ):  
    

        # Base parameters
        parser = NSParser()
        args = parser.parse_args()


        if modified_dataset:
            if 'backward' not in grounder:
                continue
            else:
                level = grounder[-1]
                dataset_name = dataset_name+'_reason_'+level
        if not os.path.exists(os.path.join(base_path, dataset_name)):
            continue



        if 'countries' in dataset_name:
            # task is the last two letters of the dataset name
            task = dataset_name[-2:]
            if task == 's2' and (grounder == 'full'): # or grounder == 'domainbody'): domainbody sometimes gives problems
                print('skipping, grounder too heavy', run_vars)
                continue
            elif task == 's3' and (grounder == 'full' or grounder == 'domainbody'):
                print('skipping, grounder too heavy', run_vars)
                continue

        args.dataset_name = dataset_name
        args.grounder = grounder
        args.kge = kge
        args.model_name = model_name 
        args.rule_miner = rule_miner 
        args.modified_dataset = modified_dataset
        args.seed = seed
        args.kge_atom_embedding_size = e
        args.facts_file = 'facts.txt'
        args.train_file = 'train.txt'  
        args.valid_file = 'valid.txt'
        args.test_file = 'test.txt'
        args.domain_file = 'domain2constants.txt'
        args.rules_file = 'rules.txt'

        if rule_miner == 'amie':
            args.rules_file = 'rules_amie.txt'
        elif rule_miner == 'ncrl':
            args.rules_file = 'rules_ncrl.txt'
        elif rule_miner == 'None':
            args.rules_file = 'rules.txt'
        else: # raise an error if the rule miner is not recognized
            raise ValueError('Rule miner not recognized for ', dataset_name)
        if not os.path.exists(os.path.join(base_path, dataset_name, args.rules_file)):
            # print('skipping, rules not existing', run_vars)
            continue

        # Data params
        args.num_negatives = neg # 1
        args.valid_negatives = 100 #200
        args.test_negatives = None  # all possible negatives
        if dataset_name == 'pharmkg_full' or dataset_name == 'kinship_family':
            args.test_negatives = 1000
        args.ragged = True
        args.format = "functional"
        args.engine_num_negatives = 0
        args.engine_num_adaptive_constants = 0
        args.constant_embedding_size = (
            2 * args.kge_atom_embedding_size
            if args.kge == "complex" or args.kge == "rotate"
            else args.kge_atom_embedding_size)
        # KGE params
        args.dropout_rate_embedder = dropout
        args.kge_regularization = r
        # Model params
        args.learning_rate = lr
        args.epochs = epochs
        args.num_rules = 0 if model_name == "no_reasoner"  else nr
        args.loss = "binary_crossentropy"
        args.weight_loss = w_loss
        args.batch_size = -1 # Full batch only for explain.
        args.val_batch_size = -1
        args.test_batch_size = 256
        args.cdcr_use_positional_embeddings = False
        args.cdcr_num_formulas = 3
        args.valid_frequency = 3
        args.resnet = True
        args.reasoner_depth = dp if nr > 0 else 0
        args.reasoner_regularization_factor = rr
        args.reasoner_formula_hidden_embedding_size = args.kge_atom_embedding_size
        args.reasoner_dropout_rate = dropout
        args.reasoner_atom_embedding_size = args.kge_atom_embedding_size
        # DCR/R2N params
        args.signed = True
        args.temperature = 0.0
        args.aggregation_type = "max"
        args.filter_num_heads = 3
        args.filter_activity_regularization = 0.0
        # Other
        # args.adaptation_layer = "identity"  # "dense", "sigmoid","identity"
        # args.output_layer = "dense" # "wmc" or "kge" or "positive_dense" or "max"
        # args.relation_entity_grounder_max_elements = 20
        # args.semiring = "product"

        run_vars = (args.dataset_name,grounder, kge, model_name, rule_miner, modified_dataset, seed, neg,e)
        args.keys_signature = ['dataset_name','grounder', 'kge', 'model_name', 'rule_miner', 'modified_dataset', 'seed', 'neg','e']
        args.run_signature = '-'.join(f'{v}' for v in run_vars)     
        all_args.append(args)


     

    def main_wrapper(args): 
        date = str(datetime.datetime.now()).replace(":","-")
        date = str(datetime.datetime.now()).replace(":","-").replace(" ","-")
        date = date[:date.index('.')]
        log_folder :str = "results"
        log_filename_tmp = os.path.join(log_folder, '_tmp_log_{}_{}.csv'.format(date,args.run_signature))

        main(base_path,None,None,log_filename_tmp,args)


    for args in all_args:
        print('Experiment number ', all_args.index(args), ' out of ', len(all_args), ' experiments.')
        main_wrapper(args)
