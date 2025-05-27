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
# from keras.callbacks import CSVLogger
from ns_lib.logic.commons import Atom, Domain, FOL, Rule, RuleLoader
from ns_lib.grounding.grounder_factory import BuildGrounder
from ns_lib.utils import MMapModelCheckpoint, KgeLossFactory, get_arg, load_model_weights, load_kge_weights
import time
from model_utils import * 
import wandb
from wandb.integration.keras import WandbCallback
from wandb.integration.keras import WandbMetricsLogger
from ns_lib.utils import save_embeddings_from_model
from tensorflow.keras.callbacks import TensorBoard
import datetime
explain_enabled: bool = False



def main(data_path, log_filename, use_WB, args):

    sorted_args = {k: args.__dict__[k] for k in sorted(args.__dict__)}
    print('\nSignature:', args.run_signature)
    print(f"\nRunning experiment: {sorted_args}\n")

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
    
    dataset_train = data_handler.get_dataset(split="train",number_negatives=args.num_negatives)
    dataset_valid = data_handler.get_dataset(split="valid",number_negatives=args.valid_negatives, corrupt_mode=args.corrupt_mode)
    dataset_test = data_handler.get_dataset(split="test",  number_negatives=args.test_negatives,  corrupt_mode=args.corrupt_mode)
    if explain_enabled and enable_rules and (args.model_name == 'dcr' or args.model_name == 'cdcr'):
        dataset_test_positive_only = data_handler.get_dataset(split="test", number_negatives=0, corrupt_mode=args.corrupt_mode)

    fol = data_handler.fol
    domain2adaptive_constants: Dict[str, List[str]] = None
    dot_product = get_arg(args, 'engine_dot_product', False)

    num_adaptive_constants = get_arg(args, 'engine_num_adaptive_constants', 0)


    # DEFINING RULES AND GROUNDING ENGINE
    rules = []
    engine = None

    enable_rules = (args.reasoner_depth > 0 and args.num_rules > 0)
    if enable_rules:
        rules = ns.utils.read_rules(join(data_path, args.dataset_name, args.rules_file),args)
        facts = list(data_handler.train_known_facts_set)
        engine = BuildGrounder(args, rules, facts, fol, domain2adaptive_constants)
    serializer = ns.serializer.LogicSerializerFast(
        predicates=fol.predicates, domains=fol.domains,
        constant2domain_name=fol.constant2domain_name,
        domain2adaptive_constants=domain2adaptive_constants)



    # DATA GENERATORS
    print('***********Generating train data**************')
    start = time.time()
    data_gen_train = ns.dataset.DataGenerator(
        dataset_train, fol, serializer, engine,
        batch_size=args.batch_size, ragged=ragged)
    end = time.time()
    args.time_ground_train = np.round(end - start,2)
    print("Time to create data generator train: ", np.round(end - start,2),'\n************************************')
    start = time.time()
    data_gen_valid = ns.dataset.DataGenerator(
       dataset_valid, fol, serializer, engine,
       batch_size=args.val_batch_size, ragged=ragged)
    end = time.time()
    args.time_ground_valid = np.round(end - start,2)
    print("Time to create data generator valid: ",  np.round(end - start,2),'\n************************************') 

    start = time.time()
    data_gen_test = ns.dataset.DataGenerator(
        dataset_test, fol, serializer, engine,
        batch_size=args.test_batch_size, ragged=ragged)
    end = time.time()
    args.time_ground_test = np.round(end- start,2)
    print("Time to create data generator test: ",  np.round(end - start,2),'\n************************************')


    # COMPILING MODEL
    model = CollectiveModel(
        fol, rules,
        kge=args.kge,
        kge_regularization=args.kge_regularization,
        model_name=get_arg(args, 'model_name', 'dcr'),
        constant_embedding_size=args.constant_embedding_size,
        predicate_embedding_size=args.predicate_embedding_size,
        kge_atom_embedding_size=args.kge_atom_embedding_size,
        kge_dropout_rate=args.kge_dropout_rate,
        reasoner_single_model=get_arg(args, 'reasoner_single_model', False),
        reasoner_atom_embedding_size=args.reasoner_atom_embedding_size,
        reasoner_formula_hidden_embedding_size=args.reasoner_formula_hidden_embedding_size,
        reasoner_regularization=args.reasoner_regularization_factor,
        reasoner_dropout_rate=args.reasoner_dropout_rate,
        reasoner_depth=args.reasoner_depth,
        aggregation_type=args.aggregation_type,
        signed=args.signed,
        resnet=get_arg(args, 'resnet', False),
        embedding_resnet=get_arg(args, 'embedding_resnet', False),
        temperature=args.temperature,
        filter_num_heads=args.filter_num_heads,
        filter_activity_regularization=args.filter_activity_regularization,
        num_adaptive_constants=num_adaptive_constants,
        dot_product=dot_product,
        cdcr_use_positional_embeddings=get_arg(
            args, 'cdcr_use_positional_embeddings', True),
        cdcr_num_formulas=get_arg(args, 'cdcr_num_formulas', 3),
        r2n_prediction_type=get_arg(args, 'r2n_prediction_type', 'full'),
        distill=get_arg(args, 'distill', False),
    )

    if args.stop_kge_gradients:
        model.kge_model.trainable = False
        print('GRADIENTS OF KGE STOPPED')


    #LOSS
    loss_name = get_arg(args, 'loss', 'binary_crossentropy')
    loss = KgeLossFactory(loss_name)

    loss = {'concept': loss, 'task': loss}
    loss_weights = {'concept': 1-args.weight_loss, 'task': args.weight_loss }
    if args.distill_kge_labels:
        loss_weights = {'concept': 0, 'task': 0}
        print('Distilling only KGE labels')

    metrics = {
        'concept': [
            ns.utils.MRRMetric(),
            ns.utils.HitsMetric(1),
            ns.utils.HitsMetric(3),
            ns.utils.HitsMetric(10)
        ],
        'task': [
            ns.utils.MRRMetric(),
            ns.utils.HitsMetric(1),
            ns.utils.HitsMetric(3),
            ns.utils.HitsMetric(10)
        ]
    }

    optimizer,lr_scheduler = choose_optimizer_scheduler(args.optimizer,args.lr_sched,args.learning_rate)
    model.compile(optimizer=optimizer,
                    loss=loss,
                    loss_weights = loss_weights,
                    metrics=metrics,
                    run_eagerly=False)



    # CHECKPOINT HANDLING

    # create model's layers and weights to be able to load the weights 
    if not model.built:
        _ = model(next(iter(data_gen_train))[0])  # force building the model.

    # LOAD CKPT
    assert not args.load_model_ckpt or not args.load_kge_ckpt, "Only one of ckpt_load and load_kge_ckpt can be set."
    name = args.run_signature+'_seed_'+str(seed)
    current_dir = os.getcwd()
    ckpt_filepath = os.path.join(current_dir, args.ckpt_folder, name, name)
    os.makedirs(args.ckpt_folder, exist_ok=True)

    # If checkpoint_load is not None, try to load the weights
    if args.load_model_ckpt or args.load_kge_ckpt:
        path_ = ckpt_filepath if args.load_model_ckpt else ckpt_filepath+'_kge_model'
        if '_kge_model' in path_:
            # subtitute args.model_name with 'no_reasoner'
            path_ = path_.replace(args.model_name, 'no_reasoner')  
            # Find directory with no_reasoner, args.kge and args.dataset_name
            base_dir = args.ckpt_folder
            matching_dirs = [d for d in os.listdir(base_dir) 
                            if os.path.isdir(os.path.join(base_dir, d)) 
                            and 'no_reasoner' in d 
                            and args.kge in d 
                            and args.dataset_name in d
                            and 'seed_' + str(seed) in d]
            
            if not matching_dirs:
                raise ValueError(f"No directory found with 'no_reasoner', '{args.kge}' and '{args.dataset_name}'")
            
            assert len(matching_dirs) == 1, f"Multiple directories found: {matching_dirs}"
            no_reasoner_dir = os.path.join(base_dir, matching_dirs[0])
            path_ = os.path.join(no_reasoner_dir, no_reasoner_dir.split('/')[-1]+'_kge_model')
            success = load_kge_weights(model, path_, verbose=True)
        else:
            success = load_model_weights(model, path_, verbose=True)
        # update the arg that was true
        args.load_model_ckpt = success if args.load_model_ckpt else args.load_model_ckpt
        args.load_kge_ckpt = success if args.load_kge_ckpt else args.load_kge_ckpt



    # CALLBACKS
    callbacks = []
    if log_filename is not None:
        csv_logger = ns.utils.CustomCSVLogger(log_filename, append=True, separator=';') 
        callbacks.append(csv_logger)

    if args.lr_sched=='plateau':
      callbacks.append(lr_scheduler)
    
    if args.early_stopping:
        early_stopping = keras.callbacks.EarlyStopping(
            # monitor="val_loss",
            # mode='min',
            monitor="val_task_mrr",
            mode='max',
            patience=40,
            verbose=1,
            )
        callbacks.append(early_stopping)
    
    model_checkpoint = MMapModelCheckpoint(
        model=model,
        monitor='val_task_mrr',
        filepath= ckpt_filepath if args.save_model_ckpt else None,
        save_best_only=True,
        save_weights_only=True,
        name='model'
    )

    kge_model_checkpoint = MMapModelCheckpoint(
        model=model.kge_model,
        monitor='val_concept_mrr',
        filepath= ckpt_filepath+'_kge_model' if args.save_kge_ckpt else None,
        save_best_only=True,
        save_weights_only=True,
        name='kge_model'
    )

    callbacks.append(model_checkpoint)
    callbacks.append(kge_model_checkpoint)

    # # Add TensorBoard callback with profiling enabled
    # REMEMBER TO ADD THE @tf.function decorator to the model's call method
    # date = datetime.datetime.now()
    # date_log = date.strftime("%Y_%m_%d_%H_%M_%S")
    # log_dir = os.path.join("logs", "fit", date_log)
    # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='0,10')
    # callbacks.append(tensorboard_callback)

    # class ProfilingCallback(tf.keras.callbacks.Callback):
    #     def __init__(self, log_dir):
    #         self.log_dir = log_dir
    #         self.step = 0  # Initialize step counter

    #     def on_train_begin(self, logs=None):
    #         tf.profiler.experimental.start(self.log_dir)

    #     def on_train_batch_begin(self, batch, logs=None):
    #         # Start profiling step (Not always necessary with tf.function)
    #         # tf.profiler.experimental.start_step()
    #         pass  # No-op in most cases since model.fit() will use tf.function

    #     def on_train_batch_end(self, batch, logs=None):
    #         # End profiling step (Not always necessary with tf.function)
    #         # tf.profiler.experimental.stop_step()
    #         self.step += 1 # Increment the step count

    #     def on_train_end(self, logs=None):
    #         tf.profiler.experimental.stop()

    # # Create the profiling callback
    # profiling_callback = ProfilingCallback(log_dir)
    # callbacks.append(profiling_callback)

    # Initialize a W&B run
    if use_WB:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dir = os.path.join(current_dir, '../..')
        run = wandb.init(project = "Grounders-exp", 
                        group=args.run_signature,
                        name=args.run_signature+'_seed_'+str(seed),
                        dir=dir,  
                        config = dict(
                            dataset = args.dataset_name,
                            model = args.model_name,
                            grounder = args.grounder,
                            seed = seed,
                            # **vars(args),
                            config = args), 
                        tags = [args.dataset_name, args.model_name, args.grounder],
                        )
        callbacks.append(WandbMetricsLogger(log_freq=10))



    # TRAIN
    do_training = args.epochs > 0 #and not (args.load_model_ckpt or args.load_kge_ckpt):
    if do_training: 
        
        # tf.profiler.experimental.start(log_dir)

        history = model.fit(data_gen_train,
                epochs=args.epochs,
                callbacks=callbacks,
                validation_data=data_gen_valid,
                validation_freq=args.valid_frequency)
        # tf.profiler.experimental.stop()

        end_train = time.time()

        # Close the W&B run
        if use_WB:
            run.finish()
    
        args.time_train = np.round(end_train - start_train,2)
        print('Training time:', np.round(end_train - start_train,2), 'seconds')

        # Restore the best weights after training
        kge_model_checkpoint.restore_weights() if args.load_kge_ckpt else model_checkpoint.restore_weights()
        kge_model_checkpoint.write_train_time(args.time_train) if args.load_kge_ckpt else model_checkpoint.write_train_time(args.time_train)

    else:
        if use_WB:
            run.finish()
        args.time_train = 0
    
    # save_embeddings_from_model(model, fol, serializer, save_dir="./../embeddings/"+args.run_signature)    

    # EVALUATION
    print("\nEvaluation test", flush=True)
    start_inf = time.time()
    test_metrics = model.evaluate(data_gen_test, return_dict=True)
    end_inf = time.time()
    args.time_inference = np.round(end_inf - start_inf,2)

    if do_training:
        print("\nEvaluation train", flush=True)
        train_metrics = model.evaluate(data_gen_train, return_dict=True)
        print("\nEvaluation val", flush=True)
        valid_metrics = model.evaluate(data_gen_valid, return_dict=True)    
    else:
        # create a copy of test metrics, with all values set to 0
        train_metrics = valid_metrics = {k: 0 for k in test_metrics.keys()}
    print('Inference time:', np.round(end_inf - start_inf,2), 'seconds')

    training_info = history.history if do_training else None

    print('\nResults'),
    print('Metrics:',train_metrics.keys(),
          '\nTrain', np.round(np.array(list(train_metrics.values())), 3),
          '\nVal', np.round(np.array(list(valid_metrics.values())), 3),
          '\nTest', np.round(np.array(list(test_metrics.values())), 3),
          flush=True)

    if get_arg(args, 'store_ranks', False):
        from ns_lib.utils import evaluate_and_store_ranks
        evaluate_and_store_ranks(model, data_gen_test, seed, args, test_metrics['task_mrr'])
        
    if explain_enabled and enable_rules and (args.model_name == 'dcr' or args.model_name == 'cdcr'):
        model.explain_mode(True)
        print('\nExplain Train', flush=True)
        print(model.predict(data_gen_train)[-1])

        print('\nExplain Test', flush=True)
        data_gen_test_explain = ns.dataset.DataGenerator(
            dataset_test, fol, serializer, engine, batch_size=-1, ragged=ragged)
        print(model.predict(data_gen_test_explain)[-1])

        data_gen_test_positive_only = ns.dataset.DataGenerator(
            dataset_test_positive_only, fol, serializer, engine,
            batch_size=args.test_batch_size, ragged=ragged)
        for r in model.reasoning[-1].rule_embedders.values():
            r._verbose=True
        print(model.predict(data_gen_test_positive_only)[-1])
    return train_metrics,valid_metrics, test_metrics, training_info
