
# update the args in run
import argparse
def update_config(run: argparse.Namespace) -> argparse.Namespace:
    """Update configuration"""

    # run = argparse.Namespace(**updated_run, **run.__dict__)

    # Dataset-specific settings
    dataset = run.dataset_name
    run.num_rules = 0 if run.model_name == "no_reasoner" else 1
    run.valid_frequency = 1# 5 if not run.early_stopping else 1
    run.resnet = False if 'ablation' in dataset else run.resnet
    run.r2n_prediction_type = 'head' if 'ablation' in dataset else 'full'
    run.reasoner_atom_embedding_size = run.kge_atom_embedding_size
    run.reasoner_formula_hidden_embedding_size = run.kge_atom_embedding_size

    run.corrupt_mode = 'TAIL' if any(ds in dataset for ds in ['countries', 'ablation']) else 'HEAD_AND_TAIL'
    
    if dataset in {'pharmkg_full', 'wn18rr', 'FB15k237', 'kinship_family'}:
        run.seed = [0]

    if dataset in {'wn18rr', 'FB15k237', 'pharmkg_full'}:
        run.test_batch_size = 1
    elif dataset == 'kinship_family' or dataset == 'kinship':
        run.test_batch_size = 4
    elif dataset == 'countries_s3':
        run.test_batch_size = 128
    else:
        run.test_batch_size = 256

    # run.test_negatives = 1000 if run.dataset_name == 'FB15k237' else None  # all possible negatives

    # Embedding size adjustments
    kge_type = getattr(run, 'kge', '')
    atom_emb_size = getattr(run, 'kge_atom_embedding_size', 100)

    run.constant_embedding_size = (
        2 * atom_emb_size if kge_type in {"complex", "rotate"} else atom_emb_size)
    run.predicate_embedding_size = (
        2 * atom_emb_size if kge_type == "complex" else atom_emb_size)

    run_data = {
        'dataset_name': run.dataset_name,
        'grounder': run.grounder,
        'model_name': run.model_name,
        'kge': run.kge,
        'resnet': run.resnet,
        # 'test_negatives': run.test_negatives,
        'train_batch_size': run.batch_size,
        'val_batch_size': run.val_batch_size,
        'test_batch_size': run.test_batch_size,
        'rules_file': run.rules_file,
    }

    run_vars = tuple(run_data.values())

    run.run_signature = '-'.join(f'{v}' for v in run_vars)
    run.keys_signature = list(run_data.keys())
    return run