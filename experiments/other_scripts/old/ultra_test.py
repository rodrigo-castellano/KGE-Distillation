import sys
import os
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ULTRA'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))
import unittest
from dataset import KGCDataHandler
from typing import List, Tuple, Dict
import ns_lib as ns
from ns_lib.serializer import LogicSerializerFast
from ns_lib.dataset import DataGenerator, _from_strings_to_tensors
from ns_lib.utils import MMapModelCheckpoint, KgeLossFactory
from model import CollectiveModel
from model_utils import optimizer_scheduler

from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
from torch.utils import data as torch_data
import torch.nn as nn
from torch import distributed as dist


from ULTRA.ultra import tasks, util,layers,datasets
from ULTRA.ultra.tasks import build_relation_graph
from ULTRA.ultra.models import Ultra,RelNBFNet,EntityNBFNet

import numpy as np
import tensorflow as tf
import math
import itertools
import copy


'''
This is a test to compare the loaded dataset from ULTRA and the one from our framework. 
Mainly I want to check that aux_dataset (Dataset class) is the same, specifically for the attributes: 
    edge_index, edge_type, num_relations, num_nodes, num_edges, device

    target_edge_index and target_edge_type are the queries. Check them also.
    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
'''

#CHECK THE CORRRRRRRRRRRRRRRRRRRUPTIONSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
# CHECK THE EXACT IMPUT GIVEN TO ULTRA

#######################################################################################################
################################################## DATA LOADING ############################################
#######################################################################################################

class Dataset_Ultra():
    
        def __init__(self, edge_index=None, edge_type=None, num_relations=None, num_nodes=None, num_edges=None, device=None, target_edge_index=None, target_edge_type=None):
            self.edge_index = edge_index
            self.edge_type = edge_type
            self.num_relations = num_relations
            self.num_nodes = num_nodes
            self.num_edges = num_edges
            self.device = device
            self.target_edge_index = target_edge_index
            self.target_edge_type = target_edge_type
  
def obtain_queries(dataset,data_handler,serializer,engine,ragged,deterministic,global_serialization):
    queries, labels = dataset[:]
    constants_features = dataset.constants_features
    fol = data_handler.fol
    
    ((X_domains_data, A_predicates_data, A_rules_data, Q, (Q_global,A_predicates_triplets,A_predicates_textualized)), y) = _from_strings_to_tensors(
        fol=fol,
        serializer=serializer,
        queries=queries,
        labels=labels,
        engine=engine,
        ragged=ragged,
        constants_features=constants_features,
        deterministic=deterministic,
        global_serialization=global_serialization) 
    Q_global_positive = [q[0] for q in Q_global]
    # print('\nqueries positive', len(queries), [query[0] for query in queries][:20])
    # print('Q_global_positive', len(Q_global_positive), Q_global_positive[:20])
    return X_domains_data, A_predicates_data, Q_global_positive

def get_ultra_datasets(dataset_train, dataset_valid, dataset_test,data_handler,serializer,engine,ragged,deterministic,global_serialization,relation_graph=build_relation_graph):

    # Get the triplets
    X_domain_train, A_pred_train, train_triplets = obtain_queries(dataset_train,data_handler,serializer,engine,ragged,deterministic,global_serialization)
    X_domain_valid, A_pred_valid, valid_triplets = obtain_queries(dataset_valid,data_handler,serializer,engine,ragged,deterministic,global_serialization)
    X_domain_test, A_pred_test, test_triplets = obtain_queries(dataset_test,data_handler,serializer,engine,ragged,deterministic,global_serialization)

    def unique_ordered(triplets):
        return list(dict.fromkeys(tuple(t) for t in triplets))

    train_triplets = unique_ordered(train_triplets)
    valid_triplets = unique_ordered(valid_triplets)
    test_triplets = unique_ordered(test_triplets)

    # get the number of nodes and relations for the train,val,test set. Do it by getting unique the ones in train, val, test
    train_nodes = [X_domain_train[key].numpy().tolist() for key in X_domain_train]
    valid_nodes = [X_domain_valid[key].numpy().tolist() for key in X_domain_valid]
    test_nodes = [X_domain_test[key].numpy().tolist() for key in X_domain_test]

    # Flatten the lists
    train_nodes = set(itertools.chain(*train_nodes))
    valid_nodes = set(itertools.chain(*valid_nodes))
    test_nodes = set(itertools.chain(*test_nodes))
    num_node = len(train_nodes.union(valid_nodes).union(test_nodes) )

    # do the same for the relations
    train_relations = [key for key in A_pred_train]
    valid_relations = [key for key in A_pred_valid]
    test_relations = [key for key in A_pred_test]
    # take the unique number of relations
    unique_relations = list(set(train_relations+valid_relations+test_relations))
    num_relations_no_inv = torch.tensor(len(unique_relations))
    # num_relations_no_inv = len(data_handler.fol.predicates)
    
    train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
    train_target_etypes = torch.tensor([t[2] for t in train_triplets])
    train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
    train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations_no_inv])

    # valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
    valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
    valid_etypes = torch.tensor([t[2] for t in valid_triplets])

    test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
    test_etypes = torch.tensor([t[2] for t in test_triplets])

    train_data = Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                      target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations_no_inv*2)
    train_data.num_edges = train_data.edge_index.shape[1]
    valid_data = Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                      target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations_no_inv*2)
    valid_data.num_edges = valid_data.edge_index.shape[1]
    test_data = Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                      target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations_no_inv*2)
    test_data.num_edges = test_data.edge_index.shape[1]
    
    # edge_index is the sum of all target_edge_index
    edge_index = torch.cat([train_target_edges, valid_edges, test_edges], dim=1)
    edge_type = torch.cat([train_target_etypes, valid_etypes, test_etypes])
    # num_nodes is given by the train set
    num_edges = None # is not defined in Ultra for the general dataset
    device = 'cpu'
    dataset = Dataset_Ultra(edge_index, edge_type, num_relations_no_inv*2, num_node, num_edges, device)
    filtered_data = dataset 

    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    return train_data, valid_data, test_data, filtered_data

def ini_ultra():
    dataset = datasets.CountriesTransductive('~/git/ULTRA/kg-datasets/')
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    return train_data, valid_data, test_data, dataset


def ini_our_framework():
    data_handler = KGCDataHandler(dataset_name="countries_s1",
                            base_path="experiments\data",
                            format="functional",
                            domain_file='domain2constants.txt',
                            train_file='train.txt',
                            valid_file='valid.txt',
                            test_file='test.txt')
    dataset_train = data_handler.get_dataset(split="train",number_negatives=2)
    dataset_valid = data_handler.get_dataset(split="valid",number_negatives=None, corrupt_mode='HEAD_AND_TAIL')
    dataset_test = data_handler.get_dataset(split="test",  number_negatives=None,  corrupt_mode='HEAD_AND_TAIL')

    fol = data_handler.fol
    facts = list(data_handler.train_known_facts_set)
    domain2adaptive_constants: Dict[str, List[str]] = None
    
    serializer = LogicSerializerFast(
        predicates=fol.predicates, domains=fol.domains,
        constant2domain_name=fol.constant2domain_name,
        domain2adaptive_constants=domain2adaptive_constants)

    engine = None

    # do the first run to get the info for ultra
    train_ultra, valid_ultra, test_ultra, filtered_data_ultra = get_ultra_datasets(dataset_train, dataset_valid, dataset_test,data_handler,serializer,engine,ragged=True,deterministic=True,global_serialization=True)

    print('starting traning set loading')
    data_gen_train = DataGenerator(
        dataset_train, fol, serializer, engine,
        batch_size=16, ragged=True,
        use_ultra=True, use_ultra_with_kge=False, 
        global_serialization=True, dataset_ultra=train_ultra)
    
    print('train set loading done')
    data_gen_valid = DataGenerator(
       dataset_valid, fol, serializer, engine,
        batch_size=16, ragged=True,
        use_ultra=True, use_ultra_with_kge=False, 
        global_serialization=True, dataset_ultra=valid_ultra)

    print('valid set loading done')
    data_gen_test = DataGenerator(
        dataset_test, fol, serializer, engine,
        batch_size=16, ragged=True,
        use_ultra=True, use_ultra_with_kge=False, 
        global_serialization=True, dataset_ultra=test_ultra)
    print('test set loading done')

    print('\nRUNNING TRAIN SET')
    print('*********************')
    data_gen_train.__getitem__(0)
    print('*********************')
    # print('\nRUNNING VALID SET' )
    # print('*********************')
    # data_gen_valid.__getitem__(0)
    # print('*********************')
    # print('\nRUNNING TEST SET')
    # print('*********************')
    # data_gen_test.__getitem__(0)
    # print('*********************')
    return (data_gen_train, data_gen_valid, data_gen_test), fol, filtered_data_ultra


def print_info_dataset(data):
    torch.set_printoptions(profile="full")
    print('data.edge_index', data.edge_index.shape, data.edge_index[:,:50]) # num of triplets
    print('data.edge_type', data.edge_type.shape, data.edge_type[:50])
    print('data.num_relations', data.num_relations)
    print('data.num_nodes', data.num_nodes)
    print('data.num_edges', data.num_edges)
    print('\nRelation graph')
    print('relation_graph', data.relation_graph)
    print('relation_graph.num_nodes', data.relation_graph.num_nodes)
    print('relation_graph.num_edges', data.relation_graph.num_edges)
    print('relation_graph.edge_index', data.relation_graph.edge_index)
    print('relation_graph.edge_type', data.relation_graph.edge_type)



def test_dataset_loader():
    # ULTRA ORIGINAL, TRANSDUCTIVE SETTING
    train_data, valid_data, test_data, dataset = ini_ultra()
    
    # data_gen_train=data_gen_valid=data_gen_test=filtered_data_ultra=fol=None 
    (data_gen_train, data_gen_valid, data_gen_test),fol,filtered_data_ultra = ini_our_framework()
    
    # print('ULTRA')
    # for data in [train_data]:#, train_data, valid_data, test_data, dataset]:
    #     print_info_dataset(data)

    # print('OUR')
    # for data in [data_gen_train.aux_dataset]: #, data_gen_valid.aux_dataset, data_gen_test.aux_dataset, filtered_data_ultra]:
    #     print_info_dataset(data)

    return (train_data, valid_data, test_data, dataset), (data_gen_train, data_gen_valid, data_gen_test, filtered_data_ultra), fol



#######################################################################################################
################################################## METRICS ############################################
#######################################################################################################

def ragged_to_dense(labels, predictions, weights):
  """Converts given inputs from ragged tensors to dense tensors.

  Args:
    labels: A `tf.RaggedTensor` of the same shape as `predictions` representing
      relevance.
    predictions: A `tf.RaggedTensor` with shape [batch_size, (list_size)]. Each
      value is the ranking score of the corresponding example.
    weights: An optional `tf.RaggedTensor` of the same shape of predictions or a
      `tf.Tensor` of shape [batch_size, 1]. The former case is per-example and
      the latter case is per-list.

  Returns:
    A tuple (labels, predictions, weights, mask) of dense `tf.Tensor`s.
  """
  _PADDING_LABEL = -1.
  _PADDING_PREDICTION = -1e6
  _PADDING_WEIGHT = 0.
  # TODO: Add checks to validate (ragged) shapes of input tensors.
  mask = tf.cast(tf.ones_like(labels).to_tensor(0.), dtype=tf.bool)
  labels = labels.to_tensor(_PADDING_LABEL)
  if predictions is not None:
    predictions = predictions.to_tensor(_PADDING_PREDICTION)
  if isinstance(weights, tf.RaggedTensor):
    weights = weights.to_tensor(_PADDING_WEIGHT)
  return labels, predictions, weights, mask

def _get_shuffle_indices(shape, mask=None, shuffle_ties=True, seed=None):
  """Gets indices which would shuffle a tensor.

  Args:
    shape: The shape of the indices to generate.
    mask: An optional mask that indicates which entries to place first. Its
      shape should be equal to given shape.
    shuffle_ties: Whether to randomly shuffle ties.
    seed: The ops-level random seed.

  Returns:
    An int32 `Tensor` with given `shape`. Its entries are indices that would
    (randomly) shuffle the values of a `Tensor` of given `shape` along the last
    axis while placing masked items first.
  """
  # Generate random values when shuffling ties or all zeros when not.
  if shuffle_ties:
    shuffle_values = tf.random.uniform(shape, seed=seed)
  else:
    shuffle_values = tf.zeros(shape, dtype=tf.float32)

  # Since shuffle_values is always in [0, 1), we can safely increase entries
  # where mask=False with 2.0 to make sure those are placed last during the
  # argsort op.
  if mask is not None:
    shuffle_values = tf.where(mask, shuffle_values, shuffle_values + 2.0)

  # Generate indices by sorting the shuffle values.
  return tf.argsort(shuffle_values, stable=True)

def sort_by_scores(scores,
                   features_list,
                   topn=None,
                   shuffle_ties=True,
                   seed=None,
                   mask=None):
  """Sorts list of features according to per-example scores.

  Args:
    scores: A `Tensor` of shape [batch_size, list_size] representing the
      per-example scores.
    features_list: A list of `Tensor`s to be sorted. The shape of the `Tensor`
      can be [batch_size, list_size] or [batch_size, list_size, feature_dims].
      The latter is applicable for example features.
    topn: An integer as the cutoff of examples in the sorted list.
    shuffle_ties: A boolean. If True, randomly shuffle before the sorting.
    seed: The ops-level random seed used when `shuffle_ties` is True.
    mask: An optional `Tensor` of shape [batch_size, list_size] representing
      which entries are valid for sorting. Invalid entries will be pushed to the
      end.

  Returns:
    A list of `Tensor`s as the list of sorted features by `scores`.
  """
  with tf.compat.v1.name_scope(name='sort_by_scores'):
    scores = tf.cast(scores, tf.float32)
    scores.get_shape().assert_has_rank(2)
    list_size = tf.shape(input=scores)[1]
    if topn is None:
      topn = list_size
    topn = tf.minimum(topn, list_size)

    # Set invalid entries (those whose mask value is False) to the minimal value
    # of scores so they will be placed last during sort ops.
    if mask is not None:
      scores = tf.where(mask, scores, tf.reduce_min(scores))

    # Shuffle scores to break ties and/or push invalid entries (according to
    # mask) to the end.
    shuffle_ind = None
    if shuffle_ties or mask is not None:
      shuffle_ind = _get_shuffle_indices(
          tf.shape(input=scores), mask, shuffle_ties=shuffle_ties, seed=seed)
      scores = tf.gather(scores, shuffle_ind, batch_dims=1, axis=1)

    # Perform sort and return sorted feature_list entries.
    _, indices = tf.math.top_k(scores, topn, sorted=True)
    if shuffle_ind is not None:
      indices = tf.gather(shuffle_ind, indices, batch_dims=1, axis=1)
    return [tf.gather(f, indices, batch_dims=1, axis=1) for f in features_list]

class MRRMetric(tf.keras.metrics.Metric):
  """Implements mean reciprocal rank (MRR). It uses the same implementation
     of tensorflow_ranking MRRMetric but with an online check for ragged
     tensors."""
  def __init__(self, name='mrr', dtype=None, **kwargs):
      super().__init__(name, dtype, **kwargs)
      self.mrr = self.add_weight("total", initializer="zeros")
      self._count = self.add_weight("count", initializer="zeros")
      self.reset_state()

  def reset_state(self):
      self.mrr.assign(0.)
      self._count.assign(0.)

  def result(self):
      return tf.math.divide_no_nan(self.mrr, self._count)

  def update_state(self, y_true, y_pred, sample_weight=None):
    mrrs = self._compute(y_true, y_pred)
    self.mrr.assign_add(tf.reduce_sum(mrrs))
    self._count.assign_add(tf.reduce_sum(tf.ones_like(mrrs)))

  def _compute(self, labels, predictions):
    if any(isinstance(tensor, tf.RaggedTensor)
           for tensor in [labels, predictions]):
      labels, predictions, _, _ = ragged_to_dense(labels, predictions, None)

    topn = tf.shape(predictions)[1] #  number of predictions per sample, which is the size of the second dimension of the predictions tensor
    sorted_labels, = sort_by_scores(predictions, [labels], topn=topn, mask=None) # sort the labels by the predictions
    sorted_list_size = tf.shape(input=sorted_labels)[1] # usually is the same as topn, unless for example I only care about the top 3 predictions
    # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
    relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32) # if the label is greater or equal to 1, then the relevance is 1, otherwise 0
    reciprocal_rank = 1.0 / tf.cast(
        tf.range(1, sorted_list_size + 1), dtype=tf.float32) #  This generates a range of ranks from 1 to the size of the sorted list. The reciprocal rank is 1/rank
    # MRR has a shape of [batch_size, 1].
    # Element-wise Multiplication: relevance * reciprocal_rank computes the reciprocal rank for relevant items (i.e., where relevance is 1.0)
    # Maximum Reciprocal Rank: tf.reduce_max(..., axis=1, keepdims=True) finds the maximum reciprocal rank for each sample across the list of predictions. This is because MRR considers the highest (earliest) rank of a relevant item.
    mrr = tf.reduce_max(
        input_tensor=relevance * reciprocal_rank, axis=1, keepdims=True) 
    return mrr



def preprocess_tf_metrics(h_pred, t_pred, pos_h_index, pos_t_index, num_h_negative, num_t_negative, h_batch, t_batch, h_mask, t_mask):
    """
    Preprocess data for TensorFlow metrics calculation.
    
    This function processes head and tail predictions, creates labels, filters predictions,
    and calculates Mean Reciprocal Rank (MRR) for both head and tail predictions.
    
    Args:
    h_pred, t_pred: Tensor, predictions for heads and tails
    pos_h_index, pos_t_index: Tensor, indices of positive samples for heads and tails
    num_h_negative, num_t_negative: Tensor, number of negative samples for heads and tails
    h_batch, t_batch: Tensor, batch data for heads and tails
    h_mask, t_mask: Tensor, masks for filtering valid samples
    
    Returns:
    mrr_head, mrr_tail: float, Mean Reciprocal Rank for head and tail predictions
    """
    
    # Helper function to process both head and tail data
    def process_batch(pred, batch, mask, pos_index, num_negative):
        pred_np = pred.cpu().detach().numpy()
        pos_index_np = pos_index.cpu().numpy()
        num_negative_np = num_negative.cpu().numpy()
        
        labels = []
        predictions_filtered = []
        
        for i, sample in enumerate(batch.cpu().numpy()[:,:,0] if batch.shape[-1] == 1 else batch.cpu().numpy()[:,:,1]):
            # Create label vector: 1 for positive sample, 0 for others
            label = np.zeros(len(sample))
            label[pos_index_np[i]] = 1
            
            # Ensure positive sample is included in the mask
            mask[i][pos_index_np[i]] = True
            
            # Filter predictions and labels based on the mask
            pred_filtered = pred_np[i][mask[i].cpu().numpy()]
            label_filtered = label[mask[i].cpu().numpy()]
            
            # Verify the number of filtered samples
            assert num_negative_np[i] + 1 == len(pred_filtered), 'Mismatch in number of valid samples'
            
            predictions_filtered.append(pred_filtered)
            labels.append(label_filtered)
        
        return labels, predictions_filtered
    
    # Process head and tail data
    labels_h, predictions_h_filtered = process_batch(h_pred, h_batch, h_mask, pos_h_index, num_h_negative)
    labels_t, predictions_t_filtered = process_batch(t_pred, t_batch, t_mask, pos_t_index, num_t_negative)
    
    # Convert to TensorFlow ragged tensors
    labels_tensor_h = tf.ragged.constant(labels_h, dtype=tf.float32)
    labels_tensor_t = tf.ragged.constant(labels_t, dtype=tf.float32)
    predictions_tensor_h = tf.ragged.constant(predictions_h_filtered, dtype=tf.float32)
    predictions_tensor_t = tf.ragged.constant(predictions_t_filtered, dtype=tf.float32)
    
    # Compute Mean Reciprocal Rank (MRR)
    mrr_metric = MRRMetric()
    mrr_metric.update_state(labels_tensor_h, predictions_tensor_h)
    mrr_head = mrr_metric.result().numpy()
    
    mrr_metric.reset_states()
    mrr_metric.update_state(labels_tensor_t, predictions_tensor_t)
    mrr_tail = mrr_metric.result().numpy()
    
    # Print results
    print(f'TF MRR head: {mrr_head:.3f}, TF MRR tail: {mrr_tail:.3f}, TF MRR average: {(mrr_head + mrr_tail) / 2:.3f}')
    
    return mrr_head, mrr_tail

 

#######################################################################################################
################################################## EVALUATIONS ########################################
#######################################################################################################

@torch.no_grad()
def test(model, test_data, device, filtered_data=None, return_metrics=True):
    world_size = util.get_world_size()
    rank = util.get_rank()
    # test_triplets=[n_triplets, 3]
    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, 16, sampler=sampler)
    model.eval()
    rankings = []
    num_negatives = []
    tail_rankings, num_tail_negs = [], []  # for explicit tail-only evaluation needed for 5 datasets
    mrr_head_all, mrr_tail_all = [], []
    for i,batch in enumerate(test_loader):
        print('\nbatch',i, 'of',len(test_loader), batch.shape)#,batch)
        t_batch, h_batch = tasks.all_negative(test_data, batch) # t_batch=[batch_size=16, 271, 3] all the negative tails. For each of the 16 heads, 1594 negative tails are sampled
        # print('t_batch', t_batch.shape, 'h_batch', h_batch.shape)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch) # h_pred=[batch_size,271] scores for each of the 271 negative heads
        # if we are in transductive: filtered is the whole dataset, i.e., we check that in corruptions there's not any triple from the whole dataset
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)

        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)
 
        # GET THE METRICS FROM TF
        mrr_head, mrr_tail = preprocess_tf_metrics(h_pred, t_pred, pos_h_index, pos_t_index, num_h_negative, num_t_negative, h_batch, t_batch, h_mask, t_mask)
        mrr_head_all.append(mrr_head)
        mrr_tail_all.append(mrr_tail)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]
    
    print('\n TF avg MRR head', np.round(np.mean(mrr_head_all),3), 'TF avg MRR tail', np.round(np.mean(mrr_tail_all),3), 'TF avg MRR', np.round((np.mean(mrr_head_all)+np.mean(mrr_tail_all))/2,3))

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    # ugly repetitive code for tail-only ranks processing
    tail_ranking = torch.cat(tail_rankings)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    # obtaining all ranks 
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    # the same for tails-only ranks
    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

    print('\nMetrics from Ultra:')
    metrics = {}
    if rank == 0:
        for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
            if "-tail" in metric:
                _metric_name, direction = metric.split("-")
                if direction != "tail":
                    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking 
                _num_neg = all_num_negative 
                _metric_name = metric
            
            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            print("%s: %g" % (metric, score))
            metrics[metric] = score
    mrr = (1 / all_ranking.float()).mean()

    return mrr if not return_metrics else metrics



def eval_ULTRA_framework(datasets,rel_model_cfg, entity_model_cfg, ultra=True):
    
    train_data, valid_data, test_data, dataset = datasets
    model = Ultra(rel_model_cfg, entity_model_cfg)
    
    ckp = None
    ckp = 'C:\\Users\\rodri\\Downloads\\PhD_code\\Review_grounders\\keras_ns_grounders\\ULTRA\\ckpts\\countries_ft_epoch_10.pth'
    if ckp is not None:
        state = torch.load(ckp, map_location="cpu")
        model.load_state_dict(state["model"])
    model = model.to('cpu')

    if ultra:
      filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=train_data.num_nodes)
    else:
      filtered_data = dataset

    val_filtered_data = test_filtered_data = filtered_data
    print('\nvalid data, ultra framework')
    metrics = test(model, valid_data, 'cpu', filtered_data=filtered_data, return_metrics=True)
    print('\ntest data, ultra framework')
    metrics = test(model, test_data, 'cpu', filtered_data=filtered_data, return_metrics=True)

    return metrics



################################################
################## MAIN ########################
################################################

# global rel_model_cfg
rel_model_cfg = {
        "class": "RelNBFNet",
        "input_dim": 64,
        "hidden_dims": [64, 64, 64, 64, 64, 64],
        "message_func": "distmult",
        "aggregate_func": "sum",
        "short_cut": "yes",
        "layer_norm": "yes"
    }


# global entity_model_cfg 
entity_model_cfg= {
        "class": "EntityNBFNet",
        "input_dim": 64,
        "hidden_dims": [64, 64, 64, 64, 64, 64],
        "message_func": "distmult",
        "aggregate_func": "sum",
        "short_cut": "yes",
        "layer_norm": "yes"
    }

ULTRA_datasets, OUR_datasets, fol = test_dataset_loader()

# print('\n\nULTRA RESULTS')
# metrics_ultra = eval_ULTRA_framework(ULTRA_datasets,copy.deepcopy(rel_model_cfg), copy.deepcopy(entity_model_cfg))

# print('\n\nOUR RESULTS')  
# (data_gen_train, data_gen_valid, data_gen_test, filtered_data_ultra) = OUR_datasets
# OUR_datasets_aux = (data_gen_train.aux_dataset, data_gen_valid.aux_dataset, data_gen_test.aux_dataset, filtered_data_ultra)
# metrics_our_framework = eval_ULTRA_framework(OUR_datasets_aux,copy.deepcopy(rel_model_cfg), copy.deepcopy(entity_model_cfg),ultra=False)

# metrics_our_framework = eval_ULTRA_framework(OUR_datasets,ultra=False)
# metrics_our_framework = eval_ourframework_OURMETRIC(OUR_datasets,fol)





