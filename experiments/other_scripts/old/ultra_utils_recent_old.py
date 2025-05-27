from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
import torch.nn as nn
from ULTRA.ultra import layers
from ULTRA.ultra.models import RelNBFNet,BaseNBFNet

import numpy as np
import tensorflow as tf

from collections import defaultdict

def nested_dict(n, type):
    """
    Creates a nested defaultdict with n levels.

    Parameters:
    n (int): The number of nested levels for the defaultdict.
    type: The default data type for the defaultdict.

    Returns:
    defaultdict: A nested defaultdict with n levels.
    """
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))


def pytorch_to_tf(pytorch_tensor):
    numpy_array = pytorch_tensor.detach().cpu().numpy()
    tf_tensor = tf.convert_to_tensor(numpy_array)
    return tf_tensor

def tf_to_pytorch(tf_tensor,type='float'):

    if tf.executing_eagerly():
        # Eager mode: Directly use numpy()
        numpy_array = tf_tensor.numpy()
    else:
        # Graph mode: Use eval() within a session
        with tf.compat.v1.Session() as sess:
            # Convert RaggedTensor to Tensor
            tensor = tf.RaggedTensor.to_tensor(tf_tensor)
            # Evaluate the tensor in the session
            numpy_array = tensor.eval()

    if type == 'float':
        pytorch_tensor = torch.tensor(numpy_array,dtype=torch.float32)  
    elif type == 'int':
        pytorch_tensor = torch.tensor(numpy_array,dtype=torch.int64)  

    return pytorch_tensor



class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1,get_scores=False, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )
        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        self.get_scores = get_scores
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        # self.mlp = nn.Sequential(*mlp)
        self.mlp_embedd = nn.Sequential(*mlp)
        if get_scores:
            mlp.append(nn.Linear(feature_dim, 1))
        self.mlp_score = nn.Sequential(*mlp)
    
    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)
        # initialize queries (relation types of the given triples)
        # print('torch.arange(batch_size, device=r_index.device)',torch.arange(batch_size, device=r_index.device).shape)
        # print('r_index:',r_index.shape)
        # create a vector which is [range(len of the batch),relations index]. That vector is used to select the self.query(relation embeddings) for each query
        # In few words, for each query, I get the relation embedding of the relation of the query, and that is used as ini of the tail node of each query
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index] #[n_nodes=tail node of each query,embedd_size of each node]
        # print('query:',query.shape, query)
        # print('h_index.unsqueeze(-1)',h_index.unsqueeze(-1).shape)
        # index goes from [n_queries,1] to [n_queries,node_reperesentation_dim]. It repeats the same index n_nodes times
        index = h_index.unsqueeze(-1).expand_as(query)
        # print('index:',index.shape,index)
        # THE PROBLEM IS THAT THE NUMBER OF NODES IS 248, BUT IN REALITY THERE ARE UP TO 270
        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # print('boundary:',boundary.shape,boundary)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        # print('index.unsqueeze(1):',index.unsqueeze(1).shape)
        # print('query.unsqueeze(1):',query.unsqueeze(1).shape)
        # to boundary, I sum the query based on the index. Index_i says to what positions of boundary I should add the query_i
        # if in index I have the idx 270, in boundary the first dim needs to have shape 270, instead of 248.
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary
        for layer in self.layers:
            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states 
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch, atom_repr=False):
        h_index, t_index, r_index = batch.unbind(-1)
        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)
            # UPDATE THE NUMBER OF EDGES
            data.num_edges = data.edge_index.shape[1]

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        # here Im supposed to have head with all the same indices for every query
        # IN TRAINING, I GET AN ERROR IN H_INDEX, BECAUSE I SHOULD HAVE ONLY HEAD OR TAIL CORRUPTION, BUT IF I HAVE THEM MIXED, I GET AN ERROR. I NEED TO ADAPT THE FUNCTION
        #  TO PUT ALL THE NEGATIVES IN THE TAIL SIDE
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations of the head entities 
        # print('h_index:',h_index.shape)
        # print('r_index:',r_index.shape)
        # print('h_index:',h_index[:, 0].shape,h_index[:, 0])
        # print('r_index:',r_index[:, 0].shape,r_index[:, 0])
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]

        if atom_repr:                                                                                                                                                                                                                     
            # Get the tail indices [19,num_negative+1], then unesqueeze it to [19,num_negative+1,1], then expand (repeat) it to [19,num_negative+1,64] 
            index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
            # here, for every query (pos and neg), I get the tail embeddings
            feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)
            # From these embeddings of the pos and negatives queries, Im just interested in the postive query (self representation)
            feature = feature[:, :1, :]

        embedd = self.mlp_embedd(feature)
        scores = self.mlp_score(feature) if self.get_scores else None
        # print('embedd:',embedd.shape)
        # print('scores:',scores.shape)
        return embedd, scores







class Ultra(nn.Module):

    def __init__(self):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()
        self.relation_model = RelNBFNet(input_dim=64, hidden_dims=[64, 64, 64, 64, 64, 64], message_func='distmult', aggregate_func='sum', short_cut=True, layer_norm=True)
        self.entity_model = EntityNBFNet(input_dim=64, hidden_dims=[64, 64, 64, 64, 64, 64], message_func='distmult', aggregate_func='sum', short_cut=True, layer_norm=True,get_scores=True)

    def split_head_tail_negatives(self, batch, relation_representations):
        '''
        Problem: In the same list of corrptions, I cannot have positives and negatives, so I need to split it in two queries
        From batch:[n_queries,n_corruptions,3] I output [n_queries+splits,n_corruptions,3] where the first dim is for head and the second for tail corruptions
        Given a batch of queries, check for every query, check for each corruption if it is a head or tail corruption. 
            - Initialize both dimensions to -1
            - If it is a head corruption, put it in the first dim, if it is a tail corruption, put it in the second dim.
            - If there are head and tail corruptions, put them in the first and second dim respectively

        return:[n_queries+splits,n_corruptions,3], where the first dim is for head and the second for tail corruptions. Usually, n_queries+splits = 2*n_queries    

        Comment: in training it only does corruptions to tail, and in testing it does corruptions to head and tail, but in a separate array. In our framework, during training,
          both head and tail corruptions are in the same array, so I need to split them.

        Bonus: if I increate the queries, I have to adjust the relation representations
        '''
        n_queries, n_corruptions, _ = batch.shape
        original_head = batch[:, 0, 0]  # [n_queries]
        negative_heads = batch[:, :, 0]  # [n_queries, n_corruptions]
        is_head = (original_head[:, None] == negative_heads)  # bool: [n_queries, n_corruptions]

        # Precompute the conditions for which no modifications are needed
        no_modifications_needed = torch.all(is_head, dim=1) | torch.all(~is_head, dim=1)

        # Now, what I do is, for each query, and if is_head of that query isn't all true or all false,
        # I go through every corruption, and if it is true, I leave it as it is, otherwise, I flip it by
        # inverting the head and tail, and for the relation I add the number of relations
        new_queries = []
        new_relation_representation = []

        for i in range(n_queries):
            if no_modifications_needed[i]:
                new_queries.append(batch[i])
                new_relation_representation.append(relation_representations[i])
            else:
                heads_pos = [batch[i][0].unsqueeze(0)]  # Add the positive triple
                tails_pos = [batch[i][0].unsqueeze(0)]  # Add the positive triple
                for j in range(1, n_corruptions):  # for every corruption
                    if is_head[i][j]:
                        heads_pos.append(batch[i][j].unsqueeze(0))
                    else:
                        tails_pos.append(batch[i][j].unsqueeze(0))
                if len(heads_pos) > 1:  # if there are tail corruptions
                    new_queries.append(torch.cat(heads_pos, dim=0))
                    new_relation_representation.append(relation_representations[i])
                if len(tails_pos) > 1:  # if there are head corruptions
                    new_queries.append(torch.cat(tails_pos, dim=0))
                    new_relation_representation.append(relation_representations[i])
        new_queries = torch.stack(new_queries)
        relation_representations = torch.stack(new_relation_representation)
        return new_queries, relation_representations
    
    def convert_to_domain(self, entity_representations, fol):
        '''
        Given the entity representations, I put them in a dict with different domains. I could take the dic with the index for every constant and domain, but it is easier and faster this way
        The order followed here is the same order followed when creating constant_to_global_index
        '''
        # self.constant_to_global_index = defaultdict(OrderedDict)
        entity_embeddings = {domain.name:[] for domain in fol.domains}
        for domain in fol.domains:
            # Add fixed constants.
            for i, constant in enumerate(domain.constants):
                # self.constant_to_global_index[domain.name][constant] = i
                # entity_embeddings[domain.name].append(entity_representations[i].detach().numpy())
                entity_embeddings[domain.name].append(entity_representations[i].detach().cpu().numpy())
        
        # # convert to tf tensors
        # for key in entity_embeddings.keys():
        #     entity_embeddings[key] = tf.convert_to_tensor(entity_embeddings[key])

        return entity_embeddings
    
    def split_corruptions(self, batch):
        '''
        Given a batch, split it in subbatches with the same number of corruptions in each subbatch. In this way it is easy to convert it to a pytorch version.
        '''
        n_queries = len(batch)
        subbatches = {}
        for i in range(n_queries):
            n_corruptions = len(batch[i])
            if n_corruptions not in subbatches:
                subbatches[n_corruptions] = []
            subbatches[n_corruptions].append([batch[i]])
        return subbatches

    def forward(self, data, batch, atom_repr=False):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs

        # Batch: [16=batch_size,1594,3]=[16 heads(tails), 1594 is, for each head(tail), the number of negative tails(heads), 3 is indeces full negative triple(h, t, r)]
        # For all the heads(tails) in the batch, take the first element of the negatives(for all negatives, the relation is the same), and the relation index
        # query_rels: torch.Size([16]) 
        # print('number of queries/A_pred:',len(batch),batch[:10])
        batches = self.split_corruptions(batch)
        for key in batches.keys():
            batches[key] = np.concatenate(batches[key], axis=0)
            batches[key] = torch.tensor(batches[key], dtype=torch.int64)

        

        # For each batch, get the relation representations
        all_relation_representations = []
        all_entity_representations = []
        all_scores = []
        for key,batch in batches.items():
            # print('Batch_i:',batch.shape)
            # if the number of dimensions is 2,add a dimension in the middle (it would mean that there are no negatives, only positives). This is thought for the atom repersentation
            if len(batch.shape) == 2:
                batch = batch.unsqueeze(1)
            query_rels = batch[:, 0, 2]

            # For each query, get a representation of all the relations of dim 64
            # relation_representations:  torch.Size([16, 360, 64])=(queries,n_relationsx2,dim_embedd)
            relation_representations = self.relation_model(data.relation_graph, query=query_rels)
            # print('relation_representations:',relation_representations.shape)
 
            # Given the batch [16,1594,3], do a prediction, for each of the 16 heads(tails) in the queries, of all possible tails(heads) candidates
            # score:  torch.Size([16, 1594])

            # Instead of an array of [n_queries, n_corruptions, 3], I get [n_queries*2, n_corruptions/2, 3] in training, where for each q, half of the corrup. are negatives (head corr) and the other half are tail corruptions
            # I put them as a two different queries, so that I can pass them to the entity model
            # print('batch:',batch.shape)
            # print('relation_representations:',relation_representations.shape)
            batch,relation_representations = self.split_head_tail_negatives(batch,relation_representations)
            # It may be that the queries have indices of entities greater than the number of entities in the graph, so I need to filter them. SHUOLD NOT HAPPEN!!!
            entity_representations, scores = self.entity_model(data, relation_representations, batch,atom_repr=atom_repr) # [16,1594,64] = [batch_size, num_negatives, embedd_size]
            # PLEASE make sure that in entity_representations:[1,354=n_nodes,64], the first node correspond to the first index and so on
            # print('entity_representations:',entity_representations.shape)
            all_relation_representations.append(relation_representations)
            all_entity_representations.append(entity_representations)
            all_scores.append(scores)

        # print('all_entity_representations:')
        # for i in range(len(all_entity_representations)):
        #     print(i,all_entity_representations[i].shape)
        # print('all_scores:')
        # for i in range(len(all_scores)):
        #     print(i,all_scores[i].shape)
        # print('all_relation_representations:')
        # for i in range(len(all_relation_representations)):
        #     print(i,all_relation_representations[i].shape)

        if not atom_repr:
            # print('relation_representations before:',relation_representations.shape)
            # print('entity_representations before:',entity_representations.shape)
            # By now, as a temporary solution, I will take the average of hte relative embeddings of each entity
            relation_representations = relation_representations[:,data.num_relations//2:,:] # Dont take the inverse relations
            relation_representations = torch.mean(relation_representations, dim=0)
            relation_representations = pytorch_to_tf(relation_representations)

            entity_representations = torch.mean(entity_representations, dim=0)
            # print('entity_representations after:',entity_representations.shape)
            # Here, if the entities have different domains, I put them in a dict with different domains
            entity_representations = self.convert_to_domain(entity_representations,data.fol)
            # print('relation_representations after:',relation_representations.shape)
            return entity_representations,relation_representations
        else:
            # For the relations, I shuold take the one associated to the query
            # squeeze the dim in the middle, given we have only the postive query
            # assert that there is only a positive query, otherwise say that there should be no negatives
            assert entity_representations.shape[1] == 1, 'There should be only positive queries given to Ultra'
            entity_representations = entity_representations.squeeze(1)
            # convert the atom embedds to tf (I dont need convert_to_domain because I directly get the atom embedds, not the cte embedds)
            atom_representations = entity_representations.detach().cpu().numpy()

            assert scores.shape[1] == scores.shape[2] == 1, 'Problem with the shape of the tail entity scores'
            scores = scores.squeeze(1)
            scores = scores.detach().cpu().numpy()
            
            # I DONT CARE ABOUT THE RELATIONS, THIS IS THE ATOMS REPRESENATIONS
            # print('scores.shape:',scores.shape)
            # print('entity_representations.shape',entity_representations.shape)
            return scores, atom_representations
        
 




