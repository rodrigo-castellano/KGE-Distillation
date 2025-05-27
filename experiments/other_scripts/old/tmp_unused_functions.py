    # Function to filter and retrieve embeddings as a dense tensor
    def get_embeddings_idx(self,sparse_tensor, target_indices):
        # Initialize an empty list to store the embeddings
        embeddings_list = []
        # Iterate over the target indices
        for index in target_indices.numpy():
            # Extract the sub-tensor for the current index
            mask = tf.equal(sparse_tensor.indices[:, 0], index)
            # tf.print('mask', mask)
            # sub_indices = tf.boolean_mask(sparse_tensor.indices, mask)
            sub_values = tf.boolean_mask(sparse_tensor.values, mask)
            embeddings_list.append(tf.expand_dims(sub_values, 0))
            # tf.print('index',index,'tf.expand_dims(sub_values, 0)',tf.expand_dims(sub_values, 0).shape)
            # if sub_values.shape == 0:
            #     tf.print('mask',mask)
            #     tf.print('sparse_tensor',sparse_tensor)
            #     tf.print('sub_values',sub_values) 
        # Concatenate all embeddings to form the final dense tensor if target_indices is not empty, else return an empty tensor
        dense_embeddings = tf.concat(embeddings_list, axis=0) if embeddings_list else tf.constant([], dtype=tf.float32)
        
        return dense_embeddings
    
    # check in chatgpt this alternative. I can first extract only the indices of embedds that I need, then do a mapping from embedds to my list
    # def get_embeddings_idx(self,sparse_tensor, target_indices):
    #     # Create a mask that identifies which entries in sparse_tensor.indices[:, 0] are in target_indices
    #     mask = tf.reduce_any(tf.equal(tf.expand_dims(sparse_tensor.indices[:, 0], 1), target_indices), axis=1)

    #     # Use the mask to extract the relevant values
    #     sub_values = tf.boolean_mask(sparse_tensor.values, mask)
        
    #     # Get the indices of the filtered values
    #     filtered_indices = tf.boolean_mask(sparse_tensor.indices, mask)

    #     # Map the target indices to their corresponding embeddings
    #     expanded_target_indices = tf.expand_dims(target_indices, axis=1)
    #     embeddings_per_target = tf.reduce_sum(tf.cast(tf.equal(filtered_indices[:, 0], expanded_target_indices), tf.int32), axis=0)
        
    #     # Prepare the list to store the embeddings
    #     split_indices = tf.cumsum(embeddings_per_target)
    #     embeddings_list = tf.split(sub_values, split_indices[:-1])

    #     # Concatenate all embeddings to form the final dense tensor
    #     dense_embeddings = tf.concat([tf.expand_dims(embed, axis=0) for embed in embeddings_list], axis=0)
        
    #     return dense_embeddings





def get_negative_and_outputs(self, model, dataset, batch, atom_repr=True):
        batch_positive = torch.tensor([q[0] for q in batch], dtype=torch.int64)
        t_batch, h_batch = tasks.all_negative(self.aux_dataset, batch_positive)
        t_pred = self.Ultra(self.aux_dataset, t_batch)
        h_pred = self.Ultra(self.aux_dataset, h_batch)
        filtered_data = self.aux_dataset
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
        
        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]



    def define_ultra_dataset(self):
        """
        Get all the information of the dataset graph, as well as the relational graph.
        Also, create the triplets for the queries.

        Returns:
            None
        """
        queries, labels = self.dataset[:]
        constants_features = self.dataset.constants_features
        # I select engine=None because I dont want to ground the rules, only the queries    
        ((X_domains_data, A_predicates_data, _, _, (Q_global,_,_)), _) = _from_strings_to_tensors(
            fol=self.fol,
            serializer=self.serializer,
            queries=queries,
            labels=labels,
            engine=None,
            ragged=self.ragged,
            constants_features=constants_features,
            deterministic=self.deterministic,
            global_serialization=self.global_serialization)
        
        self.Q_global = Q_global    
        # Here Im interested in Q_global, to get the general info that I pass to ultra. From Q_global I can get the triplets of the queries only by taking the first element in every query
        # (the rest are the negative representations)
        self.queries_global = np.array([q[0] for q in Q_global])  

        self.edge_index = self.queries_global[:, :2].T # For all the queries, this takes in the first dim the head, and in the second the tail
        self.edge_type = self.queries_global[:, 2] # For all the queries, it takes the relation
        self.num_relations_no_inv = len(A_predicates_data)
        self.num_relations = len(A_predicates_data)*2
        self.num_nodes = sum(len(X_domains_data[key]) for key in X_domains_data)
        self.num_edges = self.edge_index.shape[1] # it is the number of queries
        self.device = 'cpu' 
    
        # convert edge_index and edge_type to torch tensor to feed it to ULTRA
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        self.edge_type = torch.tensor(self.edge_type, dtype=torch.long)

        self.aux_dataset = Dataset_Ultra(edge_index=self.edge_index, edge_type=self.edge_type, num_relations=self.num_relations, 
                                    num_nodes=self.num_nodes, num_edges=self.num_edges, device=self.device)
        self.aux_dataset.device = 'cpu'
        self.aux_dataset = tasks.build_relation_graph(self.aux_dataset)
        self.aux_dataset.fol = self.fol

        return None








    def get_ultra_outputs(self, model, dataset, batch, atom_repr=True):
        """
        Get the outputs of the ultra model for the queries.
        Args:
            model: the ultra model
            dataset: the dataset
            queries: the queries
            atom_repr: if True, the output is the atom representation. If False, the output is the scores.

        Returns:
            The outputs of the ultra model for the queries.
        """
        # print('Batch:', len(batch),'corruptions:', [len(b) for b in batch])
        batches = self.split_by_corruptions(batch)
        for key in batches.keys():
            if key != 1: # avoid only postivies when the corruption is only tail
                batches[key] = np.concatenate(batches[key], axis=0)
                batches[key] = torch.tensor(batches[key], dtype=torch.int64)
        # print('Batches:', batches.keys(), [b.shape for b in batches.values()])

        # For each batch, get the relation representations
        all_relation_representations = []
        all_entity_representations = []
        all_scores = []
        for key,batch in batches.items():
            # print('Batch_i:',batch.shape)
            # if the number of dimensions is 2,add a dimension in the middle (it would mean that there are no negatives, only positives). This is thought for the atom repersentation
            if len(batch.shape) == 2:
                batch = batch.unsqueeze(1)

            # call ultra
            scores = self.Ultra(self.aux_dataset, batch)
            # query_rels = batch[:, 0, 2]
            # relation_representations = self.relation_model(dataset.relation_graph, query=query_rels)
            # batch,relation_representations = self.split_head_tail_negatives(batch,relation_representations)
            # entity_representations, scores = self.entity_model(dataset, relation_representations, batch,atom_repr=atom_repr) # [16,1594,64] = [batch_size, num_negatives, embedd_size]
            # all_relation_representations.append(relation_representations)
            # all_entity_representations.append(entity_representations)
            all_scores.append(scores)
        print('All scores:', len(all_scores), [s.shape for s in all_scores])

        return all_scores









class ConstantEmbeddingsGlobal_old(Layer):
    """Calls the constant rules_embedders, differenciating the behavior of
       the single domains."""
    def __init__(self, domains: List[Domain],
                 constant_embedding_sizes_per_domain: Dict[str, int],
                 regularization: float=0.0,
                 has_features: bool=False):
        super().__init__()
        self.embedder = {}
        self.domains = domains
        self.constant_embedding_sizes_per_domain = constant_embedding_sizes_per_domain
        self.regularization = regularization
        self.has_features = has_features
        # I could make this more efficient: for domain 1, it goes from 0 to len(domain1), for dom2, it goes from (dom1,dom1+dom2)...
        max_index = sum([len(domain.constants) for domain in domains]) 
        for domain in domains:
            if self.has_features:
                # This should be replaced with the actual embedder,
                # make this a factory function call.
                self.embedder[domain.name] = Sequential([
                    Dense(self.constant_embedding_sizes_per_domain[domain.name],
                        kernel_regularizer=L2(self.regularization))])
            else:
                self.embedder[domain.name] = Embedding(
                    max_index+ 1,
                    self.constant_embedding_sizes_per_domain[domain.name],
                    embeddings_regularizer=L2(self.regularization))

    # domain_inputs is Dict domain->tensor of idx
    def call(self, domain_inputs: Dict[str, tf.Tensor], **kwargs):
        domain_features = {}
        cte_embeddings = {}
        for domain in self.domains:
            tf.print('X_Domain:', domain.name,summarize=-1)
            tf.print('Domain inputs:', domain_inputs[domain.name],summarize=-1)
            if domain_inputs[domain.name].shape[0] != 0:
                # if self.has_features:
                #     # This should be replaced with the actual embedder,
                #     # make this a factory function call.
                #     self.embedder[domain.name] = Sequential([
                #         Dense(self.constant_embedding_sizes_per_domain[domain.name],
                #             kernel_regularizer=L2(self.regularization))])
                # else:
                #     self.embedder[domain.name] = Embedding(
                #         tf.reduce_max(domain_inputs[domain.name]) + 1,
                #         self.constant_embedding_sizes_per_domain[domain.name],
                #         embeddings_regularizer=L2(self.regularization))

                domain_features[domain.name] = self.embedder[domain.name](
                    domain_inputs[domain.name])  #CE
                
                # Get the embeddings for the domain [200]
                embedding_size = self.constant_embedding_sizes_per_domain[domain.name] 
                # Assuming embeddings is a 2D tensor of shape (168, 200)
                embeddings = domain_features[domain.name]
                # tf.print('Embeddings:', embeddings.shape, embeddings[:,:3], summarize=-1)
                # Flatten the embeddings to have the shape (168 * 200,)
                embeddings_flattened = tf.reshape(embeddings, [-1])

                # The indices tensor
                indices = domain_inputs[domain.name]
                # Create the coordinates for each embedding. Each index should correspond to all 200 dimensions of an embedding
                indices_expanded = tf.expand_dims(indices, axis=1)
                coords = tf.concat([tf.repeat(indices_expanded, embedding_size, axis=0), 
                                    tf.tile(tf.range(embedding_size, dtype=tf.int32)[:, tf.newaxis], [indices.shape[0], 1])], axis=1)
                coords = tf.cast(coords, tf.int64)

                # The dense shape should reflect the maximum index and the size of each embedding
                dense_shape = [tf.reduce_max(indices).numpy() + 1, self.constant_embedding_sizes_per_domain[domain.name]]

                # Create the sparse tensor
                sparse_tensor = tf.SparseTensor(
                    indices=coords,  # List of coordinates
                    values=embeddings_flattened,  # List of values at those coordinates
                    dense_shape=dense_shape  # Shape of the dense tensor if it were fully populated
                )

                cte_embeddings[domain.name] = sparse_tensor 
            else:
                cte_embeddings[domain.name] = None
            print('Embeddings:', cte_embeddings[domain.name])
        return cte_embeddings