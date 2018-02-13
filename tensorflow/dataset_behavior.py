"""
Build behavior dataset manager from itemlist file and behaviorlist file
"""

import pickle as pkl
import time

import numpy as np

import utils

__author__ = "Daheng Wang"
__email__ = "dwang8@nd.edu"


class BehaviorDatasetManager:

    def __init__(self):
        """
        Initialization of dataset manager attributes.
        """
        '''Basic'''
        self.number_of_items = 0
        self.number_of_item_types = 0
        self.number_of_behaviors = 0

        '''Mappings between item and item index'''
        self.item_lst = []
        self.item2itemidx_dict = {}

        '''Mappings between type and type index'''
        self.itype_lst = []
        self.itype2itypeidx_dict = {}

        '''Mapping from item index to item type index'''
        self.itemidx2itypeidx_dict = {}

        '''Mappng from item type index to all indices of items belong to that type'''
        self.itypeidx2typeitemindiceslst_dict = {}

        '''Mappng from item type index to alias sampler of items indices belong to that type'''
        self.itypeidx2typeitemindicessampler_dict = {}

        '''Dict of behaviors (each value is a list of item indices constitute that behavior)'''
        self.behavior2behavioritemindiceslst_dict = {}

    def read_itemlist_file(self, itemlist_file):
        """
        Read in file of all items in this behavior dataset
        Each line in format: <item>\t<item_type>
        Item and item_type should be int values
        """
        print('Input itemlist file: {}'.format(itemlist_file))
        t_s = time.time()

        itype_set = set()
        itypeidx2itemindicesset_dict = {}

        item_index_counter = 0  # Set counter for indexing items

        with open(itemlist_file, 'r') as f:
            for line_ind, line in enumerate(f):

                tokens = line.strip('\n').split()
                if len(tokens) == 2:  # If line format is correct
                    '''Note: token should be integer'''
                    item = int(tokens[0])
                    itype = int(tokens[1])

                    item_idx = item_index_counter
                    self.item_lst.append(item)
                    self.item2itemidx_dict[item] = item_idx

                    '''If encounter a new item type'''
                    if itype not in itype_set:
                        itype_idx = len(itype_set)  # index new type before adding to itype_set
                        itype_set.add(itype)

                        self.itype_lst.append(itype)
                        self.itype2itypeidx_dict[itype] = itype_idx

                        itypeidx2itemindicesset_dict[itype_idx] = set()

                    itype_idx = self.itype2itypeidx_dict[itype]
                    self.itemidx2itypeidx_dict[item_idx] = itype_idx
                    itypeidx2itemindicesset_dict[itype_idx].add(item_idx)

                    item_index_counter += 1
                else:
                    print('File line format wrong!')
                    break

        self.number_of_items = len(self.item_lst)
        self.number_of_item_types = len(self.itype_lst)

        print('Indexing...')
        for itypeidx, itemindicesset in itypeidx2itemindicesset_dict.items():
            self.itypeidx2typeitemindiceslst_dict[itypeidx] = list(itemindicesset)

            '''Build alias sampler for uniform sampling of item indices in a given type'''
            item_indices_probs = np.zeros(self.number_of_items)
            for item_ind in itemindicesset:
                item_indices_probs[item_ind] = 1
            item_indices_probs = item_indices_probs / np.sum(item_indices_probs)  # Normalize probs
            item_indices_sampler = utils.AliasMethod(prob_lst=item_indices_probs)
            self.itypeidx2typeitemindicessampler_dict[itypeidx] = item_indices_sampler

        t_e = time.time()
        print('Finished ({:.2f} sec)'.format(t_e - t_s))

    def read_behaviorlist_file(self, behaviorlist_file):
        """
        Read in behaviors (itemset).
        Each line follows format <behavior>\t<item1>,<item2>,<item3>...
        """
        print('Input behaviorlist file: {}'.format(behaviorlist_file))
        t_s = time.time()

        item_set = set(self.item_lst)
        unknown_item_set = set()

        with open(behaviorlist_file, 'r') as f:
            for line_ind, line in enumerate(f):
                tokens = line.strip('\n').split()
                if len(tokens) == 2:  # If format is correct
                    behavior = int(tokens[0])
                    items = tokens[1]

                    behavior_itemindicesset_lst = []
                    for item in items.split(','):
                        '''Discard item not in item_set'''
                        if int(item) in item_set:
                            item_idx = self.item2itemidx_dict[int(item)]
                            behavior_itemindicesset_lst.append(item_idx)
                        else:
                            unknown_item_set.add(int(item))
                    self.behavior2behavioritemindiceslst_dict[behavior] = behavior_itemindicesset_lst
                else:
                    print('File line format wrong!')
                    break
            self.number_of_behaviors = len(self.behavior2behavioritemindiceslst_dict.keys())
        t_e = time.time()
        print('Finished ({:.2f} sec)'.format(t_e - t_s))
        if len(unknown_item_set):
            print('Warning: {} unknown items ignored!'.format(len(unknown_item_set)))

    def sample_batch_behaviors(self, batch_size=1, negative=5, mode='1'):
        """
        Sample a batch of behaviors
        :param batch_size: number of positive samplings
        :param negative: number of negative behaviors sampled FOR EACH positive behavior
        :param mode: 1 for size-constrained; 2 for type-distribution constrained
        :return:
        """
        batch_behaviors_item_indices = []
        batch_behaviors_item_type_indices = []
        batch_behaviors_labels = []

        pos_label = 1
        neg_label = 0

        pos_behavior_key_indices = np.random.randint(0, self.number_of_behaviors, size=batch_size)

        '''For each positive sampling'''
        for pos_behavior_key_ind in pos_behavior_key_indices:
            behaviors_item_indices = []
            behaviors_item_type_indices = []
            behaviors_labels = []

            '''Sample a positive behavior'''
            pos_behavior = list(self.behavior2behavioritemindiceslst_dict.keys())[pos_behavior_key_ind]
            pos_behavior_item_indices = self.behavior2behavioritemindiceslst_dict[pos_behavior]
            behaviors_item_indices.append(pos_behavior_item_indices)

            pos_behavior_item_type_indices = []
            for pos_behavior_item_idx in pos_behavior_item_indices:
                itype_idx = self.itemidx2itypeidx_dict[pos_behavior_item_idx]
                pos_behavior_item_type_indices.append(itype_idx)
            behaviors_item_type_indices.append(pos_behavior_item_type_indices)

            behaviors_labels.append(pos_label)

            '''Sample multiple negative behaviors'''
            for _ in range(negative):
                neg_behavior_item_indices, neg_behavior_item_type_indices = \
                    self.sample_negative_behavior_based_on_positive_behavior(pos_behavior_item_indices, mode=mode)

                behaviors_item_indices.append(neg_behavior_item_indices)
                behaviors_item_type_indices.append(neg_behavior_item_type_indices)
                behaviors_labels.append(neg_label)

            batch_behaviors_item_indices.append(behaviors_item_indices)
            batch_behaviors_item_type_indices.append(behaviors_item_type_indices)
            batch_behaviors_labels.append(behaviors_labels)

        '''Necessary to unify all lists to same length'''
        batch_behaviors_item_indices = self.unify_input_dimensions(batch_behaviors_item_indices)
        batch_behaviors_item_type_indices = self.unify_input_dimensions(batch_behaviors_item_type_indices)

        return batch_behaviors_item_indices, batch_behaviors_item_type_indices, batch_behaviors_labels

    def sample_negative_behavior_based_on_positive_behavior(self, pos_behavior_item_indices, mode):
        """
        Generate a negative behavior based on the item composition of a given positive behavior
        :param pos_behavior_item_indices: list of item indices of the positive behavior
        :param mode: 1 for size-constrained; 2 for type-distribution constrained
        :return:
        """
        sampled_neg_behavior_item_indices = []
        sampled_neg_behavior_item_type_indices = []

        pos_behavior_item_num = len(pos_behavior_item_indices)

        '''Build up sampled_neg_behavior_item_indices'''
        if mode == '1':
            '''
            Size constrained mode
            Simply sample the same number of items to form a negative behavior
            '''

            '''Random samples with fixed sum'''
            random_neg_behavior_type_item_numbers = \
                np.random.multinomial(pos_behavior_item_num,
                                      np.ones(self.number_of_item_types)/self.number_of_item_types,
                                      size=1)[0]

            '''For each type, sample the same number of random items'''
            for itype_idx, neg_behavior_type_item_numbers in enumerate(random_neg_behavior_type_item_numbers):
                neg_behavior_type_item_indices = [self.itypeidx2typeitemindicessampler_dict[itype_idx].sample()
                                                  for _ in range(neg_behavior_type_item_numbers)]
                # Make sure to use extend instead of append
                sampled_neg_behavior_item_indices.extend(neg_behavior_type_item_indices)
        elif mode == '2':
            '''
            Type-distribution constrained mode
            Follow the type distribution of items in the positive behavior to sample a negative behavior
            '''

            '''Positive behavior item type distribution'''
            pos_behavior_item_type_count = [0] * self.number_of_item_types
            for pos_behavior_item_idx in pos_behavior_item_indices:
                itype_idx = self.itemidx2itypeidx_dict[pos_behavior_item_idx]
                pos_behavior_item_type_count[itype_idx] += 1

            '''For each type, sample the same number of random items'''
            for itype_idx, item_type_count in enumerate(pos_behavior_item_type_count):
                neg_behavior_type_item_indices = [self.itypeidx2typeitemindicessampler_dict[itype_idx].sample()
                                                  for _ in range(item_type_count)]

                # Make sure to use extend instead of append
                sampled_neg_behavior_item_indices.extend(neg_behavior_type_item_indices)
        else:
            print('Sampling mode error!')

        '''Build up sampled_neg_behavior_item_type_indices'''
        for sampled_neg_behavior_item_idx in sampled_neg_behavior_item_indices:
            itype_idx = self.itemidx2itypeidx_dict[sampled_neg_behavior_item_idx]
            sampled_neg_behavior_item_type_indices.append(itype_idx)

        return sampled_neg_behavior_item_indices, sampled_neg_behavior_item_type_indices

    @staticmethod
    def unify_input_dimensions(batch_behaviors_indices, fill_in=-1):
        """
        Simple function to unify all vectors within a batch to the same length
        """
        unified_batch_behaviors_indices = []

        max_len = 0
        '''Cast each behavior into list type and determine max length'''
        for behaviors in batch_behaviors_indices:
            behaviors_lst = []
            for behavior in behaviors:
                behaviors_lst.append(list(behavior))
                if len(behavior) > max_len:
                    max_len = len(behavior)
            unified_batch_behaviors_indices.append(behaviors_lst)

        for behaviors in unified_batch_behaviors_indices:
            for behavior in behaviors:
                if len(behavior) < max_len:
                    extra = [fill_in] * (max_len - len(behavior))
                    behavior.extend(extra)

        return unified_batch_behaviors_indices

    def output_embedding(self, target_embeddings, output_file_name, mode='pkl'):
        """
        Mapping item to embedding. Write to output file.
        """
        normalized_embeddings = target_embeddings

        item2embedding_dict = {}

        for item in self.item_lst:
            item_idx = self.item2itemidx_dict[item]
            item_embedding = normalized_embeddings[item_idx]
            item2embedding_dict[item] = item_embedding

        if mode == 'pkl':
            output_pkl_file = '{}.dict.pkl'.format(output_file_name)
            with open(output_pkl_file, 'wb') as f:
                pkl.dump(item2embedding_dict, f)
        else:
            output_txt_file = '{}.txt'.format(output_file_name)
            with open(output_txt_file, 'w') as f:
                # Write header line: number_of_nodes dimension_of_embedding
                f.write('{}\t{}\n'.format(self.number_of_items, len(normalized_embeddings[0])))
                for item in self.item_lst:
                    item_embedding = item2embedding_dict[item]
                    item_embedding_str = '\t'.join([str(dim) for dim in item_embedding])
                    f.write('{}\t{}\n'.format(item, item_embedding_str))
