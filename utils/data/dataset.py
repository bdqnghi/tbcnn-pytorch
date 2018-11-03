import numpy as np
import os
from os import listdir
from os.path import isfile, join
import collections
import re
from tqdm import trange
from tqdm import *
import random
import pickle
import pyarrow
from torch.utils import data
import torch

# def load_program_tree_from_directory(directory, n_classes=3):
#   for i in trange(1, 1+n_classes):
#     path = os.path.join(directory, str(i))

def build_tree(script):
    """Builds an AST from a script."""
   
    with open(script, 'rb') as file_handler:
        data_source = pickle.load(file_handler)
    return data_source
   

def _traverse_tree(root):
    num_nodes = 1
    queue = [root]

    root_json = {
        "node": str(root.kind),

        "children": []
    }
    queue_json = [root_json]
    while queue:
      
        current_node = queue.pop(0)
        num_nodes += 1
        # print (_name(current_node))
        current_node_json = queue_json.pop(0)


        children = [x for x in current_node.child]
        queue.extend(children)
       
        for child in children:
            # print "##################"
            #print child.kind

            child_json = {
                "node": str(child.kind),
                "children": []
            }

            current_node_json['children'].append(child_json)
            queue_json.append(child_json)
            # print current_node_json
   
    return root_json, num_nodes



def _onehot(i, total):
    return [1.0 if j == i else 0.0 for j in range(total)]


def process_tree(tree, labels, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    # encode labels as one-hot vectors

    nodes = []
    children = []
    label = tree['label']

    queue = [(tree['tree'], -1)]
    # print queue
    while queue:
        # print "############"
        node, parent_ind = queue.pop(0)
        # print node
        # print parent_ind
        node_ind = len(nodes)
        # print "node ind : " + str(node_ind)
        # add children and the parent index to the queue
        queue.extend([(child, node_ind) for child in node['children']])
        # create a list to store this node's children indices
        children.append([])
        # add this child to its parent's child list
        if parent_ind > -1:
            children[parent_ind].append(node_ind)
        
        n = str(node['node'])
        look_up_vector = vector_lookup[n]
        nodes.append(vectors[int(n)])
    
    return nodes, children, label

def load_program_data(directory, n_classes):
    result = []
    labels = []
    for i in trange(1, 1+n_classes):
        dir_path = os.path.join(directory, str(i))
        for file in listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            # print(file_path)
            splits = file_path.split("/")
            # l = splits[len(splits)-2]
            # if l == lang:
            label = splits[len(splits)-2]
            # print(label)
            ast_representation = build_tree(file_path)

            if ast_representation.HasField("element"):
                root = ast_representation.element
                tree, size = _traverse_tree(root)

            result.append({
                'tree': tree, 'label': label
            })
            labels.append(label)
    return result, list(set(labels))

def _pad_batch(nodes, children, labels):
    if not nodes:
        return [], [], []
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])

    feature_len = len(nodes[0][0])
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]
   
    return nodes, children, labels

# def main():
#     parse_raw_data_to_pickle(sys.argv[1], sys.argv[2], sys.argv[3])

# a simple custom collate function, just to show the idea
def my_collate(batch):
    nodes, children, labels = [], [], []
    for n, c, l in batch:
        nodes.append(n)
        children.append(c)
        labels.append(int(l))
    nodes, children, labels = _pad_batch(nodes, children, labels)
    nodes = torch.tensor(np.array(nodes)).double()
    children = torch.tensor(np.array(children)).double()
    labels = torch.tensor(np.array(labels))
    # nodes = np.array(nodes)
    # children = np.array(children)
    # labels = np.array(labels)
    return nodes, children, labels

class MonoLanguageProgramData(data.Dataset):
  def __init__(self, data_path, embeddings, embed_lookup, num_features, n_classes):
    trees, labels = load_program_data(data_path,n_classes)

    self.labels = labels
    self.data = trees
    self.embeddings = embeddings
    self.embed_lookup = embed_lookup

    # print(len(self.data))

  def __getitem__(self, index):
    tree = self.data[index]
    # print("------------------------------------------------------")
    nodes, children, label = process_tree(tree, self.labels, self.embeddings, self.embed_lookup)
    # print(nodes)
    label = int(label) - 1
    return nodes, children, label 

  def __len__(self):
    return len(self.data)