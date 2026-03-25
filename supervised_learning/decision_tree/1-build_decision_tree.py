#!/usr/bin/env python3
"""
Module for building and counting nodes in a Decision Tree
"""


class Node:
    """
    Represents an internal node in a decision tree
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=None, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def count_nodes_below(self, only_leaves=False):
        """
        Recursively counts the nodes or leaves below and including this node.
        """
        left_count = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves=only_leaves)

        if only_leaves:
            return left_count + right_count

        return 1 + left_count + right_count


class Leaf:
    """
    Represents a leaf node in a decision tree
    """
    def __init__(self, value, depth=None):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 since a leaf is a single node and a single leaf.
        """
        return 1


class Decision_Tree:
    """
    Represents a decision tree structure
    """
    def __init__(self, root=None, max_depth=10, min_pop=1,
                 seed=0, split_criterion="gini"):
        self.root = root
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.split_criterion = split_criterion

    def count_nodes(self, only_leaves=False):
        """
        Returns the total number of nodes or leaves in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
