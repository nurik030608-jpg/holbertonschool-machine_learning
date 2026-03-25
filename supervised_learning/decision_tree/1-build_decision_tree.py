#!/usr/bin/env python3
"""
Module for node counting in a Decision Tree
"""


class Node:
    """
    Represents an internal node in a decision tree
    """
    def __init__(self, feature, threshold, left_child, right_child,
                 depth, is_root=False):
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
        # Recursive call to both children
        left_count = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves=only_leaves)

        if only_leaves:
            # If we only want leaves, internal nodes return the sum of children
            return left_count + right_count

        # If counting all nodes, add 1 (for self) plus children's counts
        return 1 + left_count + right_count
