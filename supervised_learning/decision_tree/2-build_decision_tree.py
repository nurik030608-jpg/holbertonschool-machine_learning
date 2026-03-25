#!/usr/bin/env python3
"""
Module for building and visualizing a Decision Tree
"""


class Node:
    """ Represents an internal node in a decision tree """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=None, is_root=False):
        """ Initializes the node """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def count_nodes_below(self, only_leaves=False):
        """ Recursively counts the nodes or leaves """
        left = self.left_child.count_nodes_below(only_leaves)
        right = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left + right
        return 1 + left + right

    def left_child_add_prefix(self, text):
        """ Adds prefix for the left child visualization """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |      " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """ Adds prefix for the right child visualization """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("           " + x) + "\n"
        return new_text

    def __str__(self):
        """ Returns string representation of the node and its children """
        if self.is_root:
            out = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            out = f"node [feature={self.feature}, threshold={self.threshold}]\n"

        l_str = self.left_child_add_prefix(str(self.left_child))
        r_str = self.right_child_add_prefix(str(self.right_child))

        return out + l_str + r_str


class Leaf:
    """ Represents a leaf node in a decision tree """
    def __init__(self, value, depth=None):
        """ Initializes the leaf """
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """ Returns 1 for a leaf """
        return 1

    def __str__(self):
        """ Returns the string representation of a leaf """
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """ Represents a decision tree structure """
    def __init__(self, root=None, max_depth=10, min_pop=1,
                 seed=0, split_criterion="gini"):
        """ Initializes the decision tree """
        self.root = root

    def count_nodes(self, only_leaves=False):
        """ Returns total nodes or leaves """
        return self.root.count_nodes_below(only_leaves

    def __str__(self):
        return str(self.root)
