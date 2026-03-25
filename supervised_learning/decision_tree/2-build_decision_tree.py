#!/usr/bin/env python3
""" Module for a clean, recursive Decision Tree visualization. """


class Node:
    """ Represents an internal node. """
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
        """ Counts nodes recursively. """
        left = self.left_child.count_nodes_below(only_leaves)
        right = self.right_child.count_nodes_below(only_leaves)
        return (left + right) if only_leaves else (1 + left + right)

    def _add_prefix(self, text, is_left_child):
        """ Helper to add prefixes with correct alignment. """
        lines = text.split("\n")
        # Формируем первую строку с веткой
        new_text = "    +---> " + lines[0] + "\n"
        # Для остальных строк: если левый ребенок, рисуем | , если правый - пусто
        padding = "    |      " if is_left_child else "           "
        for line in lines[1:]:
            if line:
                new_text += padding + line + "\n"
        return new_text

    def __str__(self):
        """ Recursive tree string representation. """
        label = "root" if self.is_root else "node"
        out = f"{label} [feature={self.feature}, threshold={self.threshold}]\n"
        out += self._add_prefix(str(self.left_child), True)
        out += self._add_prefix(str(self.right_child), False)
        return out


class Leaf:
    """ Represents a leaf. """
    def __init__(self, value, depth=None):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """ Leaves always count as 1. """
        return 1

    def __str__(self):
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """ Main Tree class. """
    def __init__(self, root=None, max_depth=10, min_pop=1,
                 seed=0, split_criterion="gini"):
        self.root = root

    def count_nodes(self, only_leaves=False):
        """ Entry point for counting. """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        return str(self.root)
