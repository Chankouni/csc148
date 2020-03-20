"""
Assignment 2 starter code
CSC148, Winter 2020
Instructors: Bogdan Simion, Michael Liut, and Paul Vrbik

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2020 Bogdan Simion, Michael Liut, Paul Vrbik, Dan Zingaro
"""
from __future__ import annotations
import time
from typing import Dict, Tuple
from utils import *
from huffman import HuffmanTree
import random

# ====================
# Functions for compression
from starter.utils import bits_to_byte, int32_to_bytes


def build_frequency_dict(text: bytes) -> Dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    final = {}

    for by in text:
        if by in final:
            final[by] += 1
        else:
            final[by] = 1
    return final


def build_huffman_tree(freq_dict: Dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.
    >>> freq =  {104: 1, 101: 1, 108: 3, 111: 2, 119: 1, 114: 1, 100: 1}
    >>> t = build_huffman_tree(freq) # do something with this!


    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.right
    True
    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    """
    if len(freq_dict) == 1:
        symbol = list(freq_dict.keys())[0]

        symbol_2 = random.randint(0, 255)
        while symbol_2 in freq_dict:  # can take infinite time...
            symbol_2 = random.randint(0, 255)

        left = HuffmanTree(symbol)
        right = HuffmanTree(symbol_2)

        return HuffmanTree(None, left, right)

    else:
        huff_list = []
        for key in freq_dict:  # n time
            new_huff = HuffmanTree(key)
            new_huff.number = freq_dict.get(key)
            huff_list.append(new_huff)

        while len(huff_list) > 1:  # n time

            # sort list to easily get 2 lowest
            _huffman_insertion_sort(huff_list)

            # get lowest 2
            low_0 = huff_list.pop(0)
            low_1 = huff_list.pop(0)

            new_huffman = HuffmanTree(None, low_0, low_1)  # left has higher fre
            new_huffman.number = low_0.number + low_1.number

            # add it
            huff_list.append(new_huffman)

        _clean_huff_tree(huff_list[0])
        return huff_list[0]


def _huffman_insertion_sort(huff_list: list):
    """
    precondition:
    len(list) > 1
    sorts a list by huffman frequency using insertion algorithim
    >>> huf = [HuffmanTree(2, None, None), HuffmanTree(3, None, None), \
    HuffmanTree(7, None, None)]
    >>> huf[0].number = 5
    >>> huf[1].number = 3
    >>> huf[2].number = 4
    >>> _huffman_insertion_sort(huf)
    >>> huf
    [HuffmanTree(3, None, None), HuffmanTree(7, None, None), HuffmanTree(2, None, None)]
    """

    for i in range(1, len(huff_list)):
        to_swap = huff_list[i]
        cur_ind = i - 1

        cur_huff_number = to_swap.number
        prev_huff = huff_list[cur_ind]

        # if index > 0, then check if current number is less then previous number
        while cur_ind >= 0 and to_swap.number <= prev_huff.number:
            # then swap current with previous huff_node
            huff_list[cur_ind + 1], huff_list[cur_ind] = \
                huff_list[cur_ind], huff_list[cur_ind + 1]

            # go until index is 0, or cur_num > then prev_num
            cur_ind -= 1
            prev_huff = huff_list[cur_ind]


def _clean_huff_tree(tree: HuffmanTree):
    """
    convert all HuffmanTree.number to None after
    using it to keep list sorted
    """
    if tree:
        _clean_huff_tree(tree.left)
        _clean_huff_tree(tree.right)
        tree.number = None


def get_codes(tree: HuffmanTree) -> Dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    final = {}

    if tree.is_leaf():
        return {tree.symbol: '0'}

    else:
        _traverse_huff(tree, "", final)
        return final


def _traverse_huff(tree: HuffmanTree, code: str, huff_map: dict):
    """
    Algorithim that recursively builds and mutates the huff_map
    O(n) run time since we parse through each node
    """

    if tree.left:
        if tree.left.is_leaf():
            # assign code
            huff_map[tree.left.symbol] = code + "0"
        else:
            _traverse_huff(tree.left, code + "0", huff_map)

    if tree.right:
        if tree.right.is_leaf():
            huff_map[tree.right.symbol] = code + "1"

        else:
            _traverse_huff(tree.right, code + "1", huff_map)

    return huff_map


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    # cant be recursive because different in/out\
    # use list so we can mutate value

    if tree.is_leaf():
        # leafs arent given a number
        tree.number = None

    else:
        _recursive_numbering(tree, [0])


def _recursive_numbering(tree: HuffmanTree, cur_num: list):
    """
    Algorithm that recursively builds and mutates the huff_map
    """
    if tree.is_leaf():  # dont number leaves
        return

    elif not tree:  # blank tree, graceful termiantion
        return

    else:

        _recursive_numbering(tree.left, cur_num)
        # then right
        _recursive_numbering(tree.right, cur_num)
        # finally root
        tree.number = cur_num[0]
        cur_num[0] += 1


def avg_length(tree: HuffmanTree, freq_dict: Dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    freq, weight, final = 0, 0, 0
    codes = get_codes(tree)

    try:
        freq = sum([value for value in freq_dict.values()])
    except ValueError:
        print("bad value in freq_dict!")
        return 0.0

    for key in freq_dict:
        if codes.get(key):
            weight += len(codes.get(key)) * freq_dict[key]

    try:
        final = weight / freq
    except ZeroDivisionError:
        return 0.0

    return final


def compress_bytes(text: bytes, codes: Dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    bytes_8, final = [], []
    long_str = ""

    start_time = time.time()
    for byte in text:  # n time
        if not byte in codes:
            print("failure! byte is not yet mapped!")
            return

        else:
            # overwrites string each time
            long_str += codes[byte]

    while len(long_str) > 0:   # --- 97.15439987182617 seconds ---not goodn nm m nb
        bytes_8.append(long_str[:8])
        long_str = long_str[8:]

    for byte in bytes_8:  # n/8 time
        if len(byte) != 8:
            num_left = 8 - len(byte)
            byte += "0" * num_left

        final.append(bits_to_byte(byte))

    return bytes(final)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    final = []

    if tree.is_leaf():
        return bytes([0, 0, 0, 0])

    else:
        # user recursive function to post_travers
        _recursive_tree_to_bytes(tree, final)

        return bytes(final)


def _recursive_tree_to_bytes(tree, byte: list) -> None:
    """
    A recursive algo
    """
    if tree.is_leaf():  # dont number leaves
        return

    elif not tree:  # blank tree, graceful termiantion
        return

    else:
        _recursive_tree_to_bytes(tree.left, byte)
        _recursive_tree_to_bytes(tree.right, byte)

        if tree.left:
            if tree.left.is_leaf():
                byte.append(0)
                byte.append(tree.left.symbol)

                if tree.left.symbol > 255:
                    print(tree.left.symbol)

            else:
                byte.append(1)
                byte.append(tree.left.number)

        if tree.right:

            if tree.right.is_leaf():
                byte.append(0)
                byte.append(tree.right.symbol)
                if tree.right.symbol > 255:
                    print(tree.right.symbol)

            else:
                byte.append(1)
                byte.append(tree.right.number)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.

    >>> compress_file("files/julia_set.bmp", "files/testing.txt")

    """
    with open(in_file, "rb") as f1:
        text = f1.read()

    start_time = time.time()

    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))

    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree) +
              int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)

    with open(out_file, "wb") as f2:
        f2.write(result)

    print("--- %s seconds ---" % (time.time() - start_time))


# =====================TESTING FOR SPEED ON EACH FUNCTION ===================
def test_valid_bytes(dic):
    """

    """
    for x in dic:
        if x > 255:
            return False
    return True


def test_build_freq(in_file: str):
    """
    >>> a = test_build_freq("files/julia_set.bmp")
    around --- 0.1380312442779541 seconds ---
    >>> test_valid_bytes(a)
    True
    """
    start_time = time.time()
    with open(in_file, "rb") as f1:
        text = f1.read()

    freq = build_frequency_dict(text)
    print("--- %s seconds ---" % (time.time() - start_time))
    return freq


def test_build_tree(in_file: str):
    """
    >>> test_build_tree("files/julia_set.bmp")
    around --- 0.017004728317260742 seconds ---
    """

    with open(in_file, "rb") as f1:
        text = f1.read()

    freq = build_frequency_dict(text)

    # start timer now
    start_time = time.time()
    tree = build_huffman_tree(freq)
    print("--- %s seconds ---" % (time.time() - start_time))


def test_get_codes(in_file: str):
    """
    >>> test_get_codes("files/julia_set.bmp")
    around --- 0.0 seconds ---

    """
    with open(in_file, "rb") as f1:
        text = f1.read()

    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)

    # start timer now
    start_time = time.time()
    codes = get_codes(tree)

    print("--- %s seconds ---" % (time.time() - start_time))
    return codes


def test_number_nodes(in_file: str):
    """
    >>> test_number_nodes("files/julia_set.bmp")
    around --- 0.0009996891021728516 seconds ---

    """
    with open(in_file, "rb") as f1:
        text = f1.read()

    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)

    # start timer now
    start_time = time.time()

    number_nodes(tree)
    print("--- %s seconds ---" % (time.time() - start_time))


def test_avg_length(in_file: str):
    """
    >>> test_avg_length("files/julia_set.bmp")
    --- 0.0 seconds ---
    2.9465956719912674


    """
    with open(in_file, "rb") as f1:
        text = f1.read()

    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)

    # start timer now
    start_time = time.time()
    avg = avg_length(tree, freq)
    print(" --- %s seconds ---" % (time.time() - start_time))
    return avg


def test_tree_to_bytes(in_file: str):
    """
    Error Occuring.

    >>> a = test_tree_to_bytes("files/julia_set.bmp")
    --- 0.00099945068359375 seconds ---

    """
    with open(in_file, "rb") as f1:
        text = f1.read()

    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    number_nodes(tree)

    # start timer now
    start_time = time.time()
    a = tree_to_bytes(tree)
    print("--- %s seconds ---" % (time.time() - start_time))
    return a


def test_final_thing(in_file):
    """
    >>> test_final_thing("files/julia_set.bmp")
    """
    with open(in_file, "rb") as f1:
        text = f1.read()

    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)

    start_time = time.time()

    result = compress_bytes(text, codes)

    print("--- %s seconds ---" % (time.time() - start_time))


# ====================
# Functions for decompression

def generate_tree_general(node_lst: List[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    # TODO: Implement this function
    if len(node_lst) == 0:
        return HuffmanTree(None)

    for node in node_lst:

        if node.l_type == 0:  # if left side is a leaf
            pass
        else:
            pass

        # distinguish between num nodes and symbol. how?
        # if node.left.is_leaf() then it is the symbol!
        # same with left


def generate_tree_postorder(node_lst: List[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    # TODO: Implement this function
    pass


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    # TODO: Implement this function
    pass


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: Dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # TODO: Implement this function
    pass


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input("Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print("Compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print("Decompressed {} in {} seconds."
              .format(fname, time.time() - start))
