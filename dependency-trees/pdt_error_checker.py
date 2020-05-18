#!/usr/bin/env python3
import re
import sys

def find_errors (ref_tree_lines):

	root = ()
	if len(ref_tree_lines) > 0:
		for i, elem in enumerate(ref_tree_lines):
			match = re.match(r"^[a-zA-Z]*\.[a-zA-Z]*$", elem)
			if not match:
				print("Syntax error on line {}".format(i))
			else:
				if re.match(r"^[a-zA-Z]*\.$", match.group()):
					root = (i, elem)
	else:
		assert len(ref_tree_lines) > 0, "File is empty"

	find_root = re.findall(r"^[a-zA-Z]*\.$", ref_tree, re.M)
	if not find_root:
		print("Semantic error: No root")
	elif len(find_root) > 1:
		print("Semantic error: More than one root")

	child_parent, children_dict = [], {}
	for node in ref_tree_lines:
		child_parent.append(node.split('.'))
		children_dict.setdefault(node.split('.')[0], []).append(node.split('.')[1])

	def find_loop (original_node='', node='', first_iter=True):
		"""This function will return the first node subject to a loop (and diregards the rest)"""
		if node == '':
			return False
		elif node == original_node and not first_iter:
			print("Semantic error: loop at node <{}>".format(node))
			return True
		else:
			for parent in children_dict[node]:
				if parent not in children_dict.keys() and parent != '':
					print("Semantic error: Child <{}> has an undefined or floating node parent <{}>".format(child, parent))
				else:
					find_loop(original_node, parent, False)

	# This works assuming we have correct syntax
	for child in list(children_dict):
		if len(children_dict[child]) > 1:
			print("Semantic error: Child <{}> has more than one parent ==> {}".format(child, children_dict[child]))
		if child is root[1][:-1] and list(child) not in children_dict.values():
			print("Semantic error: Floating root <{}>".format(child))
		if list(child) == children_dict[child]:
			print("Semantic error: Floating node <{}>".format(child))
			del children_dict[child]
			continue
		find_loop(child, child, first_iter=True)

def convert_tree (ref_tree_lines):

	children, parent_dict = [], {}
	for node in ref_tree_lines:
		children.append(node.split('.')[0])
		parent_dict.setdefault(node.split('.')[1], []).append(node.split('.')[0])
	for node in children:
		if node not in parent_dict:
			parent_dict[node] = ''

	def recursive_join (children, first_iter=True, conv=''):
		
		if children == '':
			fh.write(',') #; print(',', end='')
		elif len(children) == 1 and not first_iter:
			fh.write('(' + children[0] + ')') #; print('(' + children[0] + ')', end='')
			recursive_join(parent_dict[children[0]], False, conv)
		else:
			if not first_iter:
				fh.write('(') #; print('(', end='')
			for child in children:
				fh.write(child) #; print(child, end='')
				recursive_join(parent_dict[child], False, conv)
			if not first_iter:
				fh.write('),') #; print('),', end='')

	fh = open(sys.argv[1] + ".conv", 'w')
	recursive_join(parent_dict[''])
	fh = open(sys.argv[1] + ".conv", 'r')
	conv = re.sub(r',\)', r')', fh.read())
	fh = open(sys.argv[1] + ".conv", 'w')
	fh.write(conv[:-1] + '\n')
	fh.close()
	if len(sys.argv) == 3:
		print(conv[:-1])

if __name__ == "__main__":

	with open(sys.argv[1], 'r') as fh:
		ref_tree = fh.read()
		ref_tree_lines = ref_tree.splitlines()
	
	find_errors(ref_tree_lines)
	convert_tree(ref_tree_lines)





