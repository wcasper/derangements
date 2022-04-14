#!/usr/bin/python

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random
import itertools
import networkx as nx
import sympy
import sys

# Python function to get permutations of a given list
def permutation(lst):
  # If lst is empty then there are no permutations
  if len(lst) == 0:
    return []

  # If there is only one element in lst then, only
  # one permutation is possible
  if len(lst) == 1:
    return [lst]

  # Find the permutations for lst if there are more than 1 characters
  l = [] # empty list that will store current permutation

  # Iterate the input(lst) and calculate the permutation
  for i in range(len(lst)):
    m = lst[i]

    # Extract lst[i] or m from the list. remLst is
    # remaining list
    remLst = lst[:i] + lst[i+1:]

    # Generating all permutations where m is first
    # element
    for p in permutation(remLst):
      l.append([m] + p)
  return l

nums = '123456'
clique_size = len(nums)
data = list(nums)
vertices = permutation(data)
n = len(vertices)

A = np.zeros([n,n])

print("Number of vertices:", n)

G = nx.Graph()
H = nx.Graph()
for i in range(n):
  G.add_node(i+1)
for j in range(n):
  for i in range(n):
    rhamdist = 0
    for k in range(len(vertices[0])):
      if(vertices[j][k] == vertices[i][k]):
        rhamdist += 1
#    if rhamdist != 1:
    if rhamdist == 0:
      A[j,i] = 1
      A[i,j] = 1
      G.add_edge(i+1,j+1)

print(vertices[0])

# read in the isotopy classes
ifile = open("isotopies.txt","r")
lines = ifile.readlines()
ifile.close()
cliques = []
cliques_raw = []
for line in lines:
  cliq = []
  cliq_raw = []
  for k in range(6):
    perm = []
    for j in range(6):
      perm.append(str(1 + int(line[6*k + j])))
    cliq_raw.append(perm)
    cliq.append(vertices.index(perm))
  cliques_raw.append(cliq_raw)
  cliques.append(cliq)

print(cliques)

def automorph(clique,perm1,perm2,antipode):

  newclique = []
  for vertex in np.array(vertices)[clique]:
    idx = list(np.array(perm1).astype(int)-1)
    newvertex = np.array(vertex)
    newvertex = list(newvertex[idx])
    for k in range(len(newvertex)):
      newvertex[k] = str(int(perm2[int(newvertex[k])-1]))
    if(antipode):
      newvertex = (np.array(np.array(newvertex).argsort()) + 1)
      newvertex = [str(x) for x in newvertex]
    newclique.append(vertices.index(newvertex))

  return newclique

for cidx in range(len(cliques)):
  clique = cliques[cidx]
  clique_raw = cliques_raw[cidx]
  print("##############################")
  print("finding symmetries for clique:")
  for perm in clique_raw:
    print(perm)
  print("##############################")
  nsymm = 0
  for perm1 in vertices:
    for perm2 in vertices:
      for antipode in [False, True]:
        newclique = automorph(clique,perm1,perm2,antipode)
        #print(clique, newclique)
        if set(newclique) == set(clique):
          print("detected symmetry:")
          print(perm1, perm2, antipode)
          nsymm += 1
  print("number of symmetries found:", nsymm)
  print("size of orbit:", 720*720*2/nsymm)

print("degrees:")
print(np.sum(A,1))
L = np.diag(np.sum(A,1))-A
print("spectrum:")
evals = np.real(np.linalg.eigvals(A))
evals.sort()
print(evals)
foo = n*np.max(np.abs(evals[:-1]))/evals[-1]
foo = (n*np.max(np.abs(evals[:-1]))/evals[-1])*np.sqrt((1-clique_size/n)/clique_size)
foo = foo**2
print("Disjoint from clique bound:", foo/(1+foo/n))
foo = n*np.max(np.abs(evals[:-1]))/evals[-1]
print("Disjoint pools max size:", foo/(1+foo/n))
AA = np.dot(A,A)
print(np.sort(AA))
pos = nx.kamada_kawai_layout(H)
#pos = nx.spring_layout(G,iterations=39)
#pos = nx.fruchterman_reingold_layout(G,iterations=123)
#pos = nx.nx_agraph.graphviz_layout(G,prog="neato")


nx.draw(H, pos)
#nx.draw_networkx_labels(G,pos,labels,font_size=16)
plt.show()



