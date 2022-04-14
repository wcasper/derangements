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

nums = '1234'
clique_size = len(nums)
data = list(nums)
vertices = permutation(data)
#vertices = random.sample(permutation(data),12)
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

if(False):
  cliques = list(nx.find_cliques(G))
  ncliques = len(cliques)
  clique_size = len(cliques[0])
  print("number of cliques:")
  print(ncliques)
  
if(False):
  verts = []
  for i in range(n):
    y = np.array(cliques[1])-1
    overlap = 0
    for jj in range(clique_size):
      if(i==y[jj] or A[i,y[jj]]):
        overlap += 1
    if(overlap == 0):
      H.add_node(i+1)
      verts.append(i)
  for i in range(len(verts)):
    for j in range(len(verts)):
      if(A[verts[i],verts[j]]):
        H.add_edge(verts[i]+1,verts[j]+1)


if(False):
  for i in range(ncliques):
    for j in range(i+1,ncliques):
      x = np.array(cliques[i])-1
      y = np.array(cliques[j])-1
      print(x,y)
      overlap = 0
      for ii in range(clique_size):
        for jj in range(clique_size):
          if(x[ii]==y[jj] or A[x[ii],y[jj]]):
            overlap += 1
      if(overlap == 0):
        print("separated cliques detected!\n")
        print("clique 1:")
        for kk in range(clique_size):
          print(vertices[x[kk]])
        print("clique 2:")
        for kk in range(clique_size):
          print(vertices[y[kk]])
        sys.exit(0)
      print(overlap)


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
#pos = nx.kamada_kawai_layout(G)
#pos = nx.spring_layout(G,iterations=39)
#pos = nx.fruchterman_reingold_layout(G,iterations=123)
pos = nx.nx_agraph.graphviz_layout(G,prog="neato")
pos = nx.nx_agraph.graphviz_layout(G,prog="sfdp")


nx.draw(G, pos)
#nx.draw_networkx_labels(G,pos,labels,font_size=16)
plt.show()



