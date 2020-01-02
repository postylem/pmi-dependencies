import json
from reporter_copy import *



#outputs uuas score of two trees
def get_uuas_score(edge1, edge2):
  uspan_correct=0
  uspan_total=0
  #edge1=[modif_prims_matrix_to_edges(x) for x in data1]
  #print("edge1")
  #edge2=[modif_prims_matrix_to_edges(x) for x in data2]
  #print("edge2")
  #print(len(edge1[10])==len(edge2[10]))
  uspan_correct+=sum([len(set(edge1[i]).intersection(set(edge2[i]))) for i in range(len(edge1))])
  uspan_total+=sum([len(edge1[i]) for i in range(len(edge1))])
  return uspan_correct/uspan_total

#outputes uuas score matrix

def make_em_all():
	data=[]
	for x in range(10):
		with open("dev-"+str(0)+str(x)+".predictions") as f:
			data.append(np.array(json.load(f)).flatten()) #stores the results

	for x in range(10, 24):
		with open("dev-"+str(x)+".predictions") as f:
			data.append(np.array(json.load(f)).flatten()) #stores the results

	edge=[[modif_prims_matrix_to_edges(y) for y in x ]for x in data]


	uspan_matrix=np.zeros((len(data), len(data)))
	print(len(data))
	for i in range(len(data)):
		for j in range(len(data)):
			uspan_matrix[i, j]=get_uuas_score(edge[i], edge[j])
	return uspan_matrix

