import pickle as pk 


with open("../data/networks/G_LINCS.pickle", 'rb') as istream:
	G = pk.load(istream)

print(G)
print(type(G))
