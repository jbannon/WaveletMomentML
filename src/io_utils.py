from typing import List, Tuple, Dict, NamedTuple
from utils import ICIDataSet
import pickle as pk



def unpack_parameters(
	D:Dict
	):
	if len(D.values())>1:
		return tuple(D.values())
	else:
		return tuple(D.values())[0]


def load_ICIDataSet(
    base_path:str,
    drug:str,
    tissue:str,
    gex_units:str
    ) -> ICIDataSet:
    
    if base_path[-1]=="/":
        base_path = base_path[:-1]
    
    filePath= base_path+"/"+drug+"/"+tissue+"/"+gex_units+".pickle"
    print(filePath)
    with open(filePath, 'rb') as istream:
        DS = pk.load(istream)

    return DS

def make_filepath(
	components:List[str],
	extension:str = None
	) -> str:
	if extension is not None:
		extension = "."+extension if extension[0]!="." else extension
	else:
		extension = ""
	print(extension)
	components = [comp[:-1] if comp[-1]== "/" else comp for comp in components]
	fpath = "".join(["/".join(components),extension])
	return fpath