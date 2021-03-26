import pickle

def dump_to_pickle(filename: str, data):
    f=open(filename, "wb")
    for item in data:
        pickle.dump(item,f)
    f.close()