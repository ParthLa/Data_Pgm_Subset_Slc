from my_utils import get_data
from my_core import Implyloss

num_classes = 6
if __name__ == '__main__':
	data = get_data(path, num_classes) # path will be the path of pickle file
	Il = Implyloss(data)
	Il.optimize()