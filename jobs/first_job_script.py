import pickle

a = {'hello': 'world'}
# path = '/home/mtnusf97/projects/def-cannoj9/mtnusf97/filename.pickle'
path = 'filename.pickle'

with open(path, 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(path, 'rb') as handle:
    b = pickle.load(handle)

print(a == b)
print('here it is')
