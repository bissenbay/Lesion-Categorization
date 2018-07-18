import numpy as np

# generating random x, y, z
#coordinates = np.random.shuffle(np.random.randint(10, size=(10, 3))) 

#x_coords = np.arange(10)
#y_coords = np.arange(10)
#z_coords = np.arange(10)

#np.random.shuffle(x_coords)
#np.random.shuffle(y_coords)
#np.random.shuffle(z_coords)

# numbering voxels
#voxels = np.arange(10)

# concatenating 2 arrays (coordinates and voxels)
#arr = np.column_stack((voxels, coordinates))

# generating first column for voxels 0 to 9
arr = np.arange(10)

for i in range(3):
    # generating x, y, z columns
    coords = np.arange(10)
    # shuffling
    np.random.shuffle(coords)
    # concatenating to the array as a column
    arr = np.column_stack((arr, coords))

print('v#, x, y, z')
print(arr)
print()

# slicing only the columns with voxels number and x coordinates
x_col = arr[:,:2]
# sorting by x
x_col = x_col[np.argsort(x_col[:,1])]

print('v#, x --- sorted by x coordinates')
print(x_col)
print()

# slicing only the columns with voxels number and y coordinates
y_col = arr[:,[0, 2]]
# sorting by y
y_col = y_col[np.argsort(y_col[:,1])]

print('v#, y --- sorted by y coordinates')
print(y_col)
print()

# slicing only the columns with voxels number and z coordinates
z_col = arr[:,[0, 3]]
# sorting by z
z_col = z_col[np.argsort(z_col[:,1])]

print('v#, z --- sorted by z coordinates')
print(z_col)
print()

# output dictionary
out = dict([(key, set()) for key in range(10)])
        
def fill_out(col):
    global out
    for i in range(1,9):
        # check if the voxel located at -1 position is already in the set, if not then add
        if col[i-1][0] not in out[col[i][0]]: out[col[i][0]].add(col[i-1][0])
        # check if the voxel located at +1 position is already in the set, if not then add
        if col[i+1][0] not in out[col[i][0]]: out[col[i][0]].add(col[i+1][0])
        # check if the current voxel is already in the set located at -1 position, if not then add
        if col[i][0] not in out[col[i-1][0]]: out[col[i-1][0]].add(col[i][0])
        # check if the current voxel is already in the set located at +1 position, if not then add
        if col[i][0] not in out[col[i+1][0]]: out[col[i+1][0]].add(col[i][0])

fill_out(x_col)
fill_out(y_col)
fill_out(z_col)

print('key is voxel number and the set consists of numbers of neighbor voxels')

for key, value in out.items():
    print(key, ': ', value)
    
    


# generating output array with number of voxels on the first column and other 26 columns for neightbor columns, where -1 means no neightbor
#out = np.column_stack((np.arange(10), np.full((10, 25), -1)))
 
#voxel_data_def = [('voxel_number', 'i8'), ('x', 'i8'), ('y', 'i8'), ('z', 'i8')]


