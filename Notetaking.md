'''
create a class for the cluster-cluster aggregate simulation such that it has the following:

- initializing method with lattice dimension N, flux of particles phi, Tstar, periodic option, and seed value
- M is number of particles aka flux (particles/area) times area aka N^2
- phi is not int whereas N should be int, either way, M must be an int thus round result of flux times N^2 to get M 
- then update phi based on rounded value of M by doing M/N^2 to obtain true phi value accurate for that of this sim given this M int value we got after rounding

- make a 2d array of zeros for our lattice grid w lengths of N and N
- then randomly select M many numbers (non-overlapping) between 0 and N^2-1 via self.rng.choice 
- imagine labeling each lattice box in the 2d array aka grid with a number in a continuous sequence manner going from 0 all the way to N^2-1 value
- then each randomly selected number (of the M many) corresponds to a site on the grid 
- find the indices of that site by doing division by number of columns (aka N) to get the row index and then uing remainder to get the  column index
- do this for each of the M numbers randomly choosen so that you have the M many random indices of the 2d NxN array
- we obtained a list of row indices and column indices, but we would like a single object storing the indices
- thus use use np.stack to form a 2d array (of dimension Nx2 by choosing axis=1) where we combine the row list and col list together into one 2d array (which we name self.pos)

status check: we now have an array self.pos which holds the M row and M col indices side by side and we have an NxN lattice grid (array) that is currently holding zeros for every element 

- we would like a way to "occupy" or denote "status of being occupied" for the lattice squares of our NxN grid if the lattice square has indices which are stored in our self.pos array thus we must replace the zero's in those positions with something that indicates those locations are "occupied"

- We will indicate occupation of a lattice square by having that lattice square aka the element in the array store an id for the cluster/particle occupying it and conversely we would like to indicate "empty" or "lack of occupation" by storing a zero instead (this way we can use ids to keep track of clusters that span multiple lattice squares too since each cluster will get its own id which will appear on all lattice squares that belong to it)
- So, since we would like to avoid using 0 as a cluster id (so that we can instead use it to identify postions in our lattice that are un-occupied), we should simply add 1 to each of the M random numbers and assign that as the cluster id for each of the M particles originally randomly chosen (this is the cid=p+1 line 34)
- then we are free to store cid in the lattice squares which we decided (randomly) will be occupied; these lattices are those with the indices stored in self.pos or equivalently in our rows and cols lists (using rows and cols lists is better in the flow of this program hence why we do that... specifically we use a loop to go through all the M particles/clusters we've created originally and find the row and col they correspond to, then assign the row and col to i,j for convenience of programmer UI/reading, and then we just make that corresponding element in our NxN 2d array self.lattice store cid in that spot indexed by i,j
- we also want to be able to easily determine the index of each particle/cluster and rather than searching through the entire lattice each time and checking whether or not the  cluster we are looking for is the one in the spot we are looking at and doing this for all the spots, we can just create a dictionary with keys being the cluster ids and the definitions aka values being their index...also since clusters will grow to contain multiple lattice squares, it is convenient to be able to have all of those indices located under the cluster id in a dictionary, again reducing time/complexity since you just need to find cid once and then you have all indices rather than looking through all N^2 lattice squares

'''