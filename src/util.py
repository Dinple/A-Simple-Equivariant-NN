import numpy as np
import torchvision.datasets as datasets
import networkx as nx
import scipy


def prepare_dataset(dataset="mnist", iftrain=True):
    # MNIST
    if dataset == "mnist":
        datasets.MNIST('./data', train=iftrain, download=True)
        return datasets.MNIST('./data', train=iftrain, download=False)
    # FashionMNIST
    elif dataset == "fashionmnist":
        datasets.FashionMNIST('./data', train=iftrain, download=True)
        return datasets.FashionMNIST('./data', train=iftrain, download=False)

##########################################
###        Modified from networkx      ###
##########################################
def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center

def spectral_layout(G, weight="weight", scale=1, center=None, dim=2):
    """Position nodes using the eigenvectors of the graph Laplacian.

    Using the unnormalized Laplacian, the layout shows possible clusters of
    nodes which are an approximation of the ratio cut. If dim is the number of
    dimensions then the positions are the entries of the dim eigenvectors
    corresponding to the ascending eigenvalues starting from the second one.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spectral_layout(G)

    Notes
    -----
    Directed graphs will be considered as undirected graphs when
    positioning the nodes.

    For larger graphs (>500 nodes) this will use the SciPy sparse
    eigenvalue solver (ARPACK).
    """
    # handle some special cases that break the eigensolvers
    import numpy as np

    G, center = _process_params(G, center, dim)

    eigv = None

    if len(G) <= 2:
        if len(G) == 0:
            pos = np.array([])
        elif len(G) == 1:
            pos = np.array([center])
        else:
            pos = np.array([np.zeros(dim), np.array(center) * 2.0])
        return dict(zip(G, pos))
    try:
        # Sparse matrix
        if len(G) < 500:  # dense solver is faster for small graphs
            raise ValueError
        A = nx.to_scipy_sparse_array(G, weight=weight, dtype="d")
        # Symmetrize directed graphs
        if G.is_directed():
            A = A + np.transpose(A)
        pos, eigv = _sparse_spectral(A, dim)
    except (ImportError, ValueError):
        # Dense matrix
        A = nx.to_numpy_array(G, weight=weight)
        # Symmetrize directed graphs
        if G.is_directed():
            A += A.T
        pos, eigv = _spectral(A, dim)

    pos = rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    return pos, eigv



def _spectral(A, dim=2):
    # Input adjacency matrix A
    # Uses dense eigenvalue solver from numpy
    import numpy as np

    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "spectral() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    # form Laplacian matrix where D is diagonal of degrees
    D = np.identity(nnodes, dtype=A.dtype) * np.sum(A, axis=1)
    L = D - A

    eigenvalues, eigenvectors = np.linalg.eig(L)
    # sort and keep smallest nonzero
    index = np.argsort(eigenvalues)[1 : dim + 1]  # 0 index is zero eigenvalue
    
    # print the eigenvalues
    print("eigenvalues: ", eigenvalues[index])

    return np.real(eigenvectors[:, index]), eigenvalues[index]


def _sparse_spectral(A, dim=2):
    # Input adjacency matrix A
    # Uses sparse eigenvalue solver from scipy
    # Could use multilevel methods here, see Koren "On spectral graph drawing"
    import numpy as np
    import scipy as sp
    import scipy.sparse  # call as sp.sparse
    import scipy.sparse.linalg  # call as sp.sparse.linalg

    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "sparse_spectral() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    # form Laplacian matrix
    # TODO: Rm csr_array wrapper in favor of spdiags array constructor when available
    D = sp.sparse.csr_array(sp.sparse.spdiags(A.sum(axis=1), 0, nnodes, nnodes))
    L = D - A

    print("Is my L Sparse? ", scipy.sparse.issparse(L))
    
    k = dim + 1
    # number of Lanczos vectors for ARPACK solver.What is the right scaling?
    ncv = max(2 * k + 1, int(np.sqrt(nnodes)))
    # return smallest k eigenvalues and eigenvectors
    eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(L, k, which="SM", ncv=ncv)
    index = np.argsort(eigenvalues)[1:k]  # 0 index is zero eigenvalue
    
    # print the corresponding eigenvalues
    print("eigenvalues: ", eigenvalues[index])

    return np.real(eigenvectors[:, index]), eigenvalues[index]

def rescale_layout(pos, scale=1):
    """Returns scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.

    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.

    See Also
    --------
    rescale_layout_dict
    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos