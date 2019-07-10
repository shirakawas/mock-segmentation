import numpy as np
import cv2
import numba


def prim_mst(graph, W, N):
    visited = np.zeros(N, dtype=bool)
    mst = - np.ones(N, dtype=np.int)
    cost = np.inf * np.ones(N)
    pi = np.zeros(N)

    # Start from the pixel at (0, 0)
    visited[0] = True
    mst[0] = 0
    cost[1], pi[1] = graph[0, 1], 0  # cost between (0, 0) pixel and the right pixel
    cost[W], pi[W] = graph[0, W], 0  # cost between (0, 0) pixel and the bottom pixel

    while True:
        if np.sum(visited) == N:
            break

        # Select minimal cost node
        min_cost, u = np.min(cost), np.argmin(cost)

        # Add the node having minimal cost
        visited[u] = True
        mst[u] = pi[u]
        cost[u] = np.inf

        # Update cost array
        v = u - 1  # left
        if u % W > 0 and (not visited[v]) and graph[u, v] < cost[v]:
            cost[v], pi[v] = graph[u, v], u
        v = u - W  # upper
        if v >= 0 and (not visited[v]) and graph[u, v] < cost[v]:
            cost[v], pi[v] = graph[u, v], u
        v = u + 1  # right
        if v % W > 0 and (not visited[v]) and graph[u, v] < cost[v]:
            cost[v], pi[v] = graph[u, v], u
        v = u + W  # bottom
        if v < N and (not visited[v]) and graph[u, v] < cost[v]:
            cost[v], pi[v] = graph[u, v], u

    return mst


@numba.jit
def labeling(ind, W):
    N = ind.size
    labels = - np.ones(N, dtype=np.int)
    num = 0
    for i in range(N):
        if labels[i] < 0:
            visit(i, num, ind, labels, W)
            num += 1
    return num, labels


@numba.jit
def visit(i, num, ind, labels, W):
    labels[i] = num
    j = i - 1  # left
    if i % W > 0 and labels[j] < 0 and (ind[i] == j or ind[j] == i):
        visit(j, num, ind, labels, W)
    j = i - W  # upper
    if i - W >= 0 and labels[j] < 0 and (ind[i] == j or ind[j] == i):
        visit(j, num, ind, labels, W)
    j = i + 1  # right
    if j % W > 0 and labels[j] < 0 and (ind[i] == j or ind[j] == i):
        visit(j, num, ind, labels, W)
    j = i + W  # bottom
    if j < ind.size and labels[j] < 0 and (ind[i] == j or ind[j] == i):
        visit(j, num, ind, labels, W)


class FitnessEvaluation(object):
    def __init__(self, img_arr, min_region_num=1, max_region_num=50, min_region_size=100):
        super(FitnessEvaluation, self).__init__()
        self.img_arr = img_arr  # height x width
        self.N = img_arr.shape[0] * img_arr.shape[1]
        self.W = img_arr.shape[1]
        self.min_region_num = min_region_num
        self.max_region_num = max_region_num
        self.min_region_size = min_region_size

        # For calculating edge value
        self.left_edge = np.zeros((self.img_arr.shape[0], self.img_arr.shape[1]))
        self.left_edge[:, 1:] = self.dist(self.img_arr[:, 1:], self.img_arr[:, :-1], axis=2)
        self.upper_edge = np.zeros((self.img_arr.shape[0], self.img_arr.shape[1]))
        self.upper_edge[1:, :] = self.dist(self.img_arr[1:, :], self.img_arr[:-1, :], axis=2)
        self.right_edge = np.zeros((self.img_arr.shape[0], self.img_arr.shape[1]))
        self.right_edge[:, :-1] = self.dist(self.img_arr[:, :-1], self.img_arr[:, 1:], axis=2)
        self.bottom_edge = np.zeros((self.img_arr.shape[0], self.img_arr.shape[1]))
        self.bottom_edge[:-1, :] = self.dist(self.img_arr[:-1, :], self.img_arr[1:, :], axis=2)

    @staticmethod
    def dist(x, y, axis=0):
        return np.sqrt(np.sum((x - y) ** 2, axis=axis))

    def __call__(self, ind):
        ind = ind.flatten()
        num, labels = labeling(ind, self.W)

        # Check the constraints. Set infinity if the constraint is violated.
        _, count = np.unique(labels, return_counts=True)
        if num < self.min_region_num or num > self.max_region_num or np.min(count) < self.min_region_size:
            return np.inf, np.inf

        # Overall deviation
        im_flat = self.img_arr.reshape(-1, 3)
        dev = np.sum(
            [np.sum(self.dist(im_flat[labels == n], im_flat[labels == n].mean(axis=0), axis=1)) for n in range(num)])

        # Edge
        labels_arr = labels.reshape(self.img_arr.shape[0], self.img_arr.shape[1])
        edge = 0.
        # Left
        left = np.zeros_like(labels_arr)
        left[:, 1:] = labels_arr[:, 1:] - labels_arr[:, :-1]
        left_mask = left.astype(np.int) != 0
        edge += self.left_edge[left_mask].sum()
        # Upper
        upper = np.zeros_like(labels_arr)
        upper[1:, :] = labels_arr[1:, :] - labels_arr[:-1, :]
        upper_mask = upper.astype(np.int) != 0
        edge += self.upper_edge[upper_mask].sum()
        # Right
        right = np.zeros_like(labels_arr)
        right[:, :-1] = labels_arr[:, :-1] - labels_arr[:, 1:]
        right_mask = right.astype(np.int) != 0
        edge += self.right_edge[right_mask].sum()
        # Bottom
        bottom = np.zeros_like(labels_arr)
        bottom[:-1, :] = labels_arr[:-1, :] - labels_arr[1:, :]
        bottom_mask = bottom.astype(np.int) != 0
        edge += self.bottom_edge[bottom_mask].sum()

        # Note: For edge value, the equation in the CEC paper is wrong. It does not include the division by the number
        #       of edge pixels, but it should be divided by the number.
        bound_num = (np.sum(left_mask) + np.sum(upper_mask) + np.sum(right_mask) + np.sum(bottom_mask))
        if bound_num == 0:
            return dev, 0.
        else:
            return dev, - edge / bound_num


@numba.jit
def mutation(ind, width, toolbox, mutate_rate=0.0001):
    new_ind = toolbox.clone(ind)
    mask = np.random.rand(ind.size) < mutate_rate
    for i in np.where(mask)[0]:
        r = np.random.randint(5)
        if r == 0 and i % width > 0:  # left
            new_ind[0][i] = i - 1
        elif r == 1 and i - width >= 0:  # upper
            new_ind[0][i] = i - width
        elif r == 2 and (i + 1) % width > 0:  # right
            new_ind[0][i] = i + 1
        elif r == 3 and i + width < ind.size:  # bottom
            new_ind[0][i] = i + width
        elif r == 4:
            new_ind[0][i] = i
    return new_ind


@numba.jit
def crossover(ind1, ind2, toolbox, cross_rate=0.7):
    new_ind1, new_ind2 = toolbox.clone(ind1), toolbox.clone(ind2)

    if np.random.rand() > cross_rate:
        return new_ind1, new_ind2

    mask = np.random.rand(ind1.size) < 0.5
    new_ind1[0][mask] = ind2[0][mask]
    new_ind2[0][mask] = ind1[0][mask]
    return new_ind1, new_ind2


@numba.jit
def reproduction(pop, offspring_size, width, toolbox, mutate_rate=0.0001, cross_rate=0.7):
    offspring = []
    pop_size = len(pop)
    while len(offspring) < offspring_size:
        # Random selection
        c = np.random.choice(np.arange(pop_size), 2, replace=False)
        # Crossover and mutation
        child1, child2 = crossover(pop[c[0]], pop[c[1]], toolbox, cross_rate)
        child1 = mutation(child1, width, toolbox, mutate_rate=mutate_rate)
        child2 = mutation(child2, width, toolbox, mutate_rate=mutate_rate)
        # Fitness evaluation
        child1.fitness.values = toolbox.evaluate(child1)
        child2.fitness.values = toolbox.evaluate(child2)
        # Resampling if constraint is violated
        if child1.fitness.values[0] != np.inf:
            offspring.append(child1)
        if child2.fitness.values[0] != np.inf:
            offspring.append(child2)
    return offspring[:offspring_size]


def save_segment_img(ind, W, H, file_name=None):
    # Calculate connected components
    N = W * H
    num, labels = labeling(ind, W)

    # Create segmentation image
    labels_arr = labels.reshape(H, W)
    seg_img = np.ones((H, W))
    for n in range(N):
        x, y = n % W, n // W
        # right
        if x < W - 1 and labels_arr[y, x] != labels_arr[y, x + 1]:
            seg_img[y, x] = 0
        # down
        if y < H - 1 and labels_arr[y, x] != labels_arr[y + 1, x]:
            seg_img[y, x] = 0

    img_arr = np.asarray(seg_img * 255).astype(np.uint8)
    if file_name is not None:
        cv2.imwrite(file_name, img_arr)
    return img_arr
