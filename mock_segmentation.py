import os
import argparse

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

import pandas as pd
import matplotlib
from matplotlib import pyplot
from PIL import Image

import cv2
import numba

import deap
from deap import base
from deap import creator
from deap import tools

from module import prim_mst, FitnessEvaluation, reproduction, labeling, save_segment_img


print('Versions:')
print('\tNumPy version: ', np.__version__)
print('\tSciPy version: ', scipy.__version__)
print('\tMatplotlib version: ', matplotlib.__version__)
print('\tcv2 version: ', cv2.__version__)
print('\tNumba version: ', numba.__version__)
print('\tDEAP version:', deap.__version__)
print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evolutionary Image Segmentation Based on Multiobjective Clustering')
    parser.add_argument('--image', '-i', type=str, default='paprika.png', help='Image file')
    parser.add_argument('--color', '-c', type=str, default='RGB', help='Color space (\'RGB\' or \'Lab\')')
    args = parser.parse_args()

    img_name = args.image
    color_space = args.color
    out_dir = 'out/'

    max_gen = 300
    pop_size = 50
    offspring_size = 50
    mutate_rate = 0.0001
    cross_rate = 0.7
    min_region_num = 1
    max_region_num = 50
    min_region_size = 100

    # Read image file
    img_bgr = cv2.imread(img_name, cv2.IMREAD_COLOR)  # height x width x 3 (BGR)
    if img_bgr is None:
        print('The image file \"{}\" cannot read.'.format(img_name))
        exit()

    print('Input image name: ', img_name)
    print('Input image size: ', img_bgr.shape)

    # Color space
    if color_space == 'Lab':
        print('Color space: L*a*b*')
        img_arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float)
        # Scaling
        img_arr[:, :, 0] = img_arr[:, :, 0] * 100. / 255.
        img_arr[:, :, 1] = img_arr[:, :, 1] - 128
        img_arr[:, :, 2] = img_arr[:, :, 2] - 128
    else:
        print('Color space: RGB')
        img_arr = img_bgr.astype(np.float)

    # Resize
    max_len = np.max(img_arr.shape)
    if max_len > 128:
        img_arr = cv2.resize(img_arr, (img_arr.shape[1] * 128 // max_len, img_arr.shape[0] * 128 // max_len))
        print('The image is resized to ', img_arr.shape)

    # The number of pixels
    N = img_arr.shape[0] * img_arr.shape[1]

    # Create graph composed of connections between neighboring pixels
    lil_mat = lil_matrix((N, N))
    W = img_arr.shape[1]
    H = img_arr.shape[0]
    for n in range(N):
        x, y = n % W, n // W
        if x < W - 1:  # right
            lil_mat[n, n+1] = np.sqrt(np.sum((img_arr[y, x] - img_arr[y, x+1])**2)) + 1.
        if y < H - 1:  # down
            lil_mat[n, n+W] = np.sqrt(np.sum((img_arr[y, x] - img_arr[y+1, x])**2)) + 1.
        if x > 0:  # left
            lil_mat[n, n-1] = np.sqrt(np.sum((img_arr[y, x] - img_arr[y, x-1])**2)) + 1.
        if y > 0:  # upper
            lil_mat[n, n-W] = np.sqrt(np.sum((img_arr[y, x] - img_arr[y-1, x])**2)) + 1.

    print('Creating MST...')
    mst = prim_mst(csr_matrix(lil_mat), W, N)

    # evolutionary algorithm
    evaluate = FitnessEvaluation(img_arr, min_region_num=min_region_num, max_region_num=max_region_num,
                                 min_region_size=min_region_size)

    creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
    creator.create('Individual', np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('attr', (lambda init_gene: init_gene), mst)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr, n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate)

    # Initialize population
    pop = toolbox.population(n=pop_size)

    # Fitness evaluation
    fits = toolbox.map(toolbox.evaluate, pop)
    for fit, ind in zip(fits, pop):
        ind.fitness.values = fit

    # Evolution loop
    gen = 0
    print('Start evolution loop...')
    while gen < max_gen:
        # Generation of offspring
        offspring = reproduction(pop, offspring_size, W, toolbox, mutate_rate=mutate_rate, cross_rate=cross_rate)

        # Archive truncation
        pop = tools.selSPEA2(pop + offspring, pop_size)

        gen += 1
        if gen % 10 == 0:
            print('({}/{}) '.format(gen, max_gen), end="", flush=True)

    print('')

    dev, edge = np.empty(pop_size), np.empty(pop_size)
    for i, p in enumerate(pop):
        dev[i], edge[i] = p.fitness.values[0], p.fitness.values[1]

    df = pd.DataFrame(index=np.arange(len(pop)), columns=[])

    print('Saving data...')
    # Sorting and normalization
    index = np.argsort(edge)
    pop_sort = [pop[i] for i in index]
    df['dev'] = dev[index]
    df['edge'] = edge[index]
    df['dev_norm'] = (dev[index] - np.min(dev)) / (np.max(dev) - np.min(dev))
    df['edge_norm'] = (edge[index] - np.min(edge)) / (np.max(edge) - np.min(edge))

    # Number of regions
    region_num = np.empty(len(pop_sort), dtype=np.int)
    for i, p in enumerate(pop_sort):
        num, _ = labeling(p[0], W)
        region_num[i] = num
    df['region_num'] = region_num

    # Selection
    sel = int(np.argmin(df['dev_norm'] + df['edge_norm']))
    selection = np.zeros(len(pop_sort), dtype=np.int)
    selection[sel] = 1
    df['selection'] = selection

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_dir + 'solutions.csv')

    # Plot Pareto solutions
    pyplot.figure(figsize=(5, 5))
    pyplot.scatter(df['edge_norm'], df['dev_norm'], color='black')
    pyplot.ylabel('Overall Deviation')
    pyplot.xlabel('Edge')
    pyplot.grid(True)
    pyplot.savefig(out_dir + 'solutions.pdf')

    # Save the selected segmentation image
    array = save_segment_img(pop_sort[sel][0], W, H)
    img = Image.fromarray(np.uint8(array))
    img.save(out_dir + 'select_sol.png')

    for i, p in enumerate(pop_sort):
        array = save_segment_img(p[0], W, H)
        img = Image.fromarray(np.uint8(array))
        img.save(out_dir + '{:03d}.png'.format(i))
