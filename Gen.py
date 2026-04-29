import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import time
import random
from itertools import permutations

def generate_uniform_tsp(n_cities, seed=None):
    if seed: np.random.seed(seed)
    return np.random.rand(n_cities, 2) * 100

def generate_clustered_tsp(n_cities, n_clusters=4, seed=None):
    if seed: np.random.seed(seed)
    points = []
    cities_per_cluster = n_cities // n_clusters
    for _ in range(n_clusters):
        center = np.random.rand(2) * 100
        cluster_points = center + np.random.randn(cities_per_cluster, 2) * 5
        points.extend(cluster_points)
    return np.array(points)

def generate_circular_tsp(n_cities):
    angles = np.linspace(0, 2 * np.pi, n_cities, endpoint=False)
    x = 50 + 40 * np.cos(angles)
    y = 50 + 40 * np.sin(angles)
    return np.column_stack((x, y))

def get_distance_matrix(points):
    return squareform(pdist(points, 'euclidean'))

def path_length(path, dist_matrix):
    return sum(dist_matrix[path[i], path[i-1]] for i in range(len(path)))

def solve_tsp_brute_force(dist_matrix):
    """Точный алгоритм для малых графов (N <= 10)"""
    n = len(dist_matrix)
    min_dist = float('inf')
    best_path = None
    for path in permutations(range(n)):
        dist = path_length(path, dist_matrix)
        if dist < min_dist:
            min_dist = dist
            best_path = path
    return list(best_path), min_dist

def solve_tsp_mst(dist_matrix):
    """2-приближение на основе остовного дерева"""
    n = len(dist_matrix)
    G = nx.from_numpy_array(dist_matrix)
    mst = nx.minimum_spanning_tree(G)
    path = list(nx.dfs_preorder_nodes(mst, source=0))
    return path, path_length(path, dist_matrix)

class TSPGeneticAlgorithm:
    def __init__(self, dist_matrix, pop_size=100, generations=300, 
                 mutation_rate=0.2, init_type='random', 
                 crossover_type='ox', mutation_type='inversion'):
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.init_type = init_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        
    def _create_individual(self, use_greedy=False):
        if use_greedy:
            start = random.randint(0, self.n_cities - 1)
            unvisited = set(range(self.n_cities))
            unvisited.remove(start)
            path = [start]
            current = start
            while unvisited:
                next_city = min(unvisited, key=lambda city: self.dist_matrix[current, city])
                path.append(next_city)
                unvisited.remove(next_city)
                current = next_city
            return path
        else:
            ind = list(range(self.n_cities))
            random.shuffle(ind)
            return ind
            
    def _init_population(self):
        population = []
        for i in range(self.pop_size):
            use_greedy = (self.init_type == 'mixed' and i < self.pop_size * 0.1)
            population.append(self._create_individual(use_greedy=use_greedy))
        return population
    
    def _fitness(self, individual):
        return 1.0 / path_length(individual, self.dist_matrix)
    
    def _tournament_selection(self, population, fitnesses, k=3):
        selected_indices = random.sample(range(self.pop_size), k)
        best_idx = max(selected_indices, key=lambda idx: fitnesses[idx])
        return population[best_idx]
    
    def _ordered_crossover(self, p1, p2):
        start, end = sorted(random.sample(range(self.n_cities), 2))
        child = [-1] * self.n_cities
        child[start:end] = p1[start:end]
        p2_idx, child_idx = end, end
        while -1 in child:
            if p2[p2_idx % self.n_cities] not in child:
                child[child_idx % self.n_cities] = p2[p2_idx % self.n_cities]
                child_idx += 1
            p2_idx += 1
        return child
    
    def _pmx_crossover(self, p1, p2):
        start, end = sorted(random.sample(range(self.n_cities), 2))
        child = [-1] * self.n_cities
        child[start:end] = p1[start:end]
        mapping = {p1[i]: p2[i] for i in range(start, end)}
        
        for i in range(self.n_cities):
            if start <= i < end: continue
            val = p2[i]
            while val in mapping:
                val = mapping[val]
            child[i] = val
        return child

    def _mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx1, idx2 = sorted(random.sample(range(self.n_cities), 2))
            if self.mutation_type == 'swap':
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            elif self.mutation_type == 'inversion':
                individual[idx1:idx2] = reversed(individual[idx1:idx2])
            elif self.mutation_type == 'scramble':
                sub = individual[idx1:idx2]
                random.shuffle(sub)
                individual[idx1:idx2] = sub
        return individual

    def run(self):
        population = self._init_population()
        best_path = None
        best_dist = float('inf')
        history_best = []
        
        for _ in range(self.generations):
            fitnesses = [self._fitness(ind) for ind in population]
            
            current_best_idx = np.argmax(fitnesses)
            current_best_dist = path_length(population[current_best_idx], self.dist_matrix)
            
            if current_best_dist < best_dist:
                best_dist = current_best_dist
                best_path = list(population[current_best_idx])
                
            history_best.append(best_dist)
            
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_population = [population[sorted_indices[0]], population[sorted_indices[1]]]
            
            while len(new_population) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                
                if self.crossover_type == 'pmx':
                    child = self._pmx_crossover(p1, p2)
                else:
                    child = self._ordered_crossover(p1, p2)
                    
                child = self._mutate(child)
                new_population.append(child)
                
            population = new_population
            
        return best_path, best_dist, history_best


def run_experiments():
    np.random.seed(42)
    random.seed(42)
    
    configs = [
        {'name': 'V1: Rand + OX + Swap', 'init': 'random', 'cross': 'ox', 'mut': 'swap'},
        {'name': 'V2: Rand + PMX + Inversion', 'init': 'random', 'cross': 'pmx', 'mut': 'inversion'},
        {'name': 'V3: Mixed Init + OX + Inversion', 'init': 'mixed', 'cross': 'ox', 'mut': 'inversion'}
    ]
    
    testsets = {
        "Малый граф (N=8)": get_distance_matrix(generate_uniform_tsp(8)),
        "Равномерный (N=50)": get_distance_matrix(generate_uniform_tsp(50)),
        "Кластерный (N=50)": get_distance_matrix(generate_clustered_tsp(50, 4)),
        "Кольцевой (N=50)": get_distance_matrix(generate_circular_tsp(50))
    }
    
    runs_per_test = 40
    
    for ts_name, dist_mat in testsets.items():
        print(f"\n{'='*60}\nТЕСТ-СЕТ: {ts_name}\n{'='*60}")
        
        if len(dist_mat) <= 10:
            _, opt_dist = solve_tsp_brute_force(dist_mat)
            print(f"[Точный оптимум (Полный перебор)]: {opt_dist:.2f}")
        else:
            opt_dist = None
            
        t0 = time.time()
        _, mst_dist = solve_tsp_mst(dist_mat)
        t_mst = time.time() - t0
        print(f"[MST 2-приближение]: {mst_dist:.2f} (Время: {t_mst:.4f} с)")
        
        plt.figure(figsize=(10, 5))
        plt.title(f'Сходимость алгоритмов: {ts_name}')
        plt.axhline(y=mst_dist, color='r', linestyle='--', label='MST Baseline')
        if opt_dist:
            plt.axhline(y=opt_dist, color='g', linestyle='-', label='Optimal')
            
        for conf in configs:
            dists, times = [], []
            best_hist = None
            best_overall = float('inf')
            
            for _ in range(runs_per_test):
                ga = TSPGeneticAlgorithm(dist_mat, pop_size=100, generations=400, 
                                         init_type=conf['init'], crossover_type=conf['cross'], 
                                         mutation_type=conf['mut'])
                t0 = time.time()
                _, dist, hist = ga.run()
                t_ga = time.time() - t0
                
                dists.append(dist)
                times.append(t_ga)
                if dist < best_overall:
                    best_overall = dist
                    best_hist = hist
                    
            print(f"\n[{conf['name']}] Статистика ({runs_per_test} запусков):")
            print(f"  Мин: {np.min(dists):.2f} | Макс: {np.max(dists):.2f} | Среднее: {np.mean(dists):.2f}")
            print(f"  Медиана: {np.median(dists):.2f} | Откл(Std): {np.std(dists):.2f}")
            print(f"  Ср. время: {np.mean(times):.2f} с")
            print(f"  Улучшение от MST: {((mst_dist - np.mean(dists)) / mst_dist * 100):.2f}%")
            
            plt.plot(best_hist, label=conf['name'])
            
        plt.xlabel('Поколения')
        plt.ylabel('Длина маршрута')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    run_experiments()