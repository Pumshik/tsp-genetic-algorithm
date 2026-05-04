import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import time
import random
from itertools import permutations

def generate_uniform_tsp(n_cities, seed=None):
    """Создаёт n_cities точек со случайными координатами в квадрате [0,100]."""
    if seed: np.random.seed(seed)
    return np.random.rand(n_cities, 2) * 100

def generate_clustered_tsp(n_cities, n_clusters=4, seed=None):
    """Генерирует данные с несколькими плотными кластерами."""
    if seed: np.random.seed(seed)
    points = []
    cities_per_cluster = n_cities // n_clusters
    for _ in range(n_clusters):
        center = np.random.rand(2) * 100
        cluster_points = center + np.random.randn(cities_per_cluster, 2) * 5
        points.extend(cluster_points)
    return np.array(points)

def generate_circular_tsp(n_cities):
    """Равномерно расставляет города по окружности радиуса 40 с центром (50,50)."""
    angles = np.linspace(0, 2 * np.pi, n_cities, endpoint=False)
    x = 50 + 40 * np.cos(angles)
    y = 50 + 40 * np.sin(angles)
    return np.column_stack((x, y))

def generate_erdos_renyi_tsp(n_cities, p, seed=None):
    """
    Генерирует метрический TSP-экземпляр на основе модели Эрдёша-Реньи G(n,p).
    Сначала строится случайный граф, затем вычисляются кратчайшие пути (алг. Флойд-Уоршелл)
    для получения метрической матрицы расстояний.
    """
    if seed: np.random.seed(seed)
    G = nx.erdos_renyi_graph(n_cities, p, seed=seed)
    
    if not nx.is_connected(G):
        # Добавляем случайные рёбра пока граф не станет связным
        nodes = list(G.nodes())
        while not nx.is_connected(G):
            u, v = random.sample(nodes, 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=1.0)
    
    # Вычисляем матрицу кратчайших путей
    dist_matrix = np.ones((n_cities, n_cities)) * np.inf
    np.fill_diagonal(dist_matrix, 0)
    for u, v in G.edges():
        dist_matrix[u, v] = dist_matrix[v, u] = 1.0
    
    # Алгоритм Флойда-Уоршелла для метризации
    for k in range(n_cities):
        for i in range(n_cities):
            for j in range(n_cities):
                if dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
    
    return dist_matrix

def generate_configuration_model_tsp(n_cities, degree_sequence, seed=None):
    """
    Генерирует метрический TSP на основе конфигурационной модели.
    degree_sequence: список желаемых степеней вершин.
    """
    if seed: random.seed(seed)
    G = nx.configuration_model(degree_sequence, seed=seed, create_using=nx.Graph())
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Если граф несвязный - добавляем рёбра
    if not nx.is_connected(G):
        nodes = list(G.nodes())
        while not nx.is_connected(G):
            u, v = random.sample(nodes, 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
    
    # Метризация через кратчайшие пути
    dist_matrix = nx.floyd_warshall_numpy(G)
    return dist_matrix


def get_distance_matrix(points):
    """Вычисляет матрицу евклидовых расстояний."""
    return squareform(pdist(points, 'euclidean'))

def path_length(path, dist_matrix):
    """Возвращает длину замкнутого маршрута."""
    return sum(dist_matrix[path[i], path[i-1]] for i in range(len(path)))

def solve_tsp_brute_force(dist_matrix):
    """Точный перебор всех перестановок (для N <= 10)."""
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
    """2-приближение через минимальное остовное дерево."""
    n = len(dist_matrix)
    G = nx.from_numpy_array(dist_matrix)
    mst = nx.minimum_spanning_tree(G)
    path = list(nx.dfs_preorder_nodes(mst, source=0))
    return path, path_length(path, dist_matrix)


def solve_tsp_2opt(dist_matrix, initial_path=None, max_iterations=1000):
    """
    Улучшение маршрута с помощью локального поиска 2-Opt.
    Если initial_path не задан, начинает с жадного решения.
    """
    n = len(dist_matrix)
    
    # Начальное решение: жадный алгоритм или переданный путь
    if initial_path is None:
        start = 0
        unvisited = set(range(n))
        unvisited.remove(start)
        path = [start]
        current = start
        while unvisited:
            next_city = min(unvisited, key=lambda city: dist_matrix[current, city])
            path.append(next_city)
            unvisited.remove(next_city)
            current = next_city
    else:
        path = list(initial_path)
    
    best_path = path
    best_dist = path_length(best_path, dist_matrix)
    
    # Итеративное улучшение 2-Opt
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Пробуем развернуть подотрезок [i, j]
                new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                new_dist = path_length(new_path, dist_matrix)
                if new_dist < best_dist - 1e-8:
                    best_path = new_path
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    
    return best_path, best_dist


class TSPGeneticAlgorithm:
    """Генетический алгоритм для метрической задачи коммивояжёра."""
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
        """Создаёт особь: либо жадным алгоритмом, либо случайной перестановкой."""
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
        """Формирует начальную популяцию с учётом init_type."""
        population = []
        for i in range(self.pop_size):
            use_greedy = (self.init_type == 'mixed' and i < self.pop_size * 0.1)
            population.append(self._create_individual(use_greedy=use_greedy))
        return population
    
    def _fitness(self, individual):
        """Приспособленность = 1 / длина маршрута."""
        return 1.0 / path_length(individual, self.dist_matrix)
    
    def _tournament_selection(self, population, fitnesses, k=3):
        """Турнирный отбор размера k."""
        selected_indices = random.sample(range(self.pop_size), k)
        best_idx = max(selected_indices, key=lambda idx: fitnesses[idx])
        return population[best_idx]
    
    def _ordered_crossover(self, p1, p2):
        """Ordered crossover: сохраняет подотрезок первого родителя, остальное заполняется вторым."""
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
        """Partially Mapped crossover: строит отображение между подотрезками."""
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
        """Применяет мутацию с вероятностью mutation_rate."""
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
        """Запуск ГА, возвращает лучший маршрут, его длину и историю улучшений."""
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
    """Главная функция: прогоняет эксперименты и строит графики сходимости."""
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
    
    # Построение сетки графиков
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (ts_name, dist_mat) in enumerate(testsets.items()):
        ax = axes[idx]
        ax.set_title(ts_name)
        
        print(f"\n{'='*60}\nТЕСТ-СЕТ: {ts_name}\n{'='*60}")
        
        if len(dist_mat) <= 10:
            _, opt_dist = solve_tsp_brute_force(dist_mat)
            print(f"[Точный оптимум (Полный перебор)]: {opt_dist:.2f}")
            ax.axhline(
                y=opt_dist, 
                color='purple', 
                linestyle='-.',
                linewidth=1.5,
                alpha=0.8, 
                label='Оптимум'
            )
        else:
            opt_dist = None
            
        t0 = time.time()
        _, mst_dist = solve_tsp_mst(dist_mat)
        t_mst = time.time() - t0
        print(f"[MST 2-приближение]: {mst_dist:.2f} (Время: {t_mst:.4f} с)")
        ax.axhline(y=mst_dist, color='red', linestyle='--', alpha=0.7, label='MST')
        
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
            
            ax.plot(best_hist, label=conf['name'])
        
        ax.set_xlabel('Поколения')
        ax.set_ylabel('Длина маршрута')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Для графиков с большим разбросом используем логарифмическую шкалу
        if ts_name in ["Равномерный (N=50)", "Кольцевой (N=50)"]:
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('convergence_plots.pdf', dpi=300)
    plt.show()
    
    # Тестирование масштабируемости и новых типов графов
    print(f"\n{'='*60}\nМАСШТАБИРУЕМОСТЬ И НОВЫЕ ТИПЫ ГРАФОВ\n{'='*60}")
    
    sizes = [10, 20, 30, 40, 50, 100]
    graph_types = {
        'Uniform': lambda n: get_distance_matrix(generate_uniform_tsp(n, seed=42)),
        'ER_dense (p=0.3)': lambda n: generate_erdos_renyi_tsp(n, p=0.3, seed=42),
        'ER_sparse (p=1/n)': lambda n: generate_erdos_renyi_tsp(n, p=1.0/n, seed=42),
        'ER_threshold (p=ln(n)/n)': lambda n: generate_erdos_renyi_tsp(n, p=np.log(n)/n, seed=42),
    }
    
    results = {gt: {'n': [], 'mst_dist': [], 'ga_dist': [], '2opt_dist': [], 
                    'mst_time': [], 'ga_time': [], '2opt_time': []} 
               for gt in graph_types}
    
    for gt_name, gen_func in graph_types.items():
        print(f"\n--- Тип графа: {gt_name} ---")
        for n in sizes:
            print(f"  n={n}...", end=' ', flush=True)
            dist_mat = gen_func(n)
            
            # MST
            t0 = time.time()
            _, mst_dist = solve_tsp_mst(dist_mat)
            mst_time = time.time() - t0
            
            # GA
            pop_size = 100 if n <= 50 else 50
            generations = 400 if n <= 50 else 200
            t0 = time.time()
            ga = TSPGeneticAlgorithm(dist_mat, pop_size=pop_size, generations=generations,
                                     init_type='mixed', crossover_type='ox', mutation_type='inversion')
            _, ga_dist, _ = ga.run()
            ga_time = time.time() - t0
            
            # 2-Opt
            t0 = time.time()
            mst_path, _ = solve_tsp_mst(dist_mat)
            _, opt_dist = solve_tsp_2opt(dist_mat, initial_path=mst_path)
            opt_time = time.time() - t0
            
            results[gt_name]['n'].append(n)
            results[gt_name]['mst_dist'].append(mst_dist)
            results[gt_name]['ga_dist'].append(ga_dist)
            results[gt_name]['2opt_dist'].append(opt_dist)
            results[gt_name]['mst_time'].append(mst_time)
            results[gt_name]['ga_time'].append(ga_time)
            results[gt_name]['2opt_time'].append(opt_time)
            
            print(f"MST={mst_dist:.1f}, GA={ga_dist:.1f}, 2-Opt={opt_dist:.1f}")
    
    # Построение графика масштабируемости
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # График 1: Длина маршрута от n
    ax1 = axes2[0]
    for gt_name in graph_types:
        ax1.plot(results[gt_name]['n'], results[gt_name]['mst_dist'], 
                 marker='o', label=f'{gt_name} - MST', linestyle='-')
        ax1.plot(results[gt_name]['n'], results[gt_name]['ga_dist'], 
                 marker='s', label=f'{gt_name} - GA', linestyle='--')
        ax1.plot(results[gt_name]['n'], results[gt_name]['2opt_dist'], 
                 marker='^', label=f'{gt_name} - 2-Opt', linestyle=':')
    ax1.set_xlabel('Число вершин (n)')
    ax1.set_ylabel('Длина маршрута')
    ax1.set_title('Зависимость качества решения от размера графа')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)
    
    # График 2: Время выполнения от n с логарифмической шкалой
    ax2 = axes2[1]
    for gt_name in graph_types:
        ax2.plot(results[gt_name]['n'], results[gt_name]['mst_time'], 
                 marker='o', label=f'{gt_name} - MST', linestyle='-')
        ax2.plot(results[gt_name]['n'], results[gt_name]['ga_time'], 
                 marker='s', label=f'{gt_name} - GA', linestyle='--')
        ax2.plot(results[gt_name]['n'], results[gt_name]['2opt_time'], 
                 marker='^', label=f'{gt_name} - 2-Opt', linestyle=':')
    ax2.set_xlabel('Число вершин (n)')
    ax2.set_ylabel('Время выполнения (с)')
    ax2.set_title('Зависимость времени от размера графа')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)
    
    plt.tight_layout()
    plt.savefig('scalability_plot.pdf', dpi=300)
    plt.show()
    
    # Вывод таблицы результатов
    print(f"\n{'='*60}\nТАБЛИЦА РЕЗУЛЬТАТОВ (среднее улучшение относительно MST)\n{'='*60}")
    for gt_name in graph_types:
        print(f"\n{gt_name}:")
        print(f"  {'n':>4} | {'MST':>8} | {'GA':>8} | {'2-Opt':>8} | {'GA vs MST':>10} | {'2-Opt vs MST':>12}")
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}")
        for i, n in enumerate(sizes):
            mst = results[gt_name]['mst_dist'][i]
            ga = results[gt_name]['ga_dist'][i]
            opt = results[gt_name]['2opt_dist'][i]
            ga_imp = (mst - ga) / mst * 100
            opt_imp = (mst - opt) / mst * 100
            print(f"  {n:>4} | {mst:>8.1f} | {ga:>8.1f} | {opt:>8.1f} | {ga_imp:>9.1f}% | {opt_imp:>11.1f}%")
    

if __name__ == "__main__":
    run_experiments()