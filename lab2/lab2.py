import numpy as np
import time
import psutil


def gradient_descent(a, b, x_train, y_train, lr=0.01, target_error=None):
    iters = 0
    while True:
        yhat = a + b * x_train
        error = (y_train - yhat)
        a_grad = -2 * error.mean()
        b_grad = -2 * (x_train * error).mean()
        a = a - lr * a_grad
        b = b - lr * b_grad
        final_error = ((y_train - (a + b * x_train)) ** 2).mean()
        iters += 1
        if target_error is not None and final_error < target_error:
            break

    print(a, b)
    print(f'Final error: {final_error}')
    print(f'Iterations: {iters}')


def hill_climb(a, b, x_train, y_train, step_size=0.1, target_error=None):
    iters = 0
    while True:
        a_new = a + (np.random.rand() - 0.5) * step_size
        b_new = b + (np.random.rand() - 0.5) * step_size

        old_error = ((y_train - (a + b) * x_train) ** 2).mean()
        new_error = ((y_train - (a_new + b_new * x_train)) ** 2).mean()

        if new_error < old_error:
            a, b = a_new, b_new
            final_error = new_error
        else:
            final_error = old_error

        iters += 1
        if target_error is not None and final_error < target_error:
            break

    print(a, b)
    print(f'Final error: {final_error}')
    print(f'Iterations: {iters}')


def genetic(x_train, y_train, population_size=100, mutation_rate=0.01, target_error=None):
    population = np.random.randn(population_size, 2)
    iters = 0

    while True:
        errors = np.array([((y_train - (a + b * x_train)) ** 2).mean() for a, b, in population])

        fitness_scores = 1 / (1 + errors)
        probs = fitness_scores / fitness_scores.sum()
        selected = population[np.random.choice(np.arange(population_size), size=population_size, replace=True, p=probs)]

        pairs = selected[np.random.randint(0, population_size, size=(population_size, 2))]
        new_population = pairs.mean(axis=1)

        mutations = (np.random.rand(population_size, 2) - 0.5) * mutation_rate
        new_population += mutations
        population = new_population

        best_idx = np.argmin([((y_train - (a + b * x_train)) ** 2).mean() for a, b in population])
        best_a, best_b = population[best_idx]
        final_error = errors[best_idx]

        iters += 1
        if target_error is not None and final_error < target_error:
            break

    print(best_a, best_b)
    print(f'Final error: {final_error}')
    print(f'Iterations: {iters}')


def main(target_error):
    np.random.seed(42)
    sz = 1000
    x = np.random.rand(sz, 1)
    noise = np.random.normal(loc=0.0, scale=0.1, size=sz)
    y = 4 * np.sqrt(x) + noise

    idx = np.arange(sz)
    np.random.shuffle(idx)
    train_idx = idx
    x_train, y_train = x[train_idx], y[train_idx]

    a = np.random.randn(1)
    b = np.random.randn(1)
    print(a, b)

    initial_error = ((y_train - (a + b * x_train)) ** 2).mean()
    print(f'Initial error: {initial_error}')

    print('\n---Gradient Descent---')
    start = time.perf_counter()

    process = psutil.Process(gradient_descent(a, b, x_train, y_train, lr=0.1, target_error=target_error))
    print(f'Time elapsed: {time.perf_counter() - start} seconds')
    print(f'Memory usage: {process.memory_info().rss / 1024 / 1024} MBytes')

    print('\n---Hill Climb---')
    start = time.perf_counter()
    process = psutil.Process(hill_climb(a, b, x_train, y_train, step_size=1, target_error=target_error))
    print(f'Time elapsed: {time.perf_counter() - start} seconds')
    print(f'Memory usage: {process.memory_info().rss / 1024 / 1024} MBytes')

    print('\n---Genetic---')
    start = time.perf_counter()
    process = psutil.Process(genetic(x_train, y_train, population_size=10, mutation_rate=1,
                                     target_error=target_error))
    print(f'Time elapsed: {time.perf_counter() - start} seconds')
    print(f'Memory usage: {process.memory_info().rss / 1024 / 1024} MBytes')


main(0.4)
