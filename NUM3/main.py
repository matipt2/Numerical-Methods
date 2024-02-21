import sys
import numpy as np
from functools import reduce
import time
import matplotlib.pyplot as plt

def solve_system(n):
    matrix = [[0] + [0.2] * (n - 1), [1.2] * n, [0.1 / i for i in range(1, n + 1)] + [0], [0.15 / i**2 for i in range(1, n + 1)] + [0, 0]]
    x = list(range(1, n + 1))
    start_time = time.time()

    for i in range(1, n-2):
        matrix[0][i] /= matrix[1][i - 1]
        matrix[1][i] -= matrix[0][i] * matrix[2][i - 1]
        matrix[2][i] -= matrix[0][i] * matrix[3][i - 1]
    matrix[0][n-2] /= matrix[1][n-3]
    matrix[1][n-2] -= matrix[0][n-2] * matrix[2][n-3]
    matrix[2][n-2] -= matrix[0][n-2] * matrix[3][n-3]
    matrix[0][n-1] /= matrix[1][n-2]
    matrix[1][n-1] -= matrix[0][n-1] * matrix[2][n-2]

    for i in range(1, n):
        x[i] -= matrix[0][i] * x[i - 1]

    x[n-1] /= matrix[1][n-1]
    x[n-2] = (x[n-2] - matrix[2][n-2] * x[n-1]) / matrix[1][n-2]
    for i in range(n - 3, -1, -1):
        x[i] = (x[i] - matrix[3][i] * x[i + 2] - matrix[2][i] * x[i + 1]) / matrix[1][i]

    determinant = reduce(lambda a, b: a * b, matrix[1])
    execution_time = time.time() - start_time

    return determinant, x, execution_time

def calculate_execution_time(N_values, probes):
    execution_times = []

    for n in N_values:
        times = []
        for j in range(probes):
            start_time = time.time()
            solve_system(n)
            end_time = time.time()
            times.append(end_time - start_time)
        average_time = sum(times) / probes
        execution_times.append(average_time)

    return execution_times

def plot_complexity(N_values, time_in_seconds):
    plt.plot(N_values, time_in_seconds, marker='.')
    plt.xlabel('N')
    plt.ylabel('exec time')
    plt.title('complexity')
    plt.grid(True)
    plt.show()

if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg == '1':
        N_values = list(range(124, 1200))
        execution_times = calculate_execution_time(N_values, probes=30)
        plot_complexity(N_values, execution_times)
    elif arg == '2':
        n = 124
        determinant, solution, execution_time = solve_system(n)
        print("Wyznacznik jest równy:", determinant)
        print("\nRozwiązanie:")
        for i in range(n):
            print(solution[i])

    else:
        print("Invalid argument")
