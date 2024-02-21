import matplotlib.pyplot as plt
import time
import numpy as np


def sherman_morrison(n):
    b_vector = [5] * n
    matrix_M = [[11] * n, [7] * (n - 1) + [0]]

    z_vector = [0] * n
    x_vector = [0] * n

    z_vector[n - 1] = b_vector[n - 1] / matrix_M[0][n - 1]
    x_vector[n - 1] = 1 / matrix_M[0][n - 1]

    for i in range(n - 2, -1, -1):
        z_vector[i] = (b_vector[n - 2] - matrix_M[1][i] * z_vector[i + 1]) / matrix_M[0][i]
        x_vector[i] = (1 - matrix_M[1][i] * x_vector[i + 1]) / matrix_M[0][i]

    delta = sum(z_vector) / (1 + sum(x_vector))

    result_vector = [z_vector[i] - x_vector[i] * delta for i in range(len(z_vector))]
    return result_vector


def numpySolve(n):
    A = np.ones((n, n))
    A += np.diag([11] * n)
    A += np.diag([7] * (n - 1), 1)
    b_vector = [5] * n
    x = np.linalg.solve(A, b_vector)
    print(x)


def calculate_execution_time_sherman_morrison(N_values, probes):
    execution_times = []

    for n in N_values:
        times = [timeit_sherman_morrison(n) for _ in range(probes)]
        average_time = sum(times) / probes
        execution_times.append(average_time)

    return execution_times

def timeit_sherman_morrison(n):
    start_time = time.time()
    sherman_morrison(n)
    end_time = time.time()
    return end_time - start_time

def plot_complexity_sherman_morrison(N_values, time_in_seconds):
    plt.plot(N_values, time_in_seconds, marker='*')
    plt.xlabel('n')
    plt.ylabel('exec time')
    plt.title('Sherman-Morrison complexity')
    plt.grid(True)
    plt.show()
print("my implementation:")
result_n_80 = sherman_morrison(80)
print(result_n_80)
print("numpy solve:")
numpySolve(80)
N_values_sherman_morrison = list(range(80, 3000, 20))
execution_times_sherman_morrison = calculate_execution_time_sherman_morrison(N_values_sherman_morrison, probes=50)
plot_complexity_sherman_morrison(N_values_sherman_morrison, execution_times_sherman_morrison)

