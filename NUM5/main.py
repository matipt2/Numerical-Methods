import numpy as np
import copy
import random

from matplotlib import pyplot as plt

def plot_iteration_errors(errors1, errors2):
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel("$|x(n) - x(last)|$")
    plt.yscale('log')
    plt.plot([i for i in range(1, len(errors1)+1)], errors1)
    plt.plot([i for i in range(1, len(errors2)+1)], errors2)
    plt.legend(['Jacobi', 'Gauss-Seidel'])
    plt.title('Porownanie dwoch metod')
    plt.show()

def gaussSeidel_method(x_vector, stop, c=0.15):
    gauss_Result = []
    norms = []
    norm = 0
    for _ in range(stop):
        y_vector = x_vector.copy()
        for i in range(n):
            if (i == 0):
                x_vector[i] = (b[i] - x_vector[i + 1] - c * x_vector[i + 2]) / 3
            elif (i == 1):
                x_vector[i] = (b[i] - x_vector[i - 1] - y_vector[i + 1] - c * y_vector[i + 2]) / 3
            elif (i == n - 2):
                x_vector[i] = (b[i] - x_vector[i - 1] - c * x_vector[i - 2] - y_vector[i + 1]) / 3
            elif (i == n - 1):
                x_vector[i] = (b[i] - x_vector[i - 1] - c * x_vector[i - 2]) / 3
            else:
                x_vector[i] = (b[i] - x_vector[i - 1] - c * x_vector[i - 2] - y_vector[i + 1] - c * y_vector[i + 2]) / 3

        norma1 = np.sqrt(sum(map(lambda a, b: (a - b) ** 2, x_vector, y_vector)))
        gauss_Result.append(copy.deepcopy(x_vector))
        norms.append(norma1)

        if abs(norm - norma1) < 10 ** (-12):
            break
        norm = norma1

    return gauss_Result, norms, x_vector

def jacobi_method(x_vector, stop, c=0.15):
    jacobiResult = []
    norms = []
    norm = 0
    for _ in range(stop):
        y_vector = x_vector.copy()
        for i in range(n):
            if (i == 0):
                x_vector[i] = (b[i] - y_vector[i + 1] - c * y_vector[i + 2]) / 3
            elif (i == 1):
                x_vector[i] = (b[i] - y_vector[i - 1] - y_vector[i + 1] - c * y_vector[i + 2]) / 3
            elif (i == n - 2):
                x_vector[i] = (b[i] - y_vector[i - 1] - c * y_vector[i - 2] - y_vector[i + 1]) / 3
            elif (i == n - 1):
                x_vector[i] = (b[i] - y_vector[i - 1] - c * y_vector[i - 2]) / 3
            else:
                x_vector[i] = (b[i] - y_vector[i - 1] - c * y_vector[i - 2] - y_vector[i + 1] - c * y_vector[i + 2]) / 3
        norma1 = np.sqrt(sum(map(lambda a, b: (a - b) ** 2, x_vector, y_vector)))
        jacobiResult.append(copy.deepcopy(x_vector))
        norms.append(norma1)

        if abs(norm - norma1) < 10 ** (-12):
            break
        norm = norma1

    return jacobiResult, norms, x_vector

n = 124
stop = 200
x = random.sample(range(200), 124)
b = list(range(1, n + 1))

c_jacobi = 0.15
c_gauss = 0.15

wynik_jacobi, normy1, result_jacobi = jacobi_method(x.copy(), stop, c=c_jacobi)
wynik_gauss, normy2, result_gauss = gaussSeidel_method(x.copy(), stop, c=c_gauss)

arr1 = []
last1 = wynik_jacobi[-1]
for i in range(len(wynik_jacobi) - 1):
    arr1.append(np.sqrt(sum(map(lambda a, b: (a - b) ** 2, wynik_jacobi[i], last1))))

arr2 = []
last2 = wynik_gauss[-1]
for i in range(len(wynik_gauss) - 1):
    arr2.append(np.sqrt(sum(map(lambda a, b: (a - b) ** 2, wynik_gauss[i], last2))))

A = np.diag(3 * np.ones(n)) + np.diag(1 * np.ones(n - 1), k=1) + np.diag(0.15 * np.ones(n - 2), k=2) + np.diag(
    1 * np.ones(n - 1), k=-1) + np.diag(0.15 * np.ones(n - 2), k=-2)

np.set_printoptions(precision=16)

c = np.arange(1, n + 1)

print(f"Wynik otrzymany metodą Jacobiego):")
print(result_jacobi)
print()

print(f"Wynik otrzymany metodą Gaussa-Seidela):")
print(result_gauss)
print()
plot_iteration_errors(arr1, arr2)