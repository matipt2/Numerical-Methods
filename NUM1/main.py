import sys
import numpy as np
import matplotlib.pyplot as plt

# Dokładna pochodna
def exact_derivative32(system, point):
    return (2 * point * np.float32(np.cos(point ** 2)))

# Przybliżona wartość pochodnej metodą różnic w przód
def forward_derivative32(system, point, h):
    return ((np.float32(np.sin((point + h) ** 2)) - np.float32(np.sin(point ** 2))) / np.float32(h))

# Przybliżona wartość pochodnej metodą różnic centralnych
def central_derivative32(system, point, h):
    return (np.float32(np.sin((point + h) ** 2)) - np.float32(np.sin((point - h) ** 2))) / np.float32((2 * h))

# Funkcja oblicza błędy przybliżonej pochodnej dla różnych wartości kroku h
# Dla każdej wartości h oblicza przybliżoną pochodną i oblicza błąd względem dokładnej pochodnej
# Na końcu błędy są zbierane w liście errors[]
def calculate_errors32(system, point, h_values, derivative_function):
    errors = []
    for h in h_values:
        approx_derivative = np.float32(derivative_function(system, point, h))
        exact = np.float32(exact_derivative(system, point))
        error = np.float32(np.abs(approx_derivative - exact))
        errors.append(error)
    return np.float32(errors)

def exact_derivative(system, point):
    return 2 * point * np.cos(point ** 2)

# Przybliżona wartość pochodnej metodą różnic w przód
def forward_derivative(system, point, h):
    return (np.sin((point + h) ** 2) - np.sin(point ** 2)) / h

# Przybliżona wartość pochodnej metodą różnic centralnych
def central_derivative(system, point, h):
    return (np.sin((point + h) ** 2) - np.sin((point - h) ** 2)) / (2 * h)

# Funkcja oblicza błędy przybliżonej pochodnej dla różnych wartości kroku h
# Dla każdej wartości h oblicza przybliżoną pochodną i oblicza błąd względem dokładnej pochodnej
# Na końcu błędy są zbierane w liście errors[]
def calculate_errors(system, point, h_values, derivative_function):
    exact = exact_derivative(system, point)
    errors = []
    for h in h_values:
        approx_derivative = derivative_function(system, point, h)
        error = np.abs(approx_derivative - exact)
        errors.append(error)
    return errors

# Rysowanie wykresu
def graph(h_values, errors_forward, errors_central, label):
    plt.grid(True)
    plt.title("NUM1")
    plt.xlabel('h')
    plt.ylabel("Błąd")
    plt.loglog(h_values, errors_forward, label='Metoda różnic w przód ' + label)
    plt.loglog(h_values, errors_central, label='Metoda różnic centralnych ' + label)
    plt.legend()
    plt.show()

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'float':
            # Precyzja float 10^-7
            system = np.float32
            h_values = np.logspace(-7, 0, 270, dtype=np.float32)
            point = np.float32(0.2)
            errors_forward = calculate_errors32(system, point, h_values, forward_derivative32)
            errors_central = calculate_errors32(system, point, h_values, central_derivative32)
            graph(h_values, errors_forward, errors_central, '(float)')

        elif arg == 'double':
            # Precyzja double 10^-16
            system = np.float64
            h_values = np.logspace(-16, 0, 270, dtype=np.float64)
            point = np.float64(0.2)
            errors_forward = calculate_errors(system, point, h_values, forward_derivative)
            errors_central = calculate_errors(system, point, h_values, central_derivative)
            graph(h_values, errors_forward, errors_central, '(double)')

        else:
            print("invalid")
    else:
        print("invalid")

if __name__ == "__main__":
    main()
