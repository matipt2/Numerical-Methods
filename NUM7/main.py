import numpy as np
import matplotlib.pyplot as plt

def function_factory(constant):
    return lambda x: 1 / (1 + constant * (x**2))

main_function = function_factory(50)
test1 = function_factory(27)
test2 = function_factory(14)

def point_a(n):
    return [-1 + 2*i/n for i in range(n+1)]

def point_b(n):
    return np.cos( [ ( (2 * i + 1) / (2 * (n + 1) ) ) * np.pi for i in range(n+1) ] )

def interpolation(function, point, arg, n):
    x = point(n)
    y = list(map(function, x))

    new_y = []
    for a in arg:
        value = sum(y[i] * np.prod([(a - x[k]) / (x[i] - x[k]) for k in range(n+1) if i != k]) for i in range(n+1))
        new_y.append(value)

    return new_y

def plot(function, point, arg, title, filename):
    plt.title(title)
    plt.plot(arg, function(arg), label='f(x)')
    for n in [5,7,8,12,15]:
        plt.plot(arg, interpolation(function, point, arg, n), label=f'W_{n}(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.savefig(filename)
    plt.show()

new_x = np.arange(-1.0, 1.01, 0.01)

plot(main_function, point_a, new_x, 'Wielomiany interpolacyjne dla funkcji\n' 'f(x)=1/(1+50x^2) i siatki x_i = -1+2(i/(n))', "main_a.svg")
plot(main_function, point_b, new_x, 'Wielomiany interpolacyjne dla funkcji\n' 'f(x)=1/(1+50x^2) i siatki x_i= cos((2i+1)/(2(n+1))*Pi)', "main_b.svg")
plot(test1, point_a, new_x, 'Wielomiany interpolacyjne dla funkcji\n' 'f(x)=1/(1+27x^2) i siatki x_i = -1+2(i/(n))', "maintest1_a.svg")
plot(test1, point_b, new_x, 'Wielomiany interpolacyjne dla funkcji\n' 'f(x)=1/(1+27x^2) i siatki x_i= cos((2i+1)/(2(n+1))*Pi)', "maintest1_b.svg")
plot(test2, point_a, new_x, 'Wielomiany interpolacyjne dla funkcji\n' 'f(x)=1/(1+14x^2) i siatki x_i = -1+2(i/(n))', "maintest2_a.svg")
plot(test2, point_b, new_x, 'Wielomiany interpolacyjne dla funkcji\n' 'f(x)=1/(1+14x^2) i siatki x_i= cos((2i+1)/(2(n+1))*Pi)', "maintest2_b.svg")
