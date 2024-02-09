import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def get_avg(mass, n):
    data = mass[len(mass)-n::, 0]
    avg = np.mean(data)
    return avg

def get_cmap(n, name='inferno'):
    cmap = plt.colormaps[name]
    colors = cmap(np.linspace(0, 1, n)) # Создание n равномерно распределённых цветов
    return mcolors.ListedColormap(colors)


def create_grid(x_in, y_in, n_in):
    grid = []
    X = np.linspace(x_in[0], x_in[1], n_in[0])
    X *= np.pi
    Y = np.linspace(y_in[0], y_in[1], n_in[1])

    for x in X:
        for y in Y:
            grid.append([x, y])
    return grid


def F(t, theta):
    out = np.zeros((2, ), dtype=float)
    out[0] = theta[1]
    out[1] = np.cos(t) - np.sin(theta[0]) - 0.1 * theta[1]
    return out


def runge_kutta(x_0, t_n, f, h):
    m = int(t_n / h)
    w = np.zeros((m + 2, 2), dtype=float)

    w[0][0] = x_0[0]
    w[0][1] = x_0[1]

    for i in range(m+1):
        t_i = i * h
        k_1 = h * f(t_i, w[i])
        k_2 = h * f(t_i + h / 2, w[i] + k_1 / 2)
        k_3 = h * f(t_i + h / 2, w[i] + k_2 / 2)
        k_4 = h * f(t_i + h, w[i] + k_3)
        w[i+1] = w[i] + (k_1 + 2*k_2 + 2*k_3 + k_4)/6
    return w


def main():
    t_n = 300
    h = 0.1
    nodes_count = 50

    x = (-4, 4)
    y = (-5, 5)
    grid_count = (300, 200)
    grid_x_len = (x[1] - x[0]) * np.pi / grid_count[0]
    grid_y_len = (y[1] - y[0]) / grid_count[1]
    cmap = get_cmap(15)

    grid = create_grid(x, y, grid_count)
    attractors_num = []
    attractors_color = []

    fig, axes = plt.subplots(1, 1, figsize=(13, 13))
    axes.set_xlim([-5 * np.pi, 5 * np.pi])
    axes.set_ylim([-6, 6])

    x_labels = [f"{i}π" for i in range(-4, 5)]
    x_tick_positions = [-4 * np.pi, -3 * np.pi, -2 * np.pi, -1 * np.pi, 0, 1 * np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi]
    axes.set_xticks(x_tick_positions)
    axes.set_xticklabels(x_labels)
    y_tick_positions = list(range(-5, 6))  # От -5 до 5 с шагом 1
    axes.set_yticks(y_tick_positions)
    axes.set_xlabel(r'${\theta}$')
    axes.set_ylabel(r'${\frac{d\theta}{dt}}$')

    i = 0
    for g in grid:
        data = runge_kutta(g, t_n, F, h)

        avg = get_avg(data, nodes_count)/np.pi

        nearest_pi = int(np.trunc(avg))

        if nearest_pi not in attractors_num:
            attractors_num.append(nearest_pi)
            attractors_color.append(cmap(i))
            i += 1
        index = attractors_num.index(nearest_pi)
        color = attractors_color[index]

        rect = plt.Rectangle((g[0], g[1]), grid_x_len, grid_y_len, facecolor=color)
        axes.add_patch(rect)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
     main()

