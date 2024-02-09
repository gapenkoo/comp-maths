import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scp
import random as rng
import matplotlib.colors as mcolors
from timeit import default_timer as timer


def get_cmap(n, name='jet'):
    cmap = plt.colormaps[name]
    colors = cmap(np.linspace(0, 1, n)) # Создание n равномерно распределённых цветов
    return mcolors.ListedColormap(colors)


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
        w[i+1] = w[i] + (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
    return w


def adams_moulton(x_0, t_n, f, h):
    m = int(t_n / h)
    w = np.zeros((m + 2, 2), dtype=float)

    w[0][0] = x_0[0]
    w[0][1] = x_0[1]

    w1 = runge_kutta(x_0, 2*h, f, h)
    w[1] = w1[1]
    w[2] = w1[2]

    for i in range(3, m + 2):
        t_i3 = i * h
        t_i2 = (i - 1) * h
        t_i1 = (i - 2) * h
        t_i0 = (i - 3) * h

        Y = lambda y3: -y3 + w[i-1] + (h/24) * (9 * f(t_i3, y3) + 19 * f(t_i2, w[i-1]) - 5 * f(t_i1, w[i-2]) + f(t_i0, w[i-3]))
        sol = scp.root(Y, w[i-1], jac=False, method='hybr')
        w[i] = sol.x
    return w


def milne_simpson(x_0, t_n, f, h):
    m = int(t_n / h)
    w = np.zeros((m + 2, 2), dtype=float)

    w[0][0] = x_0[0]
    w[0][1] = x_0[1]

    w1 = runge_kutta(x_0, 3*h, f, h)
    w[1] = w1[1]
    w[2] = w1[2]
    w[3] = w1[3]

    for i in range(3, m + 1):
        t_i_2 = (i - 2) * h
        t_i_1 = (i - 1) * h
        t_i = i * h
        t_i1 = (i + 1) * h

        w_predictor = w[i-3] + (4*h/3) * (2*f(t_i, w[i]) - f(t_i_1, w[i-1]) + 2*f(t_i_2, w[i-2]))
        w_corrector = w[i-1] + (h/3) * (f(t_i1, w_predictor) + 4*f(t_i, w[i]) + f(t_i_1, w[i-1]))
        w[i+1] = w_corrector
    return w

def main():
    start_amount = 15
    t_n = 100
    h = 0.1
    t = np.linspace(0, t_n, int(t_n/h) + 2)
    cmap = get_cmap(start_amount)

    x_0 = []
    runge_kutta_data = [] # Массив для хранения результатов, полученных методом Рунге-Кутте
    adams_moulton_data = [] # Массив для хранения результатов, полученных методом Адамса-Моултона
    milne_simpson_data = [] # Массив для хранения результатов, полученных методом Милне-Симпсона

    # Генерация начальных условий
    for i in range(start_amount):
        x_0.append([0, rng.uniform(1.85, 2.1)])

    # Заполнение данных для каждого метода
    start = timer()
    for i in range(start_amount):
        runge_kutta_data.append(runge_kutta(x_0[i], t_n, F, h))
    end = timer()
    print("Time for Method Runge-Kutta:", end - start)
    start = timer()
    for i in range(start_amount):
        adams_moulton_data.append(adams_moulton(x_0[i], t_n, F, h))
    end = timer()
    print("Time for Method Adams-Moulton:", end - start)
    start = timer()
    for i in range(start_amount):
        milne_simpson_data.append(milne_simpson(x_0[i], t_n, F, h))
    end = timer()
    print("Time for Method Milne-Simpson:", end - start)
    # Первое окно отображения: графики (theta - t)
        # Рисунок для метода Рунге-Кутте
    fig_rk, ax_rk = plt.subplots(figsize=(13, 5))
    ax_rk.set_xlabel("t", fontsize=14)
    ax_rk.set_ylabel(r'${\theta}$', fontsize=14)
    for i in range(start_amount):
        ax_rk.plot(t, runge_kutta_data[i][:, 0], '-', linewidth=1, color=cmap(i), label=f'{x_0[i][1]:.3f}')
    ax_rk.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
    # Рисунок для метода Адамса-Моултона
    fig_am, ax_am = plt.subplots(figsize=(13, 5))
    ax_am.set_xlabel("t", fontsize=14)
    ax_am.set_ylabel(r'${\theta}$', fontsize=14)
    for i in range(start_amount):
        ax_am.plot(t, adams_moulton_data[i][:, 0], '-', linewidth=1, color=cmap(i), label=f'{x_0[i][1]:.3f}')
    ax_am.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
    # Рисунок для метода Милне-Симпсона
    fig_ms, ax_ms = plt.subplots(figsize=(13, 5))
    ax_ms.set_xlabel("t", fontsize=14)
    ax_ms.set_ylabel(r'${\theta}$', fontsize=14)
    for i in range(start_amount):
        ax_ms.plot(t, milne_simpson_data[i][:, 0], '-', linewidth=1, color=cmap(i), label=f'{x_0[i][1]:.3f}')
    ax_ms.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')

    # Рисунок фазовых траекторий для метода Рунге-Кутте
    fig_rk_phase, ax_rk_phase = plt.subplots(figsize=(13, 5))
    ax_rk_phase.set_xlabel(r'${\theta}$', fontsize=14)
    ax_rk_phase.set_ylabel(r'${\frac{d\theta}{dt}}$', fontsize=14)
    for i in range(start_amount):
        ax_rk_phase.plot(runge_kutta_data[i][:, 0], runge_kutta_data[i][:, 1], '-', linewidth=1, color=cmap(i), label=f'{x_0[i][1]:.3f}')
    ax_rk_phase.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')

    # Рисунок фазовых траекторий для метода Адамса-Моултона
    fig_am_phase, ax_am_phase = plt.subplots(figsize=(13, 5))
    ax_am_phase.set_xlabel(r'${\theta}$', fontsize=14)
    ax_am_phase.set_ylabel(r'${\frac{d\theta}{dt}}$', fontsize=14)
    for i in range(start_amount):
        ax_am_phase.plot(adams_moulton_data[i][:, 0], adams_moulton_data[i][:, 1], '-', linewidth=1, color=cmap(i), label=f'{x_0[i][1]:.3f}')
    ax_am_phase.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')

    # Рисунок фазовых траекторий для метода Милне-Симпсона
    fig_ms_phase, ax_ms_phase = plt.subplots(figsize=(13, 5))
    ax_ms_phase.set_xlabel(r'${\theta}$', fontsize=14)
    ax_ms_phase.set_ylabel(r'${\frac{d\theta}{dt}}$', fontsize=14)
    for i in range(start_amount):
        ax_ms_phase.plot(milne_simpson_data[i][:, 0], milne_simpson_data[i][:, 1], '-', linewidth=1, color=cmap(i), label=f'{x_0[i][1]:.3f}')
    ax_ms_phase.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')

    # Зафиксированное начальное условие
    fixed_initial_condition_second = [0, 2.02]

    # Получение данных для каждого метода с зафиксированным начальным условием
    rk_fixed = runge_kutta(fixed_initial_condition_second, t_n, F, h)
    am_fixed = adams_moulton(fixed_initial_condition_second, t_n, F, h)
    ms_fixed = milne_simpson(fixed_initial_condition_second, t_n, F, h)

    # Создание третьего окна для сравнения фазовых траекторий
    fig_fixed, ax_fixed = plt.subplots(figsize=(9, 5))
    ax_fixed.set_xlabel(r'${\theta}$', fontsize=14)
    ax_fixed.set_ylabel(r'${\frac{d\theta}{dt}}$', fontsize=14)

    # Построение графиков фазовых траекторий для каждого метода
    ax_fixed.plot(rk_fixed[:, 0], rk_fixed[:, 1], '-', color='blue', label='Runge-Kutta')
    ax_fixed.plot(am_fixed[:, 0], am_fixed[:, 1], '-', color='red', label='Adams-Moulton')
    ax_fixed.plot(ms_fixed[:, 0], ms_fixed[:, 1], '-', color='green', label='Milne-Simpson')

    # Добавление легенды
    ax_fixed.legend()

    t_n1 = 200
    h_rk_values = [0.001, 1.0, 1.05]  # Значения шага h
    fixed_initial_condition_first = [0, 1.85]  # Фиксированное начальное условие

    for i in h_rk_values:
        t_1 = np.linspace(0, t_n1, int(t_n1/i) + 2)
        rk_data = runge_kutta(fixed_initial_condition_first, t_n1, F, i)  # Получение данных методом Рунге-Кутты

        # Построение графика
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.set_xlabel('t', fontsize=14)
        ax.set_ylabel(r'${\theta}$', fontsize=14)
        ax.plot(t_1, rk_data[:, 0], '-', color='blue')

    h_am_values = [0.001, 1.1, 1.15]
    for i in h_am_values:
        t_1 = np.linspace(0, t_n1, int(t_n1/i) + 2)
        am_data = adams_moulton(fixed_initial_condition_first, t_n1, F, i)  # Получение данных методом Адамса-Моултона

        # Построение графика
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.set_xlabel('t', fontsize=14)
        ax.set_ylabel(r'${\theta}$', fontsize=14)
        ax.plot(t_1, am_data[:, 0], '-', color='red')

    h_ms_values = [0.001, 0.15, 0.2]
    for i in h_ms_values:
        t_1 = np.linspace(0, t_n1, int(t_n1/i) + 2)
        ms_data = milne_simpson(fixed_initial_condition_first, t_n1, F, i)  # Получение данных методом Милна-Симпсона

        # Построение графика
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.set_xlabel('t', fontsize=14)
        ax.set_ylabel(r'${\theta}$', fontsize=14)
        ax.plot(t_1, ms_data[:, 0], '-', color='green')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

