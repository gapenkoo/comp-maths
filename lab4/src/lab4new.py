import numpy as np
import matplotlib.pyplot as plt

epsilon = 5e-1

def silvester(A):
    n = len(A)
    for k in range(1, n + 1):
        if np.linalg.det(A[:k, :k]) < epsilon:
            return False
    return True

def gauss(A, b, pivoting=True):
    n = len(A)
    if np.abs(np.linalg.det(A)) < epsilon: # Проверка на сингулярность
        raise ValueError("No solution")
    A_new = np.copy(A)
    b_new = np.copy(b)
    for k in range(n - 1):
        if pivoting:
            # Реализация метода Гаусса с частичным выбором главного элемента
            index_max_row = abs(A_new[k:, k]).argmax() + k  # Индекс строки максимального элемента
            # Перестановка для метода Гаусса с частичным выбором главного элемента
            if index_max_row != k: # Перестановка для строк
                A_new[[k, index_max_row]] = A_new[[index_max_row, k]]
                b_new[[k, index_max_row]] = b_new[[index_max_row, k]]
        else:
            if A_new[k, k] == 0: # Проверка на равенство нулю
                raise ValueError("Element on general diagonal is zero")
        for row in range(k + 1, n):
            coeff = A_new[row, k] / A_new[k, k]
            A_new[row, k:] = A_new[row, k:] - coeff * A_new[k, k:]
            b_new[row] = b_new[row] - coeff * b_new[k]
    #Обратный ход
    x = np.zeros((n, 1))
    for k in range(n - 1, -1, -1):
        x[k] = (b_new[k] - np.dot(A_new[k, k + 1:], x[k + 1:])) / A_new[k, k]
    return x

def thomas(A, b):
    n = len(A)
    gamma = np.zeros((n, 1)) # Коэффициенты гамма
    beta = np.zeros((n, 1)) # Коэффициенты бета

    for k in range(n - 1):
        if k != 0:
            A_k_previous = A[k, k - 1]
        else:
            A_k_previous = 0
        # Рассчет коэффициентов по средству рекуррентных соотношений
        # Прямой проход
        gamma[k + 1] = - A[k, k + 1] / (A_k_previous * gamma[k] + A[k, k])
        beta[k + 1] = (b[k] - A_k_previous * beta[k]) / (A_k_previous * gamma[k] + A[k, k])

    x = np.zeros((n, 1))
    # Начальное условие для обратного прохода
    x[n - 1] = (b[n - 1] - A[n - 1, n - 2] * beta[n - 1]) / (A[n - 1, n - 1] + A[n - 1, n - 2] * gamma[n - 1])
    # Обратный проход
    for i in range(n - 1, 0, -1):
        x[i - 1] = gamma[i] * x[i] + beta[i]

    return x

def cholesky(A, b):
    n = len(A)

    L = np.zeros((n, n))# Нижнетреугольная матрица в разложении Холецкого

    for i in range(n): # По строкам
        for j in range(i + 1): # По столбцам
            l_sum = sum([L[i, p] * L[j, p] for p in range(j)])  # Сумма произведений элементов предыдущих строк и столбцов матрицы L
            if i == j:  # Лежит ли элемент на главной диагонале
                radical_expression = A[i, i] - l_sum
                if radical_expression < 0:
                    raise ValueError("Negative under radical")
                elif radical_expression == 0:
                    raise ValueError("Matrix L is singular")
                L[i, j] = np.sqrt(radical_expression)
            else: # Вычисление недиагональных элементов
                L[i, j] = ((A[i, j] - l_sum) / L[j, j])

    y = np.zeros((n, 1))
    for k in range(n):
        y[k] = (b[k] - np.dot(L[k, :k], y[:k])) / L[k, k]
    x = np.zeros((n, 1))
    for k in range(n - 1, -1, -1):
        x[k] = (y[k] - np.dot(L.T[k, k + 1:], x[k + 1:])) / L.T[k, k]

    return x

# def iter_refinement(A, b):
#     n = len(A)
#     tol=1e-5
#     max_iter=10
#     # Изменение формы b для совместимости с функцией gauss
#     b = np.reshape(b, (n, 1))
#
#     x = gauss(A, b, pivoting=True)  # Начальное приближение
#
#     for _ in range(max_iter):
#         r = b - np.dot(A, x)  # Вычисление невязки
#         if np.linalg.norm(r, np.inf) < tol:  # Проверка точности
#             break
#         delta = gauss(A, r, pivoting=True)  # Решение системы для невязки
#         x += delta  # Корректировка приближения
#
#     return x.flatten()  # Преобразование результата обратно в одномерный массив


def generate_random_matrix(n, matrix_type="default"):
    #Генерация матриц общего вида
    gen_matrix = np.random.rand(n, n).astype(np.float32) * 2 - 1  # Генерация матрицы в интервале [-1, 1]
    while np.abs(np.linalg.det(gen_matrix)) < epsilon:
        gen_matrix = np.random.rand(n, n).astype(np.float32) * 2 - 1  # Генерация матрицы в интервале [-1, 1]

    #Генерация трехдиагональных матриц
    if matrix_type == "TD":
        diagonal = np.random.uniform(-1, 1, n).astype(np.float32)
        off_diagonal = np.random.uniform(-1, 1, n - 1).astype(np.float32)
        gen_matrix = np.diag(diagonal) + \
                     np.diag(off_diagonal, k=1) + \
                     np.diag(off_diagonal, k=-1)
        while np.abs(np.linalg.det(gen_matrix)) < epsilon:
            gen_matrix += np.eye(n) * 0.1
            max_elem = np.max(np.abs(gen_matrix))
            if max_elem >= 1:
                gen_matrix /= (max_elem + 0.001)
    #Генерация положительно-определенных матриц
    elif matrix_type == "PD":
        symmetrical_matrix = np.dot(gen_matrix, gen_matrix.T)

        max_elem = np.max(np.abs(symmetrical_matrix))
        if max_elem >= 1:
            symmetrical_matrix /= (max_elem + 0.001)

        while np.abs(np.linalg.det(symmetrical_matrix)) < epsilon:
            symmetrical_matrix += np.eye(n) * 0.1
            max_elem = np.max(np.abs(symmetrical_matrix))
            if max_elem >= 1:
                symmetrical_matrix /= (max_elem + 0.001)

        gen_matrix = symmetrical_matrix

    return gen_matrix

def count_rms_error(accurate, approximate):
    #Вычисление относительной погрешности с помощью среднеквадратичной нормы
    n = len(accurate)
    absolute_error = np.sum([(accurate[i] - approximate[i]) ** 2 for i in range(n)])
    normal = np.sum([(accurate[i] ** 2) for i in range(n)])
    return np.sqrt(absolute_error) / np.sqrt(normal)

def count_supremum_error(accurate, approximate):
    #Вычисление относительной погрешности с помощью супремум-нормы
    return np.max(np.abs(accurate - approximate)) / np.max(np.abs(accurate))

def check_matrices_data_type(matrices, data_type=np.float32):
    return all(matrix.dtype == data_type for matrix in matrices)

def visualize(data, label, x_label):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    axes.hist(data, bins=50, rwidth=0.8, log=True, label=label, color='red')
    axes.legend(loc='best', fontsize = 14)
    axes.set_xlabel(x_label, fontsize = 14)
    axes.set_ylabel("Quantity of matrices", fontsize = 14)
    plt.show()

def main ():
    A = np.array([[1., 2., 0.],
                  [2., 6., 5.],
                  [0., 5., 13.]])
    b = np.array([3.52971, 0.333, 1.6666])
    print("Gauss without pivoting:\n", gauss(A, b, False))
    print("Gauss with pivoting:\n", gauss(A, b))
    print("Thomas:\n", thomas(A, b))
    print("Cholesky:\n", cholesky(A, b))
    # print("Iter_Refinement:\n", iter_refinement(A, b))
    print("Answer:\n", np.linalg.solve(A, b))
    print("\n")
    n = 4
    matrix_type = "PD"  # Например, для положительно определенной матрицы
    random_matrix = generate_random_matrix(n, matrix_type)
    print(random_matrix)
    print(epsilon)

    matrix_types = ["default", "TD", "PD"]

    methods = {
        "default": lambda A, b: gauss(A, b, pivoting=False),
        "TD": thomas,
        "PD": cholesky
        }

    b_solution = np.array([1., 1., 1., 1., 1., 1.]).T

    for matrix_type in matrix_types:
        matrices = []
        for i in range(1000):
            gen_matrix = generate_random_matrix(n=6, matrix_type=matrix_type)
            if matrix_type == "PD" :
                while silvester(gen_matrix) == False:
                    gen_matrix = generate_random_matrix(n=6, matrix_type="PD")
            matrices.append(gen_matrix)
        solutions_universal = [gauss(matrix, b_solution, pivoting=True) for matrix in matrices]
        solutions_special = [methods[matrix_type](matrix, b_solution) for matrix in matrices]
        rms_error = [count_rms_error(accurate=solutions_universal[i], approximate=solutions_special[i]) for i in range(len(solutions_universal))]
        supremum_error = [count_supremum_error(accurate=solutions_universal[i], approximate=solutions_special[i]) for i in range(len(solutions_universal))]
        print(sum(rms_error)/1000)
        print(sum(supremum_error)/1000)
        visualize(rms_error, label="Relative error (RMS norm for {} matrices)".format(matrix_type), x_label="Error")
        visualize(supremum_error, label="Relative error (supremum norm for {} matrices)".format(matrix_type), x_label="Error")
        spectral_radius = [np.max(np.abs(np.linalg.eigvals(matrix))) for matrix in matrices]
        cond_numbers = [np.linalg.cond(matrix) for matrix in matrices]
        visualize(spectral_radius, label="Spectral radius for {} matrices".format(matrix_type), x_label="Spectral radius")
        visualize(cond_numbers, label="Condition numbers for {} matrices".format(matrix_type), x_label="Condition number")
        eigenvalue_ratio = [np.max(np.abs(np.linalg.eigvals(matrix))) / np.min(np.abs(np.linalg.eigvals(matrix))) for matrix in matrices]
        visualize(eigenvalue_ratio, label="Relative max eigenvalue to min for {} matrices".format(matrix_type), x_label="Relative")

if __name__ == "__main__":
    main()
