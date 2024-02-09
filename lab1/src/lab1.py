import numpy as np
import matplotlib.pyplot as plt

def draw_points(x, y):
    graph1 = plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='green', marker='o', s=5, label='Original Points')
    plt.legend(fontsize = 15)
    plt.grid(True)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.show()

def calculate_spline_coeffs(sparse, factor):
    n = len(sparse)
    A = np.zeros((n, n))
    A[0, 0] = A[-1, -1] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = factor
        A[i, i] = 4 * factor
        A[i, i + 1] = factor
    B_matrix = np.zeros(n)
    for i in range(1, n - 1):
        B_matrix[i] = (3 / factor) * (sparse[i+1] - sparse[i]) \
                      - (3 / factor) * (sparse[i] - sparse[i-1])

    c = np.linalg.solve(A, B_matrix)
    a = sparse[:-1]
    d = [(c[i + 1] - c[i]) / (3 * factor) for i in range(n - 1)]
    b = [(1 / factor) * (sparse[i + 1] - sparse[i])
         - (factor / 3) * (c[i + 1] + 2*c[i]) for i in range(n - 1)]
    return a, b, c[:-1], d

def spline(i, aj, t, t_j):
    return aj[i][0] + aj[i][1] * (t - t_j) + aj[i][2] * (t - t_j) ** 2 + aj[i][3] * (t - t_j) ** 3

def euclidean_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def calculate_distances(sparseset, aj, bj, x, y, factor):
    distances = []
    for t in range(len(x)):
        i = int(t // factor)
        if i >= len(sparseset) - 1:
            i = len(sparseset) - 2
        t_j = sparseset[i]
        x_spline = spline(i, aj, t, t_j)
        y_spline = spline(i, bj, t, t_j)
        dist = euclidean_dist(x_spline, y_spline, x[t], y[t])
        distances.append(dist)
    return distances

def generate_spline_points(sparseset, aj, bj, h, t_N, factor):
    xval, yval = [], []
    for t in np.arange(0, t_N, h):
        i = int(t // factor)
        if i >= len(sparseset) - 1:
            i = len(sparseset) - 2
        t_j = sparseset[i]
        x_approx = spline(i, aj, t, t_j)
        y_approx = spline(i, bj, t, t_j)
        xval.append(x_approx)
        yval.append(y_approx)
    return xval, yval

def draw_spline_and_points(sparseX, sparseY, x, y, xval, yval):
    graph2 = plt.figure(figsize=(12, 8))
    plt.scatter(sparseX, sparseY, color='red', marker='o', s=35, label='Sparse Points')
    plt.scatter(x, y, color='green', marker='o', s=4, label='Original Points')
    plt.plot(xval, yval, color='black', label='Cubic Spline')
    plt.legend(fontsize = 15)
    plt.grid(True)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.show()

class AutoDiffNum:
    def __init__(self, a, b):
        self.a = a  # real path
        self.b = b  # dual path

    def __add__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self.a + other.a, self.b + other.b)
        else:
            return AutoDiffNum(self.a + other, self.b)

    def __sub__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self.a - other.a, self.b - other.b)
        else:
            return AutoDiffNum(self.a - other, self.b)

    def __mul__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self.a * other.a, self.a * other.b + self.b * other.a)
        else:
            return AutoDiffNum(self.a * other, self.b * other)

    def __pow__(self, power):
        if isinstance(power, AutoDiffNum):
            raise ValueError("Power should be a real number.")
        else:
            return AutoDiffNum(self.a ** power, self.b * power * self.a ** (power - 1))

    def __repr__(self):
        return f"{self.a} + {self.b}ε"

def spline_derivative(i, aj, t, t_j):
    # Инициализация дуального числа для точки t
    t_dual = AutoDiffNum(t, 1)
    # Вычисление производной кубического сплайна в точке t
    G_t = (t_dual - t_j) * aj[i][1] + (t_dual - t_j) ** 2 * aj[i][2] + (t_dual - t_j) ** 3 * aj[i][3] + aj[i][0]
    return G_t


def calculate_tangent_vectors(sparseset, aj, bj, t_N, factor):
    G_t_values_x = []
    G_t_values_y = []
    for t in np.arange(0, t_N, factor):
        i = int(t // factor)
        if i >= len(sparseset) - 1:
            i = len(sparseset) - 2
        t_j = AutoDiffNum(sparseset[i], 0)
        G_t_x = spline_derivative(i, aj, t, t_j)
        G_t_y = spline_derivative(i, bj, t, t_j)
        G_t_values_x.append(G_t_x.b)
        G_t_values_y.append(G_t_y.b)
    return G_t_values_x, G_t_values_y

def calculate_normals(G_t_values_x, G_t_values_y):
    R_t_values_x = []
    R_t_values_y = []
    for i in range(len(G_t_values_x)):
        R_t_values_x.append(-G_t_values_y[i])
        R_t_values_y.append(G_t_values_x[i])
    return R_t_values_x, R_t_values_y

#lab1_base function
def lab1_base(filename_in:str, factor:int, filename_out:str):
    # Code from lab1_base
    P = np.loadtxt(filename_in)
    x, y = P[:, 0], P[:, 1]
    draw_points(x, y)
    sparseset = np.arange(0, len(P), factor)
    sparseX, sparseY = x[sparseset], y[sparseset]
    a_x, b_x, c_x, d_x = calculate_spline_coeffs(sparseX, factor)
    a_y, b_y, c_y, d_y = calculate_spline_coeffs(sparseY, factor)
    a_j = np.column_stack((a_x, b_x, c_x, d_x))
    b_j = np.column_stack((a_y, b_y, c_y, d_y))
    t_N = len(P) - 1
    h = 0.1
    xval, yval = generate_spline_points(sparseset, a_j, b_j, h, t_N, factor)
    distances = calculate_distances(sparseset, a_j, b_j, x, y, factor)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    draw_spline_and_points(sparseX, sparseY, x, y, xval, yval)
    print(f"Mean Distance: {mean_distance}")
    print(f"Standard Deviation: {std_distance}")
    coefficients_matrix = np.column_stack((a_j, b_j))
    np.savetxt(filename_out, coefficients_matrix, fmt='%.8f')
    return sparseset, a_j, b_j, t_N, sparseX, sparseY, x, y, xval, yval

# Main execution
filename_in, factor, filename_out = 'contour.txt', 10, 'coeffs.txt'
sparseset, a_j, b_j, t_N, sparseX, sparseY, x, y, xval, yval = lab1_base(filename_in, factor, filename_out)
G_t_values_x, G_t_values_y = calculate_tangent_vectors(sparseset, a_j, b_j, t_N, factor)

graph3 = plt.figure(figsize=(12, 8))
plt.scatter(sparseX, sparseY, color='red', marker='o', s=50, label='Sparse Points', alpha=0.7)
plt.plot(xval, yval, color='black', label='Cubic Spline', alpha=0.9)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.grid(True)


for idx, x in enumerate(G_t_values_x):
    if idx % 10 == 0:
        plt.plot()
        plt.arrow(sparseX[idx], sparseY[idx], G_t_values_x[idx] * 20,
                  G_t_values_y[idx] * 20, width=0.0001, head_width=0.0005, color='blue')
# plt.arrow(-1.08, 0.24, 0, 0, width=0.0001, head_width=0.001, color='red', label='gt')

R_t_values_x, R_t_values_y = calculate_normals(G_t_values_x, G_t_values_y)

for idx, x in enumerate(R_t_values_x):
    if idx % 10 == 0:
        plt.plot()
        plt.arrow(sparseX[idx], sparseY[idx], R_t_values_x[idx] * 20,
                  R_t_values_y[idx] * 20, width=0.0001, head_width=0.0005, color='purple')
plt.legend(fontsize = 15)
plt.xlim(-1.03,-1.0)
plt.ylim(0.24,0.27)
plt.show()
