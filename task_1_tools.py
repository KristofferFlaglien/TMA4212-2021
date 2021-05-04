"""
Tools for task 1.
"""

# Importing libraries and plotting tools:
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
from Tools.plotting_tools import *


# Setting the epsilon-parameter for task 1d:
epsilon = 0.01


# 1D Poisson equation tools (forward difference, central difference and an AMR scheme):

class PoissonEq:
    def __init__(self, init_m):

        """
        :param init_m: we have m + 2 as the number of points in the 1D grid.
        """

        # Starts in 1st iteration.
        self.M = [init_m]
        self.grid = [np.linspace(0, 1, self.M[0] + 2)]        # Creating a grid with points x0, x1, ... , xM+1
        self.spacing = [np.array([self.grid[0][i + 1] - self.grid[0][i] for i in range(self.M[0] + 1)])]

        # Starts in 0th iteration (needs calculations).
        self.solution = []      # U(x)-vector in a given iteration.
        self.error = []         # Error distribution at each point in the grid.
        self.rel_error = []     # Average relative error between numerical and analytical solutions.

    def dirichlet_neumann_umr_central_solver(self, f, left_bc, right_bc, iteration):
        h = self.spacing[iteration][0]  # Discretization step for the uniform grid.
        x = self.grid[iteration][1:]    # Array of points x1, x2, ... , xM+1.
        M = self.M[iteration]           # Number of non-boundary points.

        diagonals = [1, -2, 1]
        positions = [-1, 0, 1]
        diagonals_array = np.array(diagonals) / (h ** 2)

        # Creating the matrix A_h.
        matrix = scipy.sparse.diags(diagonals_array, positions, shape=(M + 1, M + 1)).toarray()
        matrix[-1][-3], matrix[-1][-2], matrix[-1][-1] = -1 / (2 * h), 2 / h, -3 / (2 * h)
        matrix = scipy.sparse.csr_matrix(matrix)

        # Creating the right hand side f-vector.
        vector = f(x)
        vector[0] -= left_bc / h ** 2
        vector[-1] = right_bc

        # Solving the Ax=b problem with A=A_h and b=f-vector.
        solution = np.zeros(M + 2)
        solution[0] = left_bc
        solution[1:] = scipy.sparse.linalg.spsolve(A=matrix, b=vector.transpose())

        # Saving the solution data.
        self.solution.append(solution)

    def dirichlet_dirichlet_umr_central_solver(self, f, left_bc, right_bc, iteration):
        h = self.spacing[iteration][0]  # Discretization step for the uniform grid.
        x = self.grid[iteration][1:-1]  # Array of points x1, x2, ... , xM.
        M = self.M[iteration]  # Number of non-boundary points.

        diagonals = [1, -2, 1]
        positions = [-1, 0, 1]
        diagonals_array = np.array(diagonals) / (h ** 2)

        # Creating the matrix A_h.
        matrix = scipy.sparse.diags(diagonals_array, positions, shape=(M, M)).tocsr()

        # Creating the right hand side f-vector.
        vector = f(x)
        vector[0] -= left_bc / h ** 2
        vector[-1] -= right_bc / h ** 2

        # Solving the Ax=b problem with A=A_h and b=f-vector.
        solution = np.zeros(M + 2)
        solution[0] = left_bc
        solution[-1] = right_bc
        solution[1:-1] = scipy.sparse.linalg.spsolve(A=matrix, b=vector.transpose())

        # Saving the solution data.
        self.solution.append(solution)

    def dirichlet_dirichlet_umr_forward_solver(self, f, left_bc, right_bc, iteration):
        h = self.spacing[iteration][0]  # Discretization step for the uniform grid.
        x = self.grid[iteration][1:-1]  # Array of points x1, x2, ... , xM.
        M = self.M[iteration]  # Number of non-boundary points.

        diagonals = [1, -2, 1]
        positions = [0, 1, 2]
        diagonals_array = np.array(diagonals) / (h ** 2)

        # Creating the matrix A_h.
        matrix = scipy.sparse.diags(diagonals_array, positions, shape=(M, M)).tocsr()

        # Creating the right hand side f-vector.
        vector = f(x)
        vector[-1] = 2 * right_bc / h ** 2

        # Solving the Ax=b problem with A=A_h and b=f-vector.
        solution = np.zeros(M + 2)
        solution[0] = left_bc
        solution[-1] = right_bc
        solution[1:-1] = scipy.sparse.linalg.spsolve(A=matrix, b=vector.transpose())

        # Saving the solution data.
        self.solution.append(solution)

    def dirichlet_dirichlet_amr_solver_asymm(self, f, left_bc, right_bc, iteration):
        x = self.grid[iteration][1:-1]  # Array of points x1, x2, ... , xM.
        h = self.spacing[iteration]     # Array of discretization steps.
        M = self.M[iteration]           # Number of non-boundary points.

        a = np.zeros(M)
        b = np.zeros(M)
        c = np.zeros(M + 1)

        for i in range(1, M + 1):
            if i == 1:
                a[i-1] = 0
                b[i-1] = 1 / (h[i-1]) ** 2
                c[i] = 1 / (h[i-1]) ** 2
            else:
                a[i-1] += 2 * (h[i] - h[i-1]) / ((h[i-2] + h[i-1]) * (h[i-2] + h[i-1] + h[i]) * h[i-2])
                b[i-1] += 2 * (h[i-2] + h[i-1] - h[i]) / (h[i-1] * h[i-2] * (h[i-1] + h[i]))
                c[i] += 2 * (h[i-2] + 2 * h[i-1]) / (h[i] * (h[i-1] + h[i]) * (h[i-2] + h[i-1] + h[i]))

        d1 = a[2:]
        d2 = b[1:]
        d3 = - (a + b + c[1:])
        d4 = c[:-1]

        data = np.array([d1, d2, d3, d4])
        diags = np.array([-2, -1, 0, 1])

        # Creating the matrix A_h.
        matrix = np.zeros((M, M))
        for i in range(4):
            matrix += scipy.sparse.spdiags(data[i], diags[i], M, M).toarray()
        matrix = scipy.sparse.csr_matrix(matrix)

        # Creating the right hand side f-vector.
        vector = f(x)
        vector[0] -= left_bc * b[0]
        vector[1] -= left_bc * a[1]
        vector[-1] -= right_bc * c[-1]

        # Solving the Ax=b problem with A=A_h and b=f-vector.
        solution = np.zeros(M + 2)
        solution[0] = left_bc
        solution[-1] = right_bc
        solution[1:-1] = scipy.sparse.linalg.spsolve(A=matrix, b=vector.transpose())

        # Saving the solution data.
        self.solution.append(solution)

    def dirichlet_dirichlet_amr_solver_symm(self, f, left_bc, right_bc, iteration):
        x = self.grid[iteration][1:-1]  # Array of points x1, x2, ... , xM.
        h = self.spacing[iteration]     # Array of discretization steps.
        M = self.M[iteration]           # Number of non-boundary points.

        a = np.zeros(M)
        b = np.zeros(M)
        c = np.zeros(M + 1)

        for i in range(1, M + 1):
            a[i-1] += 2 / ((h[i] + h[i-1]) * h[i-1])
            b[i-1] -= 2 / (h[i] * h[i-1])
            c[i] += 2 / ((h[i] + h[i-1]) * h[i])

        d1 = a[1:]
        d2 = b[:]
        d3 = c[:-1]

        data = np.array([d1, d2, d3])
        diags = np.array([-1, 0, 1])

        # Creating the matrix A_h.
        matrix = np.zeros((M, M))
        for i in range(3):
            matrix += scipy.sparse.spdiags(data[i], diags[i], M, M).toarray()
        matrix = scipy.sparse.csr_matrix(matrix)

        # Creating the right hand side f-vector.
        vector = f(x)
        vector[0] -= left_bc * a[0]
        vector[-1] -= right_bc * c[-1]

        # Solving the Ax=b problem with A=A_h and b=f-vector.
        solution = np.zeros(M + 2)
        solution[0] = left_bc
        solution[-1] = right_bc
        solution[1:-1] = scipy.sparse.linalg.spsolve(A=matrix, b=vector.transpose())

        # Saving the solution data.
        self.solution.append(solution)

    def get_error(self, u, iteration):
        # Defining the discrete l2 norm.
        def disc_norm(v):
            return np.sqrt(np.sum(v ** 2) / len(v))

        # Defining the continuous L2 norm.
        def cont_norm(x, v):
            return np.sqrt(np.trapz(v**2, x))

        x_disc = self.grid[iteration]        # Array of points x0, x1, ... , xM+1.
        U_disc = self.solution[iteration]    # Numerical solution, array of points U0, U1, ... , UM+1.
        u_disc = u(x_disc)                   # Analytical solution, array of points u0, u1, ... , uM+1

        # Interpolating over the numerical discrete solution to find a continuous function (cubic method).
        x_cont = np.linspace(0, 1, 1000000)
        U_cont = scipy.interpolate.interp1d(x_disc, U_disc, kind='cubic')(x_cont)
        u_cont = u(x_cont)

        # Finding the relative errors using the l2 and L2 norms respectively.
        error_disc = disc_norm(u_disc - U_disc) / disc_norm(u_disc)
        error_cont = cont_norm(x_cont, u_cont - U_cont) / cont_norm(x_cont, u_cont)

        # Saving the error data in the relative error list.
        self.rel_error.append((error_disc, error_cont))

        # Saving the error vector at each point in the grid.
        self.error.append(abs(u_disc - U_disc))

    def check_umr_convergence(self, f, u, iterations, solver, left_bc, right_bc):

        # Finding the initial l2 and L2 errors of the initial system.
        solver(f, left_bc, right_bc, 0)
        self.get_error(u, 0)

        # Checks the l2 error until it is below the given upper bound.
        for iteration in range(1, iterations):

            # Doubling the gridsize.
            self.M.append(2 * self.M[iteration - 1] + 1)
            M = self.M[iteration]
            self.spacing.append(1 / (M + 1) * np.ones(M + 1))
            self.grid.append(np.linspace(0, 1, M + 2))

            # Solving for the new system.
            solver(f, left_bc, right_bc, iteration)
            self.get_error(u, iteration)

    def check_amr_convergence(self, f, u, iterations, solver, left_bc, right_bc, percentage):

        # Finding the initial l2 and L2 errors of the initial system.
        solver(f, left_bc, right_bc, 0)
        self.get_error(u, 0)

        # Checks the l2 error 'iterations' times.
        for iteration in range(iterations):

            # Finding the largest deviation from the analytical expression.
            max_error = max(self.error[iteration])

            # Adding points where it is needed.
            old_grid = self.grid[iteration]
            new_grid = []
            for i in range(self.M[iteration] + 2):
                if self.error[iteration][i] > percentage * max_error:
                    # Adding two new points to the grid.
                    new_grid.append((2 * old_grid[i] + old_grid[i - 1]) / 3)
                    new_grid.append(old_grid[i])
                    new_grid.append((2 * old_grid[i] + old_grid[i + 1]) / 3)
                else:
                    new_grid.append(old_grid[i])

            self.grid.append(np.array(new_grid))
            self.M.append(len(new_grid) - 2)
            self.spacing.append(np.array([new_grid[i + 1] - new_grid[i] for i in range(len(new_grid) - 1)]))

            # Solving for the new system.
            solver(f, left_bc, right_bc, iteration + 1)
            self.get_error(u, iteration + 1)

    def plot_convergence(self, line_order=2):
        disc_error = [x[0] for x in self.rel_error]
        cont_error = [x[1] for x in self.rel_error]
        gridsize = [x + 2 for x in self.M]
        x_array = np.linspace(gridsize[0], gridsize[-1], 1000)

        def line(x, order, init_error):
            return init_error * x**(-order) / (x[0])**(-order)

        print('M+2 values:', gridsize)
        print('disc error:', disc_error)
        print('cont error:', cont_error)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        set_ax(ax,
               title='Convergence testing for 1D Poisson equation',
               xscale='log', yscale='log',
               x_label='Gridsize (M+2)', y_label='Relative error')

        ax.plot(gridsize, disc_error, marker='o', label=r'$l_2$ norm error', lw=3, c='#1F77B4')
        ax.plot(x_array, line(x_array, line_order, disc_error[0]), label=r'$O(h^{%s})$' % line_order,
                lw=3, ls='--', c='#1F77B4')

        ax.plot(gridsize, cont_error, marker='o', label=r'$L_2$ norm error', lw=3, c='#FF7F0E')
        ax.plot(x_array, line(x_array, line_order, cont_error[0]), label=r'$O(h^{%s})$' % line_order,
                lw=3, ls='--', c='#FF7F0E')

        ax.legend(fontsize=15)
        plt.show()
        plt.close()

    def plot_solution_comparison(self, u, step_jumps=1, suptitle='Solution to the 1D Poisson equation'):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 8))
        fig.suptitle('\n'+suptitle, fontsize=20)

        x_array = np.linspace(0, 1, 1000)
        step = 0

        for ax in axes.flat:
            step += step_jumps
            if step == step_jumps:
                ax.set_title('Step 1: M = ' + str(self.M[0]), fontsize=15)
                x = self.grid[0]
                U = self.solution[0]
                ax.plot(x, U, label='Numerical solution', lw=3)
                ax.plot(x_array, u(x_array), label='Analytical solution', ls='--', lw=3)
            else:
                ax.set_title('Step ' + str(step) + ': M = ' + str(self.M[step - 1]), fontsize=15)
                x = self.grid[step - 1]
                U = self.solution[step - 1]
                ax.plot(x, U, lw=3)
                ax.plot(x_array, u(x_array), ls='--', lw=3)

        plt.subplots_adjust(0.1, 0.1, 0.90, 0.80, 0.35, 0.35)
        fig.legend(fontsize=15, bbox_to_anchor=(0, -0.05, 1, 1))
        plt.show()
        plt.close()


# Analytical solutions u(x) for tasks 1a, 1b and 1d:

def u_a(x):
    return (1 - np.cos(2 * np.pi * x)) / (4 * np.pi ** 2) + (x ** 2 - 3) * x / 6


def u_b(x):
    return (1 - np.cos(2 * np.pi * x)) / (4 * np.pi ** 2) + (x ** 2 - 1) * x / 6 + 1


def u_d(x):
    return np.exp(-(x-0.5)**2/epsilon)


# RHS functions f(x) for tasks 1a/1b and 1d:

def f_ab(x):
    return np.cos(2 * np.pi * x) + x


def f_d(x):
    return (4*(x**2-x+1/4)/epsilon**2-2/epsilon)*np.exp(-(x-0.5)**2/epsilon)
