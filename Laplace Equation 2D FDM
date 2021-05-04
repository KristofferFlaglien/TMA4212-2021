"""
Tools for task 3.
"""

# Importing libraries and plotting tools:
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
from Tools.plotting_tools import *


# 2D Laplace equation tools (central difference, 5-stencil scheme):

class LaplaceEq:
    def __init__(self, init_m_x, init_m_y):
        """
        :param init_m_x: size of the grid in the x-direction, excluding boundary points.
        :param init_m_y: size of the grid in the y-direction, exluding boundary points.
        """

        # Grid definitions, starts in 1st iteration.
        self.M_x = [init_m_x]       # Number of non-boundary points in the x-grid.
        self.M_y = [init_m_y]       # Number of non-boundary points in the y-grid.
        self.x_grid = [np.linspace(0, 1, self.M_x[0] + 2)]  # Creating a grid including the boundary points.
        self.y_grid = [np.linspace(0, 1, self.M_y[0] + 2)]  # Creating a grid including the boundary points.

        # Solution and error values, starts in 0th iteration (needs calculations).
        self.solution = []      # U(x, y)-array in a given iteration.
        self.rel_error = []     # l2 error between numerical and analytical solutions.

    def solver(self, bc_0y, bc_x0, bc_1y, bc_x1, iteration):
        M_x = self.M_x[iteration]
        M_y = self.M_y[iteration]

        h = 1 / (M_x + 1)
        k = 1 / (M_y + 1)
        R = int(M_x * M_y)

        a = 1/h**2
        b = 1/k**2
        c = -2 * (1/h**2 + 1/k**2)

        diagonals = [b, a, c, a, b]
        positions = [-M_x, -1, 0, 1, M_x]

        A = scipy.sparse.diags(diagonals, positions,
                               shape=(R, R)).toarray()
        d_vec = np.zeros(R)
        d_vec[:M_x] -= np.array([b * bc_x0(h * (j + 1)) for j in range(M_x)])
        d_vec[-M_x:] -= np.array([b * bc_x1(h * (j + 1)) for j in range(M_x)])

        for i in range(R):
            if i % M_x == 0:
                d_vec[i] -= a * bc_0y(k * (1 + i // M_x))
                if i != 0:
                    A[i, i - 1] = 0
            elif i % M_x == M_x - 1:
                d_vec[i] -= a * bc_1y(k * (1 + i // M_x))
                if i != R - 1:
                    A[i, i + 1] = 0

        A = scipy.sparse.csr_matrix(A)

        # Initializing a solution matrix.
        solution = np.zeros((M_y + 2, M_x + 2))

        # Setting boundary conditions.
        solution[:, 0] = np.array([bc_0y(k * j) for j in range(M_y + 2)])
        solution[:, -1] = np.array([bc_1y(k * j) for j in range(M_y + 2)])
        solution[0, :] = np.array([bc_x0(h * j) for j in range(M_x + 2)])
        solution[-1, :] = np.array([bc_x1(h * j) for j in range(M_x + 2)])

        # Solving the Ax=b problem with A=A_h and b=d-vector.
        solution[1:-1, 1:-1] = scipy.sparse.linalg.spsolve(A=A, b=d_vec.transpose()).reshape((M_y, M_x))

        # Saving the solution data.
        self.solution.append(solution.ravel())

    def get_error(self, u, iteration):
        # Defining the discrete l2 norm.
        def disc_norm(v):
            return np.sqrt(np.sum(v ** 2) / len(v))

        # Defining a meshgrid from our x- and y-grids.
        x_disc = self.x_grid[iteration]
        y_disc = self.y_grid[iteration]
        X_disc, Y_disc = np.meshgrid(x_disc, y_disc)

        # Defining the numerical and analytical solution arrays
        U_disc = self.solution[iteration]
        u_disc = u(X_disc, Y_disc).ravel()

        # Finding the relative error using the l2 norm.
        error_disc = disc_norm(u_disc - U_disc) / disc_norm(u_disc)

        # Saving the error data in the relative error list.
        self.rel_error.append(error_disc)

    def check_convergence(self, u, iterations, solver, bc_0y, bc_x0, bc_1y, bc_x1, space='x'):

        # Finding the initial l2 and L2 errors of the initial system.
        solver(bc_0y, bc_x0, bc_1y, bc_x1, 0)
        self.get_error(u, 0)

        # Checks the l2 error in each iteration of increasing the gridsize in a given space.
        for iteration in range(1, iterations):

            # For x-space convergence testing.
            if space == 'x':
                # Doubling the gridsize in x-space.
                self.M_x.append(2 * self.M_x[iteration - 1] + 1)
                self.x_grid.append(np.linspace(0, 1, self.M_x[iteration] + 2))

                # Keeping the same y-grid.
                self.M_y.append(self.M_y[0])
                self.y_grid.append(self.y_grid[0])

            # For y-space convergence testing.
            elif space == 'y':
                # Doubling the gridsize in y-space.
                self.M_y.append(2 * self.M_y[iteration - 1] + 1)
                self.y_grid.append(np.linspace(0, 1, self.M_y[iteration] + 2))

                # Keeping the same x-grid.
                self.M_x.append(self.M_x[0])
                self.x_grid.append(self.x_grid[0])

            else:
                return

            # Solving for the new system.
            solver(bc_0y, bc_x0, bc_1y, bc_x1, iteration)
            self.get_error(u, iteration)

    def plot_convergence(self, line_order=2, space='x'):
        disc_error = self.rel_error

        if space == 'x':
            gridsize = self.M_x
            x_label = r'$M_x$'
            line_label = r'$O(h^{%s})$'
        elif space == 'y':
            gridsize = self.M_y
            x_label = r'$M_y$'
            line_label = r'$O(k^{%s})$'
        else:
            return

        x_array = np.linspace(gridsize[0], gridsize[-1], 1000)

        def line(x, order, init_error):
            return init_error * x**(-order) / (x[0])**(-order)

        print('M-values:', gridsize)
        print('disc error:', disc_error)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        set_ax(ax,
               title='Convergence testing for 2D Laplace equation',
               xscale='log', yscale='log',
               x_label=x_label, y_label='Relative error')

        ax.plot(gridsize, disc_error, marker='o', label=r'$l_2$ norm error', lw=3, c='#1F77B4')
        ax.plot(x_array, line(x_array, line_order, disc_error[0]), label=line_label % line_order,
                lw=3, ls='--', c='#1F77B4')

        ax.legend(fontsize=15)
        plt.show()
        plt.close()

    def plot_solution_comparison(self, u, color_map='seismic', suptitle='Solution to the 2D Laplace equation'):

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 8))
        fig.suptitle('\n' + suptitle, fontsize=20)

        step = 0
        im = None

        for ax in axes.flat:
            step += 1

            if step == 6:
                set_ax(ax, x_label='x', y_label='y', title=r'Analytical solution', fontsizes=[15, 15])
                x, y = np.meshgrid(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000))
                z = u(x, y)
            else:
                set_ax(ax, x_label='x', y_label='y',
                       title=r'Step ' + str(step) + r': $M_x = $' + str(self.M_x[step-1]) +
                             ', $M_y = $' + str(self.M_y[step-1]), fontsizes=[15, 15])
                x, y = np.meshgrid(self.x_grid[step-1], self.y_grid[step-1])
                z = self.solution[step-1].reshape((self.M_y[step-1] + 2, self.M_x[step-1] + 2))

            im = ax.contourf(x, y, z, cmap=color_map)
            im.set_clim(-0.75, 0.75)

        bar_ax = fig.add_axes([0.93, 0.1, 0.025, 0.8])
        color_bar = fig.colorbar(im, cax=bar_ax)
        color_bar.ax.set_xlabel('\n  U(x, y)', fontsize=15)

        plt.subplots_adjust(left=0.05, right=0.88, top=0.80, bottom=0.10, wspace=0.40, hspace=0.70)
        plt.show()
        plt.close()


# Analytical solutions u(x) for task 3b:

def u_3b(x, y):
    return np.sin(2*np.pi*x) * np.sinh(2*np.pi*y) / np.sinh(2*np.pi)


# Boundary conditions for task 3:

# noinspection PyUnusedLocal
def bc_3_0y(y):
    return 0


# noinspection PyUnusedLocal
def bc_3_1y(y):
    return 0


# noinspection PyUnusedLocal
def bc_3_x0(x):
    return 0


def bc_3_x1(x):
    return np.sin(2*np.pi*x)
