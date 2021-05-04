"""
Tools for task 2.
"""

# Importing libraries and plotting tools:
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import animation
from Tools.plotting_tools import *

# Setting the t0-parameter for task 2b:
t0 = 1


# 1D Heat equation tools (backward euler, crank-nicolson and an AMR scheme):

class HeatEq:
    def __init__(self, m, init_n, end_point):

        """
        :param m: gridsize in x-space.
        :param init_n: initial gridsize in t-space.
        :param end_point: final point T to stop the evolution of the system.
        """

        # Setting up the initial discretization parameters.
        self.M = m
        self.N = [init_n]
        self.T = end_point

        # Making grids.
        self.x_grid = np.linspace(0, 1, self.M + 2)
        self.t_grid = [np.linspace(0, self.T, self.N[0] + 2)]

        # Finding the spacings between each point in the grid.
        self.h = self.x_grid[1] - self.x_grid[0]
        self.t_spacing = [np.array([self.t_grid[0][i + 1] - self.t_grid[0][i] for i in range(self.N[0] + 1)])]

        # Starts in 0th iteration (needs calculations).
        self.solution = []              # U(x, T)-vector in a given iteration.
        self.reference = None           # U(x, T)-vector to be used as an "analytical" reference solution.
        self.error = []                 # Error distribution at each point in the grid.
        self.rel_error = []             # Average relative error between numerical and analytical solutions.

    def save_neumann_reference(self, left_bc, right_bc, init_c, reference_iteration, reference_title):

        # Number of non-boundary points, we update this number based on the reference iteration.
        N = self.N[0]
        for i in range(reference_iteration):
            N = 2 * N + 1
        M = self.M

        t_grid = np.linspace(0, self.T, N + 2)   # Array of points t0, t1, ... , tN+1.
        k = t_grid[1] - t_grid[0]                # Discretization step for the uniform grid.

        h = self.h
        r = k / h ** 2

        # Creating the right hand side D-matrix.
        rhs_diagonals = [r / 2, 1 - r, r / 2]
        rhs_positions = [-1, 0, 1]
        D = scipy.sparse.diags(rhs_diagonals, rhs_positions, shape=(M + 2, M + 2)).toarray()

        # Boundary condition for the left side:
        D[0][0], D[0][1] = 1 - r, r

        # Boundary condition for the right side:
        D[-1][-2], D[-1][-1] = r, 1 - r

        # Creating the left hand side A-matrix.
        lhs_diagonals = [- r / 2, 1 + r, - r / 2]
        lhs_positions = [-1, 0, 1]
        A = scipy.sparse.diags(lhs_diagonals, lhs_positions, shape=(M + 2, M + 2)).toarray()

        # Boundary condition for the left side:
        A[0][0], A[0][1] = 1 + r, - r

        # Boundary condition for the right side:
        A[-1][-2], A[-1][-1] = - r, 1 + r

        # Making the A-matrix sparse.
        A = scipy.sparse.csr_matrix(A)

        # Creating the right hand side b(t)-vector containing the initial conditions.
        def b(t):
            vector = np.zeros(M + 2)
            vector[0] = h * r * (left_bc(t) + left_bc(t + k))
            vector[-1] = - h * r * (right_bc(t) + right_bc(t + k))
            return vector

        # Initial condition U(x, 0).
        u = init_c(self.x_grid)

        # Evolving the system to the final state.
        for i in range(N + 1):
            rhs = np.dot(D, u) + b(i * k)
            u = scipy.sparse.linalg.spsolve(A, rhs.transpose())

        # Saving the reference solution data.
        # noinspection PyTypeChecker
        np.savetxt('References/' + reference_title, [self.x_grid, u])

        print('Reference saved.')

    def save_manufactured_reference(self, u, reference_title):
        x = np.linspace(0, 1, 1000000)
        # noinspection PyTypeChecker
        np.savetxt('References/' + reference_title, [x, u(x, self.T)])

        print('Reference saved.')

    def load_reference(self, reference_title):
        self.reference = np.loadtxt('References/' + reference_title)

    def neumann_neumann_crank_nicolson_solver(self, left_bc, right_bc, init_c, iteration):
        # Number of non-boundary points.
        M = self.M
        N = self.N[iteration]

        # Discretization steps for the uniform grid.
        k = self.t_spacing[iteration][0]
        h = self.h
        r = k / h ** 2

        # Creating the right hand side D-matrix.
        rhs_diagonals = [r / 2, 1 - r, r / 2]
        rhs_positions = [-1, 0, 1]
        D = scipy.sparse.diags(rhs_diagonals, rhs_positions, shape=(M + 2, M + 2)).toarray()

        # Boundary condition for the left side:
        D[0][0], D[0][1] = 1 - r, r

        # Boundary condition for the right side:
        D[-1][-2], D[-1][-1] = r, 1 - r

        # Creating the left hand side A-matrix.
        lhs_diagonals = [- r / 2, 1 + r, - r / 2]
        lhs_positions = [-1, 0, 1]
        A = scipy.sparse.diags(lhs_diagonals, lhs_positions, shape=(M + 2, M + 2)).toarray()

        # Boundary condition for the left side:
        A[0][0], A[0][1] = 1 + r, - r

        # Boundary condition for the right side:
        A[-1][-2], A[-1][-1] = - r, 1 + r

        # Making the A-matrix sparse.
        A = scipy.sparse.csr_matrix(A)

        # Creating the right hand side b(t)-vector containing the initial conditions.
        def b(t):
            vector = np.zeros(M + 2)
            vector[0] = h * r * (left_bc(t) + left_bc(t + k))
            vector[-1] = - h * r * (right_bc(t) + right_bc(t + k))
            return vector

        # Initial condition U(x, 0).
        u = init_c(self.x_grid)

        # Evolving the system to the final state.
        for i in range(N + 1):
            rhs = np.dot(D, u) + b(i * k)
            u = scipy.sparse.linalg.spsolve(A, rhs.transpose())

        # Saving the solution data.
        self.solution.append(u)

        print('Solution', iteration + 1, 'found.')

    def neumann_neumann_backward_euler_solver(self, left_bc, right_bc, init_c, iteration):
        M = self.M
        N = self.N[iteration]  # Number of non-boundary points.

        k = self.t_spacing[iteration][0]  # Discretization step for the uniform grid.
        h = self.h
        r = k / h ** 2

        # Creating the left hand side A-matrix.
        lhs_diagonals = [- r, 1 + 2 * r, - r]
        lhs_positions = [-1, 0, 1]
        A = scipy.sparse.diags(lhs_diagonals, lhs_positions, shape=(M + 2, M + 2)).toarray()

        # Boundary condition for the left side:
        A[0][0], A[0][1] = 1 + 2 * r, - 2 * r

        # Boundary condition for the right side:
        A[-1][-2], A[-1][-1] = - 2 * r, 1 + 2 * r

        # Making the A-matrix sparse.
        A = scipy.sparse.csr_matrix(A)

        # Creating the right hand side b(t)-vector containing the initial conditions.
        def b(t):
            vector = np.zeros(M + 2)
            vector[0] = 2 * h * r * left_bc(t + k)
            vector[-1] = 2 * h * r * right_bc(t + k)
            return vector

        # Initial condition U(x, 0).
        u = init_c(self.x_grid)

        # Evolving the system to the final state.
        for i in range(N + 1):
            rhs = u + b(i * k)
            u = scipy.sparse.linalg.spsolve(A, rhs.transpose())

        # Saving the solution data.
        self.solution.append(u)

        print('Solution', iteration + 1, 'found.')

    def dirichlet_dirichlet_crank_nicolson_solver(self, left_bc, right_bc, init_c, iteration):
        M = self.M
        N = self.N[iteration]  # Number of non-boundary points.

        k = self.t_spacing[iteration][0]  # Discretization step for the uniform grid.
        h = self.h
        r = k / h ** 2

        # Creating the right hand side D-matrix.
        rhs_diagonals = [r / 2, 1 - r, r / 2]
        rhs_positions = [-1, 0, 1]
        D = scipy.sparse.diags(rhs_diagonals, rhs_positions, shape=(M, M)).toarray()

        # Creating the left hand side A-matrix.
        lhs_diagonals = [- r / 2, 1 + r, - r / 2]
        lhs_positions = [-1, 0, 1]
        A = scipy.sparse.diags(lhs_diagonals, lhs_positions, shape=(M, M)).tocsr()

        # Creating the right hand side b(t)-vector containing the initial conditions.
        def b(t):
            vector = np.zeros(M)
            vector[0] = r / 2 * (left_bc(t) + left_bc(t + k))
            vector[-1] = r / 2 * (right_bc(t) + right_bc(t + k))
            return vector

        # Initial condition U(x, 0).
        u = init_c(self.x_grid[1:-1])

        # Evolving the system to the final state.
        for i in range(N + 1):
            rhs = np.dot(D, u) + b(i * k)
            u = scipy.sparse.linalg.spsolve(A, rhs.transpose())

        solution = np.zeros(M + 2)
        solution[0] = left_bc(self.T)
        solution[1:-1] = u
        solution[-1] = right_bc(self.T)

        # Saving the solution data.
        self.solution.append(solution)

        print('Solution', iteration + 1, 'found.')

    def dirichlet_dirichlet_backward_euler_solver(self, left_bc, right_bc, init_c, iteration):
        M = self.M
        N = self.N[iteration]  # Number of non-boundary points.

        k = self.t_spacing[iteration][0]  # Discretization step for the uniform grid.
        h = self.h
        r = k / h ** 2

        # Creating the left hand side A-matrix.
        lhs_diagonals = [- r, 1 + 2 * r, - r]
        lhs_positions = [-1, 0, 1]
        A = scipy.sparse.diags(lhs_diagonals, lhs_positions, shape=(M, M)).tocsr()

        # Creating the right hand side b(t)-vector containing the initial conditions.
        def b(t):
            vector = np.zeros(M)
            vector[0] = r * left_bc(t + k)
            vector[-1] = r * right_bc(t + k)
            return vector

        # Initial condition U(x, 0).
        u = init_c(self.x_grid[1:-1])

        # Evolving the system to the final state.
        for i in range(N + 1):
            rhs = u + b(i * k)
            u = scipy.sparse.linalg.spsolve(A, rhs.transpose())

        solution = np.zeros(M + 2)
        solution[0] = left_bc(self.T)
        solution[1:-1] = u
        solution[-1] = right_bc(self.T)

        # Saving the solution data.
        self.solution.append(solution)

        print('Solution', iteration + 1, 'found.')

    def get_error(self, iteration):
        # Defining the discrete l2 norm.
        def disc_norm(v):
            return np.sqrt(np.sum(v ** 2) / len(v))

        # Defining the continuous L2 norm.
        def cont_norm(x, v):
            return np.sqrt(np.trapz(v ** 2, x))

        # Finding the discrete solutions for the given iteration for the l2 norm.
        x_disc = self.x_grid
        U_disc = self.solution[iteration]
        u_disc = scipy.interpolate.interp1d(self.reference[0], self.reference[1], kind='cubic')(x_disc)

        # Interpolating with cubic method to get "continuous" solutions for the L2 norm.
        x_cont = np.linspace(0, 1, 1000000)
        U_cont = scipy.interpolate.interp1d(x_disc, U_disc, kind='cubic')(x_cont)
        u_cont = scipy.interpolate.interp1d(self.reference[0], self.reference[1], kind='cubic')(x_cont)

        # Finding the relative errors using the l2 and L2 norms respectively.
        error_disc = disc_norm(u_disc - U_disc) / disc_norm(u_disc)
        error_cont = cont_norm(x_cont, u_cont - U_cont) / cont_norm(x_cont, u_cont)

        # Saving the error data in the relative error list.
        self.rel_error.append((error_disc, error_cont))

        # Saving the error vector at each point in the grid.
        self.error.append(abs(u_disc - U_disc))

    def check_convergence(self, iterations, solver, left_bc, right_bc, init_c, reference_title):
        # Computing the reference solution:
        self.load_reference(reference_title)
        print('Reference loaded.')

        # Finding the l2 and L2 errors of the initial system.
        solver(left_bc, right_bc, init_c, 0)
        self.get_error(iteration=0)

        # Using UMR to minimize the l2 and L2 errors of a few more iterations of the system.
        for i in range(1, iterations):

            # Doubling the size of the t-grid.
            self.N.append(2 * self.N[i - 1] + 1)
            N = self.N[i]
            self.t_spacing.append(1 / (N + 1) * np.ones(N + 1))
            self.t_grid.append(np.linspace(0, 1, N + 2))

            # Solving for the new system.
            solver(left_bc, right_bc, init_c, i)
            self.get_error(i)

    def plot_convergence(self, include_cont_error=True, line_order=2):
        disc_error = [x[0] for x in self.rel_error]
        cont_error = [x[1] for x in self.rel_error]
        t_gridsize = [x + 2 for x in self.N]
        t_array = np.linspace(t_gridsize[0], t_gridsize[-1], 1000)

        def line(x, order, init_error):
            return init_error * x**(-order) / (x[0])**(-order)

        print('N+2 values:', t_gridsize)
        print('disc error:', disc_error)
        print('cont error:', cont_error)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        set_ax(ax,
               title='Convergence testing for the 1D Heat equation',
               xscale='log', yscale='log',
               x_label='Gridsize (M+2)', y_label='Relative error')

        ax.plot(t_gridsize, disc_error, marker='o', label=r'$l_2$ norm error', lw=3, c='#1F77B4')
        ax.plot(t_array, line(t_array, line_order, disc_error[0]), label=r'$O(h^{%s})$' % line_order,
                lw=3, ls='--', c='#1F77B4')

        if include_cont_error:
            ax.plot(t_gridsize, cont_error, marker='o', label=r'$L_2$ norm error', lw=3, c='#FF7F0E')
            ax.plot(t_array, line(t_array, line_order, cont_error[0]), label=r'$O(h^{%s})$' % line_order,
                    lw=3, ls='--', c='#FF7F0E')

        ax.legend(fontsize=15)
        plt.show()
        plt.close()

    def plot_solution_comparison(self, step_jumps=1, suptitle='Solution to the 1D Heat equation'):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 8))
        fig.suptitle('\n'+suptitle, fontsize=20)

        step = 0

        for ax in axes.flat:
            step += step_jumps
            ax.set_title('Step ' + str(step) + ': N = ' + str(self.N[step - 1]), fontsize=15)
            x = self.x_grid
            U = self.solution[step - 1]
            if step == step_jumps:
                ax.plot(x, U, label='Numerical solution', lw=3)
                ax.plot(self.reference[0], self.reference[1], label='Analytical solution', ls='--', lw=3)
            else:
                ax.plot(x, U, lw=3)
                ax.plot(self.reference[0], self.reference[1], ls='--', lw=3)

        plt.subplots_adjust(0.1, 0.1, 0.90, 0.80, 0.35, 0.35)
        fig.legend(fontsize=15, bbox_to_anchor=(0, -0.05, 1, 1))
        plt.show()
        plt.close()


# 1D Parabolic Burger's equation tools (semi-discretization, central difference space):

class BurgerEq:
    def __init__(self, grid_step, time_step, end_point):

        """
        :param grid_step: the discretization step h in x-space.
        :param time_step: the discretization step k in t-space.
        :param end_point: the end time to which we evolve the initial state.
        """

        # Setting the discretization steps and the simulation end point.
        self.h = grid_step
        self.k = time_step
        self.T = end_point

        # Creating a grid in x-space with discrete points x0=0, x1, ... , xM+1=1.
        self.M = int(1 / self.h) - 1
        self.x_grid = np.linspace(0, 1, self.M + 2)

        # Creating a grid in t-space with discrete points t0=0, t1, ... , tN+1=T.
        self.N = int(self.T / self.k) - 1
        self.t_grid = np.linspace(0, self.T, self.N + 2)

        # Solution data.
        self.t_star = None
        self.solution = np.zeros(shape=(self.M + 2, self.N + 2))    # Data from the simulation.

    def run_simulation(self, left_bc, right_bc, init_c):
        v0 = init_c(self.x_grid[1:-1])

        diagonals = [-1, 1]
        positions = [-1, 1]
        diagonals_array = np.array(diagonals) / (4 * self.h)

        # Creating the matrix A_h.
        A = scipy.sparse.diags(diagonals_array, positions, shape=(self.M, self.M)).toarray()

        # Right hand side of the equation v_dot = Av + b.
        # noinspection PyUnusedLocal
        def rhs(t, v):
            return - np.dot(A, v ** 2)

        # Solving the v_dot = Av + b differential equation using the RK45 method.
        sol = scipy.integrate.solve_ivp(rhs, t_span=(0, self.T), y0=v0, method='RK45',
                                        t_eval=self.t_grid, max_step=self.k)

        # Saving the solution data.
        self.solution[0, :] = left_bc(self.t_grid)
        self.solution[-1, :] = right_bc(self.t_grid)

        for i in range(self.N + 2):
            self.solution[1:-1, i] = sol.y[:, i]

    def find_breaking_time(self, init_c_diff):
        self.t_star = -1 / min(init_c_diff(self.x_grid)) - 0.01
        print('Breaking time:', round(self.t_star, 4), 's')

    def plot_characteristics(self, end_time, init_c):

        # Time interval for the analysis.
        t_array = np.linspace(0, end_time, 1000)

        # Intercept points with the x-axis for the characteristic curves.
        xi_array = np.linspace(0, 1, 150)

        # General definition of the characteristic curves.
        def gamma(f, xi, t):
            return f(xi) * t + xi

        fig = plt.figure()
        ax_1 = fig.add_subplot(211)
        ax_2 = fig.add_subplot(212)
        set_ax(ax_1, x_label='x', y_label='t', title='Initial profile')
        set_ax(ax_2, x_label='x', y_label='t', title='Characteristic curves')

        ax_1.plot(self.x_grid, init_c(self.x_grid), c='r', lw=3)

        for i in range(len(xi_array)):
            ax_2.plot(gamma(init_c, xi_array[i], t_array), t_array, c='r', lw=1.5)

        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.65, wspace=0.15)
        plt.show()

    def plot_final_state(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        set_ax(ax, x_label='x', y_label='U(x)', title='U(x) at t = '+str(round(self.t_grid[-1], 4))+' s')
        ax.plot(self.x_grid, self.solution[:, -1], lw=3)
        plt.show()

    def plot_contour(self):

        T, X = np.meshgrid(self.t_grid, self.x_grid)
        U = self.solution

        fig = plt.figure()
        ax = fig.add_subplot(111)

        single_contour(fig, ax, T, X, U, x_label='t', y_label='x', z_label='U(x, t)',
                       title="The inviscid Burger's equation")
        plt.show()
        plt.close()

    def animate_simulation(self):
        x = self.x_grid
        t = self.t_grid
        solution = self.solution

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        set_ax(ax, x_label='x', y_label='U(x)', title='Breaking bad')
        ax.set(xlim=(0, 1), ylim=(0, 1.1))
        time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        line, = ax.plot([], [], lw=3, label=f'U(x)')
        ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0, fontsize=15)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            line.set_data(x, solution[:, i])
            time_text.set_text('time = %.4f s' % t[i])

            return line, time_text

        plt.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.80, hspace=0.5, wspace=0.5)
        animation.FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=1/1000, blit=True)
        plt.show()


# Analytical solutions u(x) for task 2b:

def u_2b(x, t):
    return 1 / np.sqrt(np.pi * 4 * (t + t0)) * np.exp(- (x - 1 / 2) ** 2 / (4 * (t + t0)))


# Initial conditions for task 2:

def init_c_a(x):
    return 2 * np.pi * x - np.sin(2 * np.pi * x)


def init_c_b(x):
    return 1 / np.sqrt(np.pi * 4 * t0) * np.exp(- (x - 1 / 2) ** 2 / 4 * t0)


def init_c_c(x):
    return np.exp(-400 * (x - 1/2) ** 2)


def init_c_c_diff(x):
    return -800 * (x - 1/2) * np.exp(-400 * (x - 1/2) ** 2)


# Boundary conditions for task 2a/2c:

# noinspection PyUnusedLocal
def left_bc_ac(t):
    return 0


# noinspection PyUnusedLocal
def right_bc_ac(t):
    return 0


# Boundary conditions for task 2b:

def left_bc_b(t):
    return 1 / np.sqrt(np.pi * 4 * (t + t0)) * np.exp(- 1/(16 * (t + t0)))


def right_bc_b(t):
    return 1 / np.sqrt(np.pi * 4 * (t + t0)) * np.exp(- 1/(16 * (t + t0)))
