import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt

from plotting_tools import set_ax


class wave_eq:
    def __init__(self,init_m,init_n):

        #Stabil for N => M, dvs r <= 1

        self.M = [init_m]
        self.N = init_n

        self.x_grid = [np.linspace(0, 1,self.M[0] +2)]
        self.t_grid = np.linspace(0, 1, self.N +2)


        self.solution = []
        self.rel_error = []




    def forward_euler_solver(self, init_c,iteration):
        M = self.M[iteration]
        N = self.N

        h = self.x_grid[iteration][1] - self.x_grid[iteration][0]
        k = self.t_grid[1] - self.t_grid[0]

        r = k / h
        u = np.zeros(shape=(M + 2, N + 2))  # initial time + (N+1) forward steps

        u[:, 0] = init_c(self.x_grid[iteration])  # apply initial condition u(x,0) = g(x)

        for i in range(N + 1):
            if (i == 0):  # using Neumann condition at first time step
                u[1:-1, i + 1] = u[1:-1, i] + 0.5*(r ** 2) * (u[2:, i] - 2 * u[1:-1, i] + u[:-2, i])


            else:
                u[1:-1, i + 1] = 2 * u[1:-1, i] - u[1:-1, i - 1] + (r ** 2) * (u[2:, i] - 2 * u[1:-1, i] + u[:-2, i])


            #Apply periodic boundary condition
            u[0, i + 1] = u[-3, i + 1]  # on the left side
            u[-1, i + 1] = u[2, i + 1] # on the right side
            # Apply periodic boundary condition by using the ghost points


        u[0,:] = u[-1,:]
        self.solution.append(u[:,-1])
        print('Solution', iteration + 1, 'found.')



    def get_error(self, u,iteration):

        # Defining the discrete l2 norm.
        def disc_norm(v):
            return np.sqrt(np.sum(v ** 2) / len(v))

        # Defining the continuous L2 norm.

        # Finding the discrete solutions for the given iteration for the l2 norm.
        x_disc = self.x_grid[iteration]
        U_disc = self.solution[iteration]
        u_disc = u(x_disc, 1)


        # Finding the relative errors using the l2 and L2 norms respectively.
        error_disc = disc_norm(u_disc - U_disc) / disc_norm(u_disc)

        # Saving the error data in the relative error list.
        self.rel_error.append(error_disc)

    def check_convergence(self, iterations, u, init_c):

        # Finding the l2 error of the initial system.
        self.forward_euler_solver(init_c,0)
        self.get_error(u, 0)

        # Calculating the l2 errors of a few more iterations of the system.
        for i in range(1, iterations):
            # Doubling the size of the x-grid.
            self.M.append(2 * self.M[i - 1])
            M = self.M[i]
            self.x_grid.append(np.linspace(0,1,M+2))

            # Solving for the new system.
            self.forward_euler_solver(init_c,i)
            self.get_error(u, i)


    def plot_convergence(self, line_order=2):
        disc_error = self.rel_error
        M_array = np.linspace(self.M[0], self.M[-1], 1000)

        def line(x, order, init_error):
            return init_error * x**(-order) / (x[0])**(-order)

        print('M-values:', self.M)
        print('disc error:', disc_error)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        set_ax(ax,
               title='Convergence Testing for the Wave Equation',
               xscale='log', yscale='log',
               x_label='Number of Grid Points in x-direction', y_label='Relative error')

        ax.plot(self.M, disc_error, marker='o', label=r'$l_2$ norm error', lw=3, c='orange')
        ax.plot(M_array, line(M_array, line_order, disc_error[0]), label=r'$O(h^{%s})$' % line_order,
                lw=3, ls='--', c='#1F77B4')

        ax.legend(fontsize=15)
        plt.show()
        plt.close()

    def plot_solution_comparison(self, u, step_jumps=1, suptitle='Solution to the Wave Equation'):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 8))
        fig.suptitle('\n'+suptitle, fontsize=20)

        step = 0

        for ax in axes.flat:
            step += step_jumps
            ax.set_title('Step ' + str(step) + ': M = ' + str(self.M[step - 1]) + ', N = ' + str(self.N), fontsize=15)
            x = self.x_grid
            U = self.solution[step - 1]
            if step == step_jumps:
                ax.plot(x, U, label='Numerical solution', lw=3)
                ax.plot(u_b_1(self.x_grid[0],self.t_grid[0]), u_b_1(self.x_grid[0],self.t_grid[1]), label='Analytical solution', ls='--', lw=3)
            else:
                ax.plot(x, U, lw=3)
                ax.plot(u_b_1(self.x_grid[0],self.t_grid[0]), u_b_1(self.x_grid[0],self.t_grid[1]), ls='--', lw=3)

        plt.subplots_adjust(0.1, 0.1, 0.90, 0.80, 0.35, 0.35)
        fig.legend(fontsize=15, bbox_to_anchor=(0, -0.05, 1, 1))
        plt.show()
        plt.close()



#Initial conditions for b)
def init_c_g_1(x):
    return np.cos(4*np.pi*x)
def init_c_g_2(x):
    return np.exp(-100*(x-0.5)**2)



#Analytical solution for part b
def u_b_1(x,t):
    return (1/2)*((np.cos(4*np.pi*(x-t))) + (np.cos(4*np.pi*(x+t))))





System = wave_eq(50,50000)

System.check_convergence(10,u_b_1,init_c_g_1)
System.plot_convergence()



#nt = System.solution.shape[1]
nt = len(System.solution)
ylim = [np.min(System.solution), np.max(System.solution)]


for i in range(0,nt,200): # plot every several time step
    plt.figure()
    plt.plot(System.x_grid[0], System.solution[:,i], label = "Numerical Solution")
    plt.title("time = %i " %i)
    plt.ylim(ylim)
    plt.grid()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.legend()
    plt.show()





