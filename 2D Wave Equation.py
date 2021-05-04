import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

from plotting_tools import set_ax



class wave_eq_2D:
    def __init__(self,init_m,init_n,init_t):
        self.M = [init_m]
        self.N = [init_n]
        self.T = init_t

        self.x_grid = [np.linspace(0, 1, self.M[0] + 2)]
        self.y_grid = [np.linspace(0,1,self.N[0] + 2)]
        self.t_grid = np.linspace(0, 1, self.T + 2)

        self.solution = []
        self.rel_error = []
        self.cpu_time = []
        self.solution_time = []



    def wave_solver2(self,init_c,iteration):

        M = self.M[iteration]
        N = self.N[iteration]
        T = self.T

        h_x = self.x_grid[iteration][1] - self.x_grid[iteration][0]
        h_y = self.y_grid[iteration][1] - self.y_grid[iteration][0]
        k = self.t_grid[1] - self.t_grid[0]

        r_x = k / h_x
        r_y = k / h_y

        X,Y = np.meshgrid(self.x_grid[iteration],self.y_grid[iteration])

        u = np.zeros(shape=(M + 2, N + 2, T + 2))

        # u[:, :, 0] = init_c(X, Y)  # Use this to make 3D plot
        u[:, :, 0] = init_c(self.x_grid[iteration], self.y_grid[iteration])  # otherwise use this

        for i in range(T+1):

            u_xx = u[2:, 1:-1, i] - 2 * u[1:-1, 1:-1, i] + u[:-2, 1:-1, i]
            u_yy = u[1:-1,2:,i] -2*u[1:-1,1:-1,i] + u[1:-1,:-2,i]
            if i == 0:  # using Neumann condition at first time step

                u[1:-1,1:-1, i + 1] = u[1:-1,1:-1, i] + 0.5*(r_x ** 2) * u_xx + 0.5*(r_y**2)*u_yy

                # Apply periodic BCS in both spatial dimensions: u(x+1,y,t) = u(x,y,t) = u(x,y+1,t)
                u[0, 1:-1, i + 1] = u[0,1:-1, i] + 0.5*(r_x ** 2) * (u[1,1:-1, i] - 2 * u[0,1:-1, i] + u[-2,1:-1, i]) + (r_y**2)/2*(u[0,2:,i] -2*u[0,1:-1,i] + u[0,:-2,i])
                u[-1,:,i+1] = u[0,:,i+1]
                u[1:-1, 0, i + 1] = u[1:-1, 0, i] + 0.5 * (r_x ** 2)/2 * (u[2:, 0, i] - 2 * u[1:-1, 0, i] + u[:-2, 0, i]) + (r_y ** 2)/2 * (u[1:-1, 1, i] - 2 * u[1:-1, 0, i] + u[1:-1, -2, i])
                u[:, -1, i + 1] = u[:, 0, i + 1]

            else:

                u[1:-1,1:-1, i + 1] = 2*u[1:-1,1:-1,i] -u[1:-1,1:-1,i-1] + (r_x ** 2)/2 * u_xx + (r_y**2)/2*u_yy

                # Apply periodic BCS in both spatial dimensions: u(x+1,y,t) = u(x,y,t) = u(x,y+1,t)
                u[0, 1:-1, i + 1] = 2*u[0,1:-1,i] -u[0,1:-1,i-1] + 0.5*(r_x ** 2) * (u[1,1:-1, i] - 2 * u[0,1:-1, i] + u[-2,1:-1, i]) +(0.5*r_y**2)*(u[0,2:,i] -2*u[0,1:-1,i] + u[0,:-2,i])
                u[-1, :, i + 1] = u[0, :, i + 1]  # on the right side
                u[1:-1, 0, i + 1] = 2*u[1:-1,0,i] -u[1:-1,0,i-1] + (r_x ** 2) * (u[2:,0, i] - 2 * u[1:-1,0, i] + u[:-2,0, i]) +(r_y**2)*(u[1:-1,1,i] -2*u[1:-1,0,i] + u[1:-1,-2,i])
                u[:,-1,i+1] = u[:,0,i+1]

        self.solution.append(u[:,:,-1])
        self.solution_time = u

        print("Solution", iteration + 1, "found.")




    def get_error(self, u, iteration):
        # Defining the discrete l2 norm.
        def disc_norm(v):
            return np.sqrt(np.sum(v ** 2) / len(v))

        # Defining a meshgrid from our x- and y-grids.
        x_disc = self.x_grid[iteration]
        y_disc = self.y_grid[iteration]
        #X_disc,Y_disc = np.meshgrid(x_disc,y_disc)

        # Defining the numerical and analytical solution arrays
        U_disc = self.solution[iteration]
        u_disc = u(x_disc, y_disc,1)
        # u_disc = u(X_disc,Y_disc,1)

        # Finding the relative error using the l2 norm.
        error_disc = disc_norm(u_disc - U_disc) / disc_norm(u_disc)
        # Saving the error data in the relative error list.
        self.rel_error.append(error_disc)

    def check_convergence(self, iterations, u, init_c):

        # Finding the l2 error of the initial system.
        start_time = time.time()
        self.wave_solver2(init_c, 0)
        self.cpu_time.append(time.time() - start_time)
        self.get_error(u, 0)

        # Calculating the l2 errors of a few more iterations of the system.
        for i in range(1, iterations):
            # Doubling the size of the x-grid and y-grid.
            self.M.append(2 * self.M[i - 1])
            M = self.M[i]
            self.x_grid.append(np.linspace(0, 1, M + 2))
            self.N.append(2*self.N[i-1])
            N = self.N[i]
            self.y_grid.append(np.linspace(0,1,N + 2))

            # Solving for the new system.
            start_time = time.time()
            self.wave_solver2(init_c, i)
            self.cpu_time.append(time.time() - start_time)
            self.get_error(u, i)

    def plot_convergence(self, line_order=2, space='x'):
        disc_error = self.rel_error
        print('M-values:', self.M)

        if space == 'x':
            gridsize = self.M
            x_label = r'$M$'
            line_label = r'$O(h^{%s})$'
        elif space == 'y':
            gridsize = self.N
            x_label = r'$N$'
            line_label = r'$O(k^{%s})$'
        else:
            return

        x_array = np.linspace(gridsize[0], gridsize[-1], 1000)

        def line(x, order, init_error):
            return init_error * x ** (-order) / (x[0]) ** (-order)

        print('M-values:', gridsize)
        print('disc error:', disc_error)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        set_ax(ax,
               title='Convergence testing for 2D Wave Equation',
               xscale='log', yscale='log',
               x_label=x_label, y_label='Relative error')

        ax.plot(gridsize, disc_error, marker='o', label=r'$l_2$ norm error', lw=3, c='#1F77B4')
        ax.plot(x_array, line(x_array, line_order, disc_error[0]), label=line_label % line_order,
                lw=3, ls='--', c='#1F77B4')

        ax.legend(fontsize=15)
        plt.show()
        plt.close()



    def plot_3D(self):
        # Plot 3D

        X, Y = np.meshgrid(self.x_grid[-1], self.y_grid[-1])
        sol = self.solution

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X,Y,sol[-1], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        set_ax(ax,title = ("Numerical Solution of 2D Wave Equation"), x_label= "x", y_label = "y")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()

    def plot_error_df(self,line_order = 2,space = "x"):

        disc_error = self.rel_error
        print('M-values:', self.M)
        print("N-values:", self.N)
        k = self.t_grid[1] - self.t_grid[0]
        num = [a*b for a,b in zip(self.M,self.N)]
        df = num/k


        if space == 'x':
            line_label = r'$O(h^{%s})$'
        elif space == 'y':
            line_label = r'$O(k^{%s})$'
        else:
            return



        gridsize = df
        x_label = "Number of Degrees of Freedom"

        def line(x, order, init_error):
            return init_error * x ** (-order) / (x[0]) ** (-order)


        print('M-values:', gridsize)
        print('disc error:', disc_error)
        x_array = np.linspace(gridsize[0], gridsize[-1], 1000)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        set_ax(ax,
               title='Convergence testing for 2D Wave Equation',
               xscale = "log", yscale = "log",
               x_label=x_label, y_label='Relative error')

        ax.plot(gridsize, disc_error, marker='o', label=r'$l_2$ norm error', lw=3, c='#1F77B4')
        ax.plot(x_array, line(x_array, line_order, disc_error[0]), label=line_label % line_order,
                lw=3, ls='--', c='#1F77B4')


        ax.legend(fontsize=15)
        plt.show()
        plt.close()

    def plot_cpu_time(self):

        k = self.t_grid[1] - self.t_grid[0]
        num = [a * b for a, b in zip(self.M, self.N)]
        df = num / k
        print("Cpu time: ", self.cpu_time)
        print("Number of Degrees of Freedom: ", df)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        set_ax(ax,
               title='Computational Time as Function of Degrees of Freedom', xscale = "log",
               x_label="Number of Degrees of Freedom", y_label='Computational Time in Seconds')

        ax.plot(df,self.cpu_time,marker = "o", label = "Computational time", lw = 3, c = '#1F77B4' )
        ax.legend(fontsize=15)
        plt.show()
        plt.close()




#Boundary condition g(x,y)
def g(x,y):
    return np.cos(4* np.pi * x) * np.sin(4 * np.pi * y)

#Analytical solution
def u(x,y,t):
    return np.cos(4*np.pi*x)*np.sin(4*np.pi*y)*np.cos(np.sqrt(32)*np.pi*t)

def u_0(x,y):
    return np.cos(4*np.pi*x)*np.sin(4*np.pi*y)

System = wave_eq_2D(80,80,2000)


System.check_convergence(4,u,g)
#System.plot_convergence()
#sol = System.solution_time[-1]
#err = System.rel_error
# print("Discrete error is: ", err)

#System.plot_convergence()
System.plot_error_df()
#System.plot_cpu_time()

"""

x = y = np.linspace(0,1,1000)
X, Y = np.meshgrid(x, y)
U = np.zeros(shape = (1000,1000,2000))
U[:,:,0] = g(X,Y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
my_cmap = plt.get_cmap('hot')
surf = ax.plot_surface(X, Y, U[:,:,0], label = "Numerical Solution",cmap = my_cmap, edgecolor = "none")
fig.colorbar(surf, ax = ax,
             shrink = 0.5, aspect = 5)

plt.show()


nt = 6000
for i in range(0,nt+1,950): # plot every several time step
    X, Y = np.meshgrid(System.x_grid[-1], System.y_grid[-1])

    sol = System.solution_time
    fig = plt.figure()
    u_ex = u_0(System.x_grid[-1],System.y_grid[-1])
    #print(u_ex)

    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, sol[:,:,i], rstride=1, cstride=1,cmap =  cm.coolwarm, linewidth=0, antialiased=False)
    set_ax(ax, title=("Numerical Solution of 2D Wave Equation for time = %i " %i), x_label="x", y_label="y")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    
    
"""""