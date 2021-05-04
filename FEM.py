
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.special import roots_legendre

from plotting_tools import set_ax


class fem_poisson():

    def __init__(self,a,b,d1,d2,N):

        self.N = [N]
        self.x_grid = [np.linspace(a, b, self.N[0] + 1)]
        self.a = a
        self.b = b
        self.d1 = d1
        self.d2 = d2
        self.rel_error = []
        self.solution = []
        #self.spacing = [np.array([self.x_grid[0][i + 1] - self.x_grid[0][i] for i in range(self.N[0] + 1)])]
        #self.spacing = [np.diff(self.x_grid[0])]
        self.spacing = [np.array([self.x_grid[0][i + 1] - self.x_grid[0][i] for i in range(self.N[0])])]
        self.max_error = []



    def find_F(self,f,N,iteration):
        F = np.zeros(N + 1)
        x_grid = self.x_grid[iteration]

        for i in range(N):
            x1 = x_grid[i]
            x2 = x_grid[i+1]

            def fphi1(x):
                return f(x) * phi1(x, x1, x2)

            def fphi2(x):
                return f(x) * phi2(x, x1, x2)

            F[i] += quadrature1D(x1,x2,fphi1,20)
            F[i+1] += quadrature1D(x1,x2,fphi2,20)

        return F



    def solve(self,f,iteration):

        a = self.a
        b = self.b
        N = self.N[iteration]
        d1 = self.d1
        d2 = self.d2

        R_h = np.zeros(N+1)
        R_h[0] = d1
        R_h[-1] = d2
        h = (b-a)/N
        A = (1/h)*tridiag_sparse(-1,2,-1,N)
        rhs = self.find_F(f,N,iteration)
        rhs -= A @ R_h
        rhs1 = rhs.copy()
        A1 = A.copy()
        A1[0,1], A1[0,0] = 0,1
        A1[-1,-2], A1[-1,-1] = 0,1
        rhs1[0] = 0
        rhs1[-1] = 0


        u_hat = spsolve(A1.tocsr(),rhs1)
        u = u_hat + R_h

        self.solution.append(u)

        return u

    def solve_afem(self,f,iteration):
        N = self.N[iteration]

        A = tridiag_sparse_afem(self.x_grid[iteration])

        rhs = self.find_F(f,N,iteration)
        R_h = np.zeros(N + 1)
        R_h[0] = self.d1
        R_h[-1] = self.d2

        rhs -= A @ R_h
        rhs1 = rhs.copy()
        A1 = A.copy()
        A1[0, 1], A1[0, 0] = 0, 1
        A1[-1, -2], A1[-1, -1] = 0, 1
        rhs1[0] = 0
        rhs1[-1] = 0

        u_hat = spsolve(A1.tocsr(), rhs1)

        u = u_hat + R_h

        self.solution.append(u)

        return u



    def get_error(self, u, iteration):

        # Defining the continuous L2 norm.
        def cont_norm(x, v):
            return np.sqrt(np.trapz(v ** 2, x))

        x_disc = self.x_grid[iteration]  # Array of points x0, x1, ... , xN.
        U_disc = self.solution[iteration]  # Numerical solution, array of points U0, U1, ... , UN.
        # Interpolating over the numerical discrete solution to find a continuous function
        x_cont = np.linspace(self.a, self.b, 1000000)
        U_cont = scipy.interpolate.interp1d(x_disc, U_disc, kind='linear')(x_cont)
        u_cont = u(x_cont)

        # Finding the relative errors using the l2 and L2 norms respectively.
        error_cont = cont_norm(x_cont, u_cont - U_cont) / cont_norm(x_cont, u_cont)

        # Saving the error data in the relative error list.
        self.rel_error.append(error_cont)


    def L2norm(self,u,iteration,a,b):
        u_h = scipy.interpolate.interp1d(self.x_grid[iteration], self.solution[iteration], assume_sorted=True)
        phi = lambda x: (u(x) - u_h(x)) ** 2
        L2_umuh2 = quadrature5(phi,a, b)
        return L2_umuh2



    def plot_convergence(self, line_order=1):

        cont_error = [x for x in self.rel_error]
        gridsize = [x + 1 for x in self.N]
        x_array = np.linspace(gridsize[0], gridsize[-1], 1000)

        def line(x, order, init_error):
            return init_error * x ** (-order) / (x[0]) ** (-order)

       # print('N+1 values:', gridsize)
       # print('cont error:', cont_error)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        set_ax(ax,
               title='Convergence testing for 1D Poisson equation',
               xscale='log', yscale='log',
               x_label='Gridsize (N+1)', y_label='Relative error')

        #ax.plot(gridsize, disc_error, marker='o', label=r'$l_2$ norm error', lw=3, c='#1F77B4')
        #ax.plot(x_array, line(x_array, line_order, disc_error[0]), label=r'$O(h^{%s})$' % line_order,
                #lw=3, ls='--', c='#1F77B4')

        ax.plot(gridsize, cont_error, marker='o', label=r'$L_2$ norm error', lw=3, c='#FF7F0E')
        ax.plot(x_array, line(x_array, line_order, cont_error[0]), label=r'$O(h^{%s})$' % line_order,
                lw=3, ls='--', c='steelblue')

        ax.legend(fontsize=15)
        plt.show()
        plt.close()


    def check_convergence(self, iteration, u, f):

        # Finding the l2 error of the initial system.
        a = self.a
        b = self.b
        self.solve(f,0)
        self.get_error(u, 0)

        # Calculating the l2 errors of a few more iterations of the system.
        for i in range(1, iteration):
            # Doubling the size of the x-grid.
            self.N.append(2 * self.N[i - 1])
            N = self.N[i]
            self.x_grid.append(np.linspace(a,b,N+1))
            self.spacing.append(np.diff(self.x_grid[-1]))

            # Solving for the new system.
            self.solve(f,i)
            self.get_error(u,i)


    def check_amr_convergence_average(self, iterations,u,f, alpha = 1.0):

        # Finding the initial l2 and L2 errors of the initial system.
        self.solve_afem(f, 0)
        self.get_error(u,0)

        # Checks the l2 error 'iterations' times.
        for iteration in range(iterations):
            error_bar = self.L2norm(u,iteration, self.a, self.b)

            # Adding points where it is needed.
            old_grid = self.x_grid[iteration]
            new_grid = []
            for i in range(self.N[iteration] ):
                x1 = self.x_grid[iteration][i]
                x2 = self.x_grid[iteration][i+1]
                h = x2 - x1
                if self.L2norm(u,iteration,x1, x2) > error_bar/self.N[iteration]:
                    # Splitting element in half
                    # print("Element error is: ",self.L2norm_2(u,iteration,x1,x2))
                    # new_grid.append((old_grid[i] - h/2))
                    new_grid.append(old_grid[i])
                    new_grid.append(old_grid[i] + h/2)

                else:
                    new_grid.append(old_grid[i])

            new_grid.append(self.b)
            self.x_grid.append(np.array(new_grid))
            self.N.append(len(new_grid) - 1)  # grid has N+1 points
            self.spacing.append(np.array([new_grid[i + 1] - new_grid[i] for i in range(len(new_grid) - 1)]))

            # Solving for the new system.
            self.solve_afem(f, iteration + 1)
            self.get_error(u, iteration + 1)


    def check_amr_convergence_max(self,iterations,u,f,alpha = 0.7):

        # Finding the initial l2 and L2 errors of the initial system.
        self.solve_afem(f, 0)
        self.L2Error(u, 0)

        # Checks the l2 error 'iterations' times.
        for iteration in range(iterations):
            error_bar = self.find_max_error(iteration,u)

            # Adding points where it is needed.
            old_grid = self.x_grid[iteration]
            new_grid = []
            for i in range(self.N[iteration]):
                x1 = self.x_grid[iteration][i]
                x2 = self.x_grid[iteration][i + 1]
                h = x2 - x1
                if self.L2norm(u, iteration, x1, x2) > alpha * error_bar:
                    # Splitting element in half
                    new_grid.append(old_grid[i])
                    new_grid.append(old_grid[i] + h / 2)

                else:
                    new_grid.append(old_grid[i])

            new_grid.append(self.b)
            self.x_grid.append(np.array(new_grid))
            self.N.append(len(new_grid) - 1)  # grid has N+1 points
            self.spacing.append(np.array([new_grid[i + 1] - new_grid[i] for i in range(len(new_grid) - 1)]))

            # Solving for the new system.
            self.solve_afem(f, iteration + 1)
            self.get_error(u, iteration + 1)

    def find_max_error(self,it,u):

        m = -1
        for i in range(self.N[it]):
            x1 = self.x_grid[it][i]
            x2 = self.x_grid[it][i + 1]
            error = self.L2norm(u,it,x1,x2)
            m = max(m,error)
        return m


def tridiag_sparse(v,d,w,N):
    A = sparse.diags([v,d,w], [-1,0,1], (N+1,N+1),format="lil",dtype = np.float)
    A[0,0] = 1
    A[-1,-1] = 1

    return A

def tridiag_sparse_afem(x):
    N = len(x)

    Ak0 = np.asarray([[1, -1], [-1, 1]])

    A = sparse.lil_matrix((N, N))

    for i in range(N - 1):
        x1 = x[i]
        x2 = x[i + 1]
        hi = x2 - x1
        A[i:i + 2, i:i + 2] += 1 / hi * Ak0

    return A




def quadrature5(f, a, b):
    '''
    Calculate int_a^b f(x)dx using 5-point Gaussian Quadrature
    Print the result
    '''
    quad_x = [0., np.sqrt(5. - 2. * np.sqrt(10. / 7.)) / 3., -np.sqrt(5. - 2. * np.sqrt(10. / 7.)) / 3.,
              np.sqrt(5. + 2. * np.sqrt(10. / 7.)) / 3., -np.sqrt(5. + 2. * np.sqrt(10. / 7.)) / 3.]
    quad_w = [128. / 225., (322. + 13. * np.sqrt(70)) / 900., (322. + 13. * np.sqrt(70)) / 900.,
              (322. - 13. * np.sqrt(70)) / 900., (322. - 13. * np.sqrt(70)) / 900.]
    s_int = ((b - a) / 2.) * np.sum([quad_w[j] * f(((b - a) / 2.) * quad_x[j] + (a + b) / 2.) for j in range(5)])
    return s_int


def quadrature1D(a, b,g,Nq = 5):
    """
    Function to do a numerical 1D integral: line_int = False
    or a lineintegral: line_int = True
    Both using Gaussian quadrature
    Parameters
    ----------
    a : float or list/tuple
        lower bound or startpoint of line in the integration.
    b : float or list/tuple
        upper bound or endpoint of line in the integration.
    Nq : int
        How many points to use in the numerical integration, Nq-point rule.
    g : function pointer
        pointer to function to integrate.
    line_int : bool, optional
        Do we have a lineitegral. The default is False.
    Raises
    ------
    TypeError
        If line_int=True and a or b are not in accepted form/type, meaning a and b are not list or tuple
    Returns
    -------
    I : float
        value of the integral.
    """
    # Weights and gaussian quadrature points for refrence
    # if Nq == 1:
    #     z_q = np.zeros(1)
    #     rho_q = np.ones(1) * 2
    # if Nq == 2:
    #     c = np.sqrt(1 / 3)
    #     z_q = np.array([-c, c])
    #     rho_q = np.ones(2)
    # if Nq == 3:
    #     c = np.sqrt(3 / 5)
    #     z_q = np.array([-c, 0, c])
    #     rho_q = np.array([5, 8, 5]) / 9
    # if Nq == 4:
    #     c1 = np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7)
    #     c2 = np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7)
    #     z_q = np.array([-c1, -c2, c2, c1])
    #     k1 = 18 - np.sqrt(30)
    #     k2 = 18 + np.sqrt(30)
    #     rho_q = np.array([k1, k2, k2, k1]) / 36

    # Weights and gaussian quadrature points, also works for Nq larger than 4
    z_q, rho_q = roots_legendre(Nq)
    # check if we have a line integral, given by user

    # compute the integral numerically
    I = (b - a) / 2 * np.sum(rho_q * g(0.5 * (b - a) * z_q + 0.5 * (b + a)))
    return I

def phi1(x, x1, x2):
    # phi1(x1) = 1, phi1(x2) = 0
    return (x2 - x) / (x2 - x1)
def phi2(x, x1, x2):
    # phi1(x1) = 0, phi1(x2) = 1
    return (x1 - x) / (x1 - x2)



def f_b(x):
    return -2

def f_c(x):
    return -(40000*x**2 - 200)*np.exp(-100*x**2)

def f_d(x):
    return -(4000000*x**2 - 2000)*np.exp(-1000*x**2)

def f_e(x):
    return (2/9)*x**(-4/3)


def u_b(x):
    return x**2

def u_c(x):
    return np.exp(-100*x**2)

def u_d(x):
    return np.exp(-1000*x**2)

def u_e(x):
    return x**(2/3)



System = fem_poisson(a = 0, b = 1,d1 = 0, d2 = 1, N = 20)



#System.check_convergence(iteration = 9, u = u_e, f = f_e)

System.check_amr_convergence_max(iterations = 60,u = u_e,f = f_e)
System.plot_convergence(line_order = 2)
#print("Interpolation L2 error is: ", System.rel_error)
print("N Values: ", System.N)










