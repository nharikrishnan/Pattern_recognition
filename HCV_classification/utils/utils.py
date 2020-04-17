#utils functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import sympy
from sympy import solve
from sympy import Symbol

def side_by_side(*objs, **kwds):
    ''' Une function to print objects side by side '''
    from pandas.io.formats.printing import adjoin
    space = kwds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    print(adjoin(space, *reprs))
    
def create_df(mat1, mat2):
    df1 = pd.DataFrame(mat1)
    df2 = pd.DataFrame(mat2)
    df1.columns = ['X1', 'X2', 'X3']
    df1['Class']= 0
    df2.columns = ['X1', 'X2', 'X3']
    df2['Class'] = 1
    df = df1.append(df2)
    return df
 
#function for plotting using seaborn
def plot_class(df, 
               cols,
               title):
    ln = sns.lmplot(cols[0], cols[1], df, hue='Class', fit_reg=False,scatter_kws={"s": 100})
    fig = plot.gcf()
    fig.set_size_inches(8, 4)
    plot.title(title)
    axes = ln.axes
    #axes[0,0].set_ylim(-20, 50)
    #axes[0,0].set_xlim(-20, 50)
    plot.show()

#plotting using scatter
def plot_2(df):
    feature = df.to_numpy().T
    plot.scatter(feature[1], feature[2],alpha=1.5,s=100,
            c=df['Class'].to_numpy(), cmap='viridis')
            
#functions for generating covariances
def generate_covariance_mat(a,
                            b,
                            c,
                            alpha,
                            beta):
    cov1 = [[round(a*a,3),round(alpha*a*b,3), round(alpha*a*c,3) ], 
            [round(alpha*a*b,3), round(b*b,3), round(beta*b*c,3)], 
            [round(alpha*a*c,3), round(beta*b*c,3),(c*c)]]
    cov2 = [[round(c*c,3), round(beta*b*c,2), round(beta*a*c,3)],
            [round(beta*b*c,3), round(b*b,2),round(alpha*a*b,3)], 
            [round(beta*a*c,3), round(alpha*a*b,3), round(a*a,3)]]
    return np.array(cov1), np.array(cov2)


#simultaneous diagonalizing matrix for covariance
def generate_daig_variance_mean(
                                c1,
                                c2,
                                mean_vector1,
                                mean_vector2,
                                return_value,
                                trans = 'v'
                               ):
    
    #covariances
    val_c1, vec_c1 = np.linalg.eig(c1)
    vec_t_c1 = vec_c1.T
    diag_inv_val = daig_val = np.diag(1. / np.sqrt(val_c1))
    c1_y = np.matmul(np.matmul(vec_t_c1, c1), vec_c1)
    c2_y = np.matmul(np.matmul(vec_t_c1, c2), vec_c1)
    c1_z = np.matmul(np.matmul(diag_inv_val, c1_y), diag_inv_val)
    c2_z = np.matmul(np.matmul(diag_inv_val, c2_y), diag_inv_val)
    val_c2_z, vec_c2_z = np.linalg.eig(c2_z)
    c1_v = np.matmul(np.matmul(vec_c2_z.T, c1_z), vec_c2_z)
    c2_v = np.matmul(np.matmul(vec_c2_z.T, c2_z), vec_c2_z)

    # means
    mean1_y = np.matmul( mean_vector1, vec_t_c1)
    mean2_y = np.matmul( mean_vector2, vec_t_c1)
    mean1_z = np.matmul( mean1_y, diag_inv_val)
    mean2_z = np.matmul( mean2_y, diag_inv_val)                                
    mean1_v = np.matmul( mean1_z,vec_c2_z)
    mean2_v = np.matmul( mean2_z, vec_c2_z)

    if return_value.lower() == 'covar':    
        if trans.lower() == 'v':
            return c1_v, c2_v
        if trans.lower() =='y':
            return c1_y, c2_y
        if trans.lower() =='z':
            return c1_z, c2_z
    elif return_value.lower() == 'mean':
        if trans.lower() == 'v':
            return mean1_v, mean2_v
        if trans.lower() =='y':
            return mean1_z, mean2_z
        if trans.lower() =='z':
            return mean1_y, mean2_y

# def generate_daig_mean(c1,
#                        c2,
#                        mean1, 
#                        mean2,
#                        trans = 'v'):
#     val_c1, vec_c1 = np.linalg.eig(c1)
#     vec_t_c1 = vec_c1.T
#     diag_inv_val = daig_val = np.diag(1. / np.sqrt(val_c1)
#     mean1_y = np.mea
#     mean2_y = 
#     mean1_z =
#     mean2_z =
#     mean1_v =                                
#     mean2_v =                                
                                      
def plot_descriminant_x1_x2(A,
                            B, 
                            C,
                            matrix1,
                            matrix2, 
                            start,
                            end, 
                            solve_for = 'X',
                            z = 2,
                            root = 1):
    x = Symbol('x')
    y = Symbol('y')
    if solve_for.lower() == 'x':
        print('X')
        print(solve_for)
        solution = solve((A[0][0]*(x**2) +
               A[1][1]*(y**2) + 
               (A[0][1] +A[1][0])*x*y + 
               (z*A[0][2] + z*A[2][0] + B[0])*x + 
               (z*A[1][2] + z*A[2][1] +B[1])*y + 
               z*B[2] +A[2][2]*(z**2) +C), y,x)
        X = np.linspace(start,end, 100)
        if root == 1:
            Y = [solution[0][0].subs(x,xx) for xx in X]
        else:
            Y = [solution[1][0].subs(x, xx) for xx in X]
    else:
        print('Y')
        print(solve_for)
        solution = solve((A[0][0]*(x**2) +
               A[1][1]*(y**2) + 
               (A[0][1] +A[1][0])*x*y + 
               (z*A[0][2] + z*A[2][0] + B[0])*x + 
               (z*A[1][2] + z*A[2][1] +B[1])*y + 
               z*B[2] +A[2][2]*(z**2) +C), x, y)
        Y = np.linspace(start,end, 100)
        if root == 1:
            X = [solution[0][0].subs(y,yy) for yy in Y]
        else:
            X = [solution[1][0].subs(y, yy) for yy in Y]      
    plot.style = 'seaborn-notebook'
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.plot(X,Y, c ='black')
#    ax.set_xlim(-15, 25)
#    ax.set_ylim(-15, 40)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('X1-X2')
    plot.xlabel('X1')
    plot.ylabel('X2')
    ax.scatter(matrix1.T[0], matrix1.T[1])
    ax.scatter(matrix2.T[0], matrix2.T[2])

def contour_descriminant(A,
                            B,
                            C, 
                            matrix1, 
                            matrix2,
                            col,
                            label,
                            title,
                            xlim,
                            ylim,
                            start,
                            end):
    """
        Function to plot discriminant using contour
    """
    x = np.linspace(start, end, 100)
    y = np.linspace(start, end, 100)
    X, Y = np.meshgrid(x, y)
    Z = func_calculate_x1_x2(X, Y, A, B, C)
    plot.style = 'seaborn-notebook'
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-20, 15)
    ax.set_ylim(-20, 15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(title)
    plot.xlabel(label[0])
    plot.ylabel(label[1])
    ax.scatter(matrix1.T[col[0]], matrix1.T[col[1]], alpha = .5)
    ax.scatter(matrix2.T[col[0]], matrix2.T[col[1]], alpha = .5)
    ax.contour(X, Y, Z, 0, colors = 'black')
    plot.xlim(xlim[0], xlim[1])
    plot.ylim(ylim[0], ylim[1])

def plot_descriminant_x2_x3(A, 
                            B, 
                            C, 
                            matrix1,
                            matrix2,
                            start, 
                            end, 
                            solve_for = 'Y',
                            x = 2,
                            root = 1):
    y = Symbol('y')
    z = Symbol('z')
    if solve_for.lower() != 'y':
        print('Z')
        print(solve_for)
        solution = solve((A[0][0]*(x**2) +
               A[1][1]*(y**2) + 
               (A[0][1] +A[1][0])*x*y + 
               (z*A[0][2] + z*A[2][0] + B[0])*x + 
               (z*A[1][2] + z*A[2][1] +B[1])*y + 
               z*B[2] +A[2][2]*(z**2) +C), y,z)
        Z = np.linspace(start,end, 100)
        if root == 1:
            Y = [solution[0][0].subs(z,zz) for zz in Z]
        else:
            Y = [solution[1][0].subs(z, zz) for zz in Z]
    else:
        print('Y')
        print(solve_for)
        solution = solve((A[0][0]*(x**2) +
               A[1][1]*(y**2) + 
               (A[0][1] +A[1][0])*x*y + 
               (z*A[0][2] + z*A[2][0] + B[0])*x + 
               (z*A[1][2] + z*A[2][1] +B[1])*y + 
               z*B[2] +A[2][2]*(z**2) +C), z, y)
        Y = np.linspace(start,end, 100)
        if root == 1:
            Z = [solution[0][0].subs(y,yy) for yy in Y]
        else:
            Z = [solution[1][0].subs(y, yy) for yy in Y]      
    plot.style = 'seaborn-notebook'
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.plot(Y,Z, c ='black')
#    ax.set_xlim(-15, 25)
#    ax.set_ylim(-15, 40)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('X2-X3')
    plot.xlabel('X2')
    plot.ylabel('X3')
    ax.scatter(matrix1.T[0], matrix1.T[1])
    ax.scatter(matrix2.T[0], matrix2.T[2])
    
def func_calculate_x1_x2(x, y, A, B, C):
    k = (A[0][0]*(x**2) +
           A[1][1]*(y**2) + 
           (A[0][1] +A[1][0])*x*y + 
           (2*A[0][2] + 2*A[2][0] + B[0])*x + 
           (2*A[1][2] + 2*A[2][1] +B[1])*y + 
           2*B[2] +A[2][2]*(4) +C)
    return k

def contour_descriminant_x1_x2(A,
                               B,
                               C, 
                               matrix1, 
                               matrix2,
                               start,
                               end):
    """
        Function to plot discriminant using contour
    """
    x = np.linspace(start, end, 100)
    y = np.linspace(start, end, 100)
    X, Y = np.meshgrid(x, y)
    Z = func_calculate_x1_x2(X, Y, A, B, C)
    plot.style = 'seaborn-notebook'
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-20, 15)
    ax.set_ylim(-20, 15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('X1-X2')
    plot.xlabel('X1')
    plot.ylabel('X2')
    ax.scatter(matrix1.T[0], matrix1.T[1])
    ax.scatter(matrix2.T[0], matrix2.T[1])
    ax.contour(X, Y, Z, 0, colors = 'black')

def func_calculate_x2_x3(y,
                         z,
                         A,
                         B,
                         C):
    k = (A[0][0]*(2**2) +
           A[1][1]*(y**2) + 
           (A[0][1] +A[1][0])*2*y + 
           (z*A[0][2] + z*A[2][0] + B[0])*2 + 
           (z*A[1][2] + z*A[2][1] +B[1])*y + 
           z*B[2] +A[2][2]*(z**2) +C)
    return k

def contour_descriminant_x2_x3(A, 
                               B,
                               C,
                               matrix1,
                               matrix2,
                               start,
                               end):
    y = np.linspace(start, end, 100)
    z = np.linspace(start, end, 100)
    Y, Z = np.meshgrid(y, z)
    X = func_calculate_x2_x3(Y, Z, A, B, C)
    plot.style = 'seaborn-notebook'
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-20, 15)
    ax.set_ylim(-20, 15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('X2-X3')
    plot.xlabel('X2')
    plot.ylabel('X3')
    ax.scatter(matrix1.T[1], matrix1.T[2])
    ax.scatter(matrix2.T[1], matrix2.T[2])
    ax.contour(Y, Z, X, 0, colors = 'black')

    # function to predict accuracy
def accuracy(matrix_predicted,
             c):
    count_class1 = 0
    for i in range(len(matrix_predicted.tolist())):
        if int(matrix_predicted.tolist()[i][3]) == c:
            count_class1 = count_class1 +1
    return (count_class1/200)*100


def print_intermediate_cov(covar_a_1, covar_a_2):
    print('covarience_x1')
    
    print(np.round(covar_a_1,4))
    
    print('covarience_x2')
    
    print(np.round(covar_a_2,4))
    covar_a1_y, covar_a2_y = generate_daig_covar(covar_a_1, covar_a_2, trans='Y')
    covar_a1_z, covar_a2_z = generate_daig_covar(covar_a_1, covar_a_2, trans='Z')
    covar_a1_v, covar_a2_v = generate_daig_covar(covar_a_1, covar_a_2, trans='V')
    
    
    print('\n'+'Transformation Y ')
    
    print('\n')
    
    print('covarience_x1_y')
    
    print(np.round(covar_a1_y,4))
    
    print('covarience_x2_y')
    
    print(np.round(covar_a2_y,4))

    print('\n'+'Transformation Z ')
    
    print('\n')
    
    print('covarience_x1_z')
    
    print(np.round(covar_a1_z,4))
    
    print('covarience_x2_z')
    
    print(np.round(covar_a2_z,4))
    
    print('\n'+'Transformation V')
    
    print('\n')
    
    print('covarience_x1_v')
    
    print(np.round(covar_a1_v,4))
    
    print('covarience_x2_v')
    
    print(np.round(covar_a2_v,4))

