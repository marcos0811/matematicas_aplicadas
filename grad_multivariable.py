import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

# definimos las variables simbólicas que en estes caso trabajaremos con x,y
x_sym, y_sym = sp.symbols('x y')

func_sym = sp.sin(5 * x_sym) + sp.cos(5 * y_sym)

# se calculas las derivadas parciales respecto a x ,y
derivada_x = sp.diff(func_sym, x_sym)
derivada_y = sp.diff(func_sym, y_sym)

# funciones para evaluar la función y sus derivadas numericamente
def func(x_val, y_val):
    return np.sin(5 * x_val) + np.cos(5 * y_val)

def gradiente(x_val, y_val): # en este funcion se crea una lista de las derivadas parciales dx*i + dy*j se lo hace en froma de una lista
    grad_x = float(derivada_x.subs({x_sym: x_val, y_sym: y_val}))
    grad_y = float(derivada_y.subs({x_sym: x_val, y_sym: y_val}))
    return np.array([grad_x, grad_y])

# funcinn de descenso de gradiente
def descenso_gradiente(x0, y0, alpha, er, iter):
    """En esta funcion se procedera hacer los calculos respectivos con la equacion vista en clases """
    points = [(x0, y0)] # lista donde se almacenaran los puntos de los valores calculadis

    for i in range(iter):
        grad = gradiente(x0, y0)  # Evaluamos el gradiente
        # con el uso de la equacion 1 actualzamos los valores de x,y
        x_new = x0 - alpha * grad[0]  # Actualizamos x
        y_new = y0 - alpha * grad[1]  # Actualizamos y

        points.append((x_new, y_new)) # agreagmos los puntos optenimos a la lista

        # Graficamos en una grafica en 3 dimensiones
        x_range = np.linspace(4, 8, 100)
        y_range = np.linspace(4, 8, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = func(X, Y) # se calcula los valores de z para la graficacion

        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.4)
        
        # Usamos func para calcular los valores de Z correspondientes a los puntos
        z_values = [func(x, y) for x, y in points]
        ax.scatter(*zip(*points), z_values, color='red', s=20, alpha=1)# se grafica los puntos en la grafica

        ax.view_init(elev=5, azim=90)  # aqui ajustamos la froma de la grafica para una mejor vizualizacion  

        # Se configura la grafica para visualizarla de una mejor manera
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        plt.savefig(f'grafica_seno_cos{i}.png', dpi=300, bbox_inches='tight')
        plt.show()  # Mostramos la grafica
        
        # verificamos el criterio de convergencia
        if np.linalg.norm([x_new - x0, y_new - y0]) < er:
            break
        
        x0, y0 = x_new, y_new  # actualizamos los valores para la siguiente iteración

alpha = 0.15  # tasa de aprendizaje (0-1)
x0, y0 = 2*np.pi, 2*np.pi  # valores iniciales
er = 0.1  #error relativo 
iter = 10  # limite maximo de iteraciones

# Ejecutamos la funcion
descenso_gradiente(x0, y0, alpha, er, iter)
