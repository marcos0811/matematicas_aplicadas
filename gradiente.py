import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.symbols('x')  # Se define la variable simbólica con el paquete de sympy
func_sym = sp.sin(x)  # Esta función puede ser reemplazada por cualquier otra
derivada_sym = sp.diff(func_sym, x)  # Se calcula la derivada con la función diff

def func(x_val):
    return np.sin(x_val)  # Utilizamos np.sin para valores numéricos

def derivada(x_val):
    return float(derivada_sym.subs(x, x_val))  # Evaluamos la derivada simbólica

# Función de descenso de gradiente
def descenso_gradiente(x0, alpha, e, iter):
    x_values = [x0]

    for i in range(iter):
        grad = derivada(x0)  
        x_new = x0 - alpha * grad  

        x_values.append(x_new) # agregamos los valores a la lista

        x_range = np.linspace(0, 2 * np.pi, 100)  # Rango para la grafica
        y_values = func(x_range)

        plt.plot(x_range, y_values, label='f(x) = sin(x)', color='blue')
        # se crea un lista por comprension para la graficacion de los puntos
        plt.scatter(x_values, [func(val) for val in x_values], color='red', linewidths=5)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Descenso de Gradiente')
        plt.grid()
        plt.legend()
     
        plt.savefig(f'grafica_seno_{i}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()  # Limpiar la figura para la siguiente iteración

        # Verificar el criterio de convergencia
        if abs(x_new - x0) < e:
            print(f'Convergencia alcanzada: x = {x_new:.4f} en {i + 1} iteraciones.')
            break
        
        x0 = x_new  # Actualizar x para la siguiente iteración

    else:
        print(f'Límite máximo de iteraciones alcanzado: x = {x_new:.4f}')

# Parámetros
alpha = 0.4  # Tasa de aprendizaje
x0 = np.pi       # Valor inicial
e = 0.05     # Error mínimo
iter = 10    # Límite máximo de iteraciones

# Ejecutar el descenso de gradiente
descenso_gradiente(x0, alpha, e, iter)
