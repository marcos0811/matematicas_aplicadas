import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.symbols('x')  # Definimos la variable simbólica con el paquete de sympy
func_sym = sp.sin(x)  # Esta función puede ser reemplazada por cualquier otra por ejm con x**2
derivada_sym = sp.diff(func_sym, x)  # Se calcula la derivada con la función diff

def func(x_val):
    return np.sin(x_val)  # Utilizamos np.sin para trabajar con valores numéricos
def derivada(x_val):
    return float(derivada_sym.subs(x, x_val))  # Evaluamos la derivada simbólica
    
def descenso_gradiente(x0, alpha, e, iter): 
    """ se crea una funcion la cual estara encargada de los calculos de la funcion y de su graficacion""" 
    x_values = [x0] # lista donde se almacenara los datos calculados de los puntos del gradiente
    for i in range(iter):
        grad = derivada(x0)  
        x_new = x0 - alpha * grad # funcion del desenso del gradiente  
        x_values.append(x_new) # agregamos los valores a la lista

        x_range = np.linspace(0, 2 * np.pi, 100)  # Rango para la grafica
        y_values = func(x_range)
        
        plt.plot(x_range, y_values, label='f(x) = sin(x)', color='blue')
        # se crea un lista por comprension para la graficacion de los puntos
        plt.scatter(x_values, [func(val) for val in x_values], color='red', linewidths=5)

        #Configuramos la grafica con el titulo, el nombre de los ejes y la grilla para una mejor visualización 
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Descenso de Gradiente')
        plt.grid()
        plt.legend()
        # El siguiente comadno es opcional ya que no es relevante para los calculos ni la grafica solo para guardar las imagenes
        # plt.savefig(f'grafica_seno_{i}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()  # Limpiamos la figura para la siguiente iteración

        # Verificamos el criterio de convergencia
        if abs(x_new - x0) < e:
            break
        
        x0 = x_new  # Actualizamos los valores de x para la siguiente iteracinn

# Parámetros
alpha = 0.4  # Tasa de aprendizaje(este valor puede variar de 0-1)
x0 = np.pi       # Valor inicial que tomara la funcion
e = 0.01     # Error minimo (este valor se puede cambiar para tener una mejor aproximacion)
iter = 10    # Limite máximo de iteraciones para evitar una divergencia en caso de no econtrar un minimo exacto(cero)

# Ejecutamos el descenso de gradiente
descenso_gradiente(x0, alpha, e, iter)
