import math
import torch 
import numpy as np
import matplotlib.pyplot as plt

def apply_tnorm_iterative(tnorm, values):
    if values.ndim < 1:
        raise ValueError("Input values must have at least one dimension.")

    # Get the shape of the tensor except the last dimension
    leading_shape = values.shape[:-1]
    result = values.select(dim=-1, index=0)  # Start with the first value in the last dimension

    # Iteratively apply the t-norm across the last dimension
    for i in range(1, values.shape[-1]):
        # Select the next slice along the last dimension
        next_values = values.select(dim=-1, index=i)
        
        # Apply the t-norm elementwise
        result = torch.tensor(
            [tnorm(a.item(), b.item()) for a, b in zip(result.view(-1), next_values.view(-1))]
        ).view(leading_shape)

    return result

def min_tnorm(a, b):
    return min(a, b)

def product_tnorm(a, b):
    return a * b

def lukasiewicz_tnorm(a, b):
    return max(0, a + b - 1)

def drastic_tnorm(a, b):
    if a == 1:
        return b
    elif b == 1:
        return a
    else:
        return 0

def hamacherprod_tnorm(a, b):
    if a == 0 and b == 0:
        return 0
    return (a * b) / (a + b - a * b)

def nilpotentmin_tnorm(a, b):
    if a + b > 1:
        return min(a, b)
    return 0

def schweizer_sklar_tnorm(a, b, p = 2):
    if p == 0:
        return min_tnorm(a, b)
    inner_value = a**p + b**p - 1
    if inner_value < 0:
        inner_value = 0  
    return max(0, inner_value**(1 / p))

def hamacher_tnorm(a, b, p = 2):
    return a * b / (p + (1 - p) * (a + b - a * b))

def frank_tnorm(a, b, p = 2):
    if p == 1:
        return product_tnorm(a,b)
    return math.log(1 + (p**a - 1) * (p**b - 1) / (p - 1), p)

def yager_tnorm(a, b, p = 2):
    if p == 1:
        return lukasiewicz_tnorm(a,b)
    return max(0, 1 - ((1 - a)**p + (1 - b)**p)**(1 / p))

def sugeno_weber_tnorm(a, b, p = 1):
    return max(0, (a + b - 1 + p * a * b) / (1 + p))

def dombi_tnorm(a, b, p = 1):
    if a == 0 or b == 0:
        return 0
    elif p == 0:
        return drastic_tnorm(a,b)
    return (1 + ((1 - a) / a)**p + ((1 - b) / b)**p)**(-1 / p)

def aczel_alsina_tnorm(a, b, p = 1):
    if p == 0 or a == 0 or b == 0:
        return drastic_tnorm(a,b)
    return math.exp(-((abs(-math.log(a))**p + abs(-math.log(b))**p)**(1 / p)))

def plot_2variables(function, resolution=100):

    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)

    z_values = []
    for xi in x:
        row = []
        for yi in y:
            # row.append(function(torch.Tensor([xi, yi])))  
            row.append(apply_tnorm_iterative(function, torch.Tensor([xi, yi])))  
        z_values.append(row)
        
    Z = np.array(z_values)
    

    X, Y = np.meshgrid(x, y) 
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z.T, cmap='twilight') 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Plot of {function.__name__}(x, y)')

    plt.show()
    filename = f"{function.__name__}2v.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
   
    
def plot_3variables(function, resolution=30):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import torch

    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    t = np.linspace(0, 1, resolution)

    # Initialize Z values
    Z_values = np.zeros((resolution, resolution, resolution))

    # Compute Z values for each combination of x, y, and t
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            for k, ti in enumerate(t):
                Z_values[i, j, k] = apply_tnorm_iterative(function, torch.Tensor([xi, yi, ti]))

    # Create 3D scatter data
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    Z_flat = Z_values.flatten()
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    T_flat = T.flatten()

    # Filter points by a threshold
    threshold = 0
    mask = Z_flat >= threshold
    X_filtered = X_flat[mask]
    Y_filtered = Y_flat[mask]
    T_filtered = T_flat[mask]
    Z_filtered = Z_flat[mask]

    # Plot the results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(X_filtered, Y_filtered, T_filtered, c=Z_filtered, cmap='twilight', s=10)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Function Value (Z)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('T')
    ax.set_title(f'3D Scatter Plot of {function.__name__}(x, y, t)')

    plt.show()
    filename = f"{function.__name__}_3v.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")

