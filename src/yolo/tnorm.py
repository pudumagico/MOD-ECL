import torch 
import numpy as np
import matplotlib.pyplot as plt
# T Norms
def product_tnorm(fuzzy_values, req_ind=None):
    return torch.prod(fuzzy_values)

def lukaziewics_tnorm(fuzzy_values, req_ind=None):
    return max(0, torch.sum(fuzzy_values)-len(fuzzy_values) + 1)

def minimum_tnorm(fuzzy_values, req_ind=None):
    # print('fz', fuzzy_values, 'min', torch.min(fuzzy_values),'minaxis', torch.min(fuzzy_values, axis=-1))
    return torch.min(fuzzy_values)

def drastic_tnorm(fuzzy_values, req_ind=None):
    if torch.any(fuzzy_values == 1):
        return torch.min(fuzzy_values)
    else:
        return 0

def nilpotentmin_tnorm(fuzzy_values, red_ind=None):
    sum_pred_const = torch.sum(fuzzy_values)
    return torch.where(sum_pred_const > 1, torch.min(fuzzy_values), torch.zeros_like(sum_pred_const))

def hamacherprod_tnorm(fuzzy_values, red_ind=None):
    if torch.all(fuzzy_values == 0):
        return 0
    else:
        sum_pred_const = torch.sum(fuzzy_values, axis=-1)
        prod_pred_const = torch.prod(fuzzy_values, axis=-1)
        return prod_pred_const / (sum_pred_const - prod_pred_const)

# Parametric T Norms

def hamacher_tnorm(fuzzy_values, p, red_ind=None):
    if torch.isinf(p):
        return drastic_tnorm(fuzzy_values)
    elif torch.all(fuzzy_values) == 0 and p == 0:
        return 0
    else:
        sum_pred_const = torch.sum(fuzzy_values, axis=-1)
        prod_pred_const = torch.prod(fuzzy_values, axis=-1)
        return prod_pred_const / (p + (1 - p) * (sum_pred_const - prod_pred_const))
    
def frank_tnorm(fuzzy_values, p, red_ind=None):
    frank_prod = torch.pow(p, fuzzy_values) - 1
    return torch.log(1 + torch.prod(frank_prod, axis=-1) / (p - 1), p)

# def sugenoweber_tnorm(fuzzy_values, p, red_ind=None):
#     if p == -1:
#         return drastic_tnorm(fuzzy_values)
#     elif p > -1 and not torch.isinf(p):
        
#         sum_pred_const = torch.sum(fuzzy_values, axis=-1)
#         prod_pred_const = torch.prod(fuzzy_values, axis=-1)
        
#         return max(0, (sum_pred_const ) / () )

def plot_2variables(function, resolution=100):

    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)

    # Step 3: Compute z values using loops
    z_values = []
    for xi in x:
        row = []
        for yi in y:
            row.append(function(torch.Tensor([xi, yi])))  # Call the function with single values
        z_values.append(row)
        
    # Convert to a numpy array for easier handling
    Z = np.array(z_values)
    

    # Step 4: Visualize the (x, y, z) graph
    X, Y = np.meshgrid(x, y)  # Create the grid for plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z.T, cmap='cividis')  # Use Z.T to match array orientation

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of f(x, y)')

    # Show plot
    plt.show()
     # Save the plot with the function name
    filename = f"{function.__name__}2v.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
   
    
def plot_3variables(function):
    # Step 2: Create grids for x, y, and t values
# Step 2: Create grids for x, y, and t values
    resolution = 20  # Adjust for desired resolution
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    t = np.linspace(0, 1, resolution)

    # Step 3: Compute z values for all combinations of x, y, and t
    X, Y, T = np.meshgrid(x, y, t)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    T_flat = T.flatten()
    Z_flat = np.array([function(torch.Tensor([xi, yi, ti])) for xi, yi, ti in zip(X_flat, Y_flat, T_flat)])

    # Step 4: Filter points where Z < 0.1
    threshold = 0.1
    mask = Z_flat >= threshold  # Keep only points with Z >= 0.1
    X_filtered = X_flat[mask]
    Y_filtered = Y_flat[mask]
    T_filtered = T_flat[mask]
    Z_filtered = Z_flat[mask]

    # Step 5: Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot: X, Y, T as positions; Z as color
    scatter = ax.scatter(X_filtered, Y_filtered, T_filtered, c=Z_filtered, cmap='cividis', s=10)

    # Add color bar to show the function values
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Function Value (Z)')

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('T')
    ax.set_title('3D Scatter Plot of f(x, y, t) (Z >= 0.1)')

    # Show plot
    plt.show()
    # Save the plot with the function name
    filename = f"{function.__name__}3v.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
