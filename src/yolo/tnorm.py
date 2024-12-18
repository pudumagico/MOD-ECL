import torch 
import numpy as np
import matplotlib.pyplot as plt

def apply_tnorm_iterative(tnorm, values):
    # Ensure the input is a tensor with at least one dimension
    if values.ndim < 1:
        raise ValueError("Input values must have at least one dimension.")

    # Move the tensor to CPU and detach if it's on the GPU or requires gradients
    if values.is_cuda or values.requires_grad:
        values = values.detach().cpu()

    # Initialize the result with the first element along the last dimension
    result = values.select(dim=-1, index=0)  # Select the first slice along the last dimension

    # Iterate over the remaining elements along the last dimension and apply the t-norm
    for i in range(1, values.shape[-1]):
        result = torch.tensor([tnorm(a.item(), b.item()) for a, b in zip(result, values.select(dim=-1, index=i))])

    return result

def min_tnorm(a, b):
    return min(a, b)

def product_tnorm(a, b):
    return a * b

def lukaziewics_tnorm(a, b):
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
    threshold = 0
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

