import math
import torch
import numpy as np
import matplotlib.pyplot as plt

DEBUG = 0


def apply_tnorm_iterative(tnorm, values):
    if values.ndim < 1:
        raise ValueError("Input values must have at least one dimension.")

    leading_shape = values.shape[:-1]
    result = values.select(dim=-1, index=0)
    for i in range(1, values.shape[-1]):
        next_values = values.select(dim=-1, index=i)

        result_list = [tnorm(a, b) for a, b in zip(result.view(-1), next_values.view(-1))]
        result = torch.stack(result_list, axis=-1).view(leading_shape)

        # result = torch.tensor(
        #     [
        #         tnorm(a.item(), b.item())
        #         for a, b in zip(result.view(-1), next_values.view(-1))
        #     ],
        #     device=values.device,
        #     requires_grad=True,
        #     dtype=torch.float,
        # ).view(leading_shape)

    return result

# Fast, multi-variable version of the t-norms


def min_tnorm_tensor(fv):
    return torch.min(fv, axis=-1)[0]

def product_tnorm_tensor(fv):
    return torch.prod(fv, axis=-1)

def lukasiewicz_tnorm_tensor(fv):
    return torch.relu(torch.sum(fv, axis=-1) - fv.size(-1) + 1)

def drastic_tnorm_tensor(fv):
    none_one = fv < 1
    return torch.where(none_one.sum(-1) <= 1, torch.min(fv, axis=-1)[0], 0)


def log_with_base(x, base):
    return torch.log(x) / torch.log(torch.tensor(base, device=x.device))

def min_tnorm(a, b):
    return torch.min(a, b)


def product_tnorm(a, b):
    return a * b


def lukasiewicz_tnorm(a, b):
    return torch.max(torch.zeros_like(a), a + b - 1)


def drastic_tnorm(a, b):
    if a == 1:
        return b
    elif b == 1:
        return a
    else:
        return torch.zeros_like(a)


def hamacherprod_tnorm(a, b):
    if a == 0 and b == 0:
        return torch.zeros_like(a)
    return (a * b) / (a + b - a * b)


def nilpotentmin_tnorm(a, b):
    if a + b > 1:
        return torch.min(a, b)
    return torch.zeros_like(a)


def schweizer_sklar_tnorm(a, b, p=2):
    if p == 0:
        return min_tnorm(a, b)
    inner_value = a**p + b**p - 1
    if inner_value < 0:
        inner_value = torch.zeros_like(a)
    return torch.max(torch.zeros_like(a), inner_value ** (1 / p))


def hamacher_tnorm(a, b, p=2):
    return a * b / (p + (1 - p) * (a + b - a * b))


def frank_tnorm(a, b, p=2):
    if p == 1:
        return product_tnorm(a, b)
    return log_with_base(1 + (p**a - 1) * (p**b - 1) / (p - 1), p)


def yager_tnorm(a, b, p=2):
    if p == 1:
        return lukasiewicz_tnorm(a, b)
    return torch.max(torch.zeros_like(a), 1 - ((1 - a) ** p + (1 - b) ** p) ** (1 / p))


def sugeno_weber_tnorm(a, b, p=1):
    return torch.max(torch.zeros_like(a), (a + b - 1 + p * a * b) / (1 + p))


def dombi_tnorm(a, b, p=1):
    if a == 0 or b == 0:
        return torch.zeros_like(a)
    elif p == 0:
        return drastic_tnorm(a, b)
    return (1 + ((1 - a) / a) ** p + ((1 - b) / b) ** p) ** (-1 / p)


def aczel_alsina_tnorm(a, b, p=1):
    if p == 0 or a == 0 or b == 0:
        return drastic_tnorm(a, b)
    return torch.exp(-((torch.abs(-torch.log(a)) ** p + torch.abs(-torch.log(b)) ** p) ** (1 / p)))


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
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z.T, cmap="twilight")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Plot of {function.__name__}(x, y)")

    plt.show()
    filename = f"{function.__name__}2v.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
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
                Z_values[i, j, k] = apply_tnorm_iterative(
                    function, torch.Tensor([xi, yi, ti])
                )

    # Create 3D scatter data
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
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
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X_filtered, Y_filtered, T_filtered, c=Z_filtered, cmap="twilight", s=10
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Function Value (Z)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("T")
    ax.set_title(f"3D Scatter Plot of {function.__name__}(x, y, t)")

    plt.show()
    filename = f"{function.__name__}_3v.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {filename}")


if DEBUG:

    B, N, V = 3, 2, 2  # Batch size and number of values
    fuzzy_tensor = torch.rand(B, N, V)  # Random tensor with values in [0, 1]

    result_minimum = apply_tnorm_iterative(min_tnorm, fuzzy_tensor)
    result_prod = apply_tnorm_iterative(product_tnorm, fuzzy_tensor)
    result_luk = apply_tnorm_iterative(lukasiewicz_tnorm, fuzzy_tensor)
    result_dras = apply_tnorm_iterative(drastic_tnorm, fuzzy_tensor)
    result_nil = apply_tnorm_iterative(nilpotentmin_tnorm, fuzzy_tensor)
    result_hamprod = apply_tnorm_iterative(hamacherprod_tnorm, fuzzy_tensor)

    print("Input Tensor:")
    print(fuzzy_tensor)
    print("\nResult with Minimum T-norm:")
    print(result_minimum)
    print("\nResult with Product T-norm:")
    print(result_prod)
    print("\nResult with Luk T-norm:")
    print(result_luk)
    print("\nResult with nil T-norm:")
    print(result_nil)
    print("\nResult with hamprod T-norm:")
    print(result_hamprod)
    print("\nResult with drastic T-norm:")
    print(result_dras)

    result_minimum = apply_tnorm_iterative(min_tnorm, fuzzy_tensor)
    result_prod = apply_tnorm_iterative(product_tnorm, fuzzy_tensor)
    result_luk = apply_tnorm_iterative(lukasiewicz_tnorm, fuzzy_tensor)
    result_dras = apply_tnorm_iterative(drastic_tnorm, fuzzy_tensor)
    result_nil = apply_tnorm_iterative(nilpotentmin_tnorm, fuzzy_tensor)
    result_hamprod = apply_tnorm_iterative(hamacherprod_tnorm, fuzzy_tensor)

    print("Input Tensor:")
    print(fuzzy_tensor)
    print("\nResult with Minimum T-norm:")
    print(result_minimum)
    print("\nResult with Product T-norm:")
    print(result_prod)
    print("\nResult with Luk T-norm:")
    print(result_luk)
    print("\nResult with nil T-norm:")
    print(result_nil)
    print("\nResult with hamprod T-norm:")
    print(result_hamprod)
    print("\nResult with drastic T-norm:")
    print(result_dras)

    # plot_2variables(product_tnorm)
    # plot_2variables(min_tnorm)
    # plot_2variables(lukasiewicz_tnorm)
    # plot_2variables(drastic_tnorm)
    # plot_2variables(nilpotentmin_tnorm)
    # plot_2variables(hamacherprod_tnorm)

    # plot_2variables(frank_tnorm)
    # plot_2variables(yager_tnorm)
    # plot_2variables(aczel_alsina_tnorm)
    # plot_2variables(sugeno_weber_tnorm)
    # plot_2variables(dombi_tnorm)
    # plot_2variables(schweizer_sklar_tnorm)
    # plot_2variables(hamacher_tnorm)

    # plot_3variables(product_tnorm)
    # plot_3variables(min_tnorm)
    # plot_3variables(lukasiewicz_tnorm)
    # plot_3variables(drastic_tnorm)
    # plot_3variables(nilpotentmin_tnorm)
    # plot_3variables(hamacherprod_tnorm)

    # plot_3variables(frank_tnorm)
    # plot_3variables(yager_tnorm)
    # plot_3variables(aczel_alsina_tnorm)
    # plot_3variables(sugeno_weber_tnorm)
    # plot_3variables(dombi_tnorm)
    # plot_3variables(schweizer_sklar_tnorm)
    # plot_3variables(hamacher_tnorm)




# Added for checking purposes
def traverse_computation_graph(tensor, indent=0):
    """
    Recursively traverses the computation graph of a PyTorch tensor.
    :param tensor: The tensor whose computation graph is to be explored.
    :param indent: Current level of indentation for pretty printing.
    """
    if tensor.grad_fn is None:
        print(" " * indent + "Leaf tensor (no grad_fn)")
        return

    print(" " * indent + f"{type(tensor.grad_fn).__name__}")
    for next_func, _ in tensor.grad_fn.next_functions:
        if next_func is not None:
            print(" " * (indent + 4) + f"Connected to: {type(next_func).__name__}")
            # Recursively traverse the connected functions
            traverse_computation_graph_from_grad_fn(next_func, indent + 8)

def traverse_computation_graph_from_grad_fn(grad_fn, indent=0):
    """
    Traverses a computation graph starting from a grad_fn node.
    :param grad_fn: A grad_fn node in the computation graph.
    :param indent: Current level of indentation for pretty printing.
    """
    print(" " * indent + f"{type(grad_fn).__name__}")
    for next_func, _ in grad_fn.next_functions:
        if next_func is not None:
            print(" " * (indent + 4) + f"Connected to: {type(next_func).__name__}")
            traverse_computation_graph_from_grad_fn(next_func, indent + 8)



if False:
    input_tensor = torch.tensor([[1.0, 0.5, 0.2, 0.5], [0.7, 1.0, 0.8, 0.1], [1,1,1,0.5], [0.6, 0.6, 0.6, 0.6]], device="cuda:0", requires_grad=True)
    input_tensor = input_tensor * 1
    tnorm1, tnorm1_fast = min_tnorm, min_tnorm_tensor
    tnorm1, tnorm1_fast = product_tnorm, product_tnorm_tensor
    tnorm1, tnorm1_fast = lukasiewicz_tnorm, lukasiewicz_tnorm_tensor
    tnorm1, tnorm1_fast = drastic_tnorm, drastic_tnorm_tensor
    # tnorm1, tnorm1_fast = hamacherprod_tnorm, hamacherprod_tnorm_tensor
    # tnorm1, tnorm1_fast = nilpotentmin_tnorm, nilpotentmin_tnorm_tensor
    print("Input Tensor:")
    print(input_tensor)
    print("\nResult with Iterative T-norm:")
    result = apply_tnorm_iterative(tnorm1, input_tensor)
    print(result)

    traverse_computation_graph(result)

    print("\nResult with Fast T-norm:")
    result = tnorm1_fast(input_tensor)
    print(result)