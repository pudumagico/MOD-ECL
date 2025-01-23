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

def apply_tnorm_iterative2(tnorm, values):
    if values.ndim < 1:
        raise ValueError("Input values must have at least one dimension.")

    # Initialize the result with the first element along the last dimension
    result = values.select(dim=-1, index=0)

    # Iteratively apply the t-norm across the remaining elements
    for i in range(1, values.shape[-1]):
        next_values = values.select(dim=-1, index=i)
        result = tnorm(result, next_values)

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

def drastic_tnorm_batch(a, b):
    none_one = (a == 1) | (b == 1)
    return torch.where(none_one, torch.min(a, b), torch.zeros_like(a))


def hamacherprod_tnorm(a, b):
    if a == 0 and b == 0:
        return torch.zeros_like(a)
    return (a * b) / (a + b - a * b)


def hamacherprod_tnorm_batch(a, b):
    return torch.where(a + b == 0, torch.zeros_like(a), (a * b) / torch.clamp(a + b - a * b, min=1e-6, max=1))

def nilpotentmin_tnorm(a, b):
    if a + b > 1:
        return torch.min(a, b)
    return torch.zeros_like(a)


def nilpotentmin_tnorm_batch(a, b):
    return torch.where(a + b > 1, torch.min(a, b), torch.zeros_like(a))

def schweizer_sklar_tnorm(a, b, p=2):
    if p == 0:
        return min_tnorm(a, b)
    inner_value = a**p + b**p - 1
    if inner_value < 0:
        inner_value = torch.zeros_like(a)
    return torch.max(torch.zeros_like(a), inner_value ** (1 / p))

# Fixed
def schweizer_sklar_tnorm_batch(a, b, p=2):
    if p == 0:
        return min_tnorm(a, b)
    inner_value = a**p + b**p - 1
    return torch.relu(inner_value) ** (1 / p)


def hamacher_tnorm(a, b, p=2):
    return a * b / (p + (1 - p) * (a + b - a * b))


def frank_tnorm(a, b, p=2):
    if p == 1:
        return product_tnorm(a, b)
    return log_with_base(1 + (p**a - 1) * (p**b - 1) / (p - 1), p)


def yager_tnorm(a, b, p=5):
    if p == 1:
        return lukasiewicz_tnorm(a, b)
    return torch.max(torch.zeros_like(a), 1 - ((1 - a) ** p + (1 - b) ** p) ** (1 / p))


# Fixed
def yager_tnorm_batch(a, b, p=2):
    if p == 1:
        return lukasiewicz_tnorm(a, b)
    return torch.relu(1 - ((1 - a) ** p + (1 - b) ** p)) ** (1 / p)
    # return torch.max(torch.zeros_like(a), 1 - torch.pow(torch.clamp(torch.pow(1 - a, p) + torch.pow(1 - b, p), min=0, max=1), (1 / p)))

def sugeno_weber_tnorm(a, b, p=1):
    return torch.max(torch.zeros_like(a), (a + b - 1 + p * a * b) / (1 + p))


# Dombi T-Norm is not working as of now!
# def dombi_tnorm(a, b, p=2):
#     if a == 0 or b == 0:
#         return torch.zeros_like(a)
#     elif p == 0:
#         return drastic_tnorm(a, b)
#     return 1/(1 + (((1 - a) / a) ** p + ((1 - b) / b) ** p) ** (1 / p))


# def dombi_tnorm_batch(a, b, p=2):
#     if p == 0:
#         return drastic_tnorm_batch(a, b)
#     none_zero = (a >= 1e-6) & (b >= 1e-6)
#     # a, b = torch.clamp(a, min=1e-2, max=1-1e-2), torch.clamp(b, min=1e-2, max=1-1e-2)
#     return torch.where(none_zero, 1/(1 + (((1 - a) / a) ** p + ((1 - b) / b) ** p) ** (1 / p)), (a+b) * 0)

def aczel_alsina_tnorm(a, b, p=1):
    if p == 0 or a == 0 or b == 0:
        return drastic_tnorm(a, b)
    return torch.exp(-((torch.abs(-torch.log(a)) ** p + torch.abs(-torch.log(b)) ** p) ** (1 / p)))

# Fixed
def aczel_alsina_tnorm_batch(a, b, p=1):
    if p == 0:
        return drastic_tnorm_batch(a, b)
    none_zero = (a >= 1e-6) & (b >= 1e-6)
    a, b = torch.clamp(a, min=1e-6), torch.clamp(b, min=1e-6)
    return torch.where(none_zero, torch.exp(-( (torch.abs(-torch.log(a)) ** p + torch.abs(-torch.log(b)) ** p) ** (1 / p))), torch.zeros_like(a))

def drastic_tnorm_tensor(fv):
    # The drastic t-norm operates on all elements along the last dimension.
    # If any element is 1, return the minimum of the rest.
    # If no element is 1, return 0.

    # Compute the max value along the last dimension.
    max_values = torch.max(fv, dim=-1)[0]
    
    # Apply the drastic t-norm logic:
    drastic_result = torch.where(
        max_values == 1, 
        torch.min(fv, dim=-1)[0],  # If any value is 1, return the minimum of the last dimension
        torch.zeros(fv.size()[:-1], device=fv.device)  # Otherwise, return 0, collapsed over the last dimension
    )
    return drastic_result

def hamacherprod_tnorm_tensor(fv):
    # The Hamacher product t-norm operates on all pairs along the last dimension.
    # Compute the generalized Hamacher product for the last dimension.
    sum_fv = torch.sum(fv, dim=-1)
    prod_fv = torch.prod(fv, dim=-1)
    denominator = sum_fv - prod_fv

    return torch.where(
        denominator == 0, 
        torch.zeros_like(denominator),  # If denominator is 0, return 0
        prod_fv / denominator  # Otherwise, compute the t-norm
    )

def nilpotentmin_tnorm_tensor(fv):
    # Compute pairwise sums and check the condition along the last dimension
    pairwise_sums = torch.sum(fv, dim=-1)  # Sum of all values along the last dimension
    min_values = torch.min(fv, dim=-1)[0]  # Minimum value along the last dimension
    
    # Apply the nilpotent minimum condition
    return torch.where(pairwise_sums > 1, min_values, torch.zeros_like(min_values))

def schweizer_sklar_tnorm_tensor(fv, p=2):
    # Handle the case where p == 0 by applying the min t-norm
    if p == 0:
        return torch.min(fv, dim=-1)[0]

    # Compute the Schweizer-Sklar t-norm for the given tensor
    pth_powers = fv**p
    inner_value = torch.sum(pth_powers, dim=-1) - fv.size(-1)  # Sum a^p + b^p - 1 along the last dimension
    
    # Apply max(0, inner_value) and take the p-th root
    return torch.pow(torch.maximum(inner_value, torch.zeros_like(inner_value)), 1 / p)

def hamacher_tnorm_tensor(fv, p=2):
    # Compute pairwise sum, product, and the Hamacher t-norm for the tensor along the last dimension
    sum_fv = torch.sum(fv, dim=-1)
    prod_fv = torch.prod(fv, dim=-1)
    
    # Calculate the denominator (p + (1 - p) * (sum - product))
    denominator = p + (1 - p) * (sum_fv - prod_fv)
    
    # Apply the Hamacher t-norm formula
    return prod_fv / denominator

def frank_tnorm_tensor(fv, p=2):
    if p == 1:
        # Use the product t-norm if p == 1
        return product_tnorm_tensor(fv)
    
    # Compute Frank t-norm for each pair of values in the last dimension
    p_a = torch.pow(p, fv)
    frank_values = (p_a - 1) * (torch.pow(p, fv) - 1) / (p - 1)
    
    # Apply the Frank T-norm formula
    result = torch.log(1 + frank_values) / torch.log(torch.tensor(p, device=fv.device))
    
    # Reduce across the last dimension to match the iterative output
    result = torch.prod(result, dim=-1)
    
    return result

import torch

def yager_tnorm_tensor(fv, p=2):
    if p == 1:
        # Use Lukasiewicz t-norm if p == 1
        return torch.relu(torch.sum(fv, dim=-1) - fv.size(-1) + 1)

    # Compute the Yager t-norm for each pair of values in the last dimension
    one_minus_fv = 1 - fv
    pth_powers = torch.pow(one_minus_fv, p)
    inner_value = torch.sum(pth_powers, dim=-1)

    # Apply the Yager t-norm formula
    return torch.maximum(torch.zeros_like(inner_value), 1 - torch.pow(inner_value, 1 / p))


def sugeno_weber_tnorm_tensor(fv, p=1):
    # Compute the Sugeno-Weber t-norm for each pair of values in the last dimension
    sum_fv = torch.sum(fv, dim=-1)
    prod_fv = torch.prod(fv, dim=-1)
    
    # Apply the Sugeno-Weber t-norm formula
    return torch.maximum(torch.zeros_like(sum_fv), (sum_fv - 1 + p * prod_fv) / (1 + p))

def dombi_tnorm_tensor(fv, p=1):
    # Handle the case where any value is 0 (resulting in 0)
    zero_mask = (fv == 0)
    if torch.any(zero_mask):
        return torch.zeros_like(fv[..., 0])

    # Handle the case where p == 0 (use drastic t-norm)
    if p == 0:
        return drastic_tnorm_tensor(fv)

    # Compute the Dombi t-norm for each pair of values in the last dimension
    one_minus_fv = 1 - fv
    inv_fv = one_minus_fv / fv
    dombi_values = (1 + torch.pow(inv_fv, p)) ** (-1 / p)

    # Reduce across the last dimension to match the iterative output
    result = torch.prod(dombi_values, dim=-1)

    return result

def aczel_alsina_tnorm_tensor(fv, p=1):
    # Handle the case where p == 0 or any value is 0 (use drastic t-norm)
    zero_mask = (fv == 0)
    
    if torch.any(zero_mask) or p == 0:
        return drastic_tnorm_tensor(fv)
    
    # Compute the Aczel-Alcina t-norm for each pair of values in the last dimension
    log_fv = -torch.log(fv.abs())
    aczel_alsina_values = torch.exp(-torch.pow(torch.abs(log_fv), p).sum(dim=-1) ** (1 / p))
    
    return aczel_alsina_values


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
    for i in range(1):
        B, N, V = 4, 4, 4  # Batch size and number of values
        fuzzy_tensor = torch.rand(B, V)
        print(fuzzy_tensor)
        # Random tensor with values in [0, 1]
        # fuzzy_tensor = torch.tensor([[1.0, 0.5, 0.2, 0.5], [0.7, 1.0, 0.8, 0.1], [1,1,1,0.5], [0.6, 0.6, 0.6, 0.6]], device="cuda:0", requires_grad=True)
        # fuzzy_tensor = fuzzy_tensor * 1
        # result_minimum = apply_tnorm_iterative(min_tnorm, fuzzy_tensor)
        # result_prod = apply_tnorm_iterative(product_tnorm, fuzzy_tensor)
        # result_luk = apply_tnorm_iterative(lukasiewicz_tnorm, fuzzy_tensor)
        # result_dras = apply_tnorm_iterative(drastic_tnorm, fuzzy_tensor)
        # result_nil = apply_tnorm_iterative(nilpotentmin_tnorm, fuzzy_tensor)
        # result_hamprod = apply_tnorm_iterative(hamacherprod_tnorm, fuzzy_tensor)

        # print("Input Tensor:")
        # print(fuzzy_tensor)
        # print("\nResult with Minimum T-norm:")
        # print(result_minimum)
        # print("\nResult with Product T-norm:")
        # print(result_prod)
        # print("\nResult with Luk T-norm:")
        # print(result_luk)
        # print("\nResult with nil T-norm:")
        # print(result_nil)
        # print("\nResult with hamprod T-norm:")
        # print(result_hamprod)
        # print("\nResult with drastic T-norm:")
        # print(result_dras)

        # result_minimum = apply_tnorm_iterative(min_tnorm, fuzzy_tensor)
        # result_prod = apply_tnorm_iterative(product_tnorm, fuzzy_tensor)
        # result_luk = apply_tnorm_iterative(lukasiewicz_tnorm, fuzzy_tensor)
        # result_dras = apply_tnorm_iterative(drastic_tnorm, fuzzy_tensor)
        # result_nil = apply_tnorm_iterative(nilpotentmin_tnorm, fuzzy_tensor)
        result_hamprod_new = apply_tnorm_iterative2(hamacherprod_tnorm_batch, fuzzy_tensor)
        result_hamprod_old = apply_tnorm_iterative(hamacherprod_tnorm, fuzzy_tensor)
        
        print('hamacherprod_tnorm_batch', result_hamprod_new)
        print('hamacherprod_tnorm', result_hamprod_old)

        # print("Input Tensor:")
        # print(fuzzy_tensor)
        # print("\nResult with Minimum T-norm:")
        # print(result_minimum)
        # print("\nResult with Product T-norm:")
        # print(result_prod)
        # print("\nResult with Luk T-norm:")
        # print(result_luk)
        # print("\nResult with nil T-norm:")
        # print(result_nil)
        # print("\nResult with hamprod T-norm:")
        # print(result_hamprod)
        # print("\nResult with drastic T-norm:")
        # print(result_dras)



    
        
        # result_minimum_tensor = min_tnorm_tensor(fuzzy_tensor)
        # result_prod_tensor = product_tnorm_tensor(fuzzy_tensor)
        # result_luk_tensor = lukasiewicz_tnorm_tensor(fuzzy_tensor)
        # result_dras_tensor = drastic_tnorm_tensor(fuzzy_tensor)
        # result_nil_tensor = nilpotentmin_tnorm_tensor(fuzzy_tensor)
        result_hamprod_tensor = hamacherprod_tnorm_tensor(fuzzy_tensor)
        print('hamacherprod_tnorm_tensor', result_hamprod_tensor)

        # print("Input Tensor:")
        # print(fuzzy_tensor)
        # print("\nResult with Minimum T-norm:")
        # print(result_minimum_tensor)
        # print("\nResult with Product T-norm:")
        # print(result_prod_tensor)
        # print("\nResult with Luk T-norm:")
        # print(result_luk_tensor)
        # print("\nResult with nil T-norm:")
        # print(result_nil_tensor)
        # print("\nResult with hamprod T-norm:")
        # print(result_hamprod_tensor)
        # print("\nResult with drastic T-norm:")
        # print(result_dras_tensor)
        
        # print("EQUALITY CHECK")
        

        # print("\nResult with Minimum T-norm:")
        # print(result_minimum_tensor == result_minimum)
        # print("\nResult with Product T-norm:")
        # print(result_prod_tensor == result_prod)
        # print("\nResult with Luk T-norm:")
        # comparison = np.allclose(result_luk_tensor.cpu().detach().numpy(), result_luk.cpu().detach().numpy())
        # print(comparison)
        # print("\nResult with nil T-norm:")
        # print(result_nil_tensor == result_nil)
        print("\nResult with hamprod T-norm:")
        print(result_hamprod_tensor == result_hamprod_new)
        print(result_hamprod_tensor == result_hamprod_old)
        print(result_hamprod_new == result_hamprod_old)
        # print("\nResult with drastic T-norm:")
        # print(result_dras_tensor == result_dras)
        
        
        
        # # Schweitzer-Sklar T-norm
        # result_schweizer_sklar = apply_tnorm_iterative(lambda a, b: schweizer_sklar_tnorm(a, b, p=2), fuzzy_tensor)
        # result_schweizer_sklar_tensor = schweizer_sklar_tnorm_tensor(fuzzy_tensor, p=2)
        # print("\nResult with Schweizer-Sklar T-norm (Iterative):")
        # print(result_schweizer_sklar)
        # print("\nResult with Schweizer-Sklar T-norm (Tensor):")
        # print(result_schweizer_sklar_tensor)
        # print("\nEquality Check for Schweizer-Sklar T-norm:")
        # print(np.allclose(result_schweizer_sklar_tensor.cpu().detach().numpy(), result_schweizer_sklar.cpu().detach().numpy()))

        # # Hamacher T-norm
        # result_hamacher = apply_tnorm_iterative(lambda a, b: hamacher_tnorm(a, b, p=2), fuzzy_tensor)
        # result_hamacher_tensor = hamacher_tnorm_tensor(fuzzy_tensor, p=2)
        # print("\nResult with Hamacher T-norm (Iterative):")
        # print(result_hamacher)
        # print("\nResult with Hamacher T-norm (Tensor):")
        # print(result_hamacher_tensor)
        # print("\nEquality Check for Hamacher T-norm:")
        # print(np.allclose(result_hamacher_tensor.cpu().detach().numpy(), result_hamacher.cpu().detach().numpy()))

        # # Frank T-norm
        # result_frank = apply_tnorm_iterative(lambda a, b: frank_tnorm(a, b, p=2), fuzzy_tensor)
        # result_frank_tensor = frank_tnorm_tensor(fuzzy_tensor, p=2)
        # print("\nResult with Frank T-norm (Iterative):")
        # print(result_frank)
        # print("\nResult with Frank T-norm (Tensor):")
        # print(result_frank_tensor)
        # print("\nEquality Check for Frank T-norm:")
        # print(result_frank_tensor == result_frank)

        # # Yager T-norm
        # result_yager = apply_tnorm_iterative(lambda a, b: yager_tnorm(a, b, p=2), fuzzy_tensor)
        # result_yager_tensor = yager_tnorm_tensor(fuzzy_tensor, p=2)
        # print("\nResult with Yager T-norm (Iterative):")
        # print(result_yager)
        # print("\nResult with Yager T-norm (Tensor):")
        # print(result_yager_tensor)
        # print("\nEquality Check for Yager T-norm:")
        # print(np.allclose(result_yager_tensor.cpu().detach().numpy(), result_yager.cpu().detach().numpy()))

        # # Sugeno-Weber T-norm
        # result_sugeno_weber = apply_tnorm_iterative(lambda a, b: sugeno_weber_tnorm(a, b, p=1), fuzzy_tensor)
        # result_sugeno_weber_tensor = sugeno_weber_tnorm_tensor(fuzzy_tensor, p=1)
        # print("\nResult with Sugeno-Weber T-norm (Iterative):")
        # print(result_sugeno_weber)
        # print("\nResult with Sugeno-Weber T-norm (Tensor):")
        # print(result_sugeno_weber_tensor)
        # print("\nEquality Check for Sugeno-Weber T-norm:")
        # print(np.allclose(result_sugeno_weber_tensor.cpu().detach().numpy(), result_sugeno_weber.cpu().detach().numpy()))

        # # Dombi T-norm
        # result_dombi = apply_tnorm_iterative(lambda a, b: dombi_tnorm(a, b, p=1), fuzzy_tensor)
        # result_dombi_tensor = dombi_tnorm_tensor(fuzzy_tensor, p=1)
        # print("\nResult with Dombi T-norm (Iterative):")
        # print(result_dombi)
        # print("\nResult with Dombi T-norm (Tensor):")
        # print(result_dombi_tensor)
        # print("\nEquality Check for Dombi T-norm:")
        # print(result_dombi_tensor == result_dombi)

        # # Aczel-Alsina T-norm
        # result_aczel_alsina = apply_tnorm_iterative(lambda a, b: aczel_alsina_tnorm(a, b, p=1), fuzzy_tensor)
        # result_aczel_alsina_tensor = aczel_alsina_tnorm_tensor(fuzzy_tensor, p=1)
        # print("\nResult with Aczel-Alsina T-norm (Iterative):")
        # print(result_aczel_alsina)
        # print("\nResult with Aczel-Alsina T-norm (Tensor):")
        # print(result_aczel_alsina_tensor)
        # print("\nEquality Check for Aczel-Alsina T-norm:")
        # print(np.allclose(result_aczel_alsina_tensor.cpu().detach().numpy(), result_aczel_alsina.cpu().detach().numpy()))
    
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
    # tnorm1, tnorm1_fast = min_tnorm, min_tnorm_tensor
    # tnorm1, tnorm1_fast = product_tnorm, product_tnorm_tensor
    # tnorm1, tnorm1_fast = lukasiewicz_tnorm, lukasiewicz_tnorm_tensor
    # tnorm1, tnorm1_fast = drastic_tnorm_batch, drastic_tnorm_tensor
    tnorm1, tnorm1_fast = hamacherprod_tnorm, hamacherprod_tnorm_tensor
    # tnorm1, tnorm1_fast = nilpotentmin_tnorm, nilpotentmin_tnorm_tensor

    # tnorm1, tnorm1_batch = dombi_tnorm, dombi_tnorm_batch
    # tnorm1, tnorm1_batch = aczel_alsina_tnorm, aczel_alsina_tnorm_batch
    # tnorm1, tnorm1_batch = sugeno_weber_tnorm, sugeno_weber_tnorm
    # tnorm1, tnorm1_batch = yager_tnorm, yager_tnorm
    # tnorm1, tnorm1_batch = frank_tnorm, frank_tnorm
    # tnorm1, tnorm1_batch = hamacher_tnorm, hamacher_tnorm
    # tnorm1, tnorm1_batch = schweizer_sklar_tnorm, schweizer_sklar_tnorm_batch
    # tnorm1, tnorm1_batch = drastic_tnorm, drastic_tnorm_batch
    # tnorm1, tnorm1_batch = nilpotentmin_tnorm, nilpotentmin_tnorm_batch
    tnorm1, tnorm1_batch = hamacherprod_tnorm, hamacherprod_tnorm_batch

    print("Input Tensor:")
    print(input_tensor)
    print("\nResult with Iterative T-norm:")
    result = apply_tnorm_iterative2(tnorm1_batch, input_tensor)
    print(result)
    result = apply_tnorm_iterative(tnorm1, input_tensor)
    print(result)

    # traverse_computation_graph(result)

    print("\nResult with Fast T-norm:")
    result = tnorm1_fast(input_tensor)
    print(result)
    # traverse_computation_graph(result)
    # traverse_computation_graph_from_grad_fn(result.grad_fn)