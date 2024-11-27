# PyTorch Functions for Building Transformers: From Basics to Advanced

## Foundational Tensor Operations
1. [**Tensor Creation and Manipulation**](#tensor-creation-and-manipulation)
   - `torch.tensor()`: Create tensors from existing data
   - `torch.zeros()`, `torch.ones()`: Create tensors filled with zeros or ones
   - `torch.randn()`: Create tensors with random values from normal distribution
   - `.shape`: Get tensor dimensions
   - `.view()` or `.reshape()`: Reshape tensors
   - `.unsqueeze()`: Add dimensions to tensors
   - `.squeeze()`: Remove single-dimensional entries from tensor shape

2. [**Tensor Mathematical Operations**](#tensor-mathematical-operations)
   - `torch.matmul()`: Matrix multiplication
   - `torch.bmm()`: Batch matrix multiplication
   - `.transpose()`: Swap tensor dimensions
   - `torch.sum()`: Sum tensor elements
   - `torch.mean()`: Calculate mean of tensor
   - `torch.softmax()`: Apply softmax function

## Neural Network Building Blocks
3. **Basic Layer Operations**
   - `nn.Linear()`: Fully connected (dense) layer
   - `nn.Embedding()`: Embedding layer for converting tokens to dense vectors
   - `nn.LayerNorm()`: Layer normalization
   - `nn.ModuleList()`: List of modules for easy management
   - `nn.Parameter()`: Create learnable parameters

4. **Attention Mechanisms**
   - `torch.nn.functional.scaled_dot_product_attention()`: Scaled dot-product attention
   - Manual attention implementation using matrix multiplication
   - Masking techniques with `.masked_fill()`

5. **Activation Functions**
   - `torch.nn.functional.relu()`: ReLU activation
   - `torch.nn.functional.gelu()`: GELU activation (commonly used in transformers)
   - `torch.tanh()`: Hyperbolic tangent activation

## Advanced Transformer-Specific Operations
6. **Positional Encoding**
   - Creating sinusoidal position embeddings
   - `torch.arange()`: Generate position indices
   - Trigonometric functions like `torch.sin()`, `torch.cos()`

7. **Complex Tensor Manipulations**
   - `.repeat_interleave()`: Repeat tensor elements
   - `torch.stack()`: Stack tensors along a new dimension
   - `torch.cat()`: Concatenate tensors along existing dimensions

8. **Training and Optimization Utilities**
   - `torch.no_grad()`: Disable gradient computation
   - `.requires_grad_()`: Enable gradient tracking
   - `torch.optim.Adam()`: Adam optimizer
   - Learning rate schedulers like `torch.optim.lr_scheduler`

9. **Advanced Attention and Transformer Techniques**
   - Multi-head attention implementation
   - Dropout layers: `nn.Dropout()`
   - Residual connections with addition
   - Handling padding masks in attention

10. **Performance and Memory Optimization**
    - `torch.cuda.empty_cache()`: Clear GPU memory
    - `.to('cuda')`: Move tensors to GPU
    - Mixed precision training with `torch.cuda.amp`

## Bonus: Transformer-Specific Tricks
11. **Advanced Techniques**
    - Label smoothing implementation
    - Gradient clipping
    - Custom loss functions
    - Learning rate warmup strategies
   





## Tensor Creation and Manipulation

## 1. `torch.tensor()`: Create Tensors from Existing Data

```python
import torch

# Example
data = [1, 2, 3, 4]
tensor = torch.tensor(data)

print("Tensor:", tensor)
print("Tensor Type:", tensor.dtype)
```

**Output:**
```
Tensor: tensor([1, 2, 3, 4])
Tensor Type: torch.int64
```

**How it works under the hood:**
- Creates a new tensor from the provided data
- Automatically infers data type (dtype) unless specified
- Copies data to tensor's memory, so modifying original data won't affect the tensor

## 2. `torch.zeros()` and `torch.ones()`: Create Tensors Filled with Zeros or Ones

```python
# Example
zeros_tensor = torch.zeros((2, 3))  # Create a 2x3 tensor of zeros
ones_tensor = torch.ones((3, 2))   # Create a 3x2 tensor of ones

print("Zeros Tensor:\n", zeros_tensor)
print("Ones Tensor:\n", ones_tensor)
```

**Output:**
```
Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
Ones Tensor:
 tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
```

**How it works under the hood:**
- Allocate memory for the tensor
- Initialize all elements to 0 or 1
- Shape specified as a tuple (e.g., (2, 3))
- Default dtype is torch.float32

## 3. `torch.randn()`: Create Tensors with Random Values from Normal Distribution

```python
# Example
random_tensor = torch.randn((3, 3))  # 3x3 tensor with values from N(0, 1)

print("Random Tensor:\n", random_tensor)
```

**Output:**
```
Random Tensor:
 tensor([[ 0.4918, -1.1643,  0.3266],
        [ 0.4838,  1.5982,  0.2438],
        [-1.4327, -0.5987,  0.7543]])
```

**How it works under the hood:**
- Generates tensor with elements sampled from standard normal distribution
- Mean = 0, Standard Deviation = 1
- Uses efficient C++ implementations for value generation

## 4. `.shape`: Get Tensor Dimensions

```python
# Example
tensor = torch.randn((2, 4, 3))
print("Tensor Shape:", tensor.shape)
```

**Output:**
```
Tensor Shape: torch.Size([2, 4, 3])
```

**How it works under the hood:**
- Property of tensor object
- Returns dimensions as torch.Size (tuple-like structure)

## 5. `.view()` or `.reshape()`: Reshape Tensors

```python
# Example
tensor = torch.arange(12)  # Create a tensor with values [0, 1, ..., 11]
reshaped_tensor = tensor.view(3, 4)  # Reshape to 3x4
reshaped_tensor2 = tensor.reshape(4, 3)  # Reshape to 4x3

print("Original Tensor:\n", tensor)
print("Reshaped Tensor (3x4):\n", reshaped_tensor)
print("Reshaped Tensor (4x3):\n", reshaped_tensor2)
```

**Output:**
```
Original Tensor:
 tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
Reshaped Tensor (3x4):
 tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
Reshaped Tensor (4x3):
 tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
```

**How it works under the hood:**
- Rearrange tensor's memory layout without copying data (if possible)
- `.view()` requires contiguous memory
- `.reshape()` works with non-contiguous memory (may involve copying)

## 6. `unsqueeze()`: Add Dimensions to Tensors

```python
# Example
tensor = torch.tensor([1, 2, 3])
unsqueezed_tensor = tensor.unsqueeze(0)  # Add a dimension at position 0

print("Original Tensor:\n", tensor)
print("Unsqueezed Tensor:\n", unsqueezed_tensor)
```

**Output:**
```
Original Tensor:
 tensor([1, 2, 3])
Unsqueezed Tensor:
 tensor([[1, 2, 3]])
```

**How it works under the hood:**
- Adds a dimension of size 1 at specified position
- Modifies tensor's metadata without changing data

## 7. `squeeze()`: Remove Single-Dimensional Entries from Tensor Shape

```python
# Example
tensor = torch.randn((1, 3, 1, 5))
squeezed_tensor = tensor.squeeze()  # Remove all dimensions of size 1

print("Original Tensor Shape:", tensor.shape)
print("Squeezed Tensor Shape:", squeezed_tensor.shape)
```

**Output:**
```
Original Tensor Shape: torch.Size([1, 3, 1, 5])
Squeezed Tensor Shape: torch.Size([3, 5])
```

**How it works under the hood:**
- Removes dimensions with size 1
- Updates tensor's metadata without modifying underlying data

## Summary of Operations

| Operation | Description |
|-----------|-------------|
| `torch.tensor()` | Create tensors from existing data |
| `torch.zeros()` | Create tensors filled with zeros |
| `torch.ones()` | Create tensors filled with ones |
| `torch.randn()` | Create tensors with random values |
| `.shape` | Get tensor dimensions |
| `.view()` | Reshape tensor (requires contiguous memory) |
| `.reshape()` | Reshape tensor (handles non-contiguous memory) |
| `.unsqueeze()` | Add single-dimensional entries to tensor shape |
| `.squeeze()` | Remove single-dimensional entries from tensor shape |





## Tensor Mathematical Operations

### 1. `torch.matmul()`: Matrix Multiplication

```python
import torch

# Example
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
result = torch.matmul(tensor1, tensor2)

print("Result of Matrix Multiplication:\n", result)
```

**Output:**
```
Result of Matrix Multiplication:
 tensor([[19, 22],
         [43, 50]])
```

**How it works under the hood:**
- Computes the matrix product of two tensors
- Inner dimensions must match for multiplication
- Resulting tensor has dimensions of outer input matrix dimensions
- Does not automatically broadcast tensors

### 2. `torch.bmm()`: Batch Matrix Multiplication

```python
# Example
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)
result = torch.bmm(batch1, batch2)

print("Batch Matrix Multiplication Result:\n", result.size())
```

**Output:**
```
Batch Matrix Multiplication Result:
 torch.Size([10, 3, 5])
```

**How it works under the hood:**
- Performs matrix multiplication over batches of matrices
- Each matrix in the batch is multiplied independently
- Requires both input tensors to be 3D
- Batch sizes must match across both tensors

### 3. `.transpose()`: Swap Tensor Dimensions

```python
# Example
tensor = torch.tensor([[1, 2], [3, 4]])
transposed_tensor = tensor.transpose(0, 1)

print("Transposed Tensor:\n", transposed_tensor)
```

**Output:**
```
Transposed Tensor:
 tensor([[1, 3],
         [2, 4]])
```

**How it works under the hood:**
- Swaps specified dimensions of the tensor
- Crucial for reshaping tensors for specific operations
- Provides a quick way to change tensor orientation

### 4. `torch.sum()`: Sum Tensor Elements

```python
# Example
tensor = torch.tensor([[1, 2], [3, 4]])
sum_result = torch.sum(tensor)

print("Sum of Tensor Elements:", sum_result)
```

**Output:**
```
Sum of Tensor Elements: 10
```

**How it works under the hood:**
- Calculates the sum of all elements in the tensor
- Can perform summation over specific dimensions
- Useful for reducing tensor dimensionality

### 5. `torch.mean()`: Calculate Mean of Tensor

```python
# Example
tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
mean_result = torch.mean(tensor)

print("Mean of Tensor:", mean_result)
```

**Output:**
```
Mean of Tensor: 2.5
```

**How it works under the hood:**
- Calculates the arithmetic mean of tensor elements
- Can compute mean across specified dimensions
- Handles both integer and floating-point tensors

### 6. `torch.softmax()`: Apply Softmax Function

```python
# Example
tensor = torch.tensor([2.0, 1.0, 0.1])
softmax_result = torch.softmax(tensor, dim=0)

print("Softmax Result:", softmax_result)
```

**Output:**
```
Softmax Result: tensor([0.6590, 0.2424, 0.0986])
```

**How it works under the hood:**
- Applies the softmax function along a specified dimension
- Converts a vector of values into a probability distribution
- Ensures output values sum to 1
- Commonly used in machine learning for classification tasks

