# PyTorch Functions for Building Transformers: From Basics to Advanced

## Foundational Tensor Operations
1. **Tensor Creation and Manipulation**
   - `torch.tensor()`: Create tensors from existing data
   - `torch.zeros()`, `torch.ones()`: Create tensors filled with zeros or ones
   - `torch.randn()`: Create tensors with random values from normal distribution
   - `.shape`: Get tensor dimensions
   - `.view()` or `.reshape()`: Reshape tensors
   - `.unsqueeze()`: Add dimensions to tensors
   - `.squeeze()`: Remove single-dimensional entries from tensor shape

2. **Tensor Mathematical Operations**
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
