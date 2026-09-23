import torch
import numpy as np
import inspect

def _print(matrix: torch.tensor) -> None:
    frame = inspect.currentframe().f_back
    call_line = inspect.getframeinfo(frame).code_context[0].strip()
    var_name = call_line.split('_print(')[1].split(')')[0].strip()

    array = matrix.detach().cpu().numpy() if hasattr(matrix, 'detach') else np.asarray(matrix)

    print(f"\n{var_name}:")
    print(np.array2string(array, precision=3, suppress_small=True))



# 1. Packing Real Pairs into Complex Numbers
real_tensor = torch.tensor([
    [1.0, 0.0],  # Vector along Real axis
    [0.0, 1.0],  # Vector along Imaginary axis
    [1.0, 1.0],  # 45-degree vector
    [2.0, 3.0]
])

complex_vec = torch.view_as_complex(real_tensor)
_print(complex_vec)

angles = (torch.pi / 2) * torch.ones(4)
magnitudes = torch.ones_like(angles)

rotators = torch.polar(magnitudes, angles)
_print(rotators)

rotated_complex = complex_vec * rotators
rotated_real = torch.view_as_real(rotated_complex)

_print(rotated_real.round())



q = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0]]
])  # Shape: [1, 2, 4]

print("Original q shape:", q.shape)

q_paired = q.view(1, 2, 2, 2)

q_complex = torch.view_as_complex(q_paired)
print("q_complex shape :", q_complex.shape)

theta = torch.tensor([
    [[0.0, 0.0],
     [torch.pi, torch.pi]]
])  # Shape: [1, 2, 2]

rotator = torch.polar(torch.ones_like(theta), theta)

q_rotated_complex = q_complex * rotator

q_rotated_paired = torch.view_as_real(q_rotated_complex)

q_rotated = q_rotated_paired.reshape(1, 2, 4)

print("\nq_rotated shape:", q_rotated.shape)
print("q_rotated values:\n", q_rotated.round())