# -*- coding: utf-8 -*-
"""
Composites toolbox

C - Stiffness (rigidex) matrix
S - Compliance (flexibilidade) matrix

@author: Eduardo Pelanda Amaro

TODO:
    substituir zero em valores < 1e-10
    ABD_matrices symbolic
"""

import numpy as np
import sympy as sp

# %%


def rotation_matrix(theta, size=3, display=False):
    """Compute the rotation matrix for a given theta angle."""
    rad = np.radians(theta)
    c, s = np.cos(rad), np.sin(rad)

    # 6x6 Rotation matrix
    R = np.array([[c**2, s**2, 0, 0, 0, 2 * c * s],
                  [s**2, c**2, 0, 0, 0, -2 * c * s],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, c, -s, 0],
                  [0, 0, 0, s, c, 0],
                  [-c * s, c * s, 0, 0, 0, c**2 - s**2]])

    # 3x3 Rotation matrix
    if size == 3:
        R = np.delete(np.delete(R, [2, 3, 4], axis=0), [2, 3, 4], axis=1)

    if display:
        print("\nRotation Matrix for theta =", theta, "degrees:")
        print(R)

    return R


def apply_rotation(matrix, theta, display=False):
    """Apply rotation to a matrix."""
    size = np.shape(matrix)[0]
    R = rotation_matrix(theta, size)
    rotated_matrix = R.T @ matrix @ R

    if display:
        print(f"\nRotated matrix for theta = {theta} degrees:")
        print(rotated_matrix)

    return rotated_matrix


def stiffness_matrix_C(E1, E2, v12, G12,
                       E3=None, v13=None, v23=None, G13=None, G23=None,
                       display=False):
    """Create the stiffness matrix C for 3-plane symmetry (3D orthotropic)."""
    # Reciprocal Poisson's ratios
    v21 = (E2 / E1) * v12
    v31 = (E2 / E1) * v12
    v32 = (E2 / E1) * v12

    # 3x3 Stiffness matrix C
    if E3 is None:
        C11 = E1 / (1 - v12 * v21)
        C12 = (v12 * E2) / (1 - v12 * v21)
        C22 = E2 / (1 - v12 * v21)
        C66 = G12

        C = np.array([[C11, C12, 0],
                      [C12, C22, 0],
                      [0, 0, C66]])

    # 6x6 Stiffness matrix C
    else:
        Delta = (1 - v12 * v21 - v23 * v32 - v31 * v13 - 2 * v21 * v32 * v13) / (E1 * E2 * E3)
        C11 = (1 - v23 * v32) / (E2 * E3 * Delta)
        C22 = (1 - v13 * v31) / (E1 * E3 * Delta)
        C12 = (v21 + v31 * v23) / (E2 * E3 * Delta)
        C23 = (v32 + v12 * v31) / (E1 * E3 * Delta)
        C13 = (v31 + v21 * v32) / (E2 * E3 * Delta)
        C33 = (1 - v12 * v21) / (E1 * E2 * Delta)
        C44 = G23
        C55 = G13
        C66 = G12

        C = np.array([[C11, C12, C13, 0, 0, 0],
                      [C12, C22, C23, 0, 0, 0],
                      [C13, C23, C33, 0, 0, 0],
                      [0, 0, 0, C44, 0, 0],
                      [0, 0, 0, 0, C55, 0],
                      [0, 0, 0, 0, 0, C66]])

    if display:
        print("\nStiffness matrix C:")
        print(C)

    return C


def compliance_matrix_S(E1, E2, v12, G12,
                        E3=None, v13=None, v23=None, G13=None, G23=None,
                        display=False):
    """Create the compliance matrix S for 3-plane symmetry (3D orthotropic)."""
    # Reciprocal Poisson's ratios
    v21 = (v12 * E2) / E1

    S11 = 1 / E1
    S22 = 1 / E2
    S12 = -v21 / E2
    S21 = -v12 / E1
    S66 = 1 / G12

    if E3 is not None:
        v31 = (v13 * E3) / E1
        v32 = (v23 * E3) / E2
        S33 = 1 / E3
        S13 = -v31 / E3
        S31 = -v13 / E1
        S23 = -v32 / E3
        S32 = -v23 / E2
        S44 = 1 / G23
        S55 = 1 / G13
    else:
        S33 = None
        S13 = None
        S31 = None
        S23 = None
        S32 = None
        S44 = None
        S55 = None

    # 6x6 Compliance matrix S
    S = np.array([
        [S11, S12, S13, 0, 0, 0],
        [S21, S22, S23, 0, 0, 0],
        [S31, S32, S33, 0, 0, 0],
        [0, 0, 0, S44, 0, 0],
        [0, 0, 0, 0, S55, 0],
        [0, 0, 0, 0, 0, S66]
    ])

    # 3x3 Compliance matrix S
    if E3 is None:
        S = np.delete(np.delete(S, [2, 3, 4], axis=0), [2, 3, 4], axis=1)

    if display:
        print("\nCompliance matrix S:")
        print(S)

    return S


def stress_from_strain(strain, C, theta, display=False):
    """Calculate symbolic ou numerical stress given strain, stiffness matrix,
    and rotation angle."""
    # Apply rotation to the stress tensor
    C_theta = apply_rotation(C, theta)

    # Compute the stress using the stiffness matrix rotated and strain
    stress = C_theta @ strain

    # symbolic simplify
    stress = sp.simplify(stress)

    if display:
        print("\nStress:")
        print(stress)

    return stress


def strain_from_stress(stress, S, theta, display=False):
    """Calculate symbolic ou numerical strain given stress, compliance matrix,
    and rotation angle."""
    # Apply rotation to the stress tensor
    S_theta = apply_rotation(S, theta)

    # Compute the strain using the compliance matrix rotated and stress
    strain = S_theta @ stress

    # symbolic simplify
    strain = sp.simplify(strain)

    if display:
        print("\nStrain:")
        print(strain)

    return strain


def ABD_matrices(C_matrices, thicknesses, display=False):
    """
    Calculate the A, B, and D matrices for a multi-layer laminate.

    Parameters:
    C_matrices : list of numpy arrays
        List of compliance matrices for each layer.
    thicknesses : list of float
        List of thicknesses for each layer.

    Returns:
    A : numpy array
        A matrix.
    B : numpy array
        B matrix.
    D : numpy array
        D matrix.
    """
    n = len(C_matrices)
    size = np.shape(C_matrices[0])[0]

    # Initialize A, B, and D matrices based on matrix size
    A = np.zeros((size, size))
    B = np.zeros((size, size))
    D = np.zeros((size, size))

    # Calculate the total thickness
    total_thickness = sum(thicknesses)

    # Calculate z_k for each layer
    z = np.zeros(n + 1)
    z[0] = -total_thickness / 2
    for k in range(1, n + 1):
        z[k] = z[k - 1] + thicknesses[k - 1]

    # Compute A, B, and D matrices
    for k in range(1, n + 1):
        C_k = C_matrices[k - 1]

        A += (z[k] - z[k - 1]) * C_k
        B += (1 / 2) * (z[k]**2 - z[k - 1]**2) * C_k
        D += (1 / 3) * (z[k]**3 - z[k - 1]**3) * C_k

    if display:
        # Output the results
        print("\nA Matrix:\n", A)
        print("\nB Matrix:\n", B)
        print("\nD Matrix:\n", D)

    return A, B, D


def layup_ABD_matrices(layup, thicknesses, E1, E2, v12, G12, display=False):

    C0 = stiffness_matrix_C(E1, E2, v12, G12)

    # List of stiffness matrices for layers
    C_matrices = []
    for theta in layup:
        Ci = apply_rotation(C0, theta)
        C_matrices.append(Ci)

    # Calculate A, B, D matrices
    A, B, D = ABD_matrices(C_matrices, thicknesses)

    if display:
        # Output the results
        print('\nLayup:', layup, 'degrees')
        print('Thicknesses:', thicknesses)

        print("\nA Matrix:\n", A)
        print("\nB Matrix:\n", B)
        print("\nD Matrix:\n", D)

    return A, B, D
