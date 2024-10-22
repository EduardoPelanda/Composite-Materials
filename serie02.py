# -*- coding: utf-8 -*-
"""
AE 727 – Assignment 02

@author: Eduardo Pelanda Amaro
"""

import numpy as np
import composite_toolbox as ct

# %% PROBLEM 1
print('\n====================================================================')
print('PROBLEM 1')
# Material constants [Pa]
E1 = 145e9
E2 = 10.5e9
v12 = 0.28
G12 = 7.5e9

# Thicknesses of the layers [m]
t = 0.25e-3

# laminates angles sequences [°]
# C_layup [+55_2/-45/30]_4s
C_layup = [55, 55, -45, 30] * 4
C_layup += C_layup[::-1]

laminates = {'A': [15, 0, 90, -15],
             'B': [45, -45],
             'C': C_layup}

for laminate in laminates:
    layup = laminates[laminate]
    print(f'\nLayup {laminate}')
    # Thicknesses of the layers [mm]
    thicknesses = [t] * len(layup)

    # Calculate A, B, D matrices
    A, B, D = ct.layup_ABD_matrices(layup, thicknesses, E1, E2, v12, G12, display=True)

# %% PROBLEM 2
print('\n====================================================================')
print('PROBLEM 2')
# Material constants [Pa]
E1 = 142e9
E2 = 10.3e9
v12 = 0.27
G12 = 7.2e9

# Material strenghts [Pa]
Xt = 2280e6
Xc = 1440e6
Yt = 57e6
Yc = 228e6
S = 71e6

# Layup angles sequence [°]
layup = [0, 90, 90, 0]

# Thicknesses of the layers [m]
t = 0.20e-3
thicknesses = [t] * len(layup)

# Calculate A, B, D matrices
A, B, D = ct.layup_ABD_matrices(layup, thicknesses, E1, E2, v12, G12, display=True)

# Solve the linear system:
# N = A*epsilon + B*kappa
# M = B*epsilon + D*kappa
# NM = ABBD * epsilon_kappa

# force and moments vector [Nx, Ny, Nxy, Mx, My, Mxy]
# for a uniaxial x load
N = np.array([1, 0, 0])

# epsilon vector
e = np.linalg.solve(A, N)

print('\nStrains:')
print(f'epsilon_x = {e[0]} * Nx')
print(f'epsilon_y = {e[1]} * Nx')

Q0 = ct.stiffness_matrix_C(E1, E2, v12, G12)

print('\nStress for 0°:')
strain = np.array([e[0], e[1], e[2]]) * 1e3
theta = 0
stress_0 = ct.stress_from_strain(strain, Q0, theta, display=True)

print('\nStress for 90°:')
strain = [e[1], e[0], e[2]]
theta = 90
stress_90 = ct.stress_from_strain(strain, Q0, theta, display=True)

# a) Maximum stress failure criteria
Fxt0 = Xt / stress_0[0]
Fxc0 = Xc / stress_0[0]
Fyt0 = Yt / stress_0[1]
Fyc0 = Yc / stress_0[1]

Fxt90 = Xt / stress_90[0]
Fxc90 = Xc / stress_90[0]
Fyt90 = Yt / stress_90[1]
Fyc90 = Yc / stress_90[1]

Ft = min(Fxt0, Fyt0, Fxt90, Fyt90)
Fc = min(Fxc0, Fyc0, Fxc90, Fyc90)
print(f'Ft = {Ft}')
print(f'Fc = {Fc}')
