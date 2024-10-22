# -*- coding: utf-8 -*-
"""
AE 727 – Assignment 01

@author: Eduardo Pelanda Amaro
"""

import numpy as np
import sympy as sp
import composite_toolbox as ct

# %% PROBLEM 1
print('\n====================================================================')
print('PROBLEM 1')
# rotation angle [°]
theta = 45

# define symbolic variables
E1, E2, Ex45, v12, G12 = sp.symbols('E1 E2 Ex45 v12 G12')

# compliance matrix
S = ct.compliance_matrix_S(E1, E2, v12, G12, display=True)

# rotation matrix
R = ct.rotation_matrix(theta, size=3, display=True)

# rotated compliance matrix
S45 = ct.apply_rotation(S, theta, display=True)

# S45(1, 1) = 1/Extheta
S45_11 = S45[0, 0]

# solve for G12
equation = sp.Eq(S45_11, 1 / Ex45)
G12_solution = sp.solve(equation, G12)
print('\nG12 =')
sp.pprint(G12_solution)

# solve for Ex45
Ex45_solution = sp.solve(equation, Ex45)
print('\nEx45 =')
sp.pprint(Ex45_solution)

# %% PROBLEM 2
print('\n====================================================================')
print('PROBLEM 2')
# define symbolic variables
E1, E2, v12, G12, Ex30, Ex60 = sp.symbols('E1 E2 v12 G12 Ex30 Ex60')

# compliance matrix
S = ct.compliance_matrix_S(E1, E2, v12, G12, display=True)

# rotated compliance matrix
S30 = ct.apply_rotation(S, 30)
S60 = ct.apply_rotation(S, 60)

# S30(1, 1) = 1/Ex30
S30_11 = S30[0, 0]
# S60(1, 1) = 1/Ex60
S60_11 = S60[0, 0]

# solve for Ex30
equation1 = sp.Eq(S30_11, 1 / Ex30)
Ex30_solution = sp.solve(equation1, Ex30)
print('\nEx30 =')
sp.pprint(Ex30_solution)

# solve for Ex60
equation2 = sp.Eq(S60_11, 1 / Ex60)
Ex60_solution = sp.solve(equation2, Ex60)
print('\nEx60 =')
sp.pprint(Ex60_solution)

# solve for: 1 / Ex30 - 1 / Ex60
difference = 1 / Ex30_solution[0] - 1 / Ex60_solution[0]
simplified_difference = sp.simplify(difference)
print('\n1 / Ex30 - 1 / Ex60 =')
sp.pprint(simplified_difference)

# %% PROBLEM 3
print('\n====================================================================')
print('PROBLEM 3')
# define symbolic variables
E1, E2, v12, G12 = sp.symbols('E1 E2 v12 G12')

# compliance matrix
S = ct.compliance_matrix_S(E1, E2, v12, G12, display=True)

# rotation matrix
theta = sp.symbols('theta')
c, s = sp.cos(theta), sp.sin(theta)
R = sp.Matrix([
    [c**2, s**2, 2 * c * s],
    [s**2, c**2, -2 * c * s],
    [-c * s, c * s, c**2 - s**2]])
print('\nR =')
sp.pprint(R)

# global tensions
sigma_x, k = sp.symbols('sigma_x k')
sigma_theta = sp.Matrix([sigma_x, k * sigma_x, 0])
print('\nsigma_theta =')
sp.pprint(sigma_theta)

# local deformations
epsilon = S @ R @ sigma_theta
print('\nepsilon =')
sp.pprint(epsilon)

# %% PROBLEM 4
print('\n====================================================================')
print('PROBLEM 4')
# rotation angle [°]
theta = 45

# rotation matrix
R = ct.rotation_matrix(theta, size=3, display=True)

# global tensions
sigma_0 = sp.symbols('sigma_0')
sigma_theta = sp.Matrix([2 * sigma_0, -sigma_0, 0])
print('\nsigma_theta =')
sp.pprint(sigma_theta)

# local tensions
sigma = R @ sigma_theta
print('\nsigma =')
sp.pprint(sigma)

# %% PROBLEM 5
print('\n====================================================================')
print('PROBLEM 5')

# material strenght [MPa]
Xt = 2280
Xc = 1450
Yt = 59
Yc = 228
S = 69

# rotation matrix
theta = sp.symbols('theta')
c, s = sp.cos(theta), sp.sin(theta)
R = sp.Matrix([
    [c**2, s**2, 2 * c * s],
    [s**2, c**2, -2 * c * s],
    [-c * s, c * s, c**2 - s**2]])
print('\nR =')
sp.pprint(R)

# global tensions
F_0 = sp.symbols('F_0')
sigma_theta = sp.Matrix([F_0, -F_0, 0])
print('\nsigma_theta =')
sp.pprint(sigma_theta)

# local tensions
sigma = R @ sigma_theta
print('\nsigma =')
sp.pprint(sigma)

sigma_1 = sigma[0]
sigma_2 = sigma[1]
tau_12 = sigma[2]

# %% Tsai-Hill criteria
print('\nTsai-Hill criteria')
# traction condition X = Xt e Y = Yc
X = Xt
Y = Yc
tsai_hill = (sigma_1 / X)**2 - (sigma_1 * sigma_2 / X**2) + (sigma_2 / Y)**2 + (tau_12 / S)**2

# solve for F_0
equation = sp.Eq(tsai_hill, 1)
F_0_solution = sp.solve(equation, F_0)
print('\nTraction condition\nF_0 =')
print(F_0_solution)

theta_45 = np.radians(30)
F_0_at_45 = [sol.subs(theta, theta_45).evalf() for sol in F_0_solution]
print('\nF_0 at theta = 45 degrees:')
sp.pprint(F_0_at_45)

# compression condition X = Xc e Y = Yt
X = Xc
Y = Yt
tsai_hill = (sigma_1 / X)**2 - (sigma_1 * sigma_2 / X**2) + (sigma_2 / Y)**2 + (tau_12 / S)**2

# solve for F_0
equation = sp.Eq(tsai_hill, 1)
F_0_solution = sp.solve(equation, F_0)
print('\nCompression condition\nF_0 =')
print(F_0_solution)

F_0_at_45 = [sol.subs(theta, theta_45).evalf() for sol in F_0_solution]
print('\nF_0 at theta = 45 degrees:')
sp.pprint(F_0_at_45)

# %% Maximum strain criteria
print('\nMaximum strain criteria')
# compliance matrix
E1, E2, v12, G12 = sp.symbols('E1 E2 v12 G12')
Smatrix = ct.compliance_matrix_S(E1, E2, v12, G12, display=True)

# local deformations
epsilon = Smatrix @ R @ sigma_theta
print('\nepsilon =')
print(epsilon)

epsilon_1 = epsilon[0]
epsilon_2 = epsilon[1]
gamma_12 = epsilon[2]

# solve for F_0
equation = sp.Eq(epsilon_1, Xt / E1)
F_0_solution = sp.solve(equation, F_0)
print('\nF_0 =')
print(F_0_solution)

equation = sp.Eq(epsilon_1, Xc / E1)
F_0_solution = sp.solve(equation, F_0)
print('\nF_0 =')
print(F_0_solution)

equation = sp.Eq(epsilon_2, Yt / E2)
F_0_solution = sp.solve(equation, F_0)
print('\nF_0 =')
print(F_0_solution)

equation = sp.Eq(epsilon_2, Yc / E2)
F_0_solution = sp.solve(equation, F_0)
print('\nF_0 =')
print(F_0_solution)

equation = sp.Eq(gamma_12, S / G12)
F_0_solution = sp.solve(equation, F_0)
print('\nF_0 =')
print(F_0_solution)
