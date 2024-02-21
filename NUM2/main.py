import numpy as np

# Przed zaburzeniem
def przed_zaburzeniem():
    print("\nPrzed zaburzeniem:")
    solve_1 = np.linalg.solve(A1, b)
    solve_2 = np.linalg.solve(A2, b)
    print("A1:")
    print(solve_1)
    print("A2:")
    print(solve_2)


# Zaburzone
def po_zaburzeniu():
    print("\nPo zaburzeniu:")
    solve_11 = np.linalg.solve(A1, dodane)
    solve_22 = np.linalg.solve(A2, dodane)
    print("A1:")
    print(solve_11)
    print("A2:")
    print(solve_22)

# Macierz A1
A1 = np.array([[2.554219275, 0.871733993, 0.052575899, 0.240740262, 0.316022841],
               [0.871733993, 0.553460938, -0.070921727, 0.255463951, 0.707334556],
               [0.052575899, -0.070921727, 3.409888776, 0.293510439, 0.847758171],
               [0.240740262, 0.255463951, 0.293510439, 1.108336850, -0.206925123],
               [0.316022841, 0.707334556, 0.847758171, -0.206925123, 2.374094162]])

# Macierz A2
A2 = np.array([[2.645152285, 0.544589368, 0.009976745, 0.327869824, 0.424193304],
               [0.544589368, 1.730410927, 0.082334875, -0.057997220, 0.318175706],
               [0.009976745, 0.082334875, 3.429845092, 0.252693077, 0.797083832],
               [0.327869824, -0.057997220, 0.252693077, 1.191822050, -0.103279098],
               [0.424193304, 0.318175706, 0.797083832, -0.103279098, 2.502769647]])

# Wektor podany w zadaniu, ktory bedzie zaburzony
b = np.array([-0.642912346, -1.408195475, 4.595622394, -5.073473196, 2.178020609])

# Generowanie zaburzenia
delta_b = 1e-6 * np.random.rand(5)
dodane = b + delta_b

przed_zaburzeniem()
po_zaburzeniu()


