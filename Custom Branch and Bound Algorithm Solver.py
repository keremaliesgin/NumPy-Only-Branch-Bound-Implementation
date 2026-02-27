# %%
import numpy as np
import scipy.sparse as sp

# %%
def simplex(tableau):

    def has_negative_value_in_a_row(input_row):
        return np.any(input_row < 0)
    
    def obtain_highest_negative_element_in_a_row(input_row):
        min_value = np.min(input_row)
        if min_value < 0:
            return np.argmin(input_row), min_value
        return None, None

    while has_negative_value_in_a_row(tableau[0]):
        col_position, value = obtain_highest_negative_element_in_a_row(tableau[0])

        rhs = tableau[1:, -1]
        pivot_col = tableau[1:, col_position]

        valid_rows = pivot_col > 1e-9
        if not np.any(valid_rows):
            print("Unbounded problem.")
            return tableau
        
        ratios = np.full_like(rhs, np.inf)
        ratios[valid_rows] = rhs[valid_rows] / pivot_col[valid_rows]
        row_position = np.argmin(ratios) + 1

        value_at_pivot_position = tableau[row_position, col_position]

        # update pivot row
        tableau[row_position] /= value_at_pivot_position

        for i in range(tableau.shape[0]):
            if i == row_position: continue
            value_to_be_eliminated = tableau[i][col_position]
            tableau[i] -= value_to_be_eliminated * tableau[row_position]
        
    return tableau

# %%
def dual_simplex(infeasible_tableau):
    rhs = infeasible_tableau[1:, -1]

    def obtain_highest_negative_element_in_a_row(input_row):
        min_value = np.min(input_row)
        if min_value < -1e-5:
            return np.argmin(input_row), min_value
        return None, None

    while np.any(rhs < 0):
        rhs_index, value = obtain_highest_negative_element_in_a_row(rhs)

        if rhs_index is None: break

        row_position = rhs_index + 1

        if np.all(infeasible_tableau[row_position, 1:-1] >= -1e-5):
            return None 

        all_ratios = np.array([
            infeasible_tableau[0][i] / abs(infeasible_tableau[row_position][i]) 
            if infeasible_tableau[row_position][i] < -1e-5 else np.inf 
            for i in range(1, infeasible_tableau.shape[1] - 1)
        ])
        
        col_position = np.argmin(all_ratios) + 1

        if infeasible_tableau[row_position][col_position] != 0:
            infeasible_tableau[row_position] = infeasible_tableau[row_position] / infeasible_tableau[row_position][col_position]

        for i in range(infeasible_tableau.shape[0]):
            if i == row_position: continue
            infeasible_tableau[i] -= infeasible_tableau[i][col_position] * infeasible_tableau[row_position]

        rhs = infeasible_tableau[1:, -1]
    return infeasible_tableau + 0.0

# %%
def extract_solution(tableau, num_decision_variables):

    x_star = np.zeros(num_decision_variables)
    for i in range(1, num_decision_variables + 1):
        current_column = tableau[:, i]
        amount_of_zeros = np.argwhere(np.isclose(current_column, 0, atol=1e-5)).size
        amount_of_ones = np.argwhere(np.isclose(current_column, 1, atol=1e-5)).size
        if not (amount_of_zeros == tableau.shape[0] - 1 and amount_of_ones == 1):
            x_star[i - 1] = 0
        else:
            row_position_of_one = np.argwhere(current_column == 1).flatten()
            x_star[i - 1] = tableau[row_position_of_one[0]][-1]
            
    return x_star, tableau[0][-1]

# %%
def add_branch_constraint(tableau, variable_index, bound_value, sense):
    new_tableau = np.copy(tableau)[:, :-1]
    source_row_index = np.where(new_tableau[:, variable_index + 1] == 1)[0][0]

    if sense == "L": # x <= 3
        new_tableau = np.hstack((new_tableau, np.zeros(tableau.shape[0]).reshape(-1, 1), tableau[:, -1].reshape(-1, 1)))
        new_tableau = np.vstack((new_tableau, np.repeat(0, new_tableau.shape[1])))
        new_tableau[-1][variable_index + 1] = 1
        new_tableau[-1][-1] = bound_value
        new_tableau[-1][-2] = 1
        new_tableau[-1] -= new_tableau[source_row_index]

    elif sense == "G": # x >= 4  ------> -x <= -4
        new_tableau = np.hstack((new_tableau, np.zeros(tableau.shape[0]).reshape(-1, 1), tableau[:, -1].reshape(-1, 1)))
        new_tableau = np.vstack((new_tableau, np.repeat(0, new_tableau.shape[1])))
        new_tableau[-1][variable_index + 1] = -1
        new_tableau[-1][-1] = -bound_value
        new_tableau[-1][-2] = 1
        new_tableau[-1] += new_tableau[source_row_index]

    return new_tableau

# %%
def custom_branch_and_bound(problem):
    parent_c, parent_A, parent_b, parent_senses, l, u, direction = problem
    
    if direction == "minimize":
        parent_c *= -1
    parent_A = np.vstack((parent_A, np.eye(u.size)))
    parent_b = np.concatenate((parent_b, u))
    parent_senses = np.concatenate((parent_senses, np.repeat("L", u.size)))

    num_constraints, num_decision_variables = parent_A.shape

    objective_offset = 0.0

    # now, we edit lower bounds
    if np.any(l != 0):
        locations_of_nonzero_lower_bounds = np.argwhere(l != 0)
        for i in locations_of_nonzero_lower_bounds.flatten():
            objective_offset += parent_c[i] * l[i]
            for b in range(num_constraints):
                parent_b[b] -= parent_A[b][i] * l[i]

    # here, we craft a tableau
    modified_A = np.copy(parent_A)
    modified_b = np.copy(parent_b)

    # here, we fix the tableau for inappropiate sense values
    for i in range(num_constraints - 1, -1, -1):
        sense = parent_senses[i]

        if sense == "G":
            modified_A[i] *= -1
            modified_b[i] *= -1

        elif sense == "E":
            # modifying A
            copy_of_parent_A = np.delete(modified_A, i, axis=0)
            row_to_be_appended = parent_A[i]
            modified_A = np.vstack((copy_of_parent_A, row_to_be_appended, -row_to_be_appended))

            # modifying B
            copy_of_parent_b = np.delete(modified_b, i)
            value_to_be_appended = parent_b[i]
            modified_b = np.concatenate((copy_of_parent_b, [value_to_be_appended], [-value_to_be_appended]))

    parent_A = modified_A
    parent_b = modified_b
    num_constraints = parent_b.size
        
    tableau_top_row = np.concatenate(([1], -parent_c, np.repeat(0, num_constraints), [objective_offset]))
    tableau_bottom_part = np.hstack((np.repeat(0, num_constraints).reshape(-1, 1),
                                     parent_A, np.eye(num_constraints),
                                     parent_b.reshape(-1, 1)))

    original_tableau = np.vstack((tableau_top_row, tableau_bottom_part))

    def find_the_most_centric(elements):
        lowest_element = elements[0]
        lowest_score = abs(elements[0] % 1 - 0.5)
        lowest_index = 0
        for i, element in enumerate(elements):
            current_element = element
            current_score = abs(element % 1 - 0.5)
            if current_score < lowest_score:
                lowest_score = current_score
                lowest_element = current_element
                lowest_index = i
        return lowest_index, lowest_element

    first_node = simplex(original_tableau)
    incumbent_obj = -np.inf
    incumbent_x = None
    active_nodes = [first_node]
    while len(active_nodes) > 0:
        current_node = active_nodes.pop()
        current_x_star, current_obj_star = extract_solution(current_node, num_decision_variables)

        if current_x_star is None: continue
        if current_obj_star <= incumbent_obj: continue

        all_integers = True
        for x in current_x_star:
            if abs(x - round(x)) > 1e-5:
                all_integers = False
                break

        if all_integers:
            incumbent_obj = current_obj_star
            incumbent_x = current_x_star
        else:
            current_index, current_candidate = find_the_most_centric(current_x_star)
            child_node_1 = add_branch_constraint(current_node, current_index, int(current_candidate), "L")
            solved_1 = dual_simplex(child_node_1)
            child_node_2 = add_branch_constraint(current_node, current_index, int(current_candidate) + 1, "G")
            solved_2 = dual_simplex(child_node_2)

            if solved_1 is not None:
                active_nodes.append(solved_1)
            if solved_2 is not None:
                active_nodes.append(solved_2)

    return incumbent_x, incumbent_obj

# %%
# here is a small knapsack problem

values = np.array([50, 15, 40, 10, 30])
weights = np.array([5, 2, 4, 1, 6])
capacity = np.array([15])

c = values
A = weights.reshape(1, -1)
b = capacity
l = np.repeat(0, values.size)
u = np.repeat(1, values.size)
senses = np.array(["L"])
direction = "maximize"

problem = (c, A, b, senses, l, u, direction)

x_star, obj_star = custom_branch_and_bound(problem)
print(np.round(x_star, 3))
print(obj_star)

# %%
# here is a small test problem that I have manually verified
# myself. with LP relaxations, the optimum solution is (2.25, 3.75) with
# the objective 41.25. the solver is able to find the most optimal integer
# solution which is (0, 5) with the objective 40.

c = np.array([5, 8])
A = np.array([
    [1, 1],
    [5, 9],
    [2, 1]
])
b = np.array([6, 45, 10])

senses = np.array(["L", "L", "L"])

l = np.array([0, 0])
u = np.array([np.inf, np.inf])

direction = "maximize"

problem = (c, A, b, senses, l, u, direction)

x_star, obj_star = custom_branch_and_bound(problem)
print(np.round(x_star, 3))
print(obj_star)

# %%
# here is an example where the custom solver actually works
# be careful, it is extremely slow, that is why I do not
# recommend you to run this with N = 9 or above.

def sudoku(N):
    N_sq = int(np.sqrt(N))
    
    c = np.repeat(1, N**3)
    l = np.repeat(0, N**3)
    u = np.repeat(1, N**3)

    b = np.repeat(1, 4 * N**2)
    senses = np.repeat("E", 4 * N**2)
    
    # row constraint
    col_rows = np.arange(N**3)
    
    # col constraint
    col_cols = np.arange(N**3).reshape(N, N, N).swapaxes(1, 2).flatten()
    
    # cell constraint
    col_cells = np.arange(N**3).reshape(N, N**2).T.flatten()
    
    # square constraint
    col_squares = np.arange(N**3).reshape(N, N_sq, N_sq, N_sq, N_sq).swapaxes(2, 3).flatten()
    
    row = np.repeat(range(4 * N**2), N)
    col = np.concatenate((col_rows, col_cols, col_cells, col_squares))
    aij = np.repeat(1, col.size)
    
    A = sp.csr_matrix((aij, (row, col)), shape = (4*N**2, N**3))

    # if you were to set the direction to minimize, solver would not be
    # attempting. hence, you would be getting an array full of zeros
    problem = (c, A.toarray(), b, senses, l, u, "maximize")


    x_star, _ = custom_branch_and_bound(problem)
    return np.argmax(np.array(x_star).reshape(N, N, N), axis = 0) + 1

board = sudoku(4)
print(board)


