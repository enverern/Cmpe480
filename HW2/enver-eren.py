import argparse
import sys
import re
import copy


def create_problem_file(n, problem_number, problem_file):
    if problem_number == "P1":
        create_problem_file_1(n, problem_file) # n queens
    elif problem_number == "P2":
        create_problem_file_2(n, problem_file) # map coloring
    elif problem_number == "P3":
        create_problem_file_3(n, problem_file) # cryptarithmetic
    else:
        print("Invalid problem number")
        sys.exit(1)

def create_problem_file_1(n, problem_file):
    variables = [f"X{i}" for i in range(1, n+1)]
    domains = [j for j in range(1, n+1)]
    # create a constraints dictionary with keys from variables
    constraints = {}
    for v in variables:
        constraints[v] = []
    diag_constraints = []
    # add constraints to the dictionary
    for i in range(len(variables)):
        for j in range(i+1, len(variables)):
            v1 = variables[i]
            v2 = variables[j]
            constraints[v1].append(v2)
            diag_constraints.append(f"abs({v1} - {v2}) != {abs(int(v1[1:]) - int(v2[1:]))}")
            
    # write the problem to the file
    with open(problem_file, "w") as f:
        f.write("variables: ")
        for v in variables:
            f.write(f"{v} ")
        f.write("\n")
        f.write("\ndomains:\n")
        for v in variables:
            f.write(f"{v} = [")
            for d in domains[:-1]:
                f.write(f"{d}, ")
            f.write(f"{domains[-1]}]\n")
        f.write("\nconstraints:\n")
        for v in variables:
            for c in constraints[v]:
                f.write(f"{v} != {c}\n")
        for c in diag_constraints:
            f.write(f"{c}\n")

def create_problem_file_2(n, problem_file):
    variables = ["WA", "NT", "Q", "NSW", "V", "SA", "T"]
    domains = [f"c{j}" for j in range(1, n+1)]   # colors are represented as c1, c2, c3, ... cn
    constraints = {}
    for v in variables:
        constraints[v] = []
    # add constraints to the dictionary
    constraints["WA"].append("NT")
    constraints["WA"].append("SA")
    constraints["NT"].append("SA")
    constraints["NT"].append("Q")
    constraints["Q"].append("SA")
    constraints["Q"].append("NSW")
    constraints["SA"].append("NSW")
    constraints["SA"].append("V")
    constraints["NSW"].append("V")

    # write the problem to the file
    with open(problem_file, "w") as f:
        f.write("variables: ")
        for v in variables:
            f.write(f"{v} ")
        f.write("\n")
        f.write("\ndomains:\n")
        for v in variables:
            f.write(f"{v} = [")
            for d in domains[:-1]:
                f.write(f"{d}, ")
            try:
                f.write(f"{domains[-1]}]\n")
            except IndexError:
                f.write("]\n")
        f.write("\nconstraints:\n")
        for v in variables:
            for c in constraints[v]:
                f.write(f"{v} != {c}\n")

def create_problem_file_3(n, problem_file):
    variables = ["T", "O", "F", "R"]
    if n == 0:
        constraints = {"T": [], "O": [], "F": [], "R": []}
        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                constraints[variables[i]].append(variables[j])
        variables.append("X1")
        variables.append("X2")
        sum_constraints = ["O + O = R + 10 * X1", "T + T + X1 = O + 10 * X2", "F = X2"]
    elif n == 1:
        constraints = {"T": [], "W": [], "O": [], "F": [], "R": []}
        variables.append("W")
        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                constraints[variables[i]].append(variables[j])
        variables.append("X1")
        variables.append("X2")
        variables.append("X3")
        sum_constraints = ["O + O = R + 10 * X1", "W + W + X1 = W + 10 * X2", "T + T + X2 = O + 10 * X3", "F = X3"]
    elif n == 2:
        constraints = {"T": [], "W": [], "O": [], "F": [], "X": [], "R": []}
        variables.append("W")
        variables.append("X")
        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                constraints[variables[i]].append(variables[j])
        variables.append("X1")
        variables.append("X2")
        variables.append("X3")
        variables.append("X4")
        sum_constraints = ["O + O = R + 10 * X1", "W + W + X1 = X + 10 * X2", "X + X + X2 = W + 10 * X3", "T + T + X3 = O + 10 * X4", "F = X4"]
    elif n == 3:
        constraints = {"T": [], "W": [], "O": [], "F": [], "X": [], "R": [], "Y": []}
        variables.append("W")
        variables.append("X")
        variables.append("Y")
        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                constraints[variables[i]].append(variables[j])
        variables.append("X1")
        variables.append("X2")
        variables.append("X3")
        variables.append("X4")
        variables.append("X5")
        sum_constraints = ["O + O = R + 10 * X1", "W + W + X1 = Y + 10 * X2", "X + X + X2 = X + 10 * X3", "Y + Y + X3 = W + 10 * X4", "T + T + X4 = O + 10 * X5", "F = X5"]
    else:
        print("Invalid n value")
        sys.exit(1)
    
    with open(problem_file, "w") as f:
        f.write("variables: ")
        for v in variables:
            f.write(f"{v} ")
        f.write("\n")
        f.write("\ndomains:\n")
        for v in variables:
            if v == "F" or v[1:].isnumeric():
                f.write(f"{v} = [0, 1]\n")
            else:
                f.write(f"{v} = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n")
        f.write("\nconstraints:\n")
        for v in variables:
            try:
                for c in constraints[v]:
                    f.write(f"{v} != {c}\n")
            except KeyError:
                pass
        for s in sum_constraints:
            f.write(f"{s}\n")

def parse_input(file_name):
    with open(file_name, 'r') as file:
        input_text = file.read()

    lines = input_text.strip().split('\n')
    variables = []
    domains = dict()
    constraints = []

    parsing_domains = False
    parsing_constraints = False
    for line in lines:
        if line == '':
            if parsing_domains:
                parsing_domains = False
            continue
        line = line.strip()
        if line.startswith('variables:'):
            variables = line[len('variables:'):].strip().split()
            if not variables:
                print("No variables found")
                sys.exit(1)
        elif line.startswith('domains:'):
            parsing_domains = True
        elif parsing_domains:
            if ' = ' in line:
                var, domain = line.split('=')
                var = var.strip()
                domain = domain.split('[')[1].split(']')[0].split(',')
                domains[var] = []
                for d in domain:
                    d = d.strip()
                    if d.isnumeric():
                        domains[var].append(int(d))
                    else:
                        domains[var].append(d)
        elif line.startswith('constraints:'):
            parsing_constraints = True
        elif parsing_constraints:
            if "!=" in line:
                constraints.append(line)
            else:
                line = line.replace('=','==')
                constraints.append(line)
    return variables, domains, constraints 

class Constriant:
    def __init__(self, expression):
        self.expression = expression
        self.variables = self.extract_variables(expression)


    def extract_variables(self, constraint):
        return set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b(?!\s*\()', constraint)) 
    
    def is_satisfied(self, assignment):
        if not self.variables.issubset(assignment.keys()):
            return True
        
        expr = self.expression
        for var in self.variables:
            if var in assignment:
                replacement = f'"{assignment[var]}"' if isinstance(assignment[var], str) else str(assignment[var])
                expr = re.sub(rf'\b{re.escape(var)}\b', replacement, expr)
        return eval(expr)
    
    def get_variables(self):
        var_list = sorted(self.variables)
        return var_list
    
    def check_variable(self, var):
        return var in self.variables
    
    
def select_unassigned_variable(variable_domains, assignment, apply_mrv, apply_dh, constraints, unassigned_variables, neighbors):
    unassigned_vars = sorted(list(unassigned_variables.keys()))
    if apply_mrv:
        min_domain_size = min(len(unassigned_variables[var]) for var in unassigned_vars)
        unassigned_vars = [var for var in unassigned_vars if len(unassigned_variables[var]) == min_domain_size]

    if apply_dh and len(unassigned_vars) > 1:
        max_constraint_count = -1
        max_constraint_var = None
        for var in unassigned_vars:
            neighbors_count = 0
            for neighbor in neighbors[var]:
                if neighbor in unassigned_vars:
                    neighbors_count += 1
            if neighbors_count > max_constraint_count:
                max_constraint_count = neighbors_count
                max_constraint_var = var
        return max_constraint_var
    else:
        # tiebreaker alphanumeric ordering
        try:
            return sorted(unassigned_vars)[0]
        except IndexError:
            return None
        

def order_domain_values(var, variable_domains, assignment, constraints, apply_lcv):
    if apply_lcv:
        def lcv(value):
            count = 0
            for constraint in constraints:
                if constraint.check_variable(var):
                    for neighbor in constraint.get_variables():
                        if neighbor != var and neighbor not in assignment.keys():
                            for neighbor_value in variable_domains[neighbor]:
                                local_assignment = copy.deepcopy(assignment)
                                local_assignment[var] = value
                                local_assignment[neighbor] = neighbor_value
                                if not constraint.is_satisfied(local_assignment):
                                    count += 1
            return count
        return sorted(variable_domains[var], key=lcv)
        
    else:
        return sorted(variable_domains[var])
        
        
    
def is_consistent(var, value, assignment, constraints):
    infunct_assignment = copy.deepcopy(assignment)
    infunct_assignment[var] = value
    for constraint in constraints:
        if constraint.check_variable(var):
            if not constraint.is_satisfied(infunct_assignment):
                return False
    return True

def backtracking_search(csp):
    return recursive_backtracking({}, csp) 

def recursive_backtracking(assignment, csp):
    global expanded_nodes
    var_domains, constraints, neighbors, apply_mrv, apply_dh, apply_lcv, apply_cp, unassigned_variables = csp


    if len(assignment) == len(var_domains):
        return assignment

    var = select_unassigned_variable(var_domains, assignment, apply_mrv, apply_dh, constraints, unassigned_variables, neighbors)
    
    if var is None:
        return None
    expanded_nodes += 1

    for value in order_domain_values(var, var_domains, assignment, constraints, apply_lcv):
        if is_consistent(var, value, assignment, constraints):
            local_assignment = copy.deepcopy(assignment)
            local_variable_domains = copy.deepcopy(var_domains)
            local_assignment[var] = value
            local_variable_domains[var] = [value]
            local_unassigned_variables = copy.deepcopy(unassigned_variables)
            del local_unassigned_variables[var]

            if apply_mrv:
                local_unassigned_variables = update_unassigned_variables(local_unassigned_variables, var, value, local_variable_domains, constraints, neighbors)

            if apply_cp:
                local_variable_domains = AC_3(local_variable_domains, constraints, neighbors)

            result = recursive_backtracking(local_assignment, (local_variable_domains, constraints, neighbors, apply_mrv, apply_dh, apply_lcv, apply_cp, local_unassigned_variables))
            if result is not None:
                return result           
    
    return None 

def update_unassigned_variables(l_unassigned_variables, var, value, variable_domains, constraints, neighbors):
    for neighbor in neighbors[var]:
        if neighbor in l_unassigned_variables.keys():
            for constraint in constraints:
                if constraint.check_variable(neighbor) and constraint.check_variable(var):
                    for neighbor_value in variable_domains[neighbor]:
                        if not constraint.is_satisfied({var: value, neighbor: neighbor_value}) and neighbor_value in l_unassigned_variables[neighbor]:
                            l_unassigned_variables[neighbor].remove(neighbor_value)

    return l_unassigned_variables
                            


def AC_3(variable_domains, constraints, neighbors):
    queue = []

    for var in sorted(variable_domains.keys()):
        for neighbor in sorted(neighbors[var]):
            queue.append((var, neighbor)) 

    while queue:
        (xi, xj) = queue.pop(0)
        if remove_inconsistent_values(variable_domains, xi, xj, constraints):
            if not variable_domains[xi]:
                return variable_domains
            for xk in neighbors[xi]:
                queue.append((xk, xi)) 
    return variable_domains

def remove_inconsistent_values(variable_domains, xi, xj, constraints):
    removed = False
    constraints_between_xi_xj = [c for c in constraints if c.check_variable(xi) and c.check_variable(xj)]

    current_domain = copy.deepcopy(variable_domains[xi])
    for x in current_domain:
        satisfied = False
        for y in variable_domains[xj]:
            if all(c.is_satisfied({xi: x, xj: y}) for c in constraints_between_xi_xj):
                satisfied = True
                break
        if not satisfied:
            variable_domains[xi].remove(x)
            removed = True
    return removed


def find_neighbors(variables, constraints):
    neighbors = {var: set() for var in variables}
    for c in constraints:
        vars_in_constraint = c.get_variables()
        for var in vars_in_constraint:
            neighbors[var].update(vars_in_constraint)
    for var in variables:
        try:
            neighbors[var].remove(var)
        except KeyError:
            pass
        neighbors[var] = sorted(neighbors[var])  
    return neighbors
            
        
if __name__ == "__main__":
    # if the 2nd argument is integer, create a problem file
    if sys.argv[1].isdigit():
        n = int(sys.argv[1])
        problem_number = sys.argv[2]
        problem_file = sys.argv[3]
        create_problem_file(n, problem_number, problem_file)
    else:
        expanded_nodes = 0
        parser = argparse.ArgumentParser(description='CSP Solver with heuristics.')
        parser.add_argument('heuristics', nargs='*', help='Heuristics to apply: MRV, DH, LCV, CP')
        parser.add_argument('problem_file', help='Path to the problem file')
        args = parser.parse_args()
        heuristics = set(arg.upper() for arg in args.heuristics)
        apply_mrv = 'MRV' in heuristics
        apply_dh = 'DH' in heuristics
        apply_lcv = 'LCV' in heuristics
        apply_cp = 'CP' in heuristics

        variables, domains, constraints = parse_input(args.problem_file)
        variables = sorted(variables)
        variable_domains = dict(sorted(domains.items()))

        constraint_objects = [Constriant(c) for c in constraints]

        neighbors = find_neighbors(variables, constraint_objects)
        
        if apply_cp:
            variable_domains = AC_3(variable_domains, constraint_objects, neighbors)
            if any(len(dom) == 0 for dom in variable_domains.values()):
                print("No solution found")
                sys.exit(0)

        unassigned_variables = copy.deepcopy(variable_domains)
        csp = (variable_domains, constraint_objects, neighbors, apply_mrv, apply_dh, apply_lcv, apply_cp, unassigned_variables)
        result = backtracking_search(csp)

        print("Expanded nodes:", expanded_nodes)
        if result is not None:
            for key, value in result.items():
                print(f"{key}: {value}")
        else:
            print("No solution found")

