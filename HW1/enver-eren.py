import sys
import heapq

def load_board(file_name):
    with open(file_name, 'r', encoding='utf-8-sig') as f:
        board = []
        for line in f:
            line = line.strip()
            if line:
                board.append([char for char in line if char != ' '])
    return board

def save_board(f, board_state):
    for row in board_state:
        f.write(' '.join(row) + '\n')

def copy_board(board):
    return [row[:] for row in board]

def graph_search(problem):
    initial_board = problem[0]
    algorithm = problem[1]
    heuristic = problem[2]
    row_num = len(initial_board)
    col_num = len(initial_board[0])
    pawns_count = sum(cell.isdigit() for row in initial_board for cell in row)

    # For easier comparison, store the board with respect to their pawn count
    closed_dict = {}
    for i in range(0, pawns_count + 1):
        closed_dict[i] = set()

    number_of_expanded_nodes = 0
    # Store the initial locations of the agents
    agent_locations = dict()
    for i in range(0, row_num):
        for j in range(0, col_num):
            if initial_board[i][j] == "K":
                agent_locations["K"] = [i, j]
            elif initial_board[i][j] == "B":
                agent_locations["B"] = [i, j]
            elif initial_board[i][j] == "R":
                agent_locations["R"] = [i, j]
    
    # Calculate the initial heuristic costs
    initial_heuristic_cost_h1 = h1(initial_board, agent_locations, pawns_count)
    initial_heuristic_cost_h2 = h2(initial_board, agent_locations, pawns_count, 0, 0)

    # Insert the initial state to the fringe
    # fringe = [(cost, board, previous_path, pawns_count, heuristic_cost, agent_locations)]
    if algorithm == "UCS":
        fringe_costs = [0]
        heapq.heapify(fringe_costs)
        fringe_dict = {0: [(0, initial_board, [], pawns_count, 0, agent_locations)]}
    else:
        heuristic_cost = initial_heuristic_cost_h1 if heuristic == "h1" else initial_heuristic_cost_h2
        fringe_costs = [heuristic_cost]
        heapq.heapify(fringe_costs)
        fringe_dict = {heuristic_cost: [(0, initial_board, [], pawns_count, heuristic_cost, agent_locations)]}

    while True:
        # Failure
        if len(fringe_costs) == 0:
            return None
        # Pop the node with the lowest cost which is the first element of the fringe
        min_cost = fringe_costs[0]
        node = fringe_dict[min_cost].pop(0)
        if len(fringe_dict[min_cost]) == 0:
            heapq.heappop(fringe_costs)
            del fringe_dict[min_cost]

        board_tuple = board_to_tuple(node[1])
        current_pawns_count = node[3]

        # Check if the goal state is reached
        if current_pawns_count == 0:
            path = node[2] + [node[1]]
            return number_of_expanded_nodes, node[0], path, initial_heuristic_cost_h1, initial_heuristic_cost_h2, col_num

        # If the node is not in the closed list, add it to the closed list and expand it
        elif board_tuple not in closed_dict[current_pawns_count]:
            closed_dict[current_pawns_count].add(board_tuple)
            fringe_dict, fringe_costs = make_move(node, row_num, col_num, fringe_dict, fringe_costs, algorithm, heuristic)
            number_of_expanded_nodes += 1

def make_move(node, row_num, col_num, fringe_dict, fringe_costs, algorithm, heuristic):
    base_cost = node[0]
    board = node[1]
    previous_path = node[2]
    pawns_count = node[3]
    former_heuristic_cost = node[4]
    path = previous_path + [board]
    agent_locations = dict(node[5])
    knight_location = agent_locations.get("K")
    bishop_location = agent_locations.get("B")
    rook_location = agent_locations.get("R")

    if knight_location is not None:
        x = knight_location[0]
        y = knight_location[1]
        # Check the possible moves of the knight
        for (i,j) in [(2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2), (1,2), (2,1)]:
            new_x = x + i
            new_y = y + j
            if 0 <= new_x < row_num and 0 <= new_y < col_num:
                cell = board[new_x][new_y]
                if cell == ".":
                    new_board = copy_board(board)
                    new_board[new_x][new_y] = "K"
                    new_board[x][y] = "."
                    new_cost = base_cost + 6
                    agent_locations["K"] = [new_x, new_y]
                    heuristic_cost = 0 if algorithm == "UCS" else heuristic_cost_calc(new_board, agent_locations, algorithm, heuristic, pawns_count, former_heuristic_cost, 6)
                    fringe_dict, fringe_costs = update_fringe(fringe_dict, fringe_costs, new_cost, new_board, path, pawns_count, heuristic_cost, algorithm, agent_locations)
                    agent_locations["K"] = [x, y]
                elif cell.isdigit():
                    new_pawns_count = pawns_count - 1
                    new_board = copy_board(board)
                    new_board[new_x][new_y] = "K"
                    new_board[x][y] = "."
                    new_cost = base_cost + 6
                    agent_locations["K"] = [new_x, new_y]
                    heuristic_cost = 0 if algorithm == "UCS" else heuristic_cost_calc(new_board, agent_locations, algorithm, heuristic, new_pawns_count, former_heuristic_cost, 6)
                    fringe_dict, fringe_costs = update_fringe(fringe_dict, fringe_costs, new_cost, new_board, path, new_pawns_count, heuristic_cost, algorithm, agent_locations)
                    agent_locations["K"] = [x, y]

    if bishop_location is not None:
        x = bishop_location[0]
        y = bishop_location[1]
        for (i,j) in [(1,-1), (-1,-1), (-1,1), (1,1)]:
            m = 1
            while True:
                new_x = x + i * m
                new_y = y + j * m
                if 0 <= new_x < row_num and 0 <= new_y < col_num:
                    cell = board[new_x][new_y]
                    if cell == ".":
                        new_board = copy_board(board)
                        new_board[new_x][new_y] = "B"
                        new_board[x][y] = "."
                        new_cost = base_cost + 10
                        agent_locations["B"] = [new_x, new_y]
                        heuristic_cost = 0 if algorithm == "UCS" else heuristic_cost_calc(new_board, agent_locations, algorithm, heuristic, pawns_count, former_heuristic_cost, 10)
                        fringe_dict, fringe_costs = update_fringe(fringe_dict, fringe_costs, new_cost, new_board, path, pawns_count, heuristic_cost, algorithm, agent_locations)
                        agent_locations["B"] = [x, y]
                        m += 1
                    elif cell.isdigit():
                        new_pawns_count = pawns_count - 1
                        new_board = copy_board(board)
                        new_board[new_x][new_y] = "B"
                        new_board[x][y] = "."
                        new_cost = base_cost + 10
                        agent_locations["B"] = [new_x, new_y]
                        heuristic_cost = 0 if algorithm == "UCS" else heuristic_cost_calc(new_board, agent_locations, algorithm, heuristic, new_pawns_count, former_heuristic_cost, 10)
                        fringe_dict, fringe_costs = update_fringe(fringe_dict, fringe_costs, new_cost, new_board, path, new_pawns_count, heuristic_cost, algorithm, agent_locations)
                        agent_locations["B"] = [x, y]
                        break
                    else:
                        break
                else:
                    break
    if rook_location is not None:
        x = rook_location[0]
        y = rook_location[1]
        for (i,j) in [(1,0), (0,-1),(-1,0), (0,1)]:
            m = 1
            while True:
                new_x = x + i * m
                new_y = y + j * m
                if 0 <= new_x < row_num and 0 <= new_y < col_num:
                    cell = board[new_x][new_y]
                    if cell == ".":
                        new_board = copy_board(board)
                        new_board[new_x][new_y] = "R"
                        new_board[x][y] = "."
                        new_cost = base_cost + 8
                        agent_locations["R"] = [new_x, new_y]
                        heuristic_cost = 0 if algorithm == "UCS" else heuristic_cost_calc(new_board, agent_locations, algorithm, heuristic, pawns_count, former_heuristic_cost, 8)
                        fringe_dict, fringe_costs = update_fringe(fringe_dict, fringe_costs, new_cost, new_board, path, pawns_count, heuristic_cost, algorithm, agent_locations)
                        agent_locations["R"] = [x, y]
                        m += 1
                    elif cell.isdigit():
                        new_pawns_count = pawns_count - 1
                        new_board = copy_board(board)
                        new_board[new_x][new_y] = "R"
                        new_board[x][y] = "."
                        new_cost = base_cost + 8
                        agent_locations["R"] = [new_x, new_y]
                        heuristic_cost = 0 if algorithm == "UCS" else heuristic_cost_calc(new_board, agent_locations, algorithm, heuristic, new_pawns_count, former_heuristic_cost, 8)
                        fringe_dict, fringe_costs = update_fringe(fringe_dict, fringe_costs, new_cost, new_board, path, new_pawns_count, heuristic_cost, algorithm, agent_locations)
                        agent_locations["R"] = [x, y]
                        break
                    else:
                        break
                else:
                    break

    return fringe_dict, fringe_costs

def update_fringe(fringe_dict, fringe_costs, new_cost, new_board, parent_index, new_pawns_count, heuristic_cost, algorithm, agent_locations):
    new_agent_locations = dict(agent_locations)
    total_cost = new_cost if algorithm == "UCS" else (heuristic_cost if algorithm == "GS" else new_cost + heuristic_cost)
    try:
        fringe_dict[total_cost].append((new_cost, new_board, parent_index, new_pawns_count, heuristic_cost, new_agent_locations))
    except KeyError:
        fringe_dict[total_cost] = [(new_cost, new_board, parent_index, new_pawns_count, heuristic_cost, new_agent_locations)]
        heapq.heappush(fringe_costs, total_cost)

    return fringe_dict, fringe_costs

def heuristic_cost_calc(board, agent_locations, algorithm, heuristic, pawns_count, former_heuristic_cost, move_cost):
    if algorithm == "UCS":
        return 0
    elif heuristic == "h1":
        return h1(board, agent_locations, pawns_count)
    elif heuristic == "h2":
        return h2(board, agent_locations, pawns_count, former_heuristic_cost, move_cost)
            
def h1(board, agent_locations, pawns_count):
    try:
        rook_location = agent_locations["R"]
    except KeyError:
        return 0
    
    rook_row = board[rook_location[0]]
    rook_col = [board[i][rook_location[1]] for i in range(len(board))]
    if any(cell.isdigit() for cell in rook_row) or any(cell.isdigit() for cell in rook_col):
        return pawns_count * 8
    return (pawns_count + 1) * 8
        
def h2(board, agent_locations, pawns_count, former_heuristic_cost, move_cost):
    former_h_cost = former_heuristic_cost
    m_cost = move_cost
    if pawns_count == 0:
        return 0
    new_agent_locations = dict(agent_locations)
    pawns_locations = []
    row_num = len(board)
    col_num = len(board[0])
    constrained_pawns = set()
    # Create a dictionary to store the minimum cost for each pawn and which piece is responsible for that cost
    min_cost_dict = {}
    for i in range(0, row_num):
        for j in range(0, col_num):
            if board[i][j].isdigit():
                pawns_locations.append((i, j))
                min_cost_dict[(i, j)] = [float('inf'), ""]  # Infinity

    r_loc = new_agent_locations.get("R")
    b_loc = new_agent_locations.get("B")
    k_loc = new_agent_locations.get("K")

    if k_loc is not None:
        k_x = k_loc[0]
        k_y = k_loc[1]
    if b_loc is not None:
        b_x = b_loc[0]
        b_y = b_loc[1]
    if r_loc is not None:
        r_x = r_loc[0]
        r_y = r_loc[1]

    for (i, j) in pawns_locations:
        constrained = True
        for mov in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
            ni = i + mov[0]
            nj = j + mov[1]
            if 0 <= ni < row_num and 0 <= nj < col_num:
                if board[ni][nj] != "x":
                    constrained = False
                    break
        if constrained:
            constrained_pawns.add((i, j))

    for index in range(0, len(pawns_locations)):
        i, j = pawns_locations[index]
        if k_loc is not None:
            dx = abs(i - k_x)
            dy = abs(j - k_y)
            n_moves = max((dx + dy + 2) // 3, (max(dx, dy) + 1) // 2)
            knight_cost = 6 * n_moves
            if knight_cost < min_cost_dict[(i, j)][0]:
                min_cost_dict[(i, j)] = [knight_cost, "K"]

        if b_loc is not None:
            if (i, j) in constrained_pawns:
                bishop_cost = float('inf')
                if bishop_cost < min_cost_dict[(i, j)][0]:
                    min_cost_dict[(i, j)] = [bishop_cost, "B"]
            elif (abs(b_x-i) + abs(b_y-j)) % 2 == 0:
                if abs(b_x-i) == abs(b_y-j):
                    obstacle = False
                    dir_x = (i - b_x) // abs(i - b_x) if i != b_x else 0
                    dir_y = (j - b_y) // abs(j - b_y) if j != b_y else 0
                    m = 1
                    while b_x + m * dir_x != i and b_y + m * dir_y != j:
                        if board[b_x + m * dir_x][b_y + m * dir_y] == "x":
                            obstacle = True
                            break
                        m += 1
                    if obstacle:
                        bishop_cost = 30
                    else:
                        bishop_cost = 10
                    if bishop_cost < min_cost_dict[(i, j)][0]:
                        min_cost_dict[(i, j)] = [bishop_cost, "B"]
                else:
                    bishop_cost = 20
                    if bishop_cost < min_cost_dict[(i, j)][0]:
                        min_cost_dict[(i, j)] = [bishop_cost, "B"]
        
        if r_loc is not None:
            if (i, j) in constrained_pawns:
                rook_cost = float('inf')
                if rook_cost < min_cost_dict[(i, j)][0]:
                    min_cost_dict[(i, j)] = [rook_cost, "R"]
            elif i == r_x or j == r_y:
                obstacle = False
                if i == r_x:
                    for m in range(min(j, r_y) + 1, max(j, r_y)):
                        if board[i][m] == "x":
                            obstacle = True
                            break
                else:
                    for m in range(min(i, r_x) + 1, max(i, r_x)):
                        if board[m][j] == "x":
                            obstacle = True
                            break
                if obstacle:
                    rook_cost = 24
                else:
                    rook_cost = 8
                if rook_cost < min_cost_dict[(i, j)][0]:
                    min_cost_dict[(i, j)] = [rook_cost, "R"]
            else:
                rook_cost = 16
                if rook_cost < min_cost_dict[(i, j)][0]:
                    min_cost_dict[(i, j)] = [rook_cost, "R"]

        # Now check for pawn-pawn interactions
        for cursor in range(0, len(pawns_locations)):
            if index == cursor:
                continue
            p_x, p_y = pawns_locations[cursor]
            if k_loc is not None:
                dx = abs(i - p_x)
                dy = abs(j - p_y)
                n_moves = max((dx + dy + 2) // 3, (max(dx, dy) + 1) // 2)
                pawn_k_cost = 6 * n_moves
                if pawn_k_cost < min_cost_dict[(i, j)][0]:
                    min_cost_dict[(i, j)] = [pawn_k_cost, "PK"]
            if b_loc is not None:
                if (i, j) in constrained_pawns:
                    pawn_b_cost = float('inf')
                    if pawn_b_cost < min_cost_dict[(i, j)][0]:
                        min_cost_dict[(i, j)] = [pawn_b_cost, "PB"]
                elif (abs(p_x - i) + abs(p_y - j)) % 2 == 0:
                    if abs(p_x - i) == abs(p_y - j):
                        obstacle = False
                        dir_x = (i - p_x) // abs(i - p_x) if i != p_x else 0
                        dir_y = (j - p_y) // abs(j - p_y) if j != p_y else 0
                        m = 1
                        while p_x + m * dir_x != i and p_y + m * dir_y != j:
                            if board[p_x + m * dir_x][p_y + m * dir_y] == "x":
                                obstacle = True
                                break
                            m += 1
                        if obstacle:
                            pawn_b_cost = 30
                        else:
                            pawn_b_cost = 10
                        if pawn_b_cost < min_cost_dict[(i, j)][0]:
                            min_cost_dict[(i, j)] = [pawn_b_cost, "PB"]
                    else:
                        pawn_b_cost = 20
                        if pawn_b_cost < min_cost_dict[(i, j)][0]:
                            min_cost_dict[(i, j)] = [pawn_b_cost, "PB"]
            if r_loc is not None:
                if (i, j) in constrained_pawns:
                    pawn_r_cost = float('inf')
                    if pawn_r_cost < min_cost_dict[(i, j)][0]:
                        min_cost_dict[(i, j)] = [pawn_r_cost, "PR"]
                elif i == p_x or j == p_y:
                    obstacle = False
                    if i == p_x:
                        for m in range(min(j, p_y) + 1, max(j, p_y)):
                            if board[i][m] == "x":
                                obstacle = True
                                break
                    else:
                        for m in range(min(i, p_x) + 1, max(i, p_x)):
                            if board[m][j] == "x":
                                obstacle = True
                                break
                    if obstacle:
                        pawn_r_cost = 24
                    else:
                        pawn_r_cost = 8
                    if pawn_r_cost < min_cost_dict[(i, j)][0]:
                        min_cost_dict[(i, j)] = [pawn_r_cost, "PR"]
                else:
                    pawn_r_cost = 16
                    if pawn_r_cost < min_cost_dict[(i, j)][0]:
                        min_cost_dict[(i, j)] = [pawn_r_cost, "PR"]

    # Check if at least one of the values in the min_cost_dict has a piece responsible ("K", "B", or "R")
    if not any(value[1] in ["K", "B", "R"] for value in min_cost_dict.values()):
        min_dict = dict(min_cost_dict)
        for (i, j) in pawns_locations:
            # Compute costs again
            knight_cost = bishop_cost = rook_cost = float('inf')
            if k_loc is not None:
                dx = abs(i - k_x)
                dy = abs(j - k_y)
                n_moves = max((dx + dy + 2) // 3, (max(dx, dy) + 1) // 2)
                knight_cost = 6 * n_moves
            if b_loc is not None:
                if (i, j) in constrained_pawns:
                    bishop_cost = float('inf')
                elif (abs(b_x - i) + abs(b_y - j)) % 2 == 0:
                    if abs(b_x - i) == abs(b_y - j):
                        obstacle = False
                        dir_x = (i - b_x) // abs(i - b_x) if i != b_x else 0
                        dir_y = (j - b_y) // abs(j - b_y) if j != b_y else 0
                        m = 1
                        while b_x + m * dir_x != i and b_y + m * dir_y != j:
                            if board[b_x + m * dir_x][b_y + m * dir_y] == "x":
                                obstacle = True
                                break
                            m +=1
                        if obstacle:
                            bishop_cost = 30
                        else:
                            bishop_cost = 10
                    else:
                        bishop_cost = 20
            if r_loc is not None:
                if (i, j) in constrained_pawns:
                    rook_cost = float('inf')
                elif i == r_x or j == r_y:
                    obstacle = False
                    if i == r_x:
                        for m in range(min(j, r_y) + 1, max(j, r_y)):
                            if board[i][m] == "x":
                                obstacle = True
                                break
                    else:
                        for m in range(min(i, r_x) + 1, max(i, r_x)):
                            if board[m][j] == "x":
                                obstacle = True
                                break
                    if obstacle:
                        rook_cost = 24
                    else:
                        rook_cost = 8
                else:
                    rook_cost = 16

            # Update min_dict
            if k_loc is not None and knight_cost > min_dict[(i, j)][0]:
                min_dict[(i, j)] = [knight_cost, "K"]
            if b_loc is not None and bishop_cost > min_dict[(i, j)][0]:
                min_dict[(i, j)] = [bishop_cost, "B"]
            if r_loc is not None and rook_cost > min_dict[(i, j)][0]:
                min_dict[(i, j)] = [rook_cost, "R"]
     
        min_diff = [min_dict[pawns_locations[0]][0] - min_cost_dict[pawns_locations[0]][0], min_dict[pawns_locations[0]][1], pawns_locations[0]]
        for (i, j) in pawns_locations:
            diff = min_dict[(i, j)][0] - min_cost_dict[(i, j)][0]
            if diff < min_diff[0]:
                min_diff = [diff, min_dict[(i, j)][1], (i, j)]
        min_cost_dict[min_diff[2]] = [min_dict[min_diff[2]][0], min_diff[1]]
    
    new_heuristic_cost = sum(value[0] for value in min_cost_dict.values())
    if former_h_cost <= new_heuristic_cost + m_cost:
        return new_heuristic_cost
    else:
        return former_h_cost - m_cost
                
def board_to_tuple(board):
    # Convert a list of lists to a hashable tuple of tuples
    return tuple(tuple(row) for row in board)

# Read the arguments
board_file_name = sys.argv[1]
output_file_name = sys.argv[2]
algorithm = sys.argv[3]
heuristic = sys.argv[4]

# Read the board file and trim the BOM if present
loaded_board = load_board(board_file_name)

# Define the problem
problem = (loaded_board, algorithm, heuristic)

# Run the search algorithm
result = graph_search(problem)


if result is None:
    # Write "FAIL" to output file if no solution is found
    with open(output_file_name, 'w') as f:
        f.write("FAIL\n")
else:
    # Unpack the result
    number_of_expanded_nodes, cost, path, initial_heuristic_cost_h1, initial_heuristic_cost_h2, col_num = result
    # Write the result to the output file
    with open(output_file_name, 'w') as f:
        f.write("expanded: " + str(number_of_expanded_nodes) + "\n")
        f.write("path-cost: " + str(cost) +"\n")
        f.write("h1: " + str(initial_heuristic_cost_h1) +"\n")
        f.write("h2: " + str(initial_heuristic_cost_h2) +"\n")
        for board_state in path:
            save_board(f, board_state)
            f.write("*" * (2 * col_num -1) + "\n")
