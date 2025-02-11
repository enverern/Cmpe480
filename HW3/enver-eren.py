import sys

def read_map(filename):
    with open(filename, 'r') as f:
        grid = [list(line.strip('\n')) for line in f]
    rows = len(grid)
    cols = len(grid[0])
    return grid, rows, cols

def parse_query(filename):
    with open(filename, 'r') as f:
        query = f.readline().strip()

    parts = query.split(':', 1)
    query_type_and_time = parts[0].strip()
    obs_act_string = parts[1].strip()

    query_type_part, t_part = query_type_and_time.split('at t=')
    query_type = query_type_part.strip()
    t = int(t_part.strip())

    obs_act_list = [item.strip() for item in obs_act_string.split(';') if item.strip()]

    observations = []
    actions = []
    expect_observation = True

    i = 0
    while i < len(obs_act_list):
        token = obs_act_list[i]
        if ',' in token:
            parts = [x.strip() for x in token.split(',') if x.strip()]
            if len(parts) != 2:
                raise ValueError(f"Unexpected format: {token}")
            if expect_observation:
                raise ValueError("Sequence error: Observation beklenirken action+observation geldi.")
            else:
                actions.append(parts[0])    
                expect_observation = True   
                observations.append(parts[1])  
                expect_observation = False 
        else:
            if expect_observation:
                if token.startswith('bf:'):
                    observations.append(token)
                    expect_observation = False  
                else:
                    if query_type == "Prediction":
                        actions.append(token)
                        i += 1
                        while i < len(obs_act_list):
                            actions.append(obs_act_list[i])
                            i += 1
                        break
                    else:
                        raise ValueError(f"Observation was expected but '{token}' is not in observation format.")
            else:
                if token.startswith('bf:'):
                    raise ValueError(f"Action was expected but a new observation'{token}' arrived.")
                else:
                    actions.append(token)
                    expect_observation = True 

        i += 1

    return query_type, t, observations, actions

def parse_sensor_reading(reading):
    vals = reading[3:] 
    back_val = int(vals[0])
    front_val = int(vals[1])
    return back_val, front_val

def initialize_belief(grid, nrows, ncols):
    belief = zero_matrix(nrows, ncols, 4)
    total_free_spaces = 0
    for r in range(nrows):
        for c in range(ncols):
            if grid[r][c] == ' ':
                for d in range(4):
                    belief[r][c][d] = 1
                    total_free_spaces += 1
    if total_free_spaces == 0:
        result = basic_zero_matrix(nrows, ncols)
        print_highest_prob(result, grid)
        sys.exit(0)
    for r in range(nrows):
        for c in range(ncols):
            for d in range(4):
                belief[r][c][d] /= total_free_spaces
    return belief

def find_distance_to_nearest_obstacle(grid, r, c, direction):
    if direction == 0: # up
        for i in range(r-1, -1, -1):
            if grid[i][c] == 'x':
                return r - i
    elif direction == 1: # right
        for i in range(c+1, len(grid[0])):
            if grid[r][i] == 'x':
                return i - c
    elif direction == 2: # down
        for i in range(r+1, len(grid)):
            if grid[i][c] == 'x':
                return i - r
    elif direction == 3: # left
        for i in range(c-1, -1, -1):
            if grid[r][i] == 'x':
                return c - i

def sensor_direction_probability(grid, r, c, direction, sensor_val):
    nrows = len(grid)
    ncols = len(grid[0])

    dist = find_distance_to_nearest_obstacle(grid, r, c, direction)
    
    # Probability(sensor=1) = 1/(dist^2)
    # Probability(sensor=2) = 2/(dist^2)
    # Probability(sensor=0) = 1 - 3/(dist^2)
    dist_sq = dist*dist
    if dist_sq == 1:
        p1 = 1
        p2 = 0
        p0 = 0
    else:
        p1 = 1/dist_sq
        p2 = 2/dist_sq
        p0 = 1 - (3/dist_sq)
        
    if sensor_val == 0:
        return p0
    elif sensor_val == 1:
        return p1
    elif sensor_val == 2:
        return p2
    else:
        return 0.0

def zero_matrix(nrows, ncols, ndirs):
    result = []
    for r in range(nrows):
        row = []
        for c in range(ncols):
            row.append([0]*ndirs)
        result.append(row)

    return result


def update_belief_on_observation(belief, observation, grid):
    backVal, frontVal = observation

    nrows = len(grid)
    ncols = len(grid[0])

    dir_offsets = [(-1,0), (0,1), (1,0), (0,-1)] # up,right,down,left
    
    new_belief = zero_matrix(nrows, ncols, 4)

    for r in range(nrows):
        for c in range(ncols):
            # If cell is an obstacle, robot cannot be there
            if grid[r][c] == 'x':
                continue
            for d in range(4):
                # Probability of forward sensor reading given state
                forward_sensor_dir = d
                p_forward = sensor_direction_probability(grid, r, c, forward_sensor_dir, frontVal)

                # Probability of backward sensor reading given state
                backward_sensor_dir = (d + 2) % 4
                p_backward = sensor_direction_probability(grid, r, c, backward_sensor_dir, backVal)

                # Combined probability
                p_obs_given_state = p_forward * p_backward

                new_belief[r][c][d] = belief[r][c][d]* p_obs_given_state

    # Normalize
    result = normalize(new_belief)

    return result

def project_state_forward(belief, grid, nrows, ncols, action):
    # Directions: 0=up, 1=right, 2=down, 3=left
    dir_offsets = [(-1,0), (0,1), (1,0), (0,-1)]
    new_belief = zero_matrix(nrows, ncols, 4)

    for r in range(nrows):
        for c in range(ncols):
            # If it's an obstacle, no robot there
            if grid[r][c] == 'x':
                continue
            for d in range(4):
                p = belief[r][c][d]
                if p == 0:
                    continue

                # Compute probabilities
                no_move_prob = (r+1)/(2*nrows)
                drift_prob = (c+1)/(2*ncols)
                intended_prob = 1 - no_move_prob - drift_prob

                # Determine intended final state 
                dr, dc = dir_offsets[d] if action == 'forward' else (0,0)
                rF = r + dr
                cF = c + dc
                new_d = (d + 1) % 4 if action == 'cw' else (d - 1) % 4 if action == 'ccw' else d


                # Apply "no move"
                # Robot stays in (r, c, d)
                if no_move_prob > 0:
                    new_belief[r][c][d] += p * no_move_prob

                # Apply "drift"
                # From intended final (rF, cF, d), try move one step east if possible
                if drift_prob > 0:
                    cD = c+1
                    if 0 <= cD < ncols and grid[r][cD] != 'x':
                        # drift possible
                        new_belief[r][cD][d] += p * drift_prob
                    else:
                        # can't drift, stay at intended final
                        new_belief[r][c][d] += p * drift_prob

                # Apply "intended action"
                if intended_prob > 0:
                    if 0 <= rF < nrows and 0 <= cF < ncols and grid[rF][cF] != 'x':
                        new_belief[rF][cF][new_d] += p * intended_prob
                    else:
                        # intended move not possible, stay at (r, c, d)
                        new_belief[r][c][d] += p * intended_prob

    result = normalize(new_belief)
    return result


def smoothing(belief, grid, nrows, ncols, actions, observations, t, k):
    for time_step in range(1, k+1):
        belief = project_state_forward(belief, grid, nrows, ncols, actions[time_step-1])
        belief = update_belief_on_observation(belief, observations[time_step], grid)
    backward_belief = recursive_backward(belief, grid, nrows, ncols, actions, observations, t, k)

    multiplied = elementwise_multiply(belief, backward_belief)
    normalized = normalize(multiplied)
    return normalized

def elementwise_multiply(belief, backward_belief):
    n_rows = len(belief)
    n_cols = len(belief[0])
    n_dirs = len(belief[0][0])
    result = zero_matrix(n_rows, n_cols, n_dirs)
    for r in range(n_rows):
        for c in range(n_cols):
            for d in range(n_dirs):
                result[r][c][d] = belief[r][c][d] * backward_belief[r][c][d]
    return result

def normalize(multiplied):
    n_rows = len(multiplied)
    n_cols = len(multiplied[0])
    n_dirs = len(multiplied[0][0])
    total = 0
    for r in range(n_rows):
        for c in range(n_cols):
            for d in range(n_dirs):
                total += multiplied[r][c][d]
    if total == 0:
        return multiplied
    for r in range(n_rows):
        for c in range(n_cols):
            for d in range(n_dirs):
                multiplied[r][c][d] /= total
    return multiplied

def recursive_backward(belief, grid, nrows, ncols, actions, observations, t, k):
    if k == t-1:
        result_1 = zero_matrix(nrows, ncols, 4)
        for r in range(nrows):
            for c in range(ncols):
                if grid[r][c] == 'x':
                    continue
                for d in range(4):
                    result_1[r][c][d] = 1
        return result_1
    
    prob_dist_k = zero_matrix(nrows, ncols, 4)
    prob_dist_k_plus_1 = recursive_backward(belief, grid, nrows, ncols, actions, observations, t, k+1)
    for r in range(nrows):
        for c in range(ncols):
            if grid[r][c] == 'x':
                continue
            for d in range(4):
                p = 0
                for r_prime in range(nrows):
                    for c_prime in range(ncols):
                        if grid[r_prime][c_prime] == 'x':
                            continue
                        for d_prime in range(4):
                            first_term = sensor_direction_probability(grid, r_prime, c_prime, d_prime, observations[k+1][1])*sensor_direction_probability(grid, r_prime, c_prime, (d_prime + 2) % 4, observations[k+1][0])
                            second_term = transition_for_smoothing(r, c, d, r_prime, c_prime, d_prime, grid, actions[k])
                            third_term = prob_dist_k_plus_1[r_prime][c_prime][d_prime]  
                            p += first_term * second_term * third_term
                if p > 0:
                    prob_dist_k[r][c][d] = p
    return prob_dist_k

def transition_for_smoothing(r,c,d,r_prime,c_prime,d_prime,grid,action):
    p = 0
    if r == r_prime and c == c_prime and d == d_prime:
        p += (r+1)/(2*len(grid))
    if (r == r_prime and c+1 == c_prime and d == d_prime) or (r == r_prime and c == c_prime and d == d_prime and grid[r][c+1] == 'x'):
        p += (c+1)/(2*len(grid[0]))
    if action == 'forward':
        if d == d_prime:
            if (d == 0 and r-1 == r_prime and c == c_prime) or (d == 0 and r == r_prime and c == c_prime and grid[r-1][c] == 'x'):
                p += 1 -(r+1 + c+1)/(2*len(grid))
            if (d == 1 and r == r_prime and c+1 == c_prime) or (d == 1 and r == r_prime and c == c_prime and grid[r][c+1] == 'x'):
                p += 1 -(r+1 + c+1)/(2*len(grid))
            if (d == 2 and r+1 == r_prime and c == c_prime) or (d == 2 and r == r_prime and c == c_prime and grid[r+1][c] == 'x'):
                p += 1 -(r+1 + c+1)/(2*len(grid))
            if (d == 3 and r == r_prime and c-1 == c_prime) or (d == 3 and r == r_prime and c == c_prime and grid[r][c-1] == 'x'):
                p += 1 -(r+1 + c+1)/(2*len(grid))
    
    if action == 'cw':
        if r == r_prime and c == c_prime and (d + 1) % 4 == d_prime:
            p += 1 -(r+1 + c+1)/(2*len(grid))
    if action == 'ccw':
        if r == r_prime and c == c_prime and (d - 1) % 4 == d_prime:
            p += 1 -(r+1 + c+1)/(2*len(grid))
    return p

def basic_zero_matrix(nrows, ncols):
    result = []
    for r in range(nrows):
        row = []
        for c in range(ncols):
            row.append(0)
        result.append(row)

    return result
def main():
    grid, nrows, ncols = read_map('map.txt')
    result = basic_zero_matrix(nrows, ncols)

    query_type, t, observations, actions = parse_query('query.txt')

    parsed_obs = [parse_sensor_reading(o) for o in observations if o.startswith('bf:')]
    belief = initialize_belief(grid, nrows, ncols)
    belief = update_belief_on_observation(belief, parsed_obs[0], grid)

    if query_type.lower() == "filtering":
        for time_step in range(1, t+1):
            belief = project_state_forward(belief, grid, nrows, ncols, actions[time_step-1])
            belief = update_belief_on_observation(belief, parsed_obs[time_step], grid)
        for r in range(nrows):
            for c in range(ncols):
                sum = 0
                for d in range(4):
                    sum += belief[r][c][d]
                result[r][c] = sum
        print_highest_prob(result, grid)

    elif query_type.lower() == "prediction":
        for time_step in range(1, len(observations)):
            belief = project_state_forward(belief, grid, nrows, ncols, actions[time_step-1])
            belief = update_belief_on_observation(belief, parsed_obs[time_step], grid)
        for action in actions[len(observations)-1:]:
            belief = project_state_forward(belief, grid, nrows, ncols, action)
        for r in range(nrows):
            for c in range(ncols):
                sum = 0
                for d in range(4):
                    sum += belief[r][c][d]
                result[r][c] = sum
        print_highest_prob(result, grid)

    elif query_type.lower() == "smoothing":
        k = t
        t = len(parsed_obs) 
        smoothed = smoothing(belief, grid, nrows, ncols, actions, parsed_obs, t, k)
        for r in range(nrows):
            for c in range(ncols):
                sum = 0
                for d in range(4):
                    sum += smoothed[r][c][d]
                result[r][c] = sum
        print_highest_prob(result, grid)


def print_highest_prob(result, grid):
    initial_high = result[0][0]
    initial_row = 0
    initial_column = 0
    for i in range(len(result)):
        for j in range(len(result[0])):
            if result[i][j] > initial_high:
                initial_high = result[i][j]
                initial_row = i
                initial_column = j
    if initial_high <= 0:
        found = False
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == ' ':
                    initial_row = r
                    initial_column = c
                    initial_high = 0.000
                    found = True
                    break
        if not found:
            output = "All cells are obstacles."
        else:
            output = f"({initial_column+1}, {initial_row+1}): {initial_high:.3f}"
    else:
        output = f"({initial_column+1}, {initial_row+1}): {initial_high:.3f}"

    with open('output.txt', 'w') as f:
        f.write(output)


if __name__ == '__main__':
    main()
