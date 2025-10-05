import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- PARAMETERS ---
Lx, Ly = 20, 20
nx, ny = int(Lx), int(Ly)
N_robots = 5
margin = 1
steps = 2000

# --- GRID & TARGET ---
grid = -np.ones((nx, ny))  # -1 unknown, 0 visited, 2 target

num_targets = 1
targets = []
while len(targets) < num_targets:
    tx, ty = np.random.randint(margin, nx-margin), np.random.randint(margin, ny-margin)
    if (tx, ty) not in targets:
        targets.append((tx, ty))
        grid[tx, ty] = 2

# --- ROBOTS INITIAL POSITIONS ---
robots = np.column_stack((
    np.random.randint(margin, nx-margin, N_robots),
    np.random.randint(margin, ny-margin, N_robots)
))
trajectories = [robots.copy()]

# --- HELPER FUNCTIONS ---
def boustrophedon_path(x_range, y_range):
    """Generate a lawn-mower path covering given x,y ranges"""
    path = []
    for j, y in enumerate(y_range):
        if j % 2 == 0:
            for x in x_range:
                path.append((x, y))
        else:
            for x in reversed(x_range):
                path.append((x, y))
    return path

def generate_robot_paths(nx, ny, N_robots, margin):
    """Divide the grid into vertical slices and generate sweep paths"""
    slice_width = (nx - 2*margin) // N_robots
    paths = []
    for i in range(N_robots):
        x_start = margin + i*slice_width
        x_end = x_start + slice_width
        if i == N_robots-1:  # last robot takes remainder
            x_end = nx - margin
        x_range = np.arange(x_start, x_end)
        y_range = np.arange(margin, ny - margin)
        paths.append(boustrophedon_path(x_range, y_range))
    return paths

def move_step(cur_pos, goal_pos, robots, margin, nx, ny):
    """Move one step toward goal; detour if next cell is occupied.
       Target cells are allowed."""
    x, y = cur_pos
    gx, gy = goal_pos
    dx = np.sign(gx - x)
    dy = np.sign(gy - y)
    
    candidates = [(x+dx, y+dy),
                  (x+dx, y),
                  (x, y+dy),
                  (x+1, y),
                  (x-1, y),
                  (x, y+1),
                  (x, y-1)]
    
    # Filter valid cells inside margin
    candidates = [(nx_, ny_) for nx_, ny_ in candidates
                  if margin <= nx_ < nx-margin and margin <= ny_ < ny-margin]

    # Remove cells occupied by other robots
    occupied = [tuple(r) for r in robots if all(tuple(r) != cur_pos)]
    for nx_, ny_ in candidates:
        if (nx_, ny_) not in occupied:
            return nx_, ny_
    
    return x, y  # no move possible

# --- PRECOMPUTE PATHS ---
paths = generate_robot_paths(nx, ny, N_robots, margin)
indices = np.zeros(N_robots, dtype=int)

# --- SIMULATION ---
all_stopped = False

for t in range(steps):
    if all_stopped:
        trajectories.append(robots.copy())
        continue

    for i in range(N_robots):
        if indices[i] < len(paths[i]) and not all_stopped:
            robots[i] = move_step(robots[i], paths[i][indices[i]], robots, margin, nx, ny)
            
            # Compare positions safely
            cur_pos = tuple(int(v) for v in robots[i])
            goal_pos = tuple(paths[i][indices[i]])
            if cur_pos == goal_pos:
                indices[i] += 1

        x, y = robots[i]

        # Stop immediately if robot reaches target
        if (x, y) in targets:
            all_stopped = True
            print(f"🎯 Robot {i} reached target at {x, y}. All robots stop.")
            break

        if grid[int(x), int(y)] != 2:
            grid[int(x), int(y)] = 0

    trajectories.append(robots.copy())

trajectories = np.array(trajectories)

# --- ANIMATION ---
fig, ax = plt.subplots()
cmap = plt.cm.get_cmap("gray_r", 3)
cmap_colors = cmap(np.arange(3))
cmap_colors[2] = np.array([0,0,1,1])  # target blue
cmap = plt.matplotlib.colors.ListedColormap(cmap_colors)

im = ax.imshow(grid.T, origin='lower', cmap=cmap, extent=[0,Lx,0,Ly])
scat = ax.scatter(trajectories[0,:,0], trajectories[0,:,1], c='red', s=50)
ax.set_title("Swarm Full Area Scan → Stop at Target")
ax.set_xlabel("X")
ax.set_ylabel("Y")

def update(frame):
    im.set_data(grid.T)
    scat.set_offsets(trajectories[frame])
    return [im, scat]

ani = animation.FuncAnimation(fig, update, frames=len(trajectories), interval=50, blit=True)
plt.show()
