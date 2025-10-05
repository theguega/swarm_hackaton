import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- PARAMETERS ---
Lx, Ly = 20, 20
cell_size = 1.0
nx, ny = int(Lx / cell_size), int(Ly / cell_size)
N_robots = 10
steps = 800
margin = 5  # minimum distance from walls to circle

# --- OCCUPANCY GRID ---
grid = -np.ones((nx, ny))  # -1 unknown, 0 visited

# --- MULTIPLE TARGETS ---
num_targets = 2
targets = []
while len(targets) < num_targets:
    tx, ty = np.random.randint(0, nx), np.random.randint(0, ny)
    if (tx, ty) not in targets:
        targets.append((tx, ty))
        grid[tx, ty] = 2  # mark target

# --- ROBOTS ---
robots = np.column_stack(
    (np.random.randint(0, nx, N_robots), np.random.randint(0, ny, N_robots))
)
trajectories = [robots.copy()]


# --- HELPER FUNCTIONS ---
def boustrophedon_path(x_range, y_range):
    path = []
    for j, y in enumerate(y_range):
        if j % 2 == 0:
            for x in x_range:
                path.append((x, y))
        else:
            for x in reversed(x_range):
                path.append((x, y))
    return path


def wall_then_spiral(x0, y0, nx, ny):
    dists = [x0, nx - 1 - x0, y0, ny - 1 - y0]
    wall = np.argmin(dists)
    if wall == 0:
        start_x, start_y = 0, y0
        x_range = np.arange(0, nx)
        y_range = np.arange(start_y, ny)
    elif wall == 1:
        start_x, start_y = nx - 1, y0
        x_range = np.arange(nx - 1, -1, -1)
        y_range = np.arange(start_y, ny)
    elif wall == 2:
        start_x, start_y = x0, 0
        x_range = np.arange(0, nx)
        y_range = np.arange(0, ny)
    else:
        start_x, start_y = x0, ny - 1
        x_range = np.arange(0, nx)
        y_range = np.arange(ny - 1, -1, -1)

    path = []
    if wall in [0, 1]:
        step = -1 if x0 > start_x else 1
        for xx in range(x0, start_x + step, step):
            path.append((xx, y0))
    else:
        step = -1 if y0 > start_y else 1
        for yy in range(y0, start_y + step, step):
            path.append((x0, yy))
    path.extend(boustrophedon_path(x_range, y_range))
    return path


# --- PRECOMPUTE PATHS ---
paths = [wall_then_spiral(robots[i, 0], robots[i, 1], nx, ny) for i in range(N_robots)]
indices = np.zeros(N_robots, dtype=int)

# --- STATE FLAGS ---
target_found = False
chosen_target = None
surround_positions = None
all_stopped = False

# --- SIMULATION LOOP ---
for t in range(1, steps):
    if all_stopped:
        # Freeze all robots
        trajectories.append(robots.copy())
        continue

    if target_found:
        # Move robots toward surround positions
        done = True
        for i in range(N_robots):
            tx, ty = surround_positions[i]
            cx, cy = robots[i]
            if (cx, cy) != (tx, ty):
                done = False
                # Take one step closer
                if cx < tx:
                    cx += 1
                elif cx > tx:
                    cx -= 1
                if cy < ty:
                    cy += 1
                elif cy > ty:
                    cy -= 1
                robots[i] = [cx, cy]

        # If everyone reached their assigned surround position → stop all robots
        if done:
            all_stopped = True
            print(
                f"✅ All robots are surrounding target at {targets[chosen_target]} and have stopped."
            )

        trajectories.append(robots.copy())
        continue

    # Otherwise, keep searching
    for i in range(N_robots):
        if indices[i] < len(paths[i]):
            robots[i] = paths[i][indices[i]]
            indices[i] += 1

        x, y = robots[i]
        if (x, y) in targets and not target_found:
            tx, ty = x, y
            if (
                tx < margin
                or tx > nx - 1 - margin
                or ty < margin
                or ty > ny - 1 - margin
            ):
                print(
                    f"⚠️ Robot {i} found target at {tx, ty}, but too close to wall. Continuing search."
                )
            else:
                chosen_target = targets.index((tx, ty))
                target_found = True
                print(
                    f"🎯 Robot {i} found safe target at {tx, ty}. Robots are moving to surround it."
                )

                # Compute surround positions (evenly spaced around target block)
                surround_positions = {}
                radius = 3
                for j in range(N_robots):
                    angle = 2 * np.pi * j / N_robots
                    sx = tx + int(radius * np.cos(angle))
                    sy = ty + int(radius * np.sin(angle))
                    sx = min(max(0, sx), nx - 1)
                    sy = min(max(0, sy), ny - 1)
                    surround_positions[j] = (sx, sy)
                break

        elif grid[int(x), int(y)] != 2:
            grid[int(x), int(y)] = 0

    trajectories.append(robots.copy())

trajectories = np.array(trajectories)

# --- ANIMATION ---
fig, ax = plt.subplots()
cmap = plt.cm.get_cmap("gray_r", 3)
cmap_colors = cmap(np.arange(3))
cmap_colors[2] = np.array([0.0, 0.0, 1.0, 1.0])  # target = blue
cmap = plt.matplotlib.colors.ListedColormap(cmap_colors)

im = ax.imshow(grid.T, origin="lower", cmap=cmap, extent=[0, Lx, 0, Ly])
scat = ax.scatter(trajectories[0, :, 0], trajectories[0, :, 1], c="red", marker="o")
ax.set_title("Swarm: Search → Surround Target → Stop")
ax.set_xlabel("X")
ax.set_ylabel("Y")


def update(frame):
    im.set_data(grid.T)
    scat.set_offsets(trajectories[frame])
    return [im, scat]


ani = animation.FuncAnimation(
    fig, update, frames=len(trajectories), interval=100, blit=True
)
plt.show()
