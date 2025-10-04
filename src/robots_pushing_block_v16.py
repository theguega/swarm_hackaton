import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- PARAMETERS ---
Lx, Ly = 20, 20
N_robots = 8
steps = 200
dt = 0.1

block_mass = 10.0
block_size = 1.5   # square block (side length)
robot_speed = 0.2  # max move per step
push_force = 1.0   # upward force when in contact

safe_dist = 0.8    # min distance between robots
wall_margin = 0.3  # keep away from walls

# --- INITIAL BLOCK STATE ---
block_pos = np.array([10.0, 5.0])
block_vel = np.array([0.0, 0.0])

# --- ROBOTS START RANDOMLY ---
robots = np.random.rand(N_robots, 2) * np.array([Lx, Ly])

# --- TRAJECTORIES ---
block_traj = [block_pos.copy()]
robots_traj = [robots.copy()]

# --- HELPER FUNCTIONS ---
def block_corners(center, size):
    """Return corners of square block for plotting."""
    x, y = center
    s = size / 2
    return np.array([[x-s, y-s],
                     [x+s, y-s],
                     [x+s, y+s],
                     [x-s, y+s],
                     [x-s, y-s]])

def nearest_block_point(block_pos, robot_pos, size):
    """Nearest point on block boundary to robot."""
    x, y = block_pos
    rx, ry = robot_pos
    half = size / 2
    px = min(max(rx, x-half), x+half)
    py = min(max(ry, y-half), y+half)
    # Snap to closer edge
    if abs(rx-x) > abs(ry-y):
        px = x - half if rx < x else x + half
    else:
        py = y - half if ry < y else y + half
    return np.array([px, py])

# --- SIMULATION LOOP ---
for t in range(steps):
    total_force = np.zeros(2)

    new_positions = []

    for i in range(N_robots):
        # Target is nearest point on block
        target_point = nearest_block_point(block_pos, robots[i], block_size)
        direction = target_point - robots[i]
        dist = np.linalg.norm(direction)

        step = np.zeros(2)
        if dist > 1e-6:
            step = robot_speed * direction / dist
            if np.linalg.norm(step) > dist:
                step = direction

        # --- Collision avoidance with other robots ---
        repulse = np.zeros(2)
        for j in range(N_robots):
            if i == j: continue
            diff = robots[i] - robots[j]
            d = np.linalg.norm(diff)
            if d < safe_dist and d > 1e-6:
                repulse += (diff / d) * (safe_dist - d) * 0.5

        step += repulse

        # --- Wall avoidance ---
        x, y = robots[i] + step
        if x < wall_margin: x = wall_margin
        if x > Lx-wall_margin: x = Lx-wall_margin
        if y < wall_margin: y = wall_margin
        if y > Ly-wall_margin: y = Ly-wall_margin
        new_positions.append([x, y])

        # --- Push block if in contact ---
        if dist < 0.2:  
            total_force += np.array([0, push_force])

    # Update robot positions
    robots = np.array(new_positions)

    # --- Block dynamics ---
    acc = total_force / block_mass
    block_vel += acc * dt
    block_pos += block_vel * dt

    block_traj.append(block_pos.copy())
    robots_traj.append(robots.copy())

block_traj = np.array(block_traj)
robots_traj = np.array(robots_traj)

# --- ANIMATION ---
fig, ax = plt.subplots()
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_aspect('equal')

block_plot, = ax.plot([], [], 'b-', linewidth=2, label="Block")
robots_plot, = ax.plot([], [], 'ro', label="Robots")

def update(frame):
    corners = block_corners(block_traj[frame], block_size)
    block_plot.set_data(corners[:,0], corners[:,1])
    robots_plot.set_data(robots_traj[frame,:,0], robots_traj[frame,:,1])
    return block_plot, robots_plot

ani = animation.FuncAnimation(fig, update, frames=len(block_traj), interval=100, blit=True)
ax.legend()
ax.set_title("Robots Collaboratively Pushing Block with Collision Avoidance")
plt.show()

