""" XXXXXX - 3D version (clean visual, stable view) """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


#---- Parameters
N_BODIES = 40
BOX = np.array([80, 50, 80])   # width, height, depth
DT = 0.005
G = 0.075
MERGE_COLL = False
TRAIL_LENGTH = 0
np.random.seed(42)
VEL = 1.0
EPS = 2.0
FF = 35
SPIN = 0.02


# -- Particle class
class Body:
    def __init__(self, pos, vel, mass, radius, color):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.mass = mass
        self.radius = radius
        self.color = color
        self.trail = [pos.copy()]

    def update_trail(self):
        self.trail.append(self.pos.copy())
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)


#--- Init. System
bodies = []
for _ in range(N_BODIES):
    radius = np.random.uniform(0.1, 0.3)
    mass = radius ** 2 * 10
    pos = np.random.rand(3) * BOX
    vel = (np.random.rand(3) - 0.5) * VEL
    color = plt.cm.plasma(np.random.rand())
    bodies.append(Body(pos, vel, mass, radius, color))


# --- Energy helpers
def kinetic_energy(bodies):
    return sum(0.5 * b.mass * np.dot(b.vel, b.vel) for b in bodies)

def potential_energy(bodies):
    U = 0.0
    n = len(bodies)
    for i in range(n):
        for j in range(i + 1, n):
            diff = bodies[j].pos - bodies[i].pos
            r = np.linalg.norm(diff)
            if r > 0:
                U -= G * bodies[i].mass * bodies[j].mass / np.sqrt(r*r + EPS*EPS)
    return U


# --- Momentum & virial helpers
def zero_total_momentum(bodies):
    M = sum(b.mass for b in bodies)
    v_cm = sum(b.mass * b.vel for b in bodies) / (M + 1e-12)
    for b in bodies:
        b.vel -= v_cm

def virialize_to_Q(bodies, target_Q=1.0):
    K = kinetic_energy(bodies)
    U = potential_energy(bodies)
    Q = 2*K / (abs(U) + 1e-12)
    scale = np.sqrt(target_Q / (Q + 1e-12))
    for b in bodies:
        b.vel *= scale


# --- Virialize initial velocities
U = potential_energy(bodies)
K = kinetic_energy(bodies)
Q = 2*K / (abs(U) + 1e-12)
scale = np.sqrt(1.0 / (Q + 1e-12))
for b in bodies:
    b.vel *= scale


# --- Add global spin (optional)
Mtot = sum(b.mass for b in bodies)
com = sum(b.mass * b.pos for b in bodies) / Mtot
for b in bodies:
    r = b.pos - com
    tang = np.cross([0, 1, 0], r)  # spin around Y-axis
    if np.any(r):
        b.vel += SPIN * tang / (np.linalg.norm(r) + 1e-9)

# --- Remove drift and re-virialize after spin
zero_total_momentum(bodies)
virialize_to_Q(bodies, target_Q=1.0)


#-- Physics Functions
def comp_gravity(bodies):
    """Compute and return accelerations (Plummer-softened)"""
    n = len(bodies)
    acc = [np.zeros(3) for _ in bodies]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = bodies[i], bodies[j]
            diff = b.pos - a.pos
            dist2 = np.dot(diff, diff)
            if dist2 == 0:
                continue
            denom = (dist2 + EPS**2)**1.5
            fvec = G * a.mass * b.mass * diff / denom
            acc[i] +=  fvec / a.mass
            acc[j] -=  fvec / b.mass
    return acc


def update_system(bodies):
    """Leapfrog-style integrator (better energy conservation)"""
    acc = comp_gravity(bodies)
    for i, b in enumerate(bodies):
        b.vel += 0.5 * acc[i] * DT
        b.pos += b.vel * DT
    acc_new = comp_gravity(bodies)
    for i, b in enumerate(bodies):
        b.vel += 0.5 * acc_new[i] * DT

    # Reflect off 3D box boundaries
    for b in bodies:
        for dim, max_pos in enumerate(BOX):
            if b.pos[dim] < b.radius:
                b.pos[dim] = b.radius
                b.vel[dim] *= -1
            elif b.pos[dim] > max_pos - b.radius:
                b.pos[dim] = max_pos - b.radius
                b.vel[dim] *= -1
        b.update_trail()


#--- Visualization setup
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(0, BOX[0])
ax.set_ylim(0, BOX[1])
ax.set_zlim(0, BOX[2])
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_facecolor("black")
fig.patch.set_facecolor("black")

# Make 3D panes black and remove grid lines
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_color((0.0, 0.0, 0.0, 1.0))
    axis._axinfo["grid"]["color"] = (0.2, 0.2, 0.2, 0.05)
    axis.line.set_color((0, 0, 0, 0))

title = ax.set_title("3D N-Body Gravity Simulation", color="white", pad=12)
stats_text = plt.figtext(0.02, 0.95, "", color="white", fontsize=10)

# initialize scatter
positions = np.array([b.pos for b in bodies])
colors = [b.color for b in bodies]
sizes = np.array([b.radius for b in bodies])**2 * 2000
scat = ax.scatter(
    positions[:, 0], positions[:, 1], positions[:, 2],
    s=sizes, c=colors, alpha=0.9, edgecolors="white"
)


#--- Animation loop
def animate(frame):
    global bodies

    # Physics updates (fast-forward FF steps)
    for _ in range(FF):
        update_system(bodies)

    # --- Visuals
    positions = np.array([b.pos for b in bodies])
    speeds = np.array([np.linalg.norm(b.vel) for b in bodies])
    speeds = np.clip(speeds, 1e-3, None)
    v_mean, v_std = np.mean(speeds), np.std(speeds)
    norm = np.clip((speeds - v_mean) / (2*v_std + 1e-9) + 0.5, 0, 1)
    colors = plt.cm.plasma(norm)
    sizes = np.array([b.radius for b in bodies])**2 * 2000

    scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    scat.set_sizes(sizes)
    scat.set_facecolors(colors)

    # --- Energy & Q stats
    K = kinetic_energy(bodies)
    U = potential_energy(bodies)
    E = K + U
    Q = 2*K / (abs(U) + 1e-12)

    global E0
    if 'E0' not in globals() or E0 is None:
        E0 = E
    dE_rel = (E - E0) / (abs(E0) + 1e-12) * 100.0

    stats_text.set_text(f"E: {E: .4f}   ΔE: {dE_rel: .3f}%   Q={Q: .3f}")
    stats_text.set_color("limegreen" if E < 0 else "red")
    title.set_text(f"Frame {frame} | Bodies: {len(bodies)}")

    # --- Fixed camera (no spin)
    ax.view_init(elev=25, azim=45)

    return scat,


ani = FuncAnimation(fig, animate, frames=2000, interval=30, blit=False)
plt.show()