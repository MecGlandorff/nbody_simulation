""" XXXXXX """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#---- Parameters
N_BODIES = 30
W, H = 80, 50
DT = 0.005
G = 0.075
MERGE_COLL = False
TRAIL_LENGTH = 0
np.random.seed(42)
VEL = 1.0
EPS = 2.0
FF = 20
SPIN = 0.025


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
    pos = np.random.rand(2) * [W, H]
    vel = (np.random.rand(2) - 0.5) * VEL
    color = plt.cm.plasma(np.random.rand())
    bodies.append(Body(pos, vel, mass, radius, color))


# --- Energy helpers
def kinetic_energy(bodies):
    return sum(0.5 * b.mass * np.dot(b.vel, b.vel) for b in bodies)

def potential_energy(bodies):
    U = 0.0
    n = len(bodies)
    for i in range(n):
        for j in range(i+1, n):
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
    tang = np.array([-r[1], r[0]])
    if np.any(r):
        b.vel += SPIN * tang / (np.linalg.norm(r) + 1e-9)

# --- Remove drift and re-virialize after spin
zero_total_momentum(bodies)
virialize_to_Q(bodies, target_Q=1.0)


#-- Physics Functions
def comp_gravity(bodies):
    """Compute and return accelerations (Plummer-softened)"""
    n = len(bodies)
    acc = [np.zeros(2) for _ in bodies]
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
        b.vel += 0.5 * acc[i] * DT  # half-step velocity
        b.pos += b.vel * DT
    acc_new = comp_gravity(bodies)
    for i, b in enumerate(bodies):
        b.vel += 0.5 * acc_new[i] * DT
    for b in bodies:
        for dim, max_pos in enumerate([W, H]):
            if b.pos[dim] < b.radius:
                b.pos[dim] = b.radius
                b.vel[dim] *= -1
            elif b.pos[dim] > max_pos - b.radius:
                b.pos[dim] = max_pos - b.radius
                b.vel[dim] *= -1
        b.update_trail()


#--- Visualization setup
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_xticks([]); ax.set_yticks([])
ax.set_facecolor("black")

title = ax.set_title("N-Body Gravity Simulation", color="white", pad=12)
stats_text = ax.text(
    0.02, 0.97, "", color="white", fontsize=10, transform=ax.transAxes,
    verticalalignment="top", horizontalalignment="left"
)

# fix: initialize with dummy data to prevent white fallback
scat = ax.scatter([0], [0], s=[10], c=[[0, 0, 0]], alpha=0.9, edgecolors="white")
trail_lines = [ax.plot([], [], lw=1, alpha=0.4, color=b.color)[0] for b in bodies]


#--- Animation loop
def animate(frame):
    global bodies
    for _ in range(FF):
        update_system(bodies)

    # --- Visual updates: color by speed, size by radius ---
    speeds = np.array([np.linalg.norm(b.vel) for b in bodies])
    speeds = np.clip(speeds, 1e-3, None)

    # use z-score normalization to preserve contrast even if speeds are similar
    v_mean = np.mean(speeds)
    v_std = np.std(speeds)
    norm = np.clip((speeds - v_mean) / (2*v_std + 1e-9) + 0.5, 0, 1)
    colors = plt.cm.plasma(norm)

    sizes = np.array([b.radius for b in bodies])**2 * 3000

    scat.set_offsets([b.pos for b in bodies])
    scat.set_sizes(sizes)
    scat.set_facecolors(colors)

    # --- Trails (if TRAILS > 0)
    for line, body in zip(trail_lines, bodies):
        trail = np.array(body.trail)
        if len(trail) > 1:
            line.set_data(trail[:, 0], trail[:, 1])
        else:
            line.set_data([], [])

    # --- Stats display
    K = kinetic_energy(bodies)
    U = potential_energy(bodies)
    E = K + U
    Q = 2*K / (abs(U) + 1e-12)

    global E0
    if 'E0' not in globals() or E0 is None:
        E0 = E
    dE_rel = (E - E0) / (abs(E0) + 1e-12) * 100.0

    stats_text.set_text(f"E: {E: .5f}  ΔE: {dE_rel: .3f}%  Q={Q: .3f}")
    stats_text.set_color("limegreen" if E < 0 else "red")
    title.set_text(f"Frame {frame} | Bodies: {len(bodies)}")

    return scat, *trail_lines, title, stats_text


ani = FuncAnimation(fig, animate, frames=2000, interval=30, blit=True)
plt.show()