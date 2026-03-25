#!/usr/bin/env python3
"""
AME 531 Project 1 — Time Developing Flow in a Parallel Plate Channel
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'lines.linewidth': 1.5,
    'figure.figsize': (9, 6), 'figure.dpi': 150,
    'savefig.dpi': 180, 'savefig.bbox': 'tight',
})

OUT = Path(__file__).parent / "project1_final"
OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)

N_FOURIER = 200          

#  EXACT ANALYTICAL SOLUTION 

def exact_u(y, t, N=N_FOURIER):
    """u(y,t) = (1-y²)/2 − (16/π³) Σ [(-1)^n/(2n+1)³] cos((2n+1)πy/2) exp(-(2n+1)²π²t/4)"""
    y = np.asarray(y, float)
    u = 0.5 * (1.0 - y**2)
    for k in range(N):
        n = 2*k + 1
        u += -(16.0/np.pi**3) * ((-1)**k / n**3) * np.cos(n*np.pi*y/2.0) * np.exp(-n**2 * np.pi**2 * t / 4.0)
    return u

def exact_Q(t, N=N_FOURIER):
    """Q(t) = ∫₋₁¹ u dy = 2/3 − (64/π⁴) Σ exp(-(2n+1)²π²t/4)/(2n+1)⁴"""
    Q = 2.0/3.0
    for k in range(N):
        n = 2*k + 1
        Q -= (64.0/np.pi**4) * np.exp(-n**2 * np.pi**2 * t / 4.0) / n**4
    return Q

def exact_tau(t, N=N_FOURIER):
    """Wall shear at y=+1: τ = −∂u/∂y|_{y=1} = 1 − (8/π²) Σ exp(-(2n+1)²π²t/4)/(2n+1)²"""
    tau = 1.0
    for k in range(N):
        n = 2*k + 1
        tau -= (8.0/np.pi**2) * np.exp(-n**2 * np.pi**2 * t / 4.0) / n**2
    return tau

#  NUMERICAL SOLVERS

def thomas(a, b, c, d):
    """Solve tridiagonal system: a=sub, b=main, c=super, d=rhs."""
    n = len(b)
    a, b, c, d = [x.copy().astype(float) for x in [a, b, c, d]]
    for i in range(1, n):
        m = a[i-1] / b[i-1]
        b[i] -= m * c[i-1]
        d[i] -= m * d[i-1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i]*x[i+1]) / b[i]
    return x

def num_Q(u, dy):
    """Flow rate by trapezoidal rule (u[0]=u[-1]=0)."""
    return np.trapezoid(u, dx=dy)

def num_tau(u, dy):
    """Wall shear at y=+1 (right wall): τ = −∂u/∂y using 2nd-order backward difference."""
    return -(3*u[-1] - 4*u[-2] + u[-3]) / (2*dy)

def solve_explicit(Ny, dt, t_end):
    dy = 2.0/Ny;  r = dt/dy**2
    y = np.linspace(-1, 1, Ny+1)
    nsteps = int(round(t_end/dt))
    u = np.zeros(Ny+1)
    times, profiles, Qs, taus = [0.0], [u.copy()], [0.0], [0.0]
    t = 0.0
    for step in range(nsteps):
        u_new = np.zeros(Ny+1)
        u_new[1:Ny] = (1-2*r)*u[1:Ny] + r*u[0:Ny-1] + r*u[2:Ny+1] + dt
        u = u_new;  t += dt
        if np.any(np.abs(u) > 1e8):
            times.append(t); profiles.append(u.copy())
            Qs.append(np.nan); taus.append(np.nan)
            return y, np.array(times), profiles, np.array(Qs), np.array(taus), False
        times.append(t); profiles.append(u.copy())
        Qs.append(num_Q(u, dy)); taus.append(num_tau(u, dy))
    return y, np.array(times), profiles, np.array(Qs), np.array(taus), True

def solve_implicit(Ny, dt, t_end):
    dy = 2.0/Ny;  r = dt/dy**2
    y = np.linspace(-1, 1, Ny+1)
    nsteps = int(round(t_end/dt))
    m = Ny - 1
    a_lo = -r * np.ones(m-1)
    a_d  = (1+2*r) * np.ones(m)
    a_up = -r * np.ones(m-1)
    u = np.zeros(Ny+1)
    times, profiles, Qs, taus = [0.0], [u.copy()], [0.0], [0.0]
    t = 0.0
    for step in range(nsteps):
        rhs = u[1:Ny] + dt
        u[1:Ny] = thomas(a_lo, a_d, a_up, rhs)
        t += dt
        times.append(t); profiles.append(u.copy())
        Qs.append(num_Q(u, dy)); taus.append(num_tau(u, dy))
    return y, np.array(times), profiles, np.array(Qs), np.array(taus), True

def solve_rk4(Ny, dt, t_end):
    dy = 2.0/Ny
    y = np.linspace(-1, 1, Ny+1)
    nsteps = int(round(t_end/dt))
    def f(v):
        out = np.zeros_like(v)
        out[1:Ny] = 1.0 + (v[2:Ny+1] - 2*v[1:Ny] + v[0:Ny-1]) / dy**2
        return out
    u = np.zeros(Ny+1)
    times, profiles, Qs, taus = [0.0], [u.copy()], [0.0], [0.0]
    t = 0.0
    for step in range(nsteps):
        k1 = f(u)
        v = u + 0.5*dt*k1;  v[0]=v[-1]=0.0
        k2 = f(v)
        v = u + 0.5*dt*k2;  v[0]=v[-1]=0.0
        k3 = f(v)
        v = u + dt*k3;      v[0]=v[-1]=0.0
        k4 = f(v)
        u = u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        u[0] = u[-1] = 0.0;  t += dt
        if np.any(np.abs(u) > 1e8):
            times.append(t); profiles.append(u.copy())
            Qs.append(np.nan); taus.append(np.nan)
            return y, np.array(times), profiles, np.array(Qs), np.array(taus), False
        times.append(t); profiles.append(u.copy())
        Qs.append(num_Q(u, dy)); taus.append(num_tau(u, dy))
    return y, np.array(times), profiles, np.array(Qs), np.array(taus), True

#  ERROR METRICS

def Linf_error(u_num, y, t):
    return np.max(np.abs(u_num - exact_u(y, t)))

def RMS_error(u_num, y, t):
    return np.sqrt(np.mean((u_num - exact_u(y, t))**2))


#  PLOTTING

snap_times = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

def find_snap(times, target):
    return int(np.argmin(np.abs(np.array(times) - target)))

def save_csv(filepath, header, rows):
    """Save data rows to CSV file."""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'    -> data saved: {filepath.name}')

#  Plots 1-3: Velocity profiles per method.
def plot_velocity_evolution(y, times, profiles, method, fname):
    fig, ax = plt.subplots()
    ye = np.linspace(-1, 1, 300)
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(snap_times)))

    # data for CSV:
    csv_header = ['y']
    csv_data_num = [y.tolist()]
    csv_data_exact = [ye.tolist()]

    for j, ts in enumerate(snap_times):
        idx = find_snap(times, ts)
        t_actual = times[idx]
        ue = exact_u(ye, t_actual)
        ax.plot(ye, ue, '-', color=cmap[j], lw=1.5)
        ax.plot(y, profiles[idx], 'o', color=cmap[j], ms=3, label=f't* = {t_actual:.3f}')
        csv_header.append(f'u_num_t{t_actual:.4f}')
        csv_data_num.append(profiles[idx].tolist())

    uss = 0.5*(1 - ye**2)
    ax.plot(ye, uss, 'k--', lw=2, label='Steady Poiseuille')
    ax.set_xlabel('y*');  ax.set_ylabel('u*')
    ax.set_title(f'Velocity Profiles — {method}\n(lines = exact, circles = numerical)')
    ax.legend(loc='upper center', ncol=3, fontsize=8, framealpha=0.9)
    ax.set_xlim(-1, 1);  ax.set_ylim(bottom=-0.01)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT/fname); plt.close(fig)
    print(f'  {fname}')

    # numerical velocity profiles
    csv_name = fname.replace('.png', '_numerical.csv')
    rows = list(zip(*csv_data_num))
    save_csv(DATA/csv_name, csv_header, rows)

    # velocity profiles at same times
    csv_header_ex = ['y_exact']
    csv_data_ex = [ye.tolist()]
    for ts in snap_times:
        idx = find_snap(times, ts)
        t_actual = times[idx]
        ue = exact_u(ye, t_actual)
        csv_header_ex.append(f'u_exact_t{t_actual:.4f}')
        csv_data_ex.append(ue.tolist())
    csv_header_ex.append('u_steady')
    csv_data_ex.append((0.5*(1-ye**2)).tolist())
    rows_ex = list(zip(*csv_data_ex))
    csv_name_ex = fname.replace('.png', '_exact.csv')
    save_csv(DATA/csv_name_ex, csv_header_ex, rows_ex)

#  Plots 4-5: Q(t) and tau(t)
def plot_Q_tau(Q_runs, tau_runs):
    te = np.linspace(0, 1, 500)
    Qe = np.array([exact_Q(t) for t in te])
    taue = np.array([exact_tau(t) for t in te])
    styles = ['--', '-.', ':']
    cols = ['tab:blue', 'tab:red', 'tab:green']

    fig, ax = plt.subplots()
    ax.plot(te, Qe, 'k-', lw=2.5, label='Exact')
    for i, (name, t_arr, Q_arr) in enumerate(Q_runs):
        step = max(1, len(t_arr)//200)
        ax.plot(t_arr[::step], Q_arr[::step], styles[i], color=cols[i], lw=1.8, label=name)
    ax.axhline(2/3, color='gray', ls=':', alpha=0.5, label='Steady Q* = 2/3')
    ax.set_xlabel('t*'); ax.set_ylabel('Q*(t*)')
    ax.set_title('Flow Rate vs Time — All Methods')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(OUT/'plot04_flowrate.png'); plt.close(fig)
    print('  plot04_flowrate.png')

    # Q(t) data
    q_header = ['t_exact', 'Q_exact']
    q_data = [te.tolist(), Qe.tolist()]
    for name, t_arr, Q_arr in Q_runs:
        q_header.extend([f't_{name}', f'Q_{name}'])
        step = max(1, len(t_arr)//200)
        q_data.extend([t_arr[::step].tolist(), Q_arr[::step].tolist()])
    max_len = max(len(col) for col in q_data)
    for col in q_data:
        col.extend(['' for _ in range(max_len - len(col))])
    save_csv(DATA/'plot04_flowrate_data.csv', q_header, list(zip(*q_data)))

    fig, ax = plt.subplots()
    ax.plot(te, taue, 'k-', lw=2.5, label='Exact')
    for i, (name, t_arr, tau_arr) in enumerate(tau_runs):
        step = max(1, len(t_arr)//200)
        ax.plot(t_arr[::step], tau_arr[::step], styles[i], color=cols[i], lw=1.8, label=name)
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5, label='Steady τ* = 1')
    ax.set_xlabel('t*'); ax.set_ylabel('τ*(t*)  at y = +1 (right wall)')
    ax.set_title('Wall Shear Stress at y = +1 vs Time — All Methods')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(OUT/'plot05_wallshear.png'); plt.close(fig)
    print('  plot05_wallshear.png')

    #  tau(t) data
    tau_header = ['t_exact', 'tau_exact']
    tau_data = [te.tolist(), taue.tolist()]
    for name, t_arr, tau_arr in tau_runs:
        tau_header.extend([f't_{name}', f'tau_{name}'])
        step = max(1, len(t_arr)//200)
        tau_data.extend([t_arr[::step].tolist(), tau_arr[::step].tolist()])
    max_len = max(len(col) for col in tau_data)
    for col in tau_data:
        col.extend(['' for _ in range(max_len - len(col))])
    save_csv(DATA/'plot05_wallshear_data.csv', tau_header, list(zip(*tau_data)))

#  Effect Study
def effect_study(solver, method, param_label, configs, t_eval, fname, stability_note=None):
    """
    configs: list of (label_str, Ny, dt)
    Plots velocity at t_eval and prints error table.
    """
    fig, ax = plt.subplots()
    ye = np.linspace(-1, 1, 300)
    ue = exact_u(ye, t_eval)
    ax.plot(ye, ue, 'k-', lw=2.5, label='Exact')

    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
    markers = ['o', 's', '^', 'D', 'v']
    table_rows = []

    for i, (lbl, Ny, dt) in enumerate(configs):
        dy = 2.0/Ny;  r = dt/dy**2
        y, t_arr, profs, Qs, taus, stable = solver(Ny, dt, t_eval + dt)
        idx = find_snap(t_arr, t_eval)
        u_num = profs[idx]
        if stable and np.all(np.isfinite(u_num)):
            einf = Linf_error(u_num, y, t_arr[idx])
            erms = RMS_error(u_num, y, t_arr[idx])
            ax.plot(y, u_num, markers[i%5]+'-', color=colors[i%5], ms=3, lw=1, label=lbl)
            table_rows.append((lbl, f'{dt:.6f}', f'{dy:.4f}', f'{r:.4f}', f'{einf:.3e}', f'{erms:.3e}'))
        else:
            u_clip = np.clip(u_num, -1.5, 1.5) if u_num is not None else np.zeros_like(y)
            ax.plot(y, u_clip, '--', color=colors[i%5], lw=1.5, alpha=0.6, label=lbl+' (UNSTABLE)')
            table_rows.append((lbl, f'{dt:.6f}', f'{dy:.4f}', f'{r:.4f}', 'DIVERGED', 'DIVERGED'))

    title = f'Effect of {param_label} — {method} at t* = {t_eval}'
    if stability_note:
        title += f'\n{stability_note}'
    ax.set_title(title)
    ax.set_xlabel('y*'); ax.set_ylabel('u*')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    fig.tight_layout(); fig.savefig(OUT/fname); plt.close(fig)
    print(f'  {fname}')

    print(f'    {"Label":<28} {"dt":<12} {"dy":<10} {"r":<10} {"E_inf":<12} {"E_rms":<12}')
    print(f'    {"-"*84}')
    for row in table_rows:
        print(f'    {row[0]:<28} {row[1]:<12} {row[2]:<10} {row[3]:<10} {row[4]:<12} {row[5]:<12}')
    print()


    csv_err_name = fname.replace('.png', '_errors.csv')
    save_csv(DATA/csv_err_name,
             ['Label', 'dt', 'dy', 'r', 'E_inf', 'E_rms'],
             table_rows)

    # velocity profile data for each configuration
    csv_prof_name = fname.replace('.png', '_profiles.csv')
    prof_header = ['y_exact', 'u_exact']
    prof_cols = [ye.tolist(), ue.tolist()]
    for i, (lbl, Ny_c, dt_c) in enumerate(configs):
        y_c, t_arr_c, profs_c, _, _, stable_c = solver(Ny_c, dt_c, t_eval + dt_c)
        idx_c = find_snap(t_arr_c, t_eval)
        u_c = profs_c[idx_c]
        safe_lbl = lbl.replace(',', ';').replace(' ', '')
        prof_header.extend([f'y_{safe_lbl}', f'u_{safe_lbl}'])
        prof_cols.extend([y_c.tolist(), u_c.tolist()])
    max_len = max(len(col) for col in prof_cols)
    for col in prof_cols:
        col.extend(['' for _ in range(max_len - len(col))])
    save_csv(DATA/csv_prof_name, prof_header, list(zip(*prof_cols)))


#  MAIN

def main():
    print('='*65)
    print(' AME 531 PROJECT 1: Time-Developing Channel Flow')
    print(' Domain: -1 ≤ y* ≤ 1, Wall shear evaluated at y = +1')
    print('='*65)

    #  Baseline parameters
    Ny = 40;  dy = 2.0/Ny  # dy = 0.05
    r_base = 0.4;  dt_base = r_base * dy**2  
    t_final = 1.0
    print(f'\nBaseline: Ny={Ny}, dy={dy:.4f}, r={r_base}, dt={dt_base:.6f}\n')

    #  PLOTS 1-3: Velocity profiles
    print('--- Velocity Profiles ---')
    y_e, t_e, p_e, Q_e, tau_e, _ = solve_explicit(Ny, dt_base, t_final)
    plot_velocity_evolution(y_e, t_e, p_e, 'Explicit Euler', 'plot01_vel_explicit.png')

    y_i, t_i, p_i, Q_i, tau_i, _ = solve_implicit(Ny, dt_base, t_final)
    plot_velocity_evolution(y_i, t_i, p_i, 'Implicit Euler', 'plot02_vel_implicit.png')

    y_r, t_r, p_r, Q_r, tau_r, _ = solve_rk4(Ny, dt_base, t_final)
    plot_velocity_evolution(y_r, t_r, p_r, '4th-Order Runge-Kutta', 'plot03_vel_rk4.png')

    #  PLOTS 4-5: Q(t) and tau(t)
    print('\n--- Flow Rate & Wall Shear ---')
    plot_Q_tau(
        Q_runs=[
            ('Explicit Euler', t_e, Q_e),
            ('Implicit Euler', t_i, Q_i),
            ('RK4',            t_r, Q_r),
        ],
        tau_runs=[
            ('Explicit Euler', t_e, tau_e),
            ('Implicit Euler', t_i, tau_i),
            ('RK4',            t_r, tau_r),
        ],
    )

    
    #  EFFECT STUDIES

    t_eval = 0.1  

    #  EXPLICIT EULER
    print('\n--- Effect Studies: Explicit Euler ---')
    print('  (Stability limit: r ≤ 0.5)\n')

    # Effect of dt
    effect_study(solve_explicit, 'Explicit Euler', 'Δt', [
        ('Δt=0.0005, r=0.20', 40, 0.0005),
        ('Δt=0.001,  r=0.40', 40, 0.001),
        ('Δt=0.00125,r=0.50', 40, 0.00125),
        ('Δt=0.0015, r=0.60', 40, 0.0015),
    ], t_eval, 'plot06_eff_dt_explicit.png', stability_note='Stability: r ≤ 0.5')

    # Effect of dy
    effect_study(solve_explicit, 'Explicit Euler', 'Δy', [
        ('N=10, Δy=0.2',  10, 0.4*0.2**2),
        ('N=20, Δy=0.1',  20, 0.4*0.1**2),
        ('N=40, Δy=0.05', 40, 0.4*0.05**2),
        ('N=80, Δy=0.025',80, 0.4*0.025**2),
    ], t_eval, 'plot07_eff_dy_explicit.png')

    # Effect of r
    effect_study(solve_explicit, 'Explicit Euler', 'r', [
        ('r=0.10', 40, 0.10*dy**2),
        ('r=0.25', 40, 0.25*dy**2),
        ('r=0.40', 40, 0.40*dy**2),
        ('r=0.50', 40, 0.50*dy**2),
        ('r=0.60', 40, 0.60*dy**2),
    ], t_eval, 'plot08_eff_r_explicit.png', stability_note='Stability: r ≤ 0.5')

    # IMPLICIT EULER
    print('\n--- Effect Studies: Implicit Euler ---')
    print('  (Unconditionally stable \n')

    effect_study(solve_implicit, 'Implicit Euler', 'Δt', [
        ('Δt=0.0005, r=0.20', 40, 0.0005),
        ('Δt=0.001,  r=0.40', 40, 0.001),
        ('Δt=0.005,  r=2.0',  40, 0.005),
        ('Δt=0.025,  r=10.0', 40, 0.025),
        ('Δt=0.125,  r=50.0', 40, 0.125),
    ], t_eval, 'plot09_eff_dt_implicit.png', stability_note='Unconditionally stable')

    effect_study(solve_implicit, 'Implicit Euler', 'Δy', [
        ('N=10, Δy=0.2',  10, 0.4*0.2**2),
        ('N=20, Δy=0.1',  20, 0.4*0.1**2),
        ('N=40, Δy=0.05', 40, 0.4*0.05**2),
        ('N=80, Δy=0.025',80, 0.4*0.025**2),
    ], t_eval, 'plot10_eff_dy_implicit.png')

    effect_study(solve_implicit, 'Implicit Euler', 'r', [
        ('r=0.5',  40, 0.5*dy**2),
        ('r=2.0',  40, 2.0*dy**2),
        ('r=10.0', 40, 10.0*dy**2),
        ('r=50.0', 40, 50.0*dy**2),
    ], t_eval, 'plot11_eff_r_implicit.png', stability_note='Unconditionally stable')

    #  RK4
    print('\n--- Effect Studies: RK4 ---')
    print('  (Stability investigated numerically \n')

    effect_study(solve_rk4, 'RK4', 'Δt', [
        ('Δt=0.0005, r=0.20', 40, 0.0005),
        ('Δt=0.001,  r=0.40', 40, 0.001),
        ('Δt=0.0015, r=0.60', 40, 0.0015),
        ('Δt=0.002,  r=0.80', 40, 0.002),
    ], t_eval, 'plot12_eff_dt_rk4.png', stability_note='Stability limit observed numerically near r ≈ 0.7')

    effect_study(solve_rk4, 'RK4', 'Δy', [
        ('N=10, Δy=0.2',  10, 0.4*0.2**2),
        ('N=20, Δy=0.1',  20, 0.4*0.1**2),
        ('N=40, Δy=0.05', 40, 0.4*0.05**2),
        ('N=80, Δy=0.025',80, 0.4*0.025**2),
    ], t_eval, 'plot13_eff_dy_rk4.png')

    effect_study(solve_rk4, 'RK4', 'r', [
        ('r=0.25', 40, 0.25*dy**2),
        ('r=0.50', 40, 0.50*dy**2),
        ('r=0.65', 40, 0.65*dy**2),
        ('r=0.70', 40, 0.70*dy**2),
        ('r=0.80', 40, 0.80*dy**2),
    ], t_eval, 'plot14_eff_r_rk4.png', stability_note='Stability limit observed numerically near r ≈ 0.7')


    #  PLOT 15: Error convergence
    
    print('\n--- Error Convergence Analysis ---')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax_i, (solver, mname) in enumerate([(solve_explicit,'Explicit'), (solve_implicit,'Implicit'), (solve_rk4,'RK4')]):
        dts = [0.002, 0.001, 0.0005, 0.00025] if mname != 'Implicit' else [0.01, 0.005, 0.001, 0.0005]
        einfs = []
        for dtt in dts:
            r_val = dtt / dy**2
            if mname == 'Explicit' and r_val > 0.5:
                einfs.append(np.nan); continue
            yy, tt, pp, _, _, stable = solver(40, dtt, 0.5)
            idx = find_snap(tt, 0.5)
            if stable and np.all(np.isfinite(pp[idx])):
                einfs.append(Linf_error(pp[idx], yy, tt[idx]))
            else:
                einfs.append(np.nan)
        valid = [(d,e) for d,e in zip(dts, einfs) if np.isfinite(e)]
        if len(valid) >= 2:
            d_v, e_v = zip(*valid)
            axes[ax_i].loglog(d_v, e_v, 'bo-', lw=2, ms=8, label=f'{mname} E∞')
            
            d_ref = np.array(d_v)
            axes[ax_i].loglog(d_ref, e_v[0]*(d_ref/d_ref[0])**1, 'k--', lw=1, alpha=0.5, label='O(Δt)')
            if mname == 'RK4':
                axes[ax_i].loglog(d_ref, e_v[0]*(d_ref/d_ref[0])**4, 'r--', lw=1, alpha=0.5, label='O(Δt⁴)')
        axes[ax_i].set_xlabel('Δt'); axes[ax_i].set_ylabel('E∞ at t*=0.5')
        axes[ax_i].set_title(f'{mname}: Error vs Δt')
        axes[ax_i].legend(fontsize=8); axes[ax_i].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT/'plot15_error_convergence.png'); plt.close(fig)
    print('  plot15_error_convergence.png')

    # error convergence
    conv_rows = []
    for solver, mname in [(solve_explicit,'Explicit'), (solve_implicit,'Implicit'), (solve_rk4,'RK4')]:
        dts_c = [0.002, 0.001, 0.0005, 0.00025] if mname != 'Implicit' else [0.01, 0.005, 0.001, 0.0005]
        for dtt in dts_c:
            r_val = dtt / dy**2
            if mname == 'Explicit' and r_val > 0.5:
                conv_rows.append([mname, dtt, r_val, 'UNSTABLE', 'UNSTABLE'])
                continue
            yy, tt, pp, _, _, stable = solver(40, dtt, 0.5)
            idx_c = find_snap(tt, 0.5)
            if stable and np.all(np.isfinite(pp[idx_c])):
                ei = Linf_error(pp[idx_c], yy, tt[idx_c])
                er = RMS_error(pp[idx_c], yy, tt[idx_c])
                conv_rows.append([mname, dtt, r_val, f'{ei:.6e}', f'{er:.6e}'])
            else:
                conv_rows.append([mname, dtt, r_val, 'DIVERGED', 'DIVERGED'])
    save_csv(DATA/'plot15_error_convergence_data.csv',
             ['Method', 'dt', 'r', 'E_inf_at_t0.5', 'E_rms_at_t0.5'],
             conv_rows)

    
    #  SUMMARY TABLE
    
    print('\n' + '='*85)
    print(' SUMMARY ERROR TABLE (baseline: Ny=40, dt=0.001, r=0.4, evaluated at t*=1.0)')
    print('='*85)
    print(f' {"Method":<20} {"E_inf":<14} {"E_rms":<14} {"Q_num":<14} {"Q_exact":<14} {"τ_num":<14} {"τ_exact":<14}')
    print(f' {"-"*90}')
    q_ex = exact_Q(t_final);  tau_ex = exact_tau(t_final)
    for name, y_s, t_s, p_s, Q_s, tau_s in [
        ('Explicit', y_e, t_e, p_e, Q_e, tau_e),
        ('Implicit', y_i, t_i, p_i, Q_i, tau_i),
        ('RK4',      y_r, t_r, p_r, Q_r, tau_r),
    ]:
        u_final = p_s[-1]
        ei = Linf_error(u_final, y_s, t_s[-1])
        er = RMS_error(u_final, y_s, t_s[-1])
        print(f' {name:<20} {ei:<14.6e} {er:<14.6e} {Q_s[-1]:<14.8f} {q_ex:<14.8f} {tau_s[-1]:<14.8f} {tau_ex:<14.8f}')

    
    summary_rows = []
    for name, y_s, t_s, p_s, Q_s, tau_s in [
        ('Explicit', y_e, t_e, p_e, Q_e, tau_e),
        ('Implicit', y_i, t_i, p_i, Q_i, tau_i),
        ('RK4',      y_r, t_r, p_r, Q_r, tau_r),
    ]:
        u_final = p_s[-1]
        ei = Linf_error(u_final, y_s, t_s[-1])
        er = RMS_error(u_final, y_s, t_s[-1])
        summary_rows.append([name, Ny, dt_base, dy, r_base, f'{ei:.6e}', f'{er:.6e}',
                             f'{Q_s[-1]:.10f}', f'{q_ex:.10f}', f'{abs(Q_s[-1]-q_ex):.6e}',
                             f'{tau_s[-1]:.10f}', f'{tau_ex:.10f}', f'{abs(tau_s[-1]-tau_ex):.6e}'])
    save_csv(DATA/'summary_table.csv',
             ['Method','Ny','dt','dy','r','E_inf','E_rms','Q_num','Q_exact','Q_error','tau_num','tau_exact','tau_error'],
             summary_rows)

    
    for name, t_s, Q_s, tau_s in [
        ('explicit', t_e, Q_e, tau_e),
        ('implicit', t_i, Q_i, tau_i),
        ('rk4',      t_r, Q_r, tau_r),
    ]:
        step = max(1, len(t_s)//1000)  
        t_ds = t_s[::step]
        Q_ds = Q_s[::step]
        tau_ds = tau_s[::step]
        Q_ex_ds = np.array([exact_Q(t) for t in t_ds])
        tau_ex_ds = np.array([exact_tau(t) for t in t_ds])
        rows_ts = list(zip(t_ds, Q_ds, Q_ex_ds, np.abs(Q_ds-Q_ex_ds),
                           tau_ds, tau_ex_ds, np.abs(tau_ds-tau_ex_ds)))
        save_csv(DATA/f'timeseries_{name}.csv',
                 ['t', 'Q_num', 'Q_exact', 'Q_error', 'tau_num', 'tau_exact', 'tau_error'],
                 rows_ts)

    print(f'\n All {15} plots saved to: {OUT.resolve()}')
    print(f' All CSV data saved to:  {DATA.resolve()}')
    print('='*65)

if __name__ == '__main__':
    main()
