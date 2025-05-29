import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from utils_cavity_ngsolve import*
from pinn_cavity import FEMPhysicsModule

from scipy.interpolate import RegularGridInterpolator
from ngsolve import GridFunction, CoefficientFunction, NodeId, TaskManager
from ngsolve.fem import NODE_TYPE
from matplotlib import gridspec

from netgen.geom2d import unit_square
from ngsolve import Mesh, Draw

from ngsolve import*
from netgen.occ import*
from netgen.occ import X
from ngsolve import GridFunction, CoefficientFunction
from ngsolve import NodeId
from ngsolve.fem import NODE_TYPE
from netgen.occ import X as OCCX, Rectangle, Circle
from netgen.geom2d import SplineGeometry
from netgen.geom2d import unit_square
from ngsolve import x, y

def main():
    ckpt_path = "unet_checkpoints_May2_test1/last.ckpt"
    config_path = "config_cavity.yaml"
    ux, uy, pressure = get_unet_pred(ckpt_path, config_path, 2500)

    #make_2dheatmap_matrix(ux, 'ux.png')
    #make_2dheatmap_matrix(uy, 'uy.png')
    #make_2dheatmap_matrix(pressure, 'press.png')
    #make_2dheatmap_matrix(np.sqrt(ux**2 + uy**2), 'velmag.png')


    #_, _, _, gfu = run_ngsolve_plotty(u_init=np.float64(ux), v_init=np.float64(uy), p_init=np.float64(pressure), 
    #                                 nx=64, ny=64, nu=1/2500, uin_max=1.0, tau=0.001, t_iter=1)
    
    '''
    gfu =  get_gfu(u_init=np.float64(ux.T), v_init=np.float64(uy.T), p_init=np.float64(pressure.T), nx=64, ny=64, nu = 1.0/2500)
    
    save_plot_vel_heatmap(gfu, 'velmap.png')
    
    save_plot_pressure_heatmap(gfu, 'pressmap.png')
    

    _, _, _, gfu2 = run_ngsolve_plotty(u_init=np.float64(ux), v_init=np.float64(uy), p_init=np.float64(pressure), 
                                         nx=32, ny=32, nu=1/2500, uin_max=1.0, tau=0.001, t_iter=250000)

    save_plot_vel_heatmap(gfu2, 'velmap_true.png')
    
    save_plot_pressure_heatmap(gfu, 'pressmap_true.png')
    '''
    
    '''
    for i in range(2000, 3100, 100):
    	Re = i
    	print(f'Re = {Re}')
    	
    	# gfu1 = prediction, gfu2 = ground truth
    	gfu1 =  get_gfu(u_init=np.float64(ux.T), v_init=np.float64(uy.T), p_init=np.float64(pressure.T), nx=64, ny=64, nu = 1.0/Re)
    	_, _, _, gfu2 = run_ngsolve_plotty(u_init=np.float64(ux), v_init=np.float64(uy), p_init=np.float64(pressure), 
                                         nx=32, ny=32, nu=1/Re, uin_max=1.0, tau=0.003, t_iter=250000)
                                         
    	compare_velocity_gfu_contour(gfu1, gfu2, filename = f'./myplots/vcompare_{Re}.png', levels=50)
    	plot_velocity_error_gfu(gfu1, gfu2, filename = f'./myplots/verr_{Re}.png', levels=50)
    
    	compare_pressure_gfu_contour(gfu1, gfu2, filename = f'./myplots/pcompare_{Re}.png', levels=50)
    	plot_pressure_error_gfu(gfu1, gfu2, filename=f'./myplots/perr_{Re}.png', levels=50)
    '''
    Re = 2500
    gfu1 =  get_gfu(u_init=np.float64(ux.T), v_init=np.float64(uy.T), p_init=np.float64(pressure.T), nx=64, ny=64, nu = 1.0/Re)
    _, _, _, gfu2 = run_ngsolve_plotty(u_init=np.float64(ux), v_init=np.float64(uy), p_init=np.float64(pressure), 
                                         nx=32, ny=32, nu=1/Re, uin_max=1.0, tau=0.003, t_iter=250000)
    
    plot_velocity_streamlines(gfu1, gfu2, filename=f'streamlines_{Re}_test1.png', density=1.5)
    compare_velocity_gfu_contour(gfu1, gfu2, filename = f'./myplots/vcompare_{Re}_test1.png', levels=50)
    compare_pressure_gfu_contour(gfu1, gfu2, filename = f'./myplots/pcompare_{Re}_test1.png', levels=50)
    
    ckpt_path = "unet_checkpoints_May2_test2/last.ckpt"
    config_path = "config_cavity.yaml"
    ux, uy, pressure = get_unet_pred(ckpt_path, config_path, 2500)
    gfu1 =  get_gfu(u_init=np.float64(ux.T), v_init=np.float64(uy.T), p_init=np.float64(pressure.T), nx=64, ny=64, nu = 1.0/Re)
    plot_velocity_streamlines(gfu1, gfu2, filename=f'streamlines_{Re}_test2.png', density=1.5)
    compare_velocity_gfu_contour(gfu1, gfu2, filename = f'./myplots/vcompare_{Re}_test2.png', levels=50)
    compare_pressure_gfu_contour(gfu1, gfu2, filename = f'./myplots/pcompare_{Re}_test2.png', levels=50)
    
    
    
    
    
    
    
    #plot_combined_velocity_comparison(gfu1, gfu2, filename=f'velocity_comparison_combined_{Re}.png', levels=50)
    #plot_combined_pressure_comparison(gfu1, gfu2, filename=f'pressure_comparison_combined_{Re}.png', levels=50)
    
    #plot_mesh()
    

def get_gfu(u_init=None, v_init=None, p_init=None, nx=32, ny=32, nu = 1.0/2500):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.05)).Curve(3)


    num_elements = mesh.ne
    area_per_element = 1.0 / num_elements

    # Approximate grid resolution
    n_grid = int(np.sqrt(1.0 / area_per_element))  # assuming square domain
    print(n_grid)

    # Define spaces
    V = VectorH1(mesh, order=3, dirichlet="top|bottom|left|right")
    Q = H1(mesh, order=2)
    X = V * Q

    u, p = X.TrialFunction()
    v, q = X.TestFunction()

    stokes = (nu * InnerProduct(grad(u), grad(v)) + div(u) * q + div(v) * p - 1e-10 * p * q) * dx
    a = BilinearForm(stokes).Assemble()
    f = LinearForm(X).Assemble()

    gfu = GridFunction(X)


    # Apply initial conditions if provided
    if u_init is not None and v_init is not None and p_init is not None:
        #ufunc = VoxelCoefficient((0,0), (1,1), u_init, linear=True)
        #vfunc = VoxelCoefficient((0,0), (1,1), v_init, linear=True)
        #pfunc = VoxelCoefficient((0,0), (1,1), p_init, linear=True)
        #gfu.components[0].Set(CoefficientFunction((ufunc, vfunc)))
        #gfu.components[1].Set(pfunc)

        vec_voxel = CoefficientFunction((
                        VoxelCoefficient((0, 0), (1, 1), u_init.T, linear=True),
                        VoxelCoefficient((0, 0), (1, 1), v_init.T, linear=True))
                    )
        gfu.components[0].Set(vec_voxel)

        gfu.components[1].Set(VoxelCoefficient((0, 0), (1, 1), p_init, linear=True))

    return gfu


def run_ngsolve_plotty(u_init=None, v_init=None, p_init=None, nx=32, ny=32, nu=0.001, uin_max=1.0, tau=0.001, t_iter=1000):
    # Create unit square domain
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.05)).Curve(3)

    num_elements = mesh.ne
    area_per_element = 1.0 / num_elements

    # Approximate grid resolution
    n_grid = int(np.sqrt(1.0 / area_per_element))  # assuming square domain
    print(n_grid)

    # Define spaces
    V = VectorH1(mesh, order=3, dirichlet="top|bottom|left|right")
    Q = H1(mesh, order=2)
    X = V * Q

    u, p = X.TrialFunction()
    v, q = X.TestFunction()

    stokes = (nu * InnerProduct(grad(u), grad(v)) + div(u) * q + div(v) * p - 1e-10 * p * q) * dx
    a = BilinearForm(stokes).Assemble()
    f = LinearForm(X).Assemble()

    gfu = GridFunction(X)


    # Apply initial conditions if provided
    if u_init is not None and v_init is not None and p_init is not None:
        #ufunc = VoxelCoefficient((0,0), (1,1), u_init, linear=True)
        #vfunc = VoxelCoefficient((0,0), (1,1), v_init, linear=True)
        #pfunc = VoxelCoefficient((0,0), (1,1), p_init, linear=True)
        #gfu.components[0].Set(CoefficientFunction((ufunc, vfunc)))
        #gfu.components[1].Set(pfunc)

        vec_voxel = CoefficientFunction((
                        VoxelCoefficient((0, 0), (1, 1), u_init.T, linear=True),
                        VoxelCoefficient((0, 0), (1, 1), v_init.T, linear=True))
                    )
        gfu.components[0].Set(vec_voxel)

        gfu.components[1].Set(VoxelCoefficient((0, 0), (1, 1), p_init, linear=True))

    # Apply lid motion
    lid_velocity = CoefficientFunction((uin_max, 0))
    gfu.components[0].Set(lid_velocity, definedon=mesh.Boundaries("top"))

    # Solve initial system
    #inv_stokes = a.mat.Inverse(X.FreeDofs())
    #res = f.vec - a.mat * gfu.vec
    #gfu.vec.data += inv_stokes * res

    mstar = BilinearForm(u * v * dx + tau * stokes).Assemble()
    inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

    conv = BilinearForm(X, nonassemble=True)
    conv += (Grad(u) * u) * v * dx

    i = 0
    vel = gfu.components[0]

    with TaskManager():
        while i < t_iter:
            res = conv.Apply(gfu.vec) + a.mat * gfu.vec
            gfu.vec.data -= tau * inv * res
            i += 1
        
    # Replace this with your own grid sampling function
    U, V, P = sample_on_uniform_grid(gfu, nx=nx, ny=ny)
    return U, V, P, gfu

def make_2dheatmap_matrix(data, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('2D Heatmap')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.tight_layout()
    plt.savefig(filename)
    

def get_unet_pred(ckpt_path, config_path, Re):
    # Load and predict
    model = load_model(ckpt_path, config_path)
    ux, uy, pressure = predict_unet_output(model, Re = Re)
    return ux, uy, pressure

def get_true_pred():
    mesh, X, u, p, v, q, f, gfu = get_ngsolve_params()
    ux_true, uy_true, pressure_true, gfu = run_ngsolve_custom_params(nu = 1/2500.0, uin_max = 1.0, tau = 0.001, t_iter = 5000, 
                                            U_initial=np.zeros((32,32)), V_initial=np.zeros((32,32)), P_initial=np.zeros((32,32)), 
                                            mesh = mesh, X = X, u = u, p = p, v = v, q = q, f = f, gfu = gfu)
    save_plot_vel_heatmap(gfu, 'velmap_true.png')
    save_plot_pressure_heatmap(gfu, 'pmap_true.png')
    return ux_true, uy_true, pressure_true
    

# --- Load Model ---
def load_model(ckpt_path, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = FEMPhysicsModule(
        model_type=config['model']['type'],
        learning_rate=config['model']['learning_rate'],
        fem_iterations=config['training']['fem_iterations'],
        model_config=config['model']['unet_config']
    )
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model

# --- Predict UNet output ---
def predict_unet_output(model, Re=2500, device='cpu'):
    Re_norm = (Re - 2000) / (3000 - 2000)
    input_tensor = torch.ones((1, 32, 32)) * Re_norm
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        pred = model(input_tensor)[0].cpu().numpy()

    return pred[0], pred[1], pred[2]  # ux, uy, pressure


# --- Plot heatmap from FEM GridFunction ---
def save_plot_vel_heatmap(gfu, filename):
        # Grid for evaluation
    nx, ny = 32, 32
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    Vmag = np.zeros_like(X)
    velocity = gfu.components[0]

    for i in range(ny):
        for j in range(nx):
            x = x_vals[j]
            y = y_vals[i]
            
            u_val, v_val = velocity(x, y)

            Vmag[i, j] = (u_val**2 + v_val**2)**0.5

    # Velocity magnitude heatmap
    plt.figure(figsize=(4, 4))
    contour = plt.contourf(X, Y, Vmag, levels=100, cmap='jet')
    plt.colorbar(contour, label='|u|')
    plt.title(f"Velocity Magnitude")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_plot_pressure_heatmap(gfu, filename):
    # Grid for evaluation
    nx, ny = 32, 32
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    P = np.zeros_like(X)
    pressure = gfu.components[1]

    for i in range(ny):
        for j in range(nx):
            x = x_vals[j]
            y = y_vals[i]
            
            P[i, j] = pressure(x, y)

    # Pressure heatmap
    plt.figure(figsize=(4,4))
    contour = plt.contourf(X, Y, P, levels=50, cmap='coolwarm')
    plt.colorbar(contour, label='Pressure')
    plt.title(f"Pressure Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
def compare_velocity_gfu_contour(gfu1, gfu2, filename='velocity_comparison.png', levels=100):
    """
    Compare velocity magnitudes from two NGSolve GridFunctions using side-by-side contour plots.

    Parameters:
        gfu1 (GridFunction): First GridFunction (e.g., prediction).
        gfu2 (GridFunction): Second GridFunction (e.g., ground truth).
        filename (str): Output file path.
        levels (int): Number of contour levels.
    """
    nx, ny = 32, 32
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    def eval_velocity_mag(gfu):
        vel = gfu.components[0]
        Vmag = np.zeros((ny, nx))
        for i in range(ny):
            for j in range(nx):
                x = x_vals[j]
                y = y_vals[i]
                u, v = vel(x, y)
                Vmag[i, j] = np.sqrt(u**2 + v**2)
        return Vmag

    Vmag1 = eval_velocity_mag(gfu1)
    Vmag2 = eval_velocity_mag(gfu2)

    vmin = min(Vmag1.min(), Vmag2.min())
    vmax = max(Vmag1.max(), Vmag2.max())

    fig = plt.figure(figsize=(10, 4))
    spec = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=[1, 1, 0.05], wspace=0.2)

    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    cax = fig.add_subplot(spec[2])

    cs0 = ax0.contourf(X, Y, Vmag1, levels=levels, cmap='jet', vmin=vmin, vmax=vmax)
    cs1 = ax1.contourf(X, Y, Vmag2, levels=levels, cmap='jet', vmin=vmin, vmax=vmax)

    ax0.set_title("Predicted Velocity")
    ax1.set_title("True Velocity")
    for ax in [ax0, ax1]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    fig.colorbar(cs0, cax=cax)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
def compare_pressure_gfu_contour(gfu1, gfu2, filename='pressure_comparison.png', levels=50):
    """
    Compare pressure fields from two NGSolve GridFunctions using side-by-side contour plots.

    Parameters:
        gfu1 (GridFunction): First GridFunction (e.g., prediction).
        gfu2 (GridFunction): Second GridFunction (e.g., ground truth).
        filename (str): Output file path.
        levels (int): Number of contour levels.
    """
    nx, ny = 32, 32
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    def eval_pressure(gfu):
        p = gfu.components[1]
        P = np.zeros((ny, nx))
        for i in range(ny):
            for j in range(nx):
                x = x_vals[j]
                y = y_vals[i]
                P[i, j] = p(x, y)
        return P

    P1 = eval_pressure(gfu1)
    P2 = eval_pressure(gfu2)

    vmin = min(P1.min(), P2.min())
    vmax = max(P1.max(), P2.max())

    fig = plt.figure(figsize=(10, 4))
    spec = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=[1, 1, 0.05], wspace=0.2)

    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    cax = fig.add_subplot(spec[2])

    cs0 = ax0.contourf(X, Y, P1, levels=levels, cmap='coolwarm', vmin=vmin, vmax=vmax)
    cs1 = ax1.contourf(X, Y, P2, levels=levels, cmap='coolwarm', vmin=vmin, vmax=vmax)

    ax0.set_title("Predicted Pressure")
    ax1.set_title("True Pressure")
    for ax in [ax0, ax1]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    fig.colorbar(cs0, cax=cax)
    fig.colorbar(cs0, cax=cax)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_velocity_error_gfu(gfu_pred, gfu_true, filename='velocity_error.png', levels=100):
    """
    Compute and plot the pointwise velocity error between two NGSolve GridFunctions.

    Parameters:
        gfu_pred (GridFunction): Predicted velocity field.
        gfu_true (GridFunction): Ground truth velocity field.
        filename (str): Output file path for contour plot.
        levels (int): Number of contour levels.
    """
    nx, ny = 32, 32
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    vel_pred = gfu_pred.components[0]
    vel_true = gfu_true.components[0]
    error = np.zeros((ny, nx))

    for i in range(ny):
        for j in range(nx):
            x = x_vals[j]
            y = y_vals[i]
            u_pred, v_pred = vel_pred(x, y)
            u_true, v_true = vel_true(x, y)
            error[i, j] = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)

    plt.figure(figsize=(5, 4))
    contour = plt.contourf(X, Y, error, levels=levels, cmap='plasma')
    #plt.colorbar(contour, label='Velocity Error')
    plt.colorbar(contour)
    plt.title("Velocity Error")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_pressure_error_gfu(gfu_pred, gfu_true, filename='pressure_error.png', levels=100):
    """
    Compute and plot the pointwise pressure error between two NGSolve GridFunctions.

    Parameters:
        gfu_pred (GridFunction): Predicted pressure field.
        gfu_true (GridFunction): Ground truth pressure field.
        filename (str): Output file path for contour plot.
        levels (int): Number of contour levels.
    """
    nx, ny = 32, 32
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    p_pred = gfu_pred.components[1]
    p_true = gfu_true.components[1]
    error = np.zeros((ny, nx))

    for i in range(ny):
        for j in range(nx):
            x = x_vals[j]
            y = y_vals[i]
            error[i, j] = abs(p_pred(x, y) - p_true(x, y))

    plt.figure(figsize=(5, 4))
    contour = plt.contourf(X, Y, error, levels=levels, cmap='magma')
    #plt.colorbar(contour, label='Pressure Error')
    plt.colorbar(contour)
    plt.title("Pressure Error")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_combined_velocity_comparison(gfu_pred, gfu_true, filename='velocity_comparison_combined.png', levels=100):
    nx, ny = 32, 32
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    def eval_velocity_mag(gfu):
        vel = gfu.components[0]
        Vmag = np.zeros((ny, nx))
        for i in range(ny):
            for j in range(nx):
                x = x_vals[j]
                y = y_vals[i]
                u, v = vel(x, y)
                Vmag[i, j] = np.sqrt(u**2 + v**2)
        return Vmag

    Vmag_pred = eval_velocity_mag(gfu_pred)
    Vmag_true = eval_velocity_mag(gfu_true)
    vmin = min(Vmag_pred.min(), Vmag_true.min())
    vmax = max(Vmag_pred.max(), Vmag_true.max())

    # Velocity error
    error = np.sqrt((Vmag_pred - Vmag_true)**2)

    fig = plt.figure(figsize=(15, 4))
    spec = gridspec.GridSpec(ncols=5, nrows=1, width_ratios=[1, 1, 0.05, 1, 0.05], wspace=0.3)

    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    cbar_ax1 = fig.add_subplot(spec[2])
    ax2 = fig.add_subplot(spec[3])
    cbar_ax2 = fig.add_subplot(spec[4])

    cs0 = ax0.contourf(X, Y, Vmag_pred, levels=levels, cmap='jet', vmin=vmin, vmax=vmax)
    cs1 = ax1.contourf(X, Y, Vmag_true, levels=levels, cmap='jet', vmin=vmin, vmax=vmax)
    cs2 = ax2.contourf(X, Y, error, levels=levels, cmap='plasma')

    fig.colorbar(cs0, cax=cbar_ax1)
    fig.colorbar(cs2, cax=cbar_ax2)

    ax0.set_title("Predicted Velocity")
    ax1.set_title("True Velocity")
    ax2.set_title("Velocity Error")

    for ax in [ax0, ax1, ax2]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
def plot_combined_pressure_comparison(gfu_pred, gfu_true, filename='pressure_comparison_combined.png', levels=100):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import numpy as np

    nx, ny = 32, 32
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    def eval_pressure(gfu):
        p = gfu.components[1]
        P = np.zeros((ny, nx))
        for i in range(ny):
            for j in range(nx):
                x = x_vals[j]
                y = y_vals[i]
                P[i, j] = p(x, y)
        return P

    P_pred = eval_pressure(gfu_pred)
    P_true = eval_pressure(gfu_true)
    Perr = np.sqrt((P_pred - P_true)**2)

    vmin = min(P_pred.min(), P_true.min())
    vmax = max(P_pred.max(), P_true.max())

    # Setup a grid with extra room for colorbars
    fig = plt.figure(figsize=(18, 5))
    spec = gridspec.GridSpec(nrows=1, ncols=6, width_ratios=[1, 1, 0.05, 1, 0.05, 0.1], wspace=0.4)

    ax0 = fig.add_subplot(spec[0])  # Predicted
    ax1 = fig.add_subplot(spec[1])  # True
    cax1 = fig.add_subplot(spec[2])  # Shared colorbar for 0+1
    ax2 = fig.add_subplot(spec[3])  # Error
    cax2 = fig.add_subplot(spec[4])  # Colorbar for error

    # Plot pressure fields
    cs0 = ax0.contourf(X, Y, P_pred, levels=levels, cmap='coolwarm', vmin=vmin, vmax=vmax)
    cs1 = ax1.contourf(X, Y, P_true, levels=levels, cmap='coolwarm', vmin=vmin, vmax=vmax)
    cs2 = ax2.contourf(X, Y, Perr, levels=levels, cmap='magma')

    # Colorbars
    fig.colorbar(cs0, cax=cax1)
    fig.colorbar(cs2, cax=cax2)

    # Titles
    ax0.set_title("Predicted Pressure")
    ax1.set_title("True Pressure")
    ax2.set_title("Pressure Error")

    for ax in [ax0, ax1, ax2]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    plt.savefig(filename, dpi=300)
    plt.close()
    
def plot_mesh():
    # Generate and curve the mesh
    #mesh = Mesh(unit_square.GenerateMesh(maxh=0.05)).Curve(3)
    
    Lx = 2
    Ly = 0.41
    circ_centerx = 0.2
    circ_centery = 0.2
    circ_radius = 0.05
    nx = 64
    ny = 32
    shape = Rectangle(Lx,Ly).Circle(circ_centerx,circ_centery,circ_radius).Reverse().Face()
    shape.edges.name="wall"
    shape.edges.Min(OCCX).name="inlet"
    shape.edges.Max(OCCX).name="outlet"
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.07)).Curve(3)

    # Extract mesh points
    points = np.array([v.point for v in mesh.vertices])

    # Extract elements using the VOL enum
    elements = [el.vertices for el in mesh.Elements(VOL)]

    # Plot the mesh
    plt.figure(figsize=(6, 6))
    for el in elements:
        pts = points[[v.nr for v in el]]
        pts = np.vstack((pts, pts[0]))  # Close the triangle
        plt.plot(pts[:, 0], pts[:, 1], 'k-', linewidth=0.5)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.savefig("mesh_plot.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
def plot_velocity_streamlines(gfu_pred, gfu_true, filename='streamlines.png', density=1.5):
    nx, ny = 32, 32
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    def get_uv(gfu):
        vel = gfu.components[0]
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(ny):
            for j in range(nx):
                x = x_vals[j]
                y = y_vals[i]
                u, v = vel(x, y)
                U[i, j] = u
                V[i, j] = v
        return U, V

    U_pred, V_pred = get_uv(gfu_pred)
    U_true, V_true = get_uv(gfu_true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Predicted
    axes[0].streamplot(X, Y, U_pred, V_pred, color=np.sqrt(U_pred**2 + V_pred**2), cmap='viridis', density=density)
    axes[0].set_title("Predicted Streamlines")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_aspect('equal')

    # True
    axes[1].streamplot(X, Y, U_true, V_true, color=np.sqrt(U_true**2 + V_true**2), cmap='viridis', density=density)
    axes[1].set_title("True Streamlines")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_aspect('equal')

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
# --- Main ---
if __name__ == "__main__":
    main()
