from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import numpy as np
import re
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def formato_graficas():
    plt.rcParams.update({
        # Typography and Rendering
        "text.usetex": True,            # Use LaTeX for all text rendering
        "text.latex.preamble": r"\usepackage{amsmath} \boldmath \usepackage{bm} \renewcommand{\familydefault}{\sfdefault} \bfseries", # Forces global math and text bold
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,                # Standard size for journal columns

        # Axes and Titles
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 1.0,          # Professional axis border width
        "axes.grid": True,              # Enable grid by default for technical analysis

        # Lines and Markers
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "lines.markerfacecolor": "none", # Hollow markers for better B&W distinction
        "lines.markeredgewidth": 1.0,

        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,    # Minor ticks for precise data reading
        "ytick.minor.visible": True,

        # Legend
        "legend.fontsize": 10,
        "legend.frameon": True,         # Border helps visibility in dense plots
        "legend.edgecolor": "black",    
        "legend.fancybox": False,       # Rectangular corners for a formal look

        # Grid
        "grid.color": "silver",
        "grid.alpha": 0.5,
        "grid.linestyle": ":",          # Dotted lines are less intrusive

        # Figures and Saving
        "figure.figsize": [6.4, 4.8],   # Default size (can be adjusted per plot)
        "savefig.dpi": 300,             # Minimum resolution for publication
        "savefig.format": "pdf",        # Vector format to avoid pixelation
        "savefig.bbox": "tight"         # Removes unnecessary white margins
    })

def data_set(z, z_dot):

    z_train, z_test, \
    z_dot_train, z_dot_test = train_test_split(
                                z,
                                z_dot,
                                test_size=0.30,
                                shuffle=False
                            )
    np.savez("../Data/data_set.npz",    z_train=z_train, 
                                        z_test=z_test, 
                                        z_dot_train=z_dot_train, 
                                        z_dot_test=z_dot_test)

    print("Se genero y guardo con exitó")

def data_set(z, z_dot):

    z_train, z_test, \
    z_dot_train, z_dot_test = train_test_split(
                                z,
                                z_dot,
                                test_size=0.30,
                                shuffle=False
                            )
    np.savez("../Data/data_set.npz",    z_train=z_train, 
                                        z_test=z_test, 
                                        z_dot_train=z_dot_train, 
                                        z_dot_test=z_dot_test)

    print("Se genero y guardo con exitó")

def leer_metricas_txt(ruta_txt):

    with open(ruta_txt, "r", encoding="utf-8") as f:
        contenido = f.read()

    regex_num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    loss_train  = re.findall(r"loss train\s*:\s*(" + regex_num + ")", contenido)
    loss_test  = re.findall(r"loss test\s*:\s*(" + regex_num + ")", contenido)
   
    lrs = re.findall(r"lr\s*:\s*(" + regex_num + ")", contenido)

    # Conversion to float
    return (
        list(map(float, loss_train)),
        list(map(float, loss_test)),
        list(map(float, lrs))
    )

#Hamiltonian function
def Hamiltoniano_potencial(r, p, masa, G_cnst, condicional="all"):   
    Ec = 0
    if condicional in ["cm", "all"]:
        p_squared_norm = np.sum(p**2, axis=2) 
        Ec = np.sum(p_squared_norm / (2 * masa), axis=1)
        
    Ep = 0
    if condicional in ["pos", "all"]:
        n_cuerpos = masa.shape[0]
        i, j = np.triu_indices(n_cuerpos, k=1)
        diff = r[:, i, :] - r[:, j, :]
        r_scalar = np.sqrt(np.sum(diff**2, axis=2))    
        mass_products = masa[i] * masa[j]
        Ep = np.sum(-G_cnst * mass_products / r_scalar, axis=1)
                
    return Ec + Ep

def set_seed(seed = 42):
    # Python y Random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch (CPU)
    torch.manual_seed(seed)

def get_orbit(model, q_init, p_init, steps, dt, Sz, Sz_dot):
    sz_q, sz_p = Sz.flatten()[:30], Sz.flatten()[30:]
    szd_q, szd_p = Sz_dot.flatten()[:30], Sz_dot.flatten()[30:]
    
    t_span = (0.0, (steps - 1) * dt)
    t_eval = np.linspace(0, (steps - 1) * dt, steps)
    
    x0 = np.concatenate([q_init.flatten(), p_init.flatten()])

    def fvec_np(t, x):
        q_norm = x[:30] / sz_q
        p_norm = x[30:] / sz_p
        
        z_t = torch.tensor(np.concatenate([q_norm, p_norm]), dtype=torch.float64).unsqueeze(0)
        
        dz_dt = model.time_derivative_use(z_t)
        
        if isinstance(dz_dt, tuple) or isinstance(dz_dt, list):
            dz_dt = dz_dt[0]
            
        dq_dt_norm = dz_dt[:, :30]
        dp_dt_norm = dz_dt[:, 30:]
        
        dq = dq_dt_norm.detach().cpu().numpy().flatten() * szd_q
        dp = dp_dt_norm.detach().cpu().numpy().flatten() * szd_p
        
        return np.concatenate([dq, dp])

    res = solve_ivp(fvec_np, t_span, x0, t_eval=t_eval, method='DOP853', rtol=1e-15, atol=1e-13)
    
    return res.y.T

def scipy_predict_qp_model_orbits(model, q_init, p_init, steps, dt, Sz, Sz_dot):
    output = get_orbit(model, q_init, p_init, steps, dt, Sz, Sz_dot)
    return output[:, :30], output[:, 30:]
    
class Block_HNN_SL(nn.Module):
    def __init__(self, input_dim=60, num_neuronas=256, output_dim=1, func_act_V=nn.Tanh):
        super().__init__()

        # Encoder for the positions q -> q_latente
        self.w_r = nn.Parameter(torch.ones(30, dtype=torch.float64))
        self.w_rb = nn.Parameter(torch.ones(30, dtype=torch.float64))

        # Encoder for the moments p -> p_latente
        self.w_p = nn.Parameter(torch.ones(30, dtype=torch.float64))
        self.w_pb = nn.Parameter(torch.ones(30, dtype=torch.float64))

        self.MLP_V = nn.Sequential(
            nn.Linear(input_dim//2, num_neuronas),
            func_act_V(),
            nn.Linear(num_neuronas, num_neuronas),
            func_act_V(),
            nn.Linear(num_neuronas, num_neuronas),
            func_act_V(),
            nn.Linear(num_neuronas, output_dim)
        )

        # Decoder for r_dot_latente -> r_dot
        self.w_r_dot = nn.Parameter(torch.ones(30, dtype=torch.float64))

        # Decoder for p_dot_latente -> p_dot
        self.w_p_dot = nn.Parameter(torch.ones(30, dtype=torch.float64))

        # Mass invers
        self.wT = nn.Parameter(torch.ones(30, dtype=torch.float64))

        self.init_weights() 

    def encoder_r(self, r):
        return self.w_r * r + self.w_rb

    def encoder_p(self, p):
        return self.w_p * p + self.w_pb

    def decoder_r_dot(self, r_dot_latente):
        return self.w_r_dot * r_dot_latente

    def decoder_p_dot(self, p_dot_latente):
        return self.w_p_dot * p_dot_latente

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(f"Inicializando capa: {m}")
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def T_energy(self, p):
        p_latente = self.encoder_p(p)
        return torch.sum(self.wT * p_latente.pow(2), dim=1, keepdim=True), p_latente

    def V_energy(self, r):
        r_latente = self.encoder_r(r)
        return self.MLP_V(r_latente), r_latente

    def forward(self, r, p):
        # Kinetic energy
        T, p_latente = self.T_energy(p)
        
        # Potential energy
        V, r_latente = self.V_energy(r)

        # Hamiltonian Total
        return T + V, r_latente, p_latente

    def time_derivative(self, r, p, train=True):
        rh = r.detach().requires_grad_(True)
        ph = p.detach().requires_grad_(True)

        H, r_latente, p_latente = self.forward(rh, ph)

        dHdr_latente, dHdp_latente = torch.autograd.grad(
            outputs=H.sum(),
            inputs=[r_latente, p_latente],
            create_graph=train,
            retain_graph=train,
            allow_unused=False
        )

        # Decoders
        dr_dt = self.decoder_r_dot(dHdp_latente)
        dp_dt = self.decoder_p_dot(-dHdr_latente)

        return dr_dt, dp_dt

    def time_derivative_use(self, z):
        if not torch.is_tensor(z):
            z = torch.as_tensor(z, dtype=torch.float64, device=self.w_r.device)

        z = z.detach().requires_grad_(True)

        r, p = z[:, :30], z[:, 30:]

        H_net, r_latente, p_latente = self.forward(r, p)

        grad_H = torch.autograd.grad(
            outputs=H_net.sum(),
            inputs=[r_latente, p_latente],
            create_graph=False,
            retain_graph=False,
            allow_unused=False
        )

        assert grad_H[0] is not None and grad_H[1] is not None, "Error: Gradiente nulo en la integración"

        dr_dt = self.decoder_r_dot(grad_H[1])  # dH/dp
        dp_dt = self.decoder_p_dot(-grad_H[0]) # -dH/dr

        return torch.cat([dr_dt, dp_dt], dim=1)

    def criterion(self, residual):
        return torch.mean(residual.pow(2))
        
    def loss_step(self, z, z_dot, train):
        r, p = z[:, :30], z[:, 30:]
        r_dot, p_dot = z_dot[:, :30], z_dot[:, 30:]

        # Derivadas dot forward
        dr_dt_pred, dp_dt_pred = self.time_derivative(r, p, train)

        # Residuo manual sin llamadas a librerías funcionales
        res_r_dot = r_dot - dr_dt_pred
        res_p_dot = p_dot - dp_dt_pred

        # Loss physics utilizando tu método `criterion` optimizado internamente con .pow(2)
        loss_physics = self.criterion(res_r_dot) + self.criterion(res_p_dot)
                
        return loss_physics

def plot_simulation_metrics(ppr, dt_cnst, full_nn_q, full_nn_p, Zt, planets, colors, path_save, max_year):
    print(f"Simulating year {ppr}")
    steps_current_year = int(ppr / dt_cnst + 1)
    
    # Slice arrays for the current plotting period without resaving files
    nn_q_chunk = full_nn_q[:steps_current_year]
    nn_p_chunk = full_nn_p[:steps_current_year]
    
    iter_idx = nn_q_chunk.shape[0]
    Z_current = Zt[:iter_idx]
    
    r_inv_real = Z_current[:, :30].reshape(-1, 10, 3)
    p_inv_real = Z_current[:, 30:].reshape(-1, 10, 3)
    r_inv_pred = nn_q_chunk.reshape(-1, 10, 3)
    p_inv_pred = nn_p_chunk.reshape(-1, 10, 3)

    # 1. PLOT MSE Mean Squared Error
    r_mse = np.mean(np.square(r_inv_real - r_inv_pred), axis=2)
    p_mse = np.mean(np.square(p_inv_real - p_inv_pred), axis=2)
    year_axis = np.linspace(3200, 3200+ppr, r_mse.shape[0])

    if max_year == ppr:
        np.savez_compressed(path_save, r_mse=r_mse, p_mse=p_mse, year_axis=year_axis)

    # Figure for Positions MSE
    fig_pos, ax_pos = plt.subplots(figsize=(8, 5))
    for i in range(10):
        ax_pos.semilogy(year_axis, r_mse[:, i], label=planets[i])
    ax_pos.set_title(fr"\textbf{{MSE Positions - {ppr} Years}}", fontsize=14)
    ax_pos.set_xlabel(r"\textbf{Years}", fontsize=11)
    ax_pos.set_ylabel(r"\textbf{MSE (log)}", fontsize=11)
    ax_pos.grid(True, which="both", alpha=0.3)
    ax_pos.tick_params(axis='both', which='major', labelsize=10)
    ax_pos.legend(loc='best', frameon=True, shadow=False, fontsize=11, ncol=2, columnspacing=1.5)
    plt.tight_layout()
    plt.show()

    # Figure for Moments MSE
    fig_mom, ax_mom = plt.subplots(figsize=(8, 5))
    for i in range(10):
        ax_mom.semilogy(year_axis, p_mse[:, i], label=planets[i])
    fig_mom.canvas.draw()
    ax_mom.set_title(fr"\textbf{{MSE Moments - {ppr} Years}}", fontsize=14)
    ax_mom.set_xlabel(r"\textbf{Years}", fontsize=11)
    ax_mom.set_ylabel(r"\textbf{MSE (log)}", fontsize=11)
    ax_mom.grid(True, which="both", alpha=0.3)
    ax_mom.tick_params(axis='both', which='major', labelsize=10)
    ax_mom.legend(loc='best', frameon=True, shadow=False, fontsize=11, ncol=2, columnspacing=1.5)
    plt.tight_layout()
    plt.show()

    # 2. POSITIONS 2D Orbit trajectories on XY plane
    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    axes = axes.flatten()
    step = 10
    for i in range(12):
        if i < 10:
            axes[i].plot(r_inv_pred[::step, i, 0], r_inv_pred[::step, i, 1], "--", color="red", label="Predicted", alpha=0.5)
            axes[i].plot(r_inv_real[::step, i, 0], r_inv_real[::step, i, 1], "-", color="white", label="True")
            axes[i].set_title(fr"\textbf{{{planets[i]}}}", fontsize=14)
            axes[i].set_xlabel(r"\textbf{X (AU)}", fontsize=12)
            axes[i].set_ylabel(r"\textbf{Y (AU)}", fontsize=12)
            axes[i].axis('equal')
            axes[i].grid(False)
            axes[i].grid(False, which='minor')
            axes[i].set_facecolor('black')
            axes[i].tick_params(axis='both', which='major', labelsize=11, colors='black')
            axes[i].legend(loc='center', frameon=True, facecolor='white', edgecolor='black', labelcolor='black', fontsize=11, shadow=False)
        else:
            fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()

    # 3. POSITIONS 3D
    fig, axes = plt.subplots(5, 2, figsize=(12, 26), subplot_kw={'projection': '3d'})
    fig.set_facecolor('#ffffff')
    axes = axes.flatten()
    for i in range(10):
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_zticklabels([])
        axes[i].plot(r_inv_pred[::step, i, 0], r_inv_pred[::step, i, 1], r_inv_pred[::step, i, 2], "-",
                        color="red", label="Predicted")
        axes[i].plot(r_inv_real[::step, i, 0], r_inv_real[::step, i, 1], r_inv_real[::step, i, 2], "--",
                        color="k", label="True", alpha=0.5)
        
        axes[i].plot(r_inv_pred[-1, i, 0], r_inv_pred[-1, i, 1], r_inv_pred[-1, i, 2], "o", color="red")
        axes[i].plot(r_inv_real[-1, i, 0], r_inv_real[-1, i, 1], r_inv_real[-1, i, 2], "o", color="k")          
        
        axes[i].set_title(fr"\textbf{{{planets[i]}}}", fontsize=15, pad=5)           
        axes[i].set_xlabel(r"\textbf{X (AU)}", fontsize=12)
        axes[i].set_ylabel(r"\textbf{Y (AU)}", fontsize=12)
        axes[i].set_zlabel(r"\textbf{Z (AU)}", fontsize=12)          
        axes[i].legend(loc="best", fontsize=11, frameon=True, facecolor='white', edgecolor='black', labelcolor='black', shadow=False)
        
        x_limits = axes[i].get_xlim3d()
        y_limits = axes[i].get_ylim3d()
        z_limits = axes[i].get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        axes[i].set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        axes[i].set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        axes[i].set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        
        axes[i].set_box_aspect([1, 1, 1])
        
    plt.subplots_adjust(hspace=0.12, wspace=-0.05) 
    fig.suptitle(fr"\textbf{{3D Orbital Trajectories Modeling — {ppr}-Year Evolution}}", fontsize=18, y=0.91) 
    plt.show()


    # 4.1. All Planet Orbits
    fig_all, ax_all = plt.subplots(figsize=(7, 7))

    ax_all.set_facecolor("#000000")
    for i in range(10):
        ax_all.plot(r_inv_pred[::step, i, 0], r_inv_pred[::step, i, 1], color=colors[i], 
                linestyle='--', alpha=1.0, linewidth=0.5, label=f"{planets[i]} (Model)") # alpha=0.8,

        rgb = mcolors.to_rgb(colors[i])
        inverse_color = (1.0 - rgb[0], 1.0 - rgb[1], 1.0 - rgb[2])

        ax_all.plot(r_inv_real[::step, i, 0], r_inv_real[::step, i, 1], color=inverse_color, 
                linestyle='-', alpha=1.0, linewidth=0.25, label=f"{planets[i]} (True)")  # alpha=0.3,

        ax_all.scatter(r_inv_real[0, i, 0], r_inv_real[0, i, 1], 
                    color=colors[i], 
                    s=10,
                    edgecolors='white',
                    marker='o',
                    zorder=10, 
                    label=f'Start {planets[i]}')

    # Forzar límites estrictamente simétricos y cuadrados basados en el alcance del planeta más lejano
    max_val = np.max(np.abs(r_inv_real[::step, :10, :2]))
    padding = 1.05
    limit_val = max_val * padding
    ax_all.set_xlim(-limit_val, limit_val)
    ax_all.set_ylim(-limit_val, limit_val)

    ax_all.set_aspect('equal')
    ax_all.set_title(r"\textbf{Solar System Orbit}", fontsize=14, pad=15)
    ax_all.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
              fontsize=10, facecolor='#1e1e1e', labelcolor='white',
              edgecolor='white', framealpha=0.8, handlelength=3)
    ax_all.tick_params(axis='both', which='major', labelsize=10, colors='black')
    ax_all.set_xlabel(r"\textbf{X (AU)}", fontsize=11)
    ax_all.set_ylabel(r"\textbf{Y (AU)}", fontsize=11)
    ax_all.grid(False)
    
    plt.tight_layout()
    plt.show()


    # 4.2. Sun to Mars Orbits
    fig_inner, ax_inner = plt.subplots(figsize=(7, 7))
    
    ax_inner.set_facecolor("#000000")
    for i in range(5):
        ax_inner.plot(r_inv_pred[::step, i, 0], r_inv_pred[::step, i, 1], color=colors[i], 
                linestyle='--', alpha=1.0, linewidth=0.5, label=f"{planets[i]} (Model)") # alpha=0.8,

        rgb = mcolors.to_rgb(colors[i])
        inverse_color = (1.0 - rgb[0], 1.0 - rgb[1], 1.0 - rgb[2])

        ax_inner.plot(r_inv_real[::step, i, 0], r_inv_real[::step, i, 1], color=inverse_color, 
                linestyle='-', alpha=1.0, linewidth=0.25, label=f"{planets[i]} (True)")  # alpha=0.3,

        ax_inner.scatter(r_inv_real[0, i, 0], r_inv_real[0, i, 1], 
                   color=colors[i], 
                   s=10,
                   edgecolors='white',
                   marker='o',
                   zorder=10, 
                   label=f'Start {planets[i]}')
        
    ax_inner.set_xlim(-2.5, 2.5)
    ax_inner.set_ylim(-2.5, 2.5)
    
    ax_inner.set_aspect('equal')
    ax_inner.set_title(r"\textbf{Sun-to-Mars Orbit}", fontsize=14, pad=15)
    ax_inner.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
              fontsize=10, facecolor='#1e1e1e', labelcolor='white',
              edgecolor='white', framealpha=0.8, handlelength=3)
    ax_inner.tick_params(axis='both', which='major', labelsize=10, colors='black')
    ax_inner.set_xlabel(r"\textbf{X (AU)}", fontsize=11)
    ax_inner.set_ylabel(r"\textbf{Y (AU)}", fontsize=11)
    ax_inner.grid(False)
    
    plt.tight_layout()
    plt.show()

class Block_HNN_SL_240(nn.Module):
    def __init__(self, input_dim=60, num_neuronas=240, output_dim=1, func_act_V=nn.Tanh):
        super().__init__()

        # Encoder for the positions q -> q_latente
        self.w_r = nn.Parameter(torch.ones(30, dtype=torch.float64))
        self.w_rb = nn.Parameter(torch.ones(30, dtype=torch.float64))

        # Encoder for the moments p -> p_latente
        self.w_p = nn.Parameter(torch.ones(30, dtype=torch.float64))
        self.w_pb = nn.Parameter(torch.ones(30, dtype=torch.float64))

        self.MLP_V = nn.Sequential(
            nn.Linear(input_dim//2, num_neuronas),
            func_act_V(),
            nn.Linear(num_neuronas, num_neuronas),
            func_act_V(),
            nn.Linear(num_neuronas, num_neuronas),
            func_act_V(),
            nn.Linear(num_neuronas, output_dim)
        )

        # Decoder for r_dot_latente -> r_dot
        self.w_r_dot = nn.Parameter(torch.ones(30, dtype=torch.float64))

        # Decoder for p_dot_latente -> p_dot
        self.w_p_dot = nn.Parameter(torch.ones(30, dtype=torch.float64))

        # Mass invers
        self.wT = nn.Parameter(torch.ones(30, dtype=torch.float64))

        self.init_weights() 

    def encoder_r(self, r):
        return self.w_r * r + self.w_rb

    def encoder_p(self, p):
        return self.w_p * p + self.w_pb

    def decoder_r_dot(self, r_dot_latente):
        return self.w_r_dot * r_dot_latente

    def decoder_p_dot(self, p_dot_latente):
        return self.w_p_dot * p_dot_latente

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(f"Inicializando capa: {m}")
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def T_energy(self, p):
        p_latente = self.encoder_p(p)
        return torch.sum(self.wT * p_latente.pow(2), dim=1, keepdim=True), p_latente

    def V_energy(self, r):
        r_latente = self.encoder_r(r)
        return self.MLP_V(r_latente), r_latente

    def forward(self, r, p):
        # Kinetic energy
        T, p_latente = self.T_energy(p)
        
        # Potential energy
        V, r_latente = self.V_energy(r)

        # Hamiltonian Total
        return T + V, r_latente, p_latente

    def time_derivative(self, r, p, train=True):
        rh = r.detach().requires_grad_(True)
        ph = p.detach().requires_grad_(True)

        H, r_latente, p_latente = self.forward(rh, ph)

        dHdr_latente, dHdp_latente = torch.autograd.grad(
            outputs=H.sum(),
            inputs=[r_latente, p_latente],
            create_graph=train,
            retain_graph=train,
            allow_unused=False
        )

        # Decoders
        dr_dt = self.decoder_r_dot(dHdp_latente)
        dp_dt = self.decoder_p_dot(-dHdr_latente)

        return dr_dt, dp_dt

    def time_derivative_use(self, z):
        if not torch.is_tensor(z):
            z = torch.as_tensor(z, dtype=torch.float64, device=self.w_r.device)

        z = z.detach().requires_grad_(True)

        r, p = z[:, :30], z[:, 30:]

        H_net, r_latente, p_latente = self.forward(r, p)

        grad_H = torch.autograd.grad(
            outputs=H_net.sum(),
            inputs=[r_latente, p_latente],
            create_graph=False,
            retain_graph=False,
            allow_unused=False
        )

        assert grad_H[0] is not None and grad_H[1] is not None, "Error: Gradiente nulo en la integración"

        dr_dt = self.decoder_r_dot(grad_H[1])  # dH/dp
        dp_dt = self.decoder_p_dot(-grad_H[0]) # -dH/dr

        return torch.cat([dr_dt, dp_dt], dim=1)

    def criterion(self, residual):
        return torch.mean(residual.pow(2))
        
    def loss_step(self, z, z_dot, train):
        r, p = z[:, :30], z[:, 30:]
        r_dot, p_dot = z_dot[:, :30], z_dot[:, 30:]

        # Derivadas dot forward
        dr_dt_pred, dp_dt_pred = self.time_derivative(r, p, train)

        # Residuo manual sin llamadas a librerías funcionales
        res_r_dot = r_dot - dr_dt_pred
        res_p_dot = p_dot - dp_dt_pred

        # Loss physics utilizando tu método `criterion` optimizado internamente con .pow(2)
        loss_physics = self.criterion(res_r_dot) + self.criterion(res_p_dot)
                
        return loss_physics