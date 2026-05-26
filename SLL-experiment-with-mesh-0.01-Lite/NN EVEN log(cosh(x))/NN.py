#--------------------------------------------------------------------
# ------------------------PART 1-------------------------------------
#--------------------------------------------------------------------

import os

# Hardware detection
num_cores = os.cpu_count()

# Environment variables (MUST be set BEFORE importing torch)
os.environ["MKL_THREADING_LAYER"] = "INTEL"
os.environ["MKL_INTERFACE_LAYER"] = "LP64" 
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

import torch

# Torch-specific thread management
torch.set_num_threads(num_cores)       # Intra-op parallelism
torch.set_num_interop_threads(1)       # Inter-op parallelism (usually 1 is best for PINNs)
torch.set_grad_enabled(True)           # Default, but explicit

print(f"\nIntra-op threads: {torch.get_num_threads()}")
print(f"Inter-op threads: {torch.get_num_interop_threads()}")

import torch.nn as nn
import sys
import psutil
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))
import utils.funciones as fns
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time

# Configuration
name_iter = "HNN_EVEN_log(cosh(x))"
name_ubi = "SLL-experiment-with-mesh-0.01-Lite/NN EVEN log(cosh(x))"

tiempo_inicio_global = time.time()

def get_sys_info():
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = psutil.virtual_memory().percent

    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read()) / 1000.0
    except:
        temp = 0.0
        
    segundos_totales = int(time.time() - tiempo_inicio_global)
    horas, rem = divmod(segundos_totales, 3600)
    minutos, segundos = divmod(rem, 60)
    tiempo_str = f"{horas:02d}:{minutos:02d}:{segundos:02d}"
    
    return f"SISTEMA -> CPU: {cpu_usage}% | RAM: {ram_usage}% | Temp: {temp:.1f}°C | Tiempo: {tiempo_str}"

def main():
    #--------------------------------------------------------------------
    # ------------------------PART 2-------------------------------------
    #--------------------------------------------------------------------

    float_user = torch.float64
    torch.set_default_dtype(float_user)

    # Parameters
    # Train and test
    Sz = torch.tensor(np.load("Data/dt_0.01/normalization_vectors.npz")['Z_norm'].astype('float64'), dtype=float_user)
    Szd = torch.tensor(np.load("Data/dt_0.01/normalization_vectors.npz")['Z_dot_norm'].astype('float64'), dtype=float_user)
    z_train = torch.tensor(np.load("Data/dt_0.01/Z_train_opt.npz")['data'].astype('float64'), dtype=float_user)/Sz
    z_test = torch.tensor(np.load("Data/dt_0.01/Z_test_opt.npz")['data'].astype('float64'), dtype=float_user)/Sz
    z_train_dot = torch.tensor(np.load("Data/dt_0.01/Z_dot_train_opt.npz")['data'].astype('float64'), dtype=float_user)/Szd
    z_test_dot = torch.tensor(np.load("Data/dt_0.01/Z_dot_test_opt.npz")['data'].astype('float64'), dtype=float_user)/Szd

    print("Data size:")
    print("train", z_train.shape)
    print("test", z_test.shape)
    print("Z train max", z_train.max())
    print("Z test max", z_test.max())
    print("Z train min", z_train.min())
    print("Z test min", z_test.min())
    print("Z dot train max", z_train_dot.max())
    print("Z dot test max", z_test_dot.max())
    print("Z dot train min", z_train_dot.min())
    print("Z dot test min", z_test_dot.min())

    #--------------------------------------------------------------------
    # ------------------------PART 3-------------------------------------
    #--------------------------------------------------------------------

    # Seed
    fns.set_seed(0)

    class even(nn.Module):
        def __init__(self):
            super(even, self).__init__()

        def forward(self, x):
            return torch.log(torch.cosh(x))
            
    H_net = fns.Block_HNN_SL_Lite(func_act_V=even)
    H_net.double()

    # Optimizer
    optimizer = torch.optim.Adam(
        H_net.parameters(),
        lr=1e-4,
        amsgrad=False
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.75,
        patience=10
    )

    # Model summary
    print("H-Net model\n")
    print(H_net)

    model_opt = torch.compile(H_net, backend="inductor")

    #--------------------------------------------------------------------
    # ------------------------PART 4-------------------------------------
    #--------------------------------------------------------------------

    # TRAIN
    bach = 1024
    epocas = 1_000
    num_trabajadores = 0 # Nucleos
    dataset = TensorDataset(z_train, z_train_dot)
    loader = DataLoader(dataset, 
                        batch_size=bach, 
                        num_workers=num_trabajadores,
                        pin_memory=False,
                        shuffle=True)
    print("Lotes train: ", len(loader))

    # TEST
    dataset_test = TensorDataset(z_test, z_test_dot)
    loader_test = DataLoader(dataset_test, 
                                batch_size=bach, 
                                num_workers=num_trabajadores,
                                pin_memory=False,
                                shuffle=False)
    print("Lotes test: ", len(loader_test))

    #--------------------------------------------------------------------
    # ------------------------PART 5-------------------------------------
    #--------------------------------------------------------------------

    # Initialize metrics file
    with open(f"{name_ubi}/metrics/metricas_{name_iter}.txt", "w", encoding="utf-8"):
        pass
    with open(f"{name_ubi}/metrics/metricas_{name_iter}_pc.txt", "w", encoding="utf-8"):
        pass

    best_loss_train = float('inf')
    best_loss_test = float('inf')

    for epoch in range(epocas):

        with open(f"{name_ubi}/metrics/metricas_{name_iter}_temporal.txt", "w", encoding="utf-8"):
            pass

        # Mean
        loss_mean_grup = []
        loss_mean_grup_test = []

        # Train  
        model_opt.train()
        for x_train, y_train in loader:
            # Reset gradientes
            optimizer.zero_grad(set_to_none=True)

            # Rollout
            loss_train = model_opt.loss_step(   z=x_train,
                                                z_dot=y_train,
                                                train=True)

            # Backprop
            loss_train.backward()
            optimizer.step()

            # Append
            num_loss = loss_train.detach().item()
            text = f"loss train      : {num_loss}"
            loss_mean_grup.append(num_loss)
            with open(f"{name_ubi}/metrics/metricas_{name_iter}_temporal.txt", "a", encoding="utf-8") as f:
                f.write(text + "\n")

        loss_mean_epoch = np.mean(loss_mean_grup)

        scheduler.step(loss_mean_epoch)

        # Test
        for x_test, y_test in loader_test:
            model_opt.eval()
            loss_test = model_opt.loss_step(    z=x_test,
                                                z_dot=y_test,
                                                train=False)

            num_loss = loss_test.detach().item()
            loss_mean_grup_test.append(num_loss)

        loss_mean_epoch_test = np.mean(loss_mean_grup_test)

        # Logging
        batch_text = f"""
        Iteración: {epoch + 1}/{epocas}
        loss train      : {loss_mean_epoch}
        loss test       : {loss_mean_epoch_test}
        lr              : {optimizer.param_groups[0]['lr']}
        """

        with open(f"{name_ubi}/metrics/metricas_{name_iter}.txt", "a", encoding="utf-8") as f:
            f.write(batch_text)

        print(batch_text)

        sys_info = get_sys_info()
        print(sys_info)

        with open(f"{name_ubi}/metrics/metricas_{name_iter}_pc.txt", "a", encoding="utf-8") as f:
            f.write(sys_info + "\n")

        print(f"[{name_iter}]")

        checkpoint = {
            'model_state_dict': H_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss_train': loss_mean_epoch,
            'loss_test': loss_mean_epoch_test,
            'lr': optimizer.param_groups[0]['lr']
        } 

        if loss_mean_epoch < best_loss_train:
            best_loss_train = loss_mean_epoch
            torch.save(checkpoint, f'{name_ubi}/models/best_train_{name_iter}.pth')
            print(f"--- Nuevo Mejor Train en época {epoch+1} ---")

        if loss_mean_epoch_test < best_loss_test:
            best_loss_test = loss_mean_epoch_test
            torch.save(checkpoint, f'{name_ubi}/models/best_test_{name_iter}.pth')
            print(f"--- Nuevo Mejor Test en época {epoch+1} ---")

    print("Exito")

if __name__ == "__main__":
    main()