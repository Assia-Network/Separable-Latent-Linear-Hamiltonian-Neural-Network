# SLL-HNN: Separable Latent Linear Hamiltonian Neural Networks for the Solar System

## 1. Motivation

This project stems from a deep personal curiosity to understand the physics of the universe and my fascination with the development of physics-informed Machine Learning models. Exploring the cosmos through code has always been a goal of mine, but when tackling the N-body gravitational problem in the Solar System, I encountered a critical gap in current computational mechanics [[1]](#references).

Although deep learning has revolutionized function approximation, conventional models ignore the mathematical rigor demanded by celestial systems [[1]](#references). As a parallel variant to the baseline Plausible HNN (P-HNN) architecture, the development of the SLL-HNN is driven by two fundamental needs:

* **Strict Preservation of Physical Invariants:** The imperative need to respect dimensional coherence and preserve the symplectic structure to prevent the system from artificially dissipating energy during simulations over astronomical timescales.
* **Overcoming Classical Limits (The Mixing Problem):** Classical Hamiltonian Neural Networks (HNNs) fail by arbitrarily concatenating and mixing position ($q$) and momentum ($p$) variables within the same dense hidden layers. This entanglement destroys the consistency of physical units, degrading the system's geometry and promoting a catastrophic accumulation of numerical errors over the long term.

---

## 2. Experimental Framework

This repository exclusively documents the development, experimentation, and validation of the Separable Latent Linear Hamiltonian Neural Network (SLL-HNN) architecture. This model constitutes my individual contribution to the joint research, serving as an independent proof-of-concept demonstrating that explicit energy bifurcation achieves macro-scale stability.

To evaluate long-term stability and the network's robustness against High Dynamic Range (HDR) problems, a time-splitting strategy was implemented over a total dataset of 4000 years: 3200 years for training and 800 years for testing and evaluation. 

Throughout the four experimental blocks, the core components of the architecture (strict $T-V$ decoupling, Hadamard projections in the latent space, and the strict policy of not using artificial numerical stabilizers) remained unaltered. Variations were applied solely to the internal topology and activation functions:

### Phases of the Experimental Process

#### Experiment 1: Baseline Exploration (Fine Grid, $\Delta t=0.0005$)
* **Architecture:** 3 deep hidden layers of 256 neurons each, with scalar output for potential energy ($V$).
* **Data:** High temporal information density.
* **Evaluated Activation Functions:** Broad exploration using $\log(\cosh(x))$, $\log(1+x\tanh(x))$, $x\tanh(x)$, $\sqrt{x^2+1}-1$, and $\tanh(x)$.
* **Purpose:** To evaluate the initial impact of parity and geometric symmetry over a massive volume of data without temporal reduction.

#### Experiment 2: Capacity Adjustment (Coarse Grid, $\Delta t=0.01$)
* **Architecture:** 3 deep hidden layers, with the width adjusted to 240 neurons per layer.
* **Data:** Data reduction by a factor of 20 through coarser temporal discretization.
* **Evaluated Activation Functions:** Reduced to the three most promising functions: $\log(\cosh(x))$, $\log(1+x\tanh(x))$, and $\tanh(x)$.
* **Purpose:** To analyze the model's sensitivity and the accumulation of local truncation errors given a drastic reduction in temporal information density.

#### Experiment 3: "Lite" Version (Coarse Grid, $\Delta t=0.01$)
* **Architecture:** 3 deep hidden layers, with the width compacted to 128 neurons per layer.
* **Data:** Data reduction by a factor of 20.
* **Evaluated Activation Functions:** Filtered exclusively to the two highest-performing even functions: $\log(\cosh(x))$ and $\log(1+x\tanh(x))$.
* **Purpose:** To verify the viability of a model with fewer parameters per layer to retain the system's physics without losing orbital confinement.

#### Experiment 4: "Lite V2" Version (Coarse Grid, $\Delta t=0.01$) - Optimal Architecture
* **Architecture:** Structural reduction to 2 deep hidden layers, optimized with 256 neurons each.
* **Data:** Data reduction by a factor of 20.
* **Evaluated Activation Functions:** Final evaluation between even parity functions.
* **Result:** This configuration was consolidated as the definitive architecture for the SLL-HNN contribution to the scientific article. It empirically demonstrated that a more compact topology avoids the geometric instabilities of deeper networks, mitigating overfitting and numerical drift over centuries.

> **Note on Record Consistency:** While the final scientific publication reports the bounded horizon of 600 years of macro-scale stability (obtained from the optimal configuration), this repository contains the complete source codes and stability logs corresponding to the total 800 years of testing for all experimental trials.

---

## 3. Network Flowchart and Internals (SLL-HNN)

<div align="center">

<video src="https://github.com/user-attachments/assets/e8880ec7-e6ac-4ff6-9141-2df47569e2db" width="100%" autoplay loop muted playsinline></video>
<p><em>Animation of the physical and mathematical flow through the SLL-HNN architecture.</em></p>

<img src="media%20SLL-HNN/Diagrama_SLL_HNN_English_UltraHD_Arquitecto.png" alt="Static Diagram of the SLL-HNN Architecture">
<p><em>Static structural architectural diagram of the SLL-HNN presented in the scientific paper.</em></p>

</div>

The architecture's processing guarantees strict physical and dimensional decoupling, structured into the following fundamental stages:

* **Vector Normalization:** The physical input vectors corresponding to positions ($\vec{q}$) and momenta ($\vec{p}$) are initially normalized by dividing them by their respective scalar tensors of absolute maximum values ($\max(|q|)$ and $\max(|p|)$).
* **Linear Mapping (Phase Transformer):** Independent point-to-point linear projection into the latent space. Each normalized variable passes through a linear phase transformer ($\odot W+b$) to allow autonomous parametric calibration for each degree of freedom.
* **Energy Decoupling:** Completely separate computation of the fundamental energies to avoid unit entanglement:
  * *Potential Energy (V):* The latent position flow ($\tilde{q}$) enters a continuous Multi-Layer Perceptron (MLP), which compresses the information and ultimately outputs a representative scalar value of the gravitational potential.
  * *Kinetic Energy (T):* The latent momentum flow ($\tilde{p}$) interacts with an explicit parametric analytical formulation via a learned inverse mass matrix ($M^{-1}$), resulting in the kinetic energy scalar.
* **Hamiltonian Fusion:** Both scalars converge and are coherently summed to form the additive latent Hamiltonian ($H=T+V$).
* **Prediction (Symplectic Autograd):** Extraction of symplectic gradients via automatic differentiation. The system derives the Hamiltonian with respect to the latent states ($\dot{\tilde{q}}=\frac{\partial H}{\partial\tilde{p}}$ and $\dot{\tilde{p}}=-\frac{\partial H}{\partial\tilde{q}}$).
* **Inverse Scaling (Denormalization):** The resulting latent derivatives undergo a purely multiplicative linear transformation (strictly without a bias matrix) and are multiplied by their output scale factors ($\max(|\dot{q}|)$ and $\max(|\dot{p}|)$) to deliver the final physical dynamic and kinematic predictions.

### Training Visualization and Linear Weights

The internal behavior of the network during the optimization process of the best model (Experiment 4) is detailed below:

<div align="center">

<img src="media%20SLL-HNN/loss.png" alt="SLL-HNN Loss Curve">
<p><em>Evolution of the loss function over 1000 epochs for the optimal 2-layer SLL-HNN architecture.</em></p>

<img src="media%20SLL-HNN/SLL_W_B.png" alt="Linear Weights and Bias of the Encoder">
<p><em>Distribution of the linear weight matrix and bias in the encoding phase (Encoder) towards the latent space.</em></p>

<img src="media%20SLL-HNN/SLL_W_dot.png" alt="Linear Weights of the Decoder">
<p><em>Weight matrices corresponding to the decoding of the temporal derivatives following symplectic differentiation.</em></p>

</div>

---

## 4. Description of the Best Model
* **Optimized Topology:** Structural reduction to a compact configuration of 2 deep hidden layers with a width of 256 neurons each.
* **Memory Footprint:** The model is condensed into a size of 600 kB with an exact total of 74,195 parameters.
* **Explicit Bifurcation:** Kinetic energy sub-network ($T$) restricted to an explicit parametric analytical formulation, strictly separated from the potential energy sub-network ($V$) approximated via the continuous MLP.
* **Clean Physics:** The network completely dispenses with artificial numerical stabilizers, forcing the model to learn the dynamics immaculately without polluting the physical data.

## 5. Long-Term Model Predictions
* Macroscopic orbital stability proven and confined within a 600-year integration horizon.
* Despite subtle orbital variations inherent to purely linear mapping, the bodies avoid unphysical ejections and maintain a gravitationally bound state.
* The configuration empirically demonstrates that enforcing physical invariants via even activations maintains stability, even under the representational constraints of strictly linear embeddings.

### 5.1 Real-Time Dynamic Visualization (Manim)

To demonstrate the physical and geometric fidelity of the SLL-HNN model, a real-time mathematical animation has been generated using the **Manim** animation engine. This simulation visualizes the orbital behavior and evolution of celestial body trajectories under the predictions of our optimal architecture, evidencing the preservation of stability in space:

<div align="center">

<video src="https://github.com/user-attachments/assets/686a3c35-8c30-4482-b2c2-074403390510" width="100%" autoplay loop muted playsinline></video>
<p><em>Dynamic evolution of planetary trajectories predicted by the SLL-HNN.</em></p>

</div>

## 6. The Winning Activation Function
* The optimal activation function proved to be the composite even function $f(x)=\log(1+x\tanh(x))$.
* The use of this even function prevents overfitting and mitigates the severe numerical drift observed in traditional networks, aligning with the spatial symmetry of real physics, where potential energy depends on the even terms of relative distances.

## 7. Libraries Used
* PyTorch
* NumPy
* Matplotlib
* Manim *(Mathematical Orbital Visualization)*

## 8. Research Repository Ecosystem

To ensure the reproducibility and modularity of the results presented in the paper, the research code and data are distributed across three specific repositories:

* **SLL-HNN Architecture (This repository):**
  Contains exclusively my parallel investigation, the temporal stability logs, and the implementation of the latent linear mapping architecture.
* **Dataset Generation (Yoshida Symplectic Integrator):**
  [https://github.com/Assia-Network/Yoshida-Symplectic-Integrator.git](https://github.com/Assia-Network/Yoshida-Symplectic-Integrator.git)
  Repository authored by me hosting the numerical engine used to generate the 4000 years of training and evaluation data. It implements a high-order Yoshida integrator to ensure the baseline data preserves the symplectic geometry of the Solar System.
* **P-HNN and BLK-P-HNN Architectures:**
  [https://github.com/jae-hoon-daniel-lee/hamiltonian-neural-networks-for-solar-system.git](https://github.com/jae-hoon-daniel-lee/hamiltonian-neural-networks-for-solar-system.git)
  Repository managed by my co-author, Jae Hoon Lee. It documents the baseline architecture of the research (Plausible HNN) and the experiments corresponding to the sparse network focused on parameter compression (Block Sparse P-HNN).

## 9. License

This project is distributed under the **GNU General Public License v3.0 (GPL-3.0)**. 

You are free to use, copy, modify, and distribute this software for academic and commercial purposes, provided that any derivative works are also open-source and released under the exact same GPL-3.0 license. For the full legal text, please refer to the [LICENSE](LICENSE) file included in this repository.

## 10. About
* **Authors:** Jae Hoon Lee and Jesus Martin Bautista Ramirez.
* **Affiliations:** Open Stack Inc. / Universidad Peruana de Ciencias Aplicadas (UPC).
* **DOI / Paper:** [Link pending publication]

### <a name="references"></a> References

If the SLL-HNN architecture, the analytical approaches, or the code provided in this repository contribute to your academic or professional research, we kindly request that you cite our work:

**[1]** Lee, J. H., & Bautista Ramirez, J. M. (In development). *Hamiltonian Neural Networks for Solar System*. Documentation, source code, and joint development available through the repositories listed in section 8.
