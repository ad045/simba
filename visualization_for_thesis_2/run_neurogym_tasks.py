# ── connectome_neurogym.py ────────────────────────────────────────────────────
"""

    The script runs a connectome-driven reservoir computing experiment on three   
    cognitive neuroscience tasks:                                                
                                                                                    
    Setup                                                                         
    - Loads a real diffusion tractography connectome (W, shape presumably ~80–100 
    nodes) and uses it as the recurrent weight matrix of an echo-state /          
    leaky-integrator reservoir.                                                  
    - The ConnectomeReservoirWrapper replaces raw task observations with 100-D    
    reservoir states: x ← (1−α)·x + α·tanh(W_rec @ x + W_in @ u), where W_rec is
    the connectome normalized to a fixed spectral radius.                         
                                                        
    Three tasks (from neurogym)                                                   
    1. PerceptualDecisionMaking-v0 — integrate noisy evidence and pick left/right 
    2. DelayMatchSample-v0 — remember a sample stimulus across a delay, then      
    decide match/non-match                                                        
    3. ContextDecisionMaking-v0 — same stimuli in two modalities, attend to one   
    based on context cue                                                         
                                                                                    
    Training                                                  
    - Wraps each task with TrialHistoryV2 (structured trial-to-trial transition   
    probabilities, so the agent can exploit history) then the reservoir wrapper.  
    - Trains an A2C agent (two 64-unit MLP layers on top of the reservoir) for 50k
    steps on each task.                                                          
                                                                                    
    Visualisation                                                                
    1. Plots raw task trials (random agent) before training.                      
    2. Plots trained-agent trials with reservoir state traces.                    
    3. Runs 500 steps with each trained agent, applies PCA to the reservoir      
    trajectory, and saves a 3-panel scatter plot                                  
    (connectome_reservoir_summary.png) showing how the reservoir state evolves    
    through PC space over time.                                                   
                                                                                    
    The core question it's probing: can a biologically-structured connectome     
    (rather than a random reservoir) support learning of cognitive tasks, and what
    does the resulting neural geometry look like?
                                                    
"""


import warnings; warnings.filterwarnings('ignore')
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import neurogym as ngym
from neurogym import info
from neurogym.utils import plotting
from neurogym.wrappers import TrialHistoryV2
from stable_baselines3 import A2C

# ── Explore ────────────────────────────────────────────────────────────────────
info.show_all_tasks()
info.show_all_tags()
info.show_all_wrappers()

# ── Three tasks ────────────────────────────────────────────────────────────────
TASK_CONFIGS = [
    {
        "id":        "PerceptualDecisionMaking-v0",
        "label":     "Perceptual Decision Making\n(evidence integration)",
        "ob_traces": ["Fixation", "Stim L", "Stim R"],
        "timing":    {"fixation": ("constant", 300),
                      "stimulus": ("constant", 700),
                      "decision": ("constant", 200)},
        "dt": 100,
    },
    {
        "id":        "DelayMatchSample-v0",
        "label":     "Delay Match-to-Sample\n(working memory)",
        "ob_traces": ["Fixation", "Sample", "Test"],
        "timing":    {"fixation": ("constant", 300),
                      "sample":   ("constant", 500),
                      "delay":    ("constant", 1000),
                      "test":     ("constant", 500),
                      "decision": ("constant", 300)},
        "dt": 100,
    },
    {
        "id":        "ContextDecisionMaking-v0",
        "label":     "Context Decision Making\n(selective attention)",
        "ob_traces": ["Fixation", "Ctx1", "Ctx2", "Stim mod1", "Stim mod2"],
        "timing":    {"fixation":  ("constant", 300),
                      "stimulus":  ("constant", 750),
                      "delay":     ("constant", 300),
                      "decision":  ("constant", 100)},
        "dt": 100,
    },
]

# ── Visualise raw tasks (random agent, no model) ───────────────────────────────
for cfg in TASK_CONFIGS:
    env_tmp = ngym.make(cfg["id"], dt=cfg["dt"], timing=cfg["timing"])
    info.show_info(env_tmp)
    fig = plotting.plot_env(env_tmp, num_steps=200, ob_traces=cfg["ob_traces"])
    plt.suptitle(cfg["label"]); plt.tight_layout(); plt.show()
    env_tmp.close()

# ── Load your connectome ───────────────────────────────────────────────────────
W = np.load(
    "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    "data/preprocessed/kaysons_generated_networks_diffusion/"
    "01_connectomes/diffusion_10_percent.npy"
)
W = W[0,:,:]
print(f"Connectome: {W.shape}  density: {(W>0).mean():.1%}")

# ── ConnectomeReservoirWrapper ─────────────────────────────────────────────────
class ConnectomeReservoirWrapper(gym.Wrapper):
    """
    Replaces raw obs with 100-D reservoir states driven by the connectome.

        x ← (1-α)·x  +  α·tanh(W_rec @ x  +  W_in @ u)

    W_rec is the spectral-radius-normalised connectome; W_in is a fixed random
    input projection. Hidden state is zeroed on reset().
    """
    def __init__(self, env, W, spectral_radius=0.9, input_scaling=0.1,
                 leakage=0.3, seed=42):
        super().__init__(env)
        N = W.shape[0];  self.N = N;  self.leakage = leakage

        rho = np.max(np.abs(np.linalg.eigvals(W)))
        self.W_rec = (W / rho * spectral_radius).astype(np.float32)

        rng = np.random.default_rng(seed)
        obs_dim   = env.observation_space.shape[0]
        self.W_in = rng.uniform(-input_scaling, input_scaling,
                                (N, obs_dim)).astype(np.float32)
        self.x    = np.zeros(N, dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-1., high=1., shape=(N,), dtype=np.float32)

    def reset(self, **kw):
        self.x = np.zeros(self.N, dtype=np.float32)
        obs, info = self.env.reset(**kw)
        return self._update(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self._update(obs), rew, term, trunc, info

    def _update(self, obs):
        a = self.leakage
        self.x = ((1-a)*self.x
                  + a*np.tanh(self.W_rec @ self.x + self.W_in @ obs))
        return self.x.copy()


def make_env(cfg, W):
    env   = ngym.make(cfg["id"], dt=cfg["dt"], timing=cfg["timing"])
    n_ch  = len(env.unwrapped.choices)   # choices only, not fixation
    p     = 0.8
    probs = np.full((n_ch,n_ch),(1-p)/(n_ch-1)) + np.eye(n_ch)*(p-(1-p)/(n_ch-1))
    env   = TrialHistoryV2(env, probs=probs)
    env   = ConnectomeReservoirWrapper(env, W)
    return env

# ── Train A2C on all three tasks ───────────────────────────────────────────────
trained = {}
for cfg in TASK_CONFIGS:
    print(f"\n{'='*55}\nTraining: {cfg['id']}\n{'='*55}")
    env   = make_env(cfg, W)
    model = A2C("MlpPolicy", env, verbose=1,
                policy_kwargs={"net_arch": [64, 64]},
                learning_rate=7e-4, n_steps=5, gamma=0.99)
    model.learn(total_timesteps=50_000)
    trained[cfg["id"]] = model
    env.close()

# ── Visualise trained agents (reservoir state traces) ─────────────────────────
for cfg in TASK_CONFIGS:
    env = make_env(cfg, W)
    fig = plotting.plot_env(
        env,
        num_steps = 200,
        ob_traces = [f"Res {i}" for i in range(env.observation_space.shape[0])],
        model     = trained[cfg["id"]],
    )
    fig.suptitle(f"Trained agent · {cfg['label']}", y=1.01)
    plt.tight_layout(); plt.show()
    env.close()

# ── Summary: reservoir PC trajectories ────────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")
fig.suptitle("Connectome Reservoir · Trained Agent Trajectories (PC space)",
             fontsize=13, fontweight="bold")

for ax, cfg, color in zip(axes, TASK_CONFIGS, PALETTE):
    env = make_env(cfg, W)
    obs, _ = env.reset();  states = [obs]
    for _ in range(499):
        action, _ = trained[cfg["id"]].predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(int(action))
        states.append(obs)
        if term or trunc: obs, _ = env.reset()
    env.close()

    traj = PCA(n_components=2).fit_transform(np.array(states))
    sc   = ax.scatter(*traj.T, c=np.arange(500), cmap="viridis", s=4, alpha=0.7)
    ax.set_title(cfg["label"], fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="step")

plt.tight_layout()
plt.savefig("connectome_reservoir_summary.png", dpi=150, bbox_inches="tight")
plt.show()