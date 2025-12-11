# Connectome Analysis Pipeline

---
  
### ⚠️ A Note on the Current State:

> This is a preliminary and absolutely crazyyyy version of this repo. Hopefully it will be cleaner as time continues. You might need to also get my version of the GNM library - I made a few changes (for the better or worse - I am unsure about that). Lmk if this is the case. 

---

**The repo contains scripts for:**
* preprocessing,
* the evaluation of memory capacity (MC) through **ESNs** on empirical connectomes,
* **GNM** generation and evaluation (energy) - this creates nice plots (see plot below),
* the calculation of cool metrics (idk: 20-30 or so?). See 'run_evaluation.py'. 
* "dynamic GNMs" (in short **DynGNMs**) generation and evaluation (MC) (last in a version from October or so). The generation is similar to the GNM, but every next edge's placement is done in accordance to it's best location in terms of MC improvement / transfer entropy improvement / etc.

---
### The big goal (for now)

> :bulb: **The research topic:** **Memory Capacity (Or Alternatively Something Else Like *Transfer Entropy*?) as a Potential Driver of Brain Network Topology: A Reservoir Computing Approach to Understanding Long-Range Connections**

--- 
### Recommended scripts: 
* run_experiment_ALL_animals.py to generate GNMs and have the most basic metrics calculated. 
* run_evaluation.py to get more metrics 
* 