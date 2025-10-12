# Connectome Analysis Pipeline


Hi Kayson! 

If you read this, sorry for not reaching out to you earlier! I'll send you an update on Tuesday, where I'll answer the points you raised in your last email, and where I'll update you on where I currently am project-wise. This repo is a part of it, but there's obviously still lots to do, and I am unsure if my results are promissing at all so far. 

I think, in short what I have been working on, can be summarized as follows:
    - A repo that aims to be modular and expandable
    - Currently, it contains scripts for: 
        - preprocessing,
        - the evaluation of memory capacity (MC) through **ESNs** on empirical connectomes, 
        - **GNM** generation and evaluation (energy) - this creates nice plots (see plot below), 
        - "dynamic GNMs" (in short **DynGNMs**) generation and evaluation (MC). The generation is similar to the GNM, but every next edge's placement is done in accordance to it's best location in terms of MC improvement / transfer entropy improvement / etc. 

![alt text](image.png)
*Plot showing energy landscape of GNM sweep. This works now better than the results shown in the first update.*


## The big goal (for now)

> :bulb: **The research topic:**  **Memory Capacity (Or Alternatively Something Else Like Transfer Entropy?) as a Potential Driver of Brain Network Topology: A Reservoir Computing Approach to Understanding Long-Range Connections**


## Some details I like 

This pipeline includes (or includ*ed*, some may have already been lost throughout different repo versions): 

- **Modular Architecture**: Separate modules for configuration, data loading, ESN evaluation, and GNM generation 
- **Parallel Processing**: Multiprocessing for large-scale experiments
- **Incremental Saving**: Results saved progressively to prevent data loss -> this has already proved very valuable
- **YAML Configuration Files**: Makes it easy to change parameters - the YAML file is automatically copied to each experiment folder, such that it's easy to keep track of the experiments.
- **Visual and Numerical Analysis**: Built-in result analysis and visualization


Other work worked at least once, but has strong potential to be bug-gy again. This includes, but is not limited to: 
- **WANDB Integration**: This worked at least for a time, but I switched to only logging locally due to faster compute times 
- **Easy Graph Measure Integration**: 
- **Flexible Configuration**: Easy-to-modify configuration system with presets, will be soon turned to easy yaml config files, to make configuration even smoother


## Module Structure

```
src/preprocessing (run in this order)
├── get_70_connectomes_700mb.m      # Turn .mat file into something that is better workable with python
├── get_consensus_data_10_2.m       # Turn .mat file into something that is better workable with python
├── preprocess_70_connectomes.py    # Preprocess all the connectomes
└── preprocess_distance_matrix.py   # Get the distance matrix 

src/connectome_analysis
├── config.py                       # Configurations
├── data_loader.py                  # Data loading
├── esn_evaluation.py               # ESN evaluation 
├── gnm_network_generator.py        # GNM parameter fitting
├── main_pipeline_2.py              # Main orchestrator (both for the empirical and the simulated connectomes), useable with CLI
└── visualization.py                # Visualization of GNM models (Energy for grid of gammas and etas)
```

## How to run main_pipeline_2.py
- Evaluate memory capacity with ESNs on the empirical subject data: 
    `python src/connectome_analysis/main_pipeline_2.py esn --esn-random-sample-size 50 --no-wandb --experiment-name "esn_main_pipeline_2"`

- Generate GNMs in a sweep 
    `python src/connectome_analysis/main_pipeline_2.py sweep --no-wandb --experiment-name "gnm"`


- Important: I made changes to the GNM repo from Edwards, and installed it as editable... My current version can be found in this Git repo 


Pipeline for analyzing empirical connectomes using Echo State Networks (ESNs). 
Additionally, Generative Network Models (GNMs) are used to generate new connectomes. 
Then, the empirical and the generated connectomes are compared according to graph metrics and memory capacity (memory capacity is evaluated using a memory task and ESN activity). 

A design goal behind this repo is being as modular and scalable as possible. 


This goal is motivated by the fact that this repo needs to be understandable and clear (for myself and others) right now and in half a year (i.e.: at the end of the thesis) - and, in the best case, long after this, too, if someone wants to (re-)use this code. 
