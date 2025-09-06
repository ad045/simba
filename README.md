# Connectome Analysis Pipeline

⚠️ This is a preliminary state of the repo - please do not continue to read any further. I'll make some improvements, and generate some better results - and then I'll send you, Kayson a more complete update report on Tuesday evening! ⚠️

---- 

Hi Kayson! 

If you read this: Sorry for not reaching out to you earlier! I'll send you an update on Tuesday, where I'll answer the points you raised in your last email, and where I'll update you on where I currently am project-wise. This repo is a part of it, but there's obviously still lots to do, and I am unsure if my results are promissing at all so far. 

I think that what I have been working on can be summarized as follows:
    - Aiming for a repo that will be modular and expandable

Currently, the repo contains scripts for: 
        - preprocessing,
        - the evaluation of memory capacity (MC) through **ESNs** on empirical connectomes, 
        - **GNM** generation and evaluation (energy) - this creates nice plots (see plot below), 
        - "dynamic GNMs" (in short **DynGNMs**) generation and evaluation (MC). The generation is similar to the GNM, but every next edge's placement is done in accordance to it's best location in terms of MC improvement / transfer entropy improvement / etc. 

![alt text](image.png)
*Plot showing energy landscape of GNM sweep. This works now better than the results shown in the first update.*


## The big goal (for now)

> :bulb: **The research topic:**  **Memory Capacity (Or Alternatively Something Else Like *Transfer Entropy*?) as a Potential Driver of Brain Network Topology: A Reservoir Computing Approach to Understanding Long-Range Connections**