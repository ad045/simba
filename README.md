# Connectome Analysis Pipeline

---
  
### ⚠️ A Note on the Current State:

> This is a preliminary state of the repo - please do not continue to read any further.

> Kayson: I'll make some improvements, and generate some better results. You'll hopefully receive a more complete update report on Tuesday evening!

----

### **In more detail (all preliminary):**

Hi Kayson!

If you read this: Sorry for not reaching out to you earlier! I'll send you an update on Tuesday, where I'll answer the points you raised in your last email, and where I'll update you on where I currently am project-wise.

This repo is a part of it, but there's obviously still lots to do, and I am unsure if my results are promissing at all so far.


I think that what I have been working on can be summarized as follows:
* Aiming for a repo that will be modular and expandable. 


**Currently, the repo contains scripts for:**
* preprocessing,
* the evaluation of memory capacity (MC) through **ESNs** on empirical connectomes,
* **GNM** generation and evaluation (energy) - this creates nice plots (see plot below),
* "dynamic GNMs" (in short **DynGNMs**) generation and evaluation (MC). The generation is similar to the GNM, but every next edge's placement is done in accordance to it's best location in terms of MC improvement / transfer entropy improvement / etc.

![Energy landscape of GNM sweep](image.png)
*Plot showing energy landscape of GNM sweep. This works now better than the results shown in the first update.*

---
### The big goal (for now)

> :bulb: **The research topic:** **Memory Capacity (Or Alternatively Something Else Like *Transfer Entropy*?) as a Potential Driver of Brain Network Topology: A Reservoir Computing Approach to Understanding Long-Range Connections**


<!-- ---
### Notes regarding issues: 
- If one changes scripts (that run with joblib), and they are not updated because of old versions of compiled bytecode (in ``__pycache__`` directories): Delete them all with one command: ``

        find . -type d -name "__pycache__" -exec rm -r {} +
        
        `` 

 bash scripts/run_gnm_example.sh      
 
 --> 



if "fit" does not work anymore (prob after reloading the esn module): The issue might be due to different float types (all W matrices are float32...). I used this fix in the _regressor.py script (see path below), to ensure that everything has the right shape. But those changes are limited to a specific conda environment, if I understand it correctly. 
Path: 
"/opt/miniconda3/envs/ma_thesis/lib/python3.13/site-packages/echoes/esn/_regressor.py" (approx. line 190)
        # HACK: Convert X and y to match weight matrices dtype
        target_dtype = np.float32  # or whatever your weight matrices are using
        # X = X.astype(target_dtype)
        # y = y.astype(target_dtype)
        # HACK: Convert X and y to match weight matrices dtype
        X = np.array(X, np.float32) # np.float32(X) # X.astype(target_dtype)
        y = np.array(y, np.float32) # float32(X) # y.astype(target_dtype)
        self._dtype_ = target_dtype