## My tasks and contributions

### Dataset
1. create_metadata.py  
Selects a subset from the whole fastMRI dataset, including all contrasts, and creates a CSV file

2. create_fastmri_modl.py  
Uses the CSV file created above to build the dataset for modl and varnet

3. create_fastmri_ssdu.py  
Uses the CSV file to build the dataset according to SSDU learning scheme

### Implementation
1. varnet.py  
Fixed weight updation problem in cascade of unets

2. modl_dataset.py  
modified to utilize the custom dataset

3. SSDU  
fixed obsolete code and modified training and testing scripts for custom dataset

### Experiments
1. Training and testing the three architectures for different epochs and learning rates
2. Effect of different normalization on training
3. Effect of number of unrolls
4. Assessing lambda parameter and learning rates to fix modl k=10 performance
5. Comparison between the three models
