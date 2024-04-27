# Experimentation
- Found that our loss would be erratic and model outputs would pin at or near 0 for all batches and samples
- Removing "encoder layer" and only using residual blocks and the FC layers seems to alleviate this issue, and gives us reasonable loss improvements during training
    - Training got much faster after this as well
    - Mean loss stopped improving sufficiently after 1 epoch, following 5 were insufficient change and the training was stopped after 6 total epochs
 
# Data
- Cases 819 and 3704 are decent ratios of negative to positive samples
