# Experimentation
- Found that our loss would be erratic and model outputs would pin at or near 0 for all batches and samples
- Removing "encoder layer" and only using residual blocks and the FC layers seems to alleviate this issue, and gives us reasonable loss improvements during training
