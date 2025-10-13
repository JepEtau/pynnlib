from torch import nn

# Create a virtual nn.Module
# to update the NN model without error
class _VirtualEfficientTAM(nn.Module):
    def __init__(self, filepath: str, config_fp:str):
        pass
