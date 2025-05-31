import torch, copy


def get_clones(module, N):
    """
    Get N clones of a module.
    
    Args:
        module (torch.nn.Module): The module to clone.
        N (int): The number of clones to create.
    
    Returns:
        torch.nn.ModuleList: A list of cloned modules.
    """
    # deepcopy ensures that the module is cloned correctly
    # and that the parameters are not shared between the clones
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])