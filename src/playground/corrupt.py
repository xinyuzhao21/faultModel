import torch
import numpy as np
class Fault:
    def __init__(self,fault_layer,fault_index,corrupt_value=None,method='max_cap',time=0):
        self.fault_layer = fault_layer
        self.fault_index = fault_index
        self.corrupt_value = corrupt_value
        self.method=method
        self.time = time
        self.injected = False



class FaultInject:
    def __init__(self,model=None):
        self.model = model
    @staticmethod
    def weight_inject(fault,model,**kwargs):
        corrupt_model = model
        if isinstance(fault,tuple):
            fault_layer, fault_index,corrupt_value =fault
        if isinstance(fault,Fault):
            fault_layer, fault_index, corrupt_value = fault.fault_layer,fault.fault_index,fault.corrupt_value
        for name, param in corrupt_model.named_parameters():
            if fault_layer in name:
                corrupt_idx = (
                    tuple(fault_index)
                    if isinstance(fault_index, list)
                    else fault_index
                )

                if fault.method is 'max_cap':
                    corrupt_value=torch.max(param).item()
                if fault.method is 'manual':
                    corrupt_value=corrupt_value
                weight=param.data
                orig_value, injected_value= FaultInject._tensor_inject(weight,corrupt_idx,corrupt_value)
                fault.injected=True
                return orig_value, injected_value
        # param=getattr(corrupt_model,fault_layer)
        # if param:
        #     corrupt_idx = (
        #         tuple(fault_index)
        #         if isinstance(fault_index, list)
        #         else fault_index
        #     )
        #
        #     if method is 'max_cap':
        #         corrupt_value = torch.max(param)
        #     weight = param.data
        #     return self._tensor_inject(weight, corrupt_idx, corrupt_value)
        return None,None

    @staticmethod
    def _tensor_inject(tensor,index,injected_value):
        orig_value = tensor[index].item()
        tensor[index] = injected_value if not isinstance(injected_value,np.ndarray) else torch.from_numpy(injected_value)
        return orig_value,injected_value
