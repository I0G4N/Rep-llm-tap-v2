import torch
from torch import nn


class LoRAModule(torch.nn.Module):
    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name]) # functional attribute

        if not hasattr(self, "config"):
            self.config = {"model_type": "custom"}

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = getattr(self.model, "config", {"model_type": "custom"})
            if hasattr(model_config, "to_dict"):
                model_config = model_config.to_dict()

            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config

        self._find_and_replace(adapter_name) # according to 'config', replace 'named_parameters' of base model by custom LoRA layers
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError("LoraModel supports only 1 adapter with bias. When using multiple adapters, "
                             "please set bias to 'none' for all adapters.")

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)

        if self.peft_config[adapter_name].inference_mode: # if model is in inference mode, not training model, freeze the parameters of the adapter
            _freeze_adapter(self.model, adapter_name)


class Linear(nn.Module):
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            is_target_conv_1d_layer: bool = False,
            **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, adapter_name, in_features, out_features)

        self.weight.requires_grad = False  # Freeze the original weight
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)  # Reset parameters for the linear layer
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
