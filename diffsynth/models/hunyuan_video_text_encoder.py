from transformers import LlamaModel, LlamaConfig, DynamicCache, LlavaForConditionalGeneration
from copy import deepcopy
import torch


class HunyuanVideoLLMEncoder(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.auto_offload = False

    def enable_auto_offload(self, **kwargs):
        self.auto_offload = True

    def forward(self, input_ids, attention_mask, hidden_state_skip_layer=2):
        # Always use parent class forward - it handles all version differences correctly
        # The auto_offload feature will be handled by moving the entire model temporarily
        
        original_device = next(self.parameters()).device
        target_device = input_ids.device
        
        if self.auto_offload and original_device != target_device:
            # Move model to target device temporarily
            self.to(target_device)
        
        try:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Get the hidden state from the appropriate layer
            # hidden_states is a tuple of (embedding_output, layer1_output, layer2_output, ...)
            target_layer_idx = len(self.layers) - hidden_state_skip_layer
            hidden_states = outputs.hidden_states[target_layer_idx]
            
            return hidden_states
        finally:
            if self.auto_offload and original_device != target_device:
                # Move model back to original device to free GPU memory
                self.to(original_device)
                torch.cuda.empty_cache()


class HunyuanVideoMLLMEncoder(LlavaForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.auto_offload = False

    def enable_auto_offload(self, **kwargs):
        self.auto_offload = True

    # TODO: implement the low VRAM inference for MLLM.
    def forward(self, input_ids, pixel_values, attention_mask, hidden_state_skip_layer=2):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  output_hidden_states=True,
                                  pixel_values=pixel_values)
        hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
        return hidden_state
