Local GPT-2 Model
```
tensor([[[-37.7550, -38.2652, -40.4860,  ..., -47.3887, -46.8042, -36.8606],
         [-69.7059, -68.8304, -68.4345,  ..., -75.0572, -74.2558, -64.0605],
         [-50.3636, -47.9540, -51.2161,  ..., -63.0638, -60.9918, -52.0684],
         [-67.1389, -66.1321, -64.8171,  ..., -80.9309, -77.7805, -70.2395],
         [-71.4395, -71.6656, -67.7698,  ..., -85.0580, -82.6441, -74.0991]]],
       device='cuda:0', grad_fn=<UnsafeViewBackward0>)
```

Hugging Face GPT-2 Model
```
CausalLMOutputWithCrossAttentions(loss=None, logits=tensor([[[-37.7550, -38.2652, -40.4860,  ..., -47.3888, -46.8042, -36.8606],
         [-69.7059, -68.8304, -68.4345,  ..., -75.0572, -74.2558, -64.0604],
         [-50.3636, -47.9540, -51.2161,  ..., -63.0638, -60.9918, -52.0684],
         [-67.1389, -66.1321, -64.8171,  ..., -80.9309, -77.7805, -70.2395],
         [-71.4395, -71.6656, -67.7697,  ..., -85.0580, -82.6441, -74.0991]]],
       grad_fn=<UnsafeViewBackward0>), past_key_values=DynamicCache(layers=[DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer]), hidden_states=None, attentions=None, cross_attentions=None)
```