import torch
import numpy as np

def generate_gradcam(model, image_tensor, target_layer, class_idx=None):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    model.eval()
    output = model(image_tensor.unsqueeze(0))
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    loss = output[0, class_idx]
    model.zero_grad()
    loss.backward()

    grads = gradients[0]
    acts = activations[0]
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(acts.shape[1]):
        acts[0, i, :, :] *= pooled_grads[i]

    heatmap = acts[0].sum(dim=0).cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()

    handle_fw.remove()
    handle_bw.remove()
    return heatmap
