"""PAD-UFES-20 differentially private training loops.

Two mechanisms, both hand-rolled with functional per-sample gradients
(``torch.func.vmap(grad(...))``) and full-batch gradient descent:

- DP-SGD (``mode=dpsgd``): each example's ENTIRE gradient is clipped to
  ``max_grad_norm`` and Gaussian noise ``sigma * C`` is added once to the sum.

- Feature DP-SGD (``mode=feature_dpsgd``): each example's gradient is
  decomposed into a *base* (image) component — obtained by re-running the
  per-sample gradient with the metadata columns zeroed — and a *meta*
  residual component (full − base). The two components are clipped
  separately (``C_base`` / ``C_meta``) and only ``C_meta`` scales the noise,
  so the privacy cost is paid for the sensitive clinical metadata rather
  than the whole input.

The noise multiplier is calibrated to a target (epsilon, delta) with
Google's dp-accounting RDP accountant and Brent root-finding, composing a
Gaussian mechanism over ``epochs`` full-batch steps (sampling probability 1).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import grad, vmap

from medmap.pad.training import evaluate


# --------------------------------------------------------------------------- #
# Noise multiplier search (for DP-SGD / Feature DP-SGD)
# --------------------------------------------------------------------------- #
def find_noise_multiplier(sampling: float, steps: int, target_epsilon: float, target_delta: float):
    from scipy import optimize as opt
    from dp_accounting import dp_event as event
    from dp_accounting.rdp import rdp_privacy_accountant as rdp

    RDP_ORDERS = ([1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
                  + list(range(5, 64)) + [128, 256, 512])

    def objective(noise_multiplier):
        accountant = rdp.RdpAccountant(RDP_ORDERS)
        dpevent = event.SelfComposedDpEvent(
            event.PoissonSampledDpEvent(sampling, event.GaussianDpEvent(noise_multiplier)),
            steps,
        )
        accountant.compose(dpevent)
        return accountant.get_epsilon(target_delta) - target_epsilon

    return opt.brentq(objective, 1e-6, 1000)


# --------------------------------------------------------------------------- #
# DP-SGD (mode = dpsgd) — full-batch, per-sample grad clip on ALL params
# --------------------------------------------------------------------------- #
def per_sample_gradients(model, criterion, x_b, y_b):
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v for k, v in model.named_buffers()}

    def compute_loss(params, buffers, x, y):
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        logits = torch.func.functional_call(model, {**params, **buffers}, (x,))
        return criterion(logits, y)

    return vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(params, buffers, x_b, y_b)


def train_one_epoch_dpsgd(
    model,
    loader,
    criterion,
    optimizer,
    max_grad_norm,
    noise_multiplier,
    device,
    microbatch_size=16,
):
    model.train()

    total_loss, correct, total = 0.0, 0, 0

    for x_b, y_b in loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)

        B = x_b.size(0)

        optimizer.zero_grad(set_to_none=True)

        # running summed gradients
        summed_grads = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        ####################################################
        # Process microbatches
        ####################################################
        for start in range(0, B, microbatch_size):

            end = min(start + microbatch_size, B)

            x_mb = x_b[start:end]
            y_mb = y_b[start:end]

            M = x_mb.size(0)

            per_sample_grads = per_sample_gradients(
                model,
                criterion,
                x_mb,
                y_mb,
            )

            ################################################
            # global norm
            ################################################

            per_example_sq_norm = torch.zeros(
                M,
                device=device,
            )

            for g in per_sample_grads.values():
                per_example_sq_norm += g.flatten(1).pow(2).sum(dim=1)

            clip_factor = (
                max_grad_norm /
                (per_example_sq_norm.sqrt() + 1e-6)
            ).clamp(max=1.0)

            ################################################
            # accumulate
            ################################################

            for name in summed_grads:

                g = per_sample_grads[name]

                view = (M,) + (1,) * (g.ndim - 1)

                summed_grads[name] += (
                    g * clip_factor.view(view)
                ).sum(dim=0)

            del per_sample_grads
            del per_example_sq_norm
            del clip_factor

        ####################################################
        # add noise once
        ####################################################

        for name, param in model.named_parameters():

            if not param.requires_grad:
                continue

            grad_sum = summed_grads[name]

            if noise_multiplier > 0:
                grad_sum += (
                    torch.randn_like(grad_sum)
                    * noise_multiplier
                    * max_grad_norm
                )

            param.grad = grad_sum / B

        optimizer.step()

        del summed_grads

        ####################################################
        # metrics
        ####################################################

        with torch.no_grad():
            logits = model(x_b)
            loss = criterion(logits, y_b)

        total_loss += loss.item() * B
        correct += (logits.argmax(1) == y_b).sum().item()
        total += B

    return total_loss / total, correct / total


def run_training_dpsgd(model, train_loader, val_loader, device, epochs=100, lr=1e-3,
                        max_grad_norm=1.0, noise_multiplier=1.1, label='Model'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch_dpsgd(model, train_loader, criterion, optimizer,
                                                 max_grad_norm, noise_multiplier, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{label}] Epoch {epoch:>3}/{epochs}  "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  "
                  f"(σ={noise_multiplier} C={max_grad_norm})")
    return {'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc}


# --------------------------------------------------------------------------- #
# Feature DP-SGD (mode = feature_dpsgd) — separate clip for image vs meta grads
# --------------------------------------------------------------------------- #
def per_sample_gradients_featuredp(model, criterion, x_b, y_b, img_dim):
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v for k, v in model.named_buffers()}

    def compute_loss(params, buffers, x, y):
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        logits = torch.func.functional_call(model, {**params, **buffers}, (x,))
        return criterion(logits, y)

    grads_full = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(params, buffers, x_b, y_b)

    x_zero = x_b.clone()
    x_zero[:, img_dim:] = 0
    grads_base = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(params, buffers, x_zero, y_b)

    grads_meta = {k: grads_full[k] - grads_base[k] for k in grads_full}
    return grads_base, grads_meta


def train_one_epoch_feature_dpsgd(model, loader, criterion, optimizer, noise_multiplier,
                                   img_dim, C_base, C_meta, device,
                                   microbatch_size=16):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        B = x_b.size(0)

        optimizer.zero_grad(set_to_none=True)

        summed_grads = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        for start in range(0, B, microbatch_size):
            end = min(start + microbatch_size, B)

            x_mb = x_b[start:end]
            y_mb = y_b[start:end]
            M = x_mb.size(0)

            grads_base, grads_meta = per_sample_gradients_featuredp(
                model, criterion, x_mb, y_mb, img_dim
            )

            base_sq = torch.zeros(M, device=device)
            meta_sq = torch.zeros(M, device=device)

            for k in grads_base:
                base_sq += grads_base[k].flatten(1).square().sum(1)
                meta_sq += grads_meta[k].flatten(1).square().sum(1)

            base_clip = (C_base / (base_sq.sqrt() + 1e-6)).clamp(max=1)
            meta_clip = (C_meta / (meta_sq.sqrt() + 1e-6)).clamp(max=1)

            for name in summed_grads:
                g_base = grads_base[name]
                g_meta = grads_meta[name]

                shape = (M,) + (1,) * (g_base.ndim - 1)

                summed_grads[name] += (
                    g_base * base_clip.view(shape)
                    + g_meta * meta_clip.view(shape)
                ).sum(0)

            del grads_base, grads_meta
            del base_sq, meta_sq, base_clip, meta_clip

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            g = summed_grads[name]

            if noise_multiplier > 0:
                g += torch.randn_like(g) * noise_multiplier * C_meta

            param.grad = g / B

        optimizer.step()

        del summed_grads

        with torch.no_grad():
            logits = model(x_b)
            loss = criterion(logits, y_b)

        total_loss += loss.item() * B
        correct += (logits.argmax(1) == y_b).sum().item()
        total += B

    return total_loss / total, correct / total


def run_training_feature_dpsgd(model, train_loader, val_loader, device, epochs=100, lr=1e-3,
                                C_base=50.0, C_meta=1.0, noise_multiplier=1.1, img_dim=1536, label='Model'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch_feature_dpsgd(model, train_loader, criterion, optimizer,
                                                         noise_multiplier, img_dim, C_base, C_meta, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{label}] Epoch {epoch:>3}/{epochs}  "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  "
                  f"(σ={noise_multiplier} C_base={C_base} C_meta={C_meta})")
    return {'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc}
