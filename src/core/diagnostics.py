import torch
from lightning_fabric.utilities import rank_zero_info


class TrainingDiagnostics:
    """Utility class that keeps diagnostics logic out of the LightningModule."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self._checked_first_optimizer_step = False
        self._first_rqs_block = None

    def should_run_batch(self, current_epoch: int, batch_idx: int) -> bool:
        return bool(self.enabled and current_epoch == 0 and batch_idx in (0, 1))

    def _get_first_rqs_block(self, cinn_model):
        if self._first_rqs_block is not None:
            return self._first_rqs_block
        try:
            module_list = cinn_model.model.module_list
        except AttributeError:
            return None

        for mod in module_list:
            if mod.__class__.__name__ == "RationalQuadraticSplineBlock":
                self._first_rqs_block = mod
                return mod
        return None

    def log_first_block_inside_ratio(self, cinn_model, x: torch.Tensor, c: torch.Tensor, tag: str):
        if not self.enabled:
            return

        block = self._get_first_rqs_block(cinn_model)
        if block is None:
            rank_zero_info(f"[DIAGNOSTIC][{tag}] No RationalQuadraticSplineBlock found.")
            return

        if cinn_model.log_cond:
            c_norm = torch.log10(c)
        else:
            c_norm = c
        if cinn_model.pre_subnet:
            c_norm = cinn_model.pre_subnet(c_norm)

        x_log = torch.log(x + cinn_model.alpha)
        x_perm, _ = block._permute(x_log, rev=False)
        _, x2 = torch.split(x_perm, block.splits, dim=1)

        bounds = block.bounds.to(x2.device)
        inside = torch.all((x2 >= -bounds) & (x2 <= bounds), dim=-1)
        inside_count = int(inside.sum().item())
        total_count = int(inside.numel())
        inside_ratio = float(inside.float().mean().item()) if total_count > 0 else 0.0

        rank_zero_info(
            f"[DIAGNOSTIC][{tag}] first RQS inside_ratio={inside_ratio:.6f} "
            f"({inside_count}/{total_count}), bounds=[{bounds.min().item():.3f}, {bounds.max().item():.3f}]"
        )

    def _collect_gradient_diagnostics(self, lightning_module):
        trainable = []
        none_grads = []
        nonfinite_grads = []
        all_zero_grads = []
        has_zero_elements = []

        for name, param in lightning_module.named_parameters():
            if not param.requires_grad:
                continue

            trainable.append(name)
            grad = param.grad

            if grad is None:
                none_grads.append(name)
                continue

            if not torch.isfinite(grad).all():
                nonfinite_grads.append(name)

            zero_mask = grad == 0
            if zero_mask.any():
                zero_ratio = zero_mask.float().mean().item()
                has_zero_elements.append((name, zero_ratio))

            if torch.count_nonzero(grad) == 0:
                all_zero_grads.append(name)

        return {
            "num_trainable": len(trainable),
            "none_grads": none_grads,
            "nonfinite_grads": nonfinite_grads,
            "all_zero_grads": all_zero_grads,
            "has_zero_elements": has_zero_elements,
        }

    def log_gradient_diagnostics(self, lightning_module, tag: str):
        if not self.enabled:
            return

        diag = self._collect_gradient_diagnostics(lightning_module)
        rank_zero_info(
            f"[DIAGNOSTIC][{tag}] trainable parameters: {diag['num_trainable']}"
        )

        if diag["none_grads"]:
            rank_zero_info(
                f"[DIAGNOSTIC][{tag}] Parameters with grad=None: {len(diag['none_grads'])}. "
                f"Examples: {diag['none_grads'][:10]}"
            )

        if diag["nonfinite_grads"]:
            rank_zero_info(
                f"[DIAGNOSTIC][{tag}] Parameters with non-finite gradients: {len(diag['nonfinite_grads'])}. "
                f"Examples: {diag['nonfinite_grads'][:10]}"
            )

        if diag["all_zero_grads"]:
            rank_zero_info(
                f"[DIAGNOSTIC][{tag}] Parameters with all-zero gradients: {len(diag['all_zero_grads'])}. "
                f"Examples: {diag['all_zero_grads'][:10]}"
            )

        if diag["has_zero_elements"]:
            preview = [
                f"{name} (zero_ratio={ratio:.3f})"
                for name, ratio in diag["has_zero_elements"][:10]
            ]
            rank_zero_info(
                f"[DIAGNOSTIC][{tag}] Parameters containing at least one zero gradient element: "
                f"{len(diag['has_zero_elements'])}. Examples: {preview}"
            )

    def maybe_log_before_first_optimizer_step(self, lightning_module, run_diagnostics: bool):
        if not self.enabled:
            return
        if run_diagnostics and not self._checked_first_optimizer_step:
            self.log_gradient_diagnostics(lightning_module, tag="before_first_optimizer_step")
            self._checked_first_optimizer_step = True

    def log_loss_diagnostics(self, cinn_model, x, c, log_probs, batch_number: int):
        if not self.enabled:
            return

        nan_mask = ~torch.isfinite(log_probs)
        nan_count = nan_mask.sum().item()
        finite_mask = ~nan_mask
        if finite_mask.any():
            lp_min = f"{log_probs[finite_mask].min().item():.6e}"
            lp_max = f"{log_probs[finite_mask].max().item():.6e}"
        else:
            lp_min = "all_nan"
            lp_max = "all_nan"

        rank_zero_info(
            f"[DIAGNOSTIC] Batch {batch_number} NaN in log_probs: {nan_count}/{log_probs.shape[0]} samples"
        )
        rank_zero_info(
            f"  x: min={x.min().item():.6e}, max={x.max().item():.6e}, "
            f"nan_count={(~torch.isfinite(x)).sum().item()}"
        )
        rank_zero_info(
            f"  c: min={c.min().item():.6e}, max={c.max().item():.6e}, "
            f"nan_count={(~torch.isfinite(c)).sum().item()}"
        )
        rank_zero_info(
            f"  log_probs: min={lp_min}, max={lp_max}, nan_count={nan_count}"
        )

        z, log_jac_det = cinn_model.forward(x, c, rev=False)
        if isinstance(z, (tuple, list)):
            z = z[0]
        if isinstance(log_jac_det, (tuple, list)):
            log_jac_det = log_jac_det[0]

        rank_zero_info(
            f"  z: min={z.min().item():.6e}, max={z.max().item():.6e}, "
            f"nan_count={(~torch.isfinite(z)).sum().item()}, "
            f"inf_count={(torch.isinf(z)).sum().item()}"
        )
        rank_zero_info(
            f"  log_jac_det: min={log_jac_det.min().item():.6e}, max={log_jac_det.max().item():.6e}, "
            f"nan_count={(~torch.isfinite(log_jac_det)).sum().item()}"
        )
        z_sq = z**2
        rank_zero_info(
            f"  z**2: min={z_sq.min().item():.6e}, max={z_sq.max().item():.6e}, "
            f"nan_count={(~torch.isfinite(z_sq)).sum().item()}"
        )
        z_sum = torch.sum(z_sq, dim=1)
        rank_zero_info(
            f"  sum(z**2): min={z_sum.min().item():.6e}, max={z_sum.max().item():.6e}, "
            f"nan_count={(~torch.isfinite(z_sum)).sum().item()}"
        )
        mean_term = -0.5 * z_sum
        rank_zero_info(
            f"  -0.5*sum(z**2): min={mean_term.min().item():.6e}, max={mean_term.max().item():.6e}, "
            f"nan_count={(~torch.isfinite(mean_term)).sum().item()}"
        )
