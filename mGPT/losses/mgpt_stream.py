import torch
import torch.distributed as dist
import torch.nn.functional as F

from .base import BaseLosses


class StreamLosses(BaseLosses):
    def __init__(self, cfg, num_joints, **kwargs):
        super().__init__(
            cfg,
            losses=["flow_loss", "end_loss"],
            params={
                "flow_loss": float(cfg.LOSS.get("LAMBDA_FLOW", 1.0)),
                "end_loss": float(cfg.LOSS.get("LAMBDA_END", 0.1)),
            },
            losses_func={
                "flow_loss": torch.nn.MSELoss,
                "end_loss": torch.nn.BCEWithLogitsLoss,
            },
            num_joints=num_joints,
        )
        self.register_buffer("end_count", torch.tensor(0.0))

    def update(self, rs_set):
        pred = rs_set["velocity_pred"]
        target = rs_set["velocity_target"]

        if pred.shape != target.shape:
            raise ValueError("Velocity prediction and target shapes differ.")

        mask = rs_set.get("latent_mask")
        if mask is not None:
            if mask.dtype != torch.bool or mask.shape != pred.shape[:-1]:
                raise ValueError("latent_mask must match velocity positions.")
            pred, target = pred[mask], target[mask]

        if pred.numel() == 0:
            raise ValueError("Flow loss requires valid targets.")

        value = F.mse_loss(pred.float(), target.detach().float())

        logits = rs_set["end_logits"].float()
        labels = rs_set["end_targets"].float()
        end_mask = rs_set["end_mask"]
        if (
            logits.ndim != 2
            or labels.shape != logits.shape
            or end_mask.shape != logits.shape
            or end_mask.dtype != torch.bool
        ):
            raise ValueError("END logits, targets and mask must match [B, S].")

        positive = end_mask & labels.eq(1)
        negative = end_mask & labels.eq(0)
        if not (positive.sum(dim=1).eq(1) & negative.any(dim=1)).all():
            raise ValueError("Each sequence needs one END and at least one motion.")

        terms = F.binary_cross_entropy_with_logits(
            logits.masked_fill(~end_mask, 0),
            labels.masked_fill(~end_mask, 0),
            reduction="none",
        )
        # Balance END and motion positions within each sequence.
        end_value = 0.5 * (
            terms.masked_fill(~positive, 0).sum(dim=1)
            + terms.masked_fill(~negative, 0).sum(dim=1) / negative.sum(dim=1)
        ).mean()
        total = (
            self._params["flow_loss"] * value
            + self._params["end_loss"] * end_value
        )

        # Element-weighted logging, rather than averaging batch means.
        self.flow_loss.add_(value.detach() * pred.numel())
        self.count.add_(pred.numel())
        self.end_loss.add_(end_value.detach() * logits.shape[0])
        self.end_count.add_(logits.shape[0])

        return total

    def reset(self):
        super().reset()
        self.end_count.zero_()

    def compute(self, split):
        values = torch.stack(
            (self.flow_loss, self.count, self.end_loss, self.end_count)
        ).clone()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values)

        self.reset()

        if values[1].item() == 0 or values[3].item() == 0:
            return {}

        flow = values[0] / values[1]
        end = values[2] / values[3]
        return {
            f"flow/loss/{split}": flow.item(),
            f"end/loss/{split}": end.item(),
            f"total/{split}": (
                self._params["flow_loss"] * flow + self._params["end_loss"] * end
            ).item(),
        }
