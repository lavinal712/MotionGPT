from typing import List
import torch
from torch import Tensor
from torchmetrics import Metric
from mGPT.metrics.dtw_torch import dtw_pa_jpe
from mGPT.utils.human_models import smpl_x
from collections import defaultdict


class TM2TMetrics(Metric):
    def __init__(self,
                 cfg,
                 dataname='humanml3d',
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.name = "MPJPE, MPVPE DTW"
        # self.top_k = top_k
        # self.R_size = R_size
        # self.text = 'lm' in cfg.TRAIN.STAGE and cfg.model.params.task == 't2m'
        # self.diversity_times = diversity_times

        self.joint_part2idx = smpl_x.joint_part2idx
        self.vertex_part2idx = smpl_x.vertex_part2idx
        self.smplx_part2idx = {'upper_body': list(range(30)), 'lhand': list(range(30, 75)), 'rhand': list(range(75, 120)), 'hand': list(range(30, 120)), 'face': list(range(120, 133))}
        self.J_regressor_idx = smpl_x.J_regressor_idx
        self.name2scores = defaultdict(dict)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("how2sign_count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("csl_count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("phoenix_count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.add_state("how2sign_DTW_MPJPE_PA_lhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_DTW_MPJPE_PA_rhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_DTW_MPJPE_PA_body",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")

        self.add_state("csl_DTW_MPJPE_PA_lhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("csl_DTW_MPJPE_PA_rhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("csl_DTW_MPJPE_PA_body",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        
        self.add_state("phoenix_DTW_MPJPE_PA_lhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("phoenix_DTW_MPJPE_PA_rhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("phoenix_DTW_MPJPE_PA_body",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")

        self.MR_metrics = ["how2sign_DTW_MPJPE_PA_lhand", "how2sign_DTW_MPJPE_PA_rhand", "how2sign_DTW_MPJPE_PA_body", 
                           "csl_DTW_MPJPE_PA_lhand", "csl_DTW_MPJPE_PA_rhand", "csl_DTW_MPJPE_PA_body",
                           "phoenix_DTW_MPJPE_PA_lhand", "phoenix_DTW_MPJPE_PA_rhand", "phoenix_DTW_MPJPE_PA_body"]

        # All metric
        self.metrics = self.MR_metrics

        
    @torch.no_grad()
    def compute(self, sanity_flag):

        mr_metrics = {}
        for name in self.metrics:
            d = name.split('_')[0]
            mr_metrics[name] = getattr(self, name) / max(getattr(self, f'{d}_count_seq'), 1e-6)

        for name, v in mr_metrics.items():
            print(name, ': ', v)
        
        # Reset
        self.reset()
        
        return mr_metrics


    @torch.no_grad()
    def update(self, 
               feats_rst: Tensor, feats_ref: Tensor,
               joints_rst: Tensor, joints_ref: Tensor,
               vertices_rst: Tensor, vertices_ref: Tensor,
               lengths: List[int], lengths_rst: List[int],
               split: str, src: List[str], name: List[str]):
        # assert joints_rst.shape == joints_ref.shape
        # assert joints_rst.dim() == 4
        # (bs, seq, njoint=22, 3)

        B = len(lengths)
        BT, N = joints_rst.shape[:2]
        joints_rst = joints_rst.reshape(B, BT//B, N, 3)
        BT, N = joints_ref.shape[:2]
        joints_ref = joints_ref.reshape(B, BT//B, N, 3)

        BT, N = vertices_rst.shape[:2]
        vertices_rst = vertices_rst.reshape(B, BT//B, N, 3)
        BT, N = vertices_ref.shape[:2]
        vertices_ref = vertices_ref.reshape(B, BT//B, N, 3)

        # Keep geometry on the current rank's device throughout evaluation.
        device = joints_rst.device
        left_regressor = torch.as_tensor(
            smpl_x.orig_hand_regressor['left'], device=device, dtype=torch.float32
        )
        right_regressor = torch.as_tensor(
            smpl_x.orig_hand_regressor['right'], device=device, dtype=torch.float32
        )

        for i in range(B):
            cur_len, rst_len = int(lengths[i]), int(lengths_rst[i])
            if not (0 < cur_len <= joints_ref.shape[1]
                    and 0 < rst_len <= joints_rst.shape[1]):
                raise ValueError("Invalid reference/generated motion length.")
            data_src, cur_name = src[i], name[i]
            getattr(self, f"{data_src}_count_seq").add_(1)

            if split not in ('val', 'test'):
                continue

            # Preserve SOKE's body alignment over ALL joints before scoring
            # the upper-body subset. Hands use the original mesh regressors.
            with torch.autocast(device_type=device.type, enabled=False):
                mesh_gt = vertices_ref[i, :cur_len].float()
                mesh_out = vertices_rst[i, :rst_len].float()
                values = {
                    'body': dtw_pa_jpe(
                        joints_rst[i, :rst_len], joints_ref[i, :cur_len],
                        wanted=self.joint_part2idx['upper_body'],
                    ),
                    'lhand': dtw_pa_jpe(
                        torch.matmul(left_regressor, mesh_out),
                        torch.matmul(left_regressor, mesh_gt),
                    ),
                    'rhand': dtw_pa_jpe(
                        torch.matmul(right_regressor, mesh_out),
                        torch.matmul(right_regressor, mesh_gt),
                    ),
                }
            for part, value in values.items():
                key = f"{data_src}_DTW_MPJPE_PA_{part}"
                getattr(self, key).add_(value.to(getattr(self, key)))
                self.name2scores[cur_name][key] = value.item()
