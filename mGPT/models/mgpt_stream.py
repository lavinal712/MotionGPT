import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, MT5EncoderModel

from mGPT.config import instantiate_from_config
from mGPT.losses.mgpt_stream import StreamLosses
from .base import BaseModel


class StreamModel(BaseModel):
    def __init__(
        self,
        cfg,
        datamodule,
        generator,
        flow_head,
        motion_vae,
        hand_vae_cfg,
        rhand_vae_cfg,
        vae_checkpoint,
        text_model_path,
        end_latent,
        end_threshold,
        stage="stream",
        task="t2m",
        debug=False,
        metrics_dict=(),
        max_text_length=128,
        time_sampling="logit_normal",
        num_time_samples=1,
        num_flow_steps=16,
        max_new_tokens=100,
        min_new_tokens=1,
        text_dropout_prob=0.0,
        guidance_scale=1.0,
        end_mode="latent",
        end_prob_threshold=0.5,
        two_forward_max_ratio=0.0,
        two_forward_warmup_steps=10000,
        two_forward_steps=4,
        two_forward_chunk_size=256,
        latent_pack_size=1,
        df_noise_levels=(),
        df_noise_warmup_steps=10000,
    ):
        self.save_hyperparameters(ignore="datamodule", logger=False)
        self.datamodule = datamodule
        super().__init__()

        if time_sampling not in ("uniform", "logit_normal"):
            raise ValueError("Unsupported flow time sampling.")
        if not 0 <= text_dropout_prob <= 1:
            raise ValueError("text_dropout_prob must be in [0, 1].")
        if not 0 <= guidance_scale < float("inf"):
            raise ValueError("guidance_scale must be finite and nonnegative.")
        if end_mode not in ("latent", "classifier", "either"):
            raise ValueError("end_mode must be latent, classifier, or either.")
        if not 0 < end_prob_threshold < 1:
            raise ValueError("end_prob_threshold must be in (0, 1).")
        if num_time_samples < 1 or num_flow_steps < 1:
            raise ValueError("Flow sample counts must be positive.")
        if not 0 <= two_forward_max_ratio <= 1:
            raise ValueError("two_forward_max_ratio must be in [0, 1].")
        if (
            not isinstance(two_forward_warmup_steps, int)
            or two_forward_warmup_steps < 0
            or not isinstance(two_forward_steps, int)
            or two_forward_steps < 1
            or not isinstance(two_forward_chunk_size, int)
            or two_forward_chunk_size < 1
        ):
            raise ValueError("Invalid Two-Forward step or chunk counts.")
        if not 1 <= min_new_tokens <= max_new_tokens:
            raise ValueError(
                "Require 1 <= min_new_tokens <= max_new_tokens."
            )
        if not isinstance(latent_pack_size, int) or latent_pack_size < 1:
            raise ValueError("latent_pack_size must be a positive integer.")
        levels = tuple(float(level) for level in df_noise_levels)
        if levels and (
            0.0 not in levels
            or max(levels) <= 0
            or any(not 0 <= level <= 1 for level in levels)
        ):
            raise ValueError("DF noise levels must include 0 and a positive level, all within [0, 1].")
        if not isinstance(df_noise_warmup_steps, int) or df_noise_warmup_steps < 0:
            raise ValueError("df_noise_warmup_steps must be a nonnegative integer.")
        self.register_buffer(
            "_df_noise_levels", torch.tensor(levels, dtype=torch.float32),
            persistent=False,
        )

        # Frozen motion VAEs.
        self.vae = instantiate_from_config(motion_vae)
        self.hand_vae = instantiate_from_config(hand_vae_cfg)
        self.rhand_vae = instantiate_from_config(rhand_vae_cfg)

        vaes = (self.vae, self.hand_vae, self.rhand_vae)

        if [v.nfeats for v in vaes] != [43, 45, 45]:
            raise ValueError(
                "This model expects the current 133-D sign format."
            )

        for config in (motion_vae, hand_vae_cfg, rhand_vae_cfg):
            params = config["params"]
            if (
                not params.get("is_causal", False)
                or params.get("down_t", 2) != 2
                or params.get("stride_t", 2) != 2
            ):
                raise ValueError(
                    "This version requires causal VAEs with factor 4."
                )

        self.temporal_factor = 4
        self.latent_dims = [v.latent_dim for v in vaes]

        # Load a trusted checkpoint produced by this repository.
        # Lightning checkpoints contain configuration objects as well as weights.
        checkpoint = torch.load(
            vae_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint.get("state_dict", checkpoint)

        for name, vae in zip(
            ("vae", "hand_vae", "rhand_vae"), vaes
        ):
            prefix = name + "."
            weights = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            vae.load_state_dict(weights, strict=True)
            vae.requires_grad_(False).eval()

        # Frozen multilingual text encoder.
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.text_encoder = MT5EncoderModel.from_pretrained(text_model_path)
        self.text_encoder.requires_grad_(False).eval()

        # Trainable networks.
        if levels:
            generator = dict(generator)
            generator["params"] = dict(generator.get("params", {}))
            generator["params"]["history_noise_conditioning"] = True
        self.generator = instantiate_from_config(generator)
        self.flow_head = instantiate_from_config(flow_head)

        latent_dim = sum(self.latent_dims)
        packed_dim = latent_dim * latent_pack_size

        if (
            self.generator.latent_dim != packed_dim
            or self.flow_head.latent_dim != packed_dim
        ):
            raise ValueError(
                f"Generator/head latent_dim must be {packed_dim} "
                "(VAE latent dimension * latent_pack_size)."
            )
        if self.flow_head.condition_dim != self.generator.hidden_dim:
            raise ValueError(
                "Flow condition dimension does not match generator."
            )
        if (
            self.text_encoder.config.d_model
            != self.generator.text_proj.in_features
        ):
            raise ValueError(
                "Text encoder dimension does not match generator."
            )

        # Build a fixed END latent from zero encoder inputs.
        # VAEs have already loaded their weights and entered eval mode.
        if end_latent is None:
            with torch.no_grad():
                end_parts = []

                for vae in (self.vae, self.hand_vae, self.rhand_vae):
                    parameter = next(vae.parameters())

                    # Input is already in the encoder's feature space.
                    # Do not apply mean/std normalization again.
                    end_pose = torch.zeros(
                        1,
                        self.temporal_factor,
                        vae.nfeats,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )

                    # Deterministic encoding, consistent with encode_motion().
                    end_part = vae.encode(end_pose).mode()

                    if end_part.shape != (1, 1, vae.latent_dim):
                        raise ValueError(
                            "END input must encode into exactly one latent: "
                            f"got {tuple(end_part.shape)}."
                        )

                    end_parts.append(end_part[0, 0].float())

                end = torch.cat(end_parts, dim=-1)
        else:
            end = torch.as_tensor(end_latent, dtype=torch.float32)

        if end.shape != (latent_dim,) or not torch.isfinite(end).all():
            raise ValueError(
                f"END latent must be a finite vector of shape ({latent_dim},)."
            )

        if not 0 < float(end_threshold) < float("inf"):
            raise ValueError(
                "end_threshold must be positive and finite."
            )

        # Saved in state_dict and moved with the model.
        self.register_buffer("end_latent", end.detach().clone())

        self._losses = nn.ModuleDict({
            "losses_" + split: StreamLosses(
                cfg, datamodule.njoints
            )
            for split in ("train", "val", "test")
        })

    def configure_metrics(self):
        # BaseModel calls this during initialization.
        # Only build metrics used by this model.
        self.metrics = nn.ModuleDict()

        for name in self.hparams.metrics_dict:
            if name != "TM2TMetrics":
                raise ValueError(
                    "Only TM2TMetrics is supported here."
                )

            from mGPT.metrics.t2m import TM2TMetrics

            self.metrics[name] = TM2TMetrics(cfg=self.hparams.cfg)

    def train(self, mode=True):
        super().train(mode)

        # requires_grad=False alone does not disable dropout.
        for module in (
            self.vae,
            self.hand_vae,
            self.rhand_vae,
            self.text_encoder,
        ):
            module.eval()

        return self

    @torch.no_grad()
    def encode_text(self, batch):
        tokens = self.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=self.hparams.max_text_length,
            return_tensors="pt",
        ).to(self.device)

        features = self.text_encoder(**tokens).last_hidden_state

        # Target sign languages: ASL, CSL, DGS.
        mapping = {"how2sign": 0, "csl": 1, "phoenix": 2}
        language_ids = torch.tensor(
            [mapping[src] for src in batch["src"]],
            device=self.device,
        )

        return (
            features,
            tokens["attention_mask"].bool(),
            language_ids,
        )

    @torch.no_grad()
    def encode_motion(self, batch):
        sequences = []

        for motion, length in zip(batch["motion"], batch["length"]):
            length = int(length)

            if not 0 < length <= motion.shape[0]:
                raise ValueError("Invalid motion length.")

            # Crop before encoding. Batch padding must not enter the
            # final latent block. AutoencoderKL pads each clip internally.
            motion = motion[:length].unsqueeze(0)

            body = torch.cat(
                (motion[..., :30], motion[..., 120:]),
                dim=-1,
            )

            latent = torch.cat(
                (
                    self.vae.encode(body).mode(),
                    self.hand_vae.encode(motion[..., 30:75]).mode(),
                    self.rhand_vae.encode(motion[..., 75:120]).mode(),
                ),
                dim=-1,
            )[0].float()

            # Repeat the last motion latent to fill a complete motion pack.
            # These fill slots are supervised and decoded at inference.
            pack_size = self.hparams.latent_pack_size
            padding = (-len(latent)) % pack_size
            if padding:
                latent = torch.cat((latent, latent[-1:].expand(padding, -1)))
            packed = latent.reshape(-1, self.generator.latent_dim)
            # END occupies its own full pack, before masked batch padding.
            end_pack = self.end_latent.repeat(pack_size)[None]
            sequences.append(torch.cat((packed, end_pack), dim=0))

        targets = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor(
            [len(z) for z in sequences],
            device=targets.device,
        )
        mask = (
            torch.arange(targets.shape[1], device=targets.device)[None]
            < lengths[:, None]
        )

        return targets, mask

    @torch.no_grad()
    def _two_forward_inputs(self, targets, mask, text, text_mask, language_ids):
        motion = targets[:, :-1]
        ratio = self.hparams.two_forward_max_ratio
        warmup = self.hparams.two_forward_warmup_steps
        if warmup:
            progress = min(self.global_step / warmup, 1.0)
            ratio *= 0.5 * (1.0 - math.cos(math.pi * progress))
        if ratio == 0:
            return motion

        # Replace motion packs with a valid successor, including the final
        # repeated motion pack. END and batch padding are excluded.
        replace = torch.rand(motion.shape[:2], device=motion.device) < ratio
        replace &= mask[:, :-1] & mask[:, 1:]
        if not replace.any():
            return motion

        # Do not cache no-grad weight casts for the subsequent training pass.
        with torch.autocast(
            device_type=motion.device.type,
            enabled=torch.is_autocast_enabled(motion.device.type),
            dtype=torch.get_autocast_dtype(motion.device.type),
            cache_enabled=False,
        ):
            first = self.generator(
                motion, text, text_mask,
                motion_mask=mask[:, :-1], language_ids=language_ids,
            )
            # Output position i predicts target pack_i, not pack_{i+1}.
            conditions = first["condition"][:, :-1][replace]
            del first
            predictions = [
                self.sample_latent(chunk, num_steps=self.hparams.two_forward_steps)
                for chunk in conditions.split(self.hparams.two_forward_chunk_size)
            ]
        motion = motion.clone()
        motion[replace] = torch.cat(predictions).to(motion)
        return motion

    def train_forward(self, batch):
        text, text_mask, language_ids = self.encode_text(batch)
        targets, mask = self.encode_motion(batch)
        end_targets = torch.zeros_like(mask, dtype=torch.float32)
        end_targets.scatter_(1, (mask.sum(dim=1) - 1)[:, None], 1.0)

        # Drop text per sequence; retain language and motion history.
        # Validation keeps the original conditional loss.
        if self.training and self.hparams.text_dropout_prob > 0:
            drop_text = torch.rand(text.shape[0], device=text.device)
            drop_text = drop_text < self.hparams.text_dropout_prob
            text_mask = text_mask & ~drop_text[:, None]

        # StreamGenerator inserts BOS internally.
        # Shift complete packs, so no target pack enters its own condition.
        # Input:  BOS, pack0, pack1, ...
        # Target: pack0, pack1, ..., independent END pack
        motion_inputs = targets[:, :-1]
        if self.training and self.hparams.two_forward_max_ratio > 0:
            motion_inputs = self._two_forward_inputs(
                targets, mask, text, text_mask, language_ids
            )
        history_noise_levels = None
        if self.training and self._df_noise_levels.numel():
            indices = torch.randint(
                len(self._df_noise_levels), motion_inputs.shape[:2],
                device=motion_inputs.device,
            )
            history_noise_levels = self._df_noise_levels[indices]
            warmup = self.hparams.df_noise_warmup_steps
            if warmup:
                progress = min(self.global_step / warmup, 1.0)
                history_noise_levels = history_noise_levels * (
                    0.5 * (1.0 - math.cos(math.pi * progress))
                )
            # Only motion history is corrupted; END/padding and GT targets stay intact.
            history_noise_levels = history_noise_levels.masked_fill(
                ~(mask[:, :-1] & mask[:, 1:]), 0.0
            )
            strength = history_noise_levels[..., None]
            motion_inputs = (
                (1.0 - strength) * motion_inputs
                + strength * torch.randn_like(motion_inputs)
            )
        output = self.generator(
            motion_inputs,
            text,
            text_mask,
            motion_mask=mask[:, :-1],
            language_ids=language_ids,
            history_noise_levels=history_noise_levels,
        )

        # Remove padding before evaluating the flow head.
        # END remains a valid training target.
        target = targets[mask].float()
        condition = output["condition"][mask]

        repeats = self.hparams.num_time_samples
        target = target.repeat_interleave(repeats, dim=0)
        condition = condition.repeat_interleave(repeats, dim=0)

        noise = torch.randn_like(target)

        if self.hparams.time_sampling == "uniform":
            t = torch.rand(len(target), device=target.device)
        else:
            t = torch.randn(
                len(target), device=target.device
            ).sigmoid()

        x = (1 - t[:, None]) * noise + t[:, None] * target

        # Loss computation belongs to StreamLosses.
        return {
            "velocity_pred": self.flow_head(x, t, condition),
            "velocity_target": target - noise,
            "end_logits": output["end_logits"],
            "end_targets": end_targets,
            "end_mask": mask,
        }

    @torch.no_grad()
    def _flow_time_embeddings(self, batch_size, device, num_steps=None):
        # Preserve each step's original batch shape: batching time steps
        # together can change floating-point results during long rollouts.
        steps = self.hparams.num_flow_steps if num_steps is None else num_steps
        return torch.stack([
            self.flow_head.time_embed(
                torch.full((batch_size,), i / steps, device=device)
            )
            for i in range(steps)
        ])

    @torch.no_grad()
    def sample_latent(
        self, condition, unconditional_condition=None, time_embeddings=None,
        num_steps=None,
    ):
        scale = self.hparams.guidance_scale
        use_cfg = unconditional_condition is not None and scale != 1.0
        if use_cfg:
            if unconditional_condition.shape != condition.shape:
                raise ValueError("CFG conditions must have matching shapes.")
            paired_condition = torch.cat((condition, unconditional_condition))
        else:
            paired_condition = condition

        x = torch.randn(
            condition.shape[0],
            self.flow_head.latent_dim,
            device=condition.device,
            dtype=torch.float32,
        )

        steps = self.hparams.num_flow_steps if num_steps is None else num_steps
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("num_steps must be a positive integer.")
        if time_embeddings is None:
            time_embeddings = self._flow_time_embeddings(
                len(paired_condition), condition.device, steps
            )
        if (
            time_embeddings.shape != (
                steps, len(paired_condition), self.flow_head.input_proj.out_features
            )
            or time_embeddings.device != condition.device
        ):
            raise ValueError("Flow time embeddings must match steps, batch and device.")

        projected_condition = self.flow_head.condition_proj(
            paired_condition.to(self.flow_head.input_proj.weight.dtype)
        )

        # Euler integration from noise (t=0) to motion latent (t=1).
        for i in range(steps):
            context = time_embeddings[i] + projected_condition
            if use_cfg:
                # Both branches evaluate the same noisy latent and flow time.
                velocity_c, velocity_u = self.flow_head.forward_with_context(
                    torch.cat((x, x)),
                    context,
                ).float().chunk(2)
                velocity = velocity_u + scale * (velocity_c - velocity_u)
            else:
                velocity = self.flow_head.forward_with_context(x, context)
            x = x + velocity.float() / steps

        return x

    @torch.no_grad()
    def forward(self, batch):
        """Generate motion without using reference motion or its length."""
        if self.training:
            raise RuntimeError("Use eval() before generation.")

        text, text_mask, language_ids = self.encode_text(batch)
        batch_size = text.shape[0]
        limit = self.hparams.max_new_tokens
        # Generation limits retain their original (unpacked) latent units.
        pack_size = self.hparams.latent_pack_size
        pack_limit = (limit + pack_size - 1) // pack_size
        end_pack = self.end_latent.repeat(pack_size)
        end_mode = self.hparams.end_mode
        use_cfg = self.hparams.guidance_scale != 1.0
        if use_cfg:
            # Conditional rows first, text-free rows second. Each branch
            # has its own cache, but sees the same generated motion history.
            text = torch.cat((text, text))
            text_mask = torch.cat((text_mask, torch.zeros_like(text_mask)))
            language_ids = torch.cat((language_ids, language_ids))

        output = self.generator(
            text.new_empty(
                text.shape[0], 0, self.generator.latent_dim
            ),
            text,
            text_mask,
            language_ids=language_ids,
            use_cache=True,
        )

        latents = torch.zeros(
            batch_size,
            pack_limit,
            self.generator.latent_dim,
            device=self.device,
        )
        lengths = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        )
        # Local to this generation call; never reused after a weight, device,
        # precision, or sampling configuration change.
        time_embeddings = self._flow_time_embeddings(text.shape[0], text.device)

        # Inspect the next END after a full final pack, as in unpacked mode.
        # A cap inside a pack must not feed its discarded slots into history.
        inspect_final_end = end_mode != "latent" and limit % pack_size == 0
        for i in range(pack_limit + int(inspect_final_end)):
            condition = output["condition"][:, -1]
            if use_cfg:
                condition, unconditional_condition = condition.chunk(2)

            can_finish = lengths * pack_size >= self.hparams.min_new_tokens
            if end_mode != "latent":
                # END uses the conditional branch, without CFG extrapolation.
                end_prob = output["end_logits"][:batch_size, -1].float().sigmoid()
                finished |= can_finish & (end_prob >= self.hparams.end_prob_threshold)

            if finished.all() or (i == pack_limit and end_mode == "classifier"):
                break

            if use_cfg:
                latent = self.sample_latent(
                    condition, unconditional_condition, time_embeddings
                )
            else:
                latent = self.sample_latent(condition, time_embeddings=time_embeddings)

            if not torch.isfinite(latent).all():
                raise RuntimeError("Non-finite generated latent.")
            if end_mode != "classifier":
                # Compare the complete END pack. Normalize for pack size so
                # the threshold retains its original latent-distance scale.
                distance = (latent - end_pack).norm(dim=-1) / math.sqrt(pack_size)
                finished |= can_finish & (distance <= self.hparams.end_threshold)

            if i == pack_limit:
                break

            active = ~finished
            # END is excluded as a whole; retain all motion-pack fill slots.
            latents[:, i] = latent.masked_fill(~active[:, None], 0)
            lengths += active.long()

            if finished.all() or (
                i + 1 == pack_limit and (end_mode == "latent" or limit % pack_size != 0)
            ):
                break

            motion_input = latents[:, i:i + 1]
            motion_mask = active[:, None]
            if use_cfg:
                motion_input = torch.cat((motion_input, motion_input))
                motion_mask = torch.cat((motion_mask, motion_mask))
            output = self.generator(
                motion_input,
                motion_mask=motion_mask,
                past_key_values=output["past_key_values"],
                past_mask=output["attention_mask"],
                use_cache=True,
                text_cache=output["text_cache"],
            )

        motions = []
        # Keep the hard cap in original latent units, even inside a pack.
        latent_lengths = (lengths * pack_size).clamp_max(limit)

        for z, length in zip(latents, latent_lengths.tolist()):
            body, left, right = (
                z.reshape(1, -1, self.end_latent.numel())[:, :length]
                .split(self.latent_dims, dim=-1)
            )

            body = self.vae.decode(body)
            left = self.hand_vae.decode(left)
            right = self.rhand_vae.decode(right)

            motion = torch.cat(
                (
                    body[..., :30],
                    left,
                    right,
                    body[..., 30:],
                ),
                dim=-1,
            )
            motions.append(motion[0])

        return {
            "feats_rst": pad_sequence(motions, batch_first=True),
            "lengths_rst": (
                latent_lengths * self.temporal_factor
            ).tolist(),
            "finished": finished,
        }

    def allsplit_step(self, split, batch, batch_idx):
        loss = None

        if split in ("train", "val"):
            rs_set = self.train_forward(batch)
            loss = self._losses["losses_" + split].update(rs_set)

        if split == "train":
            return loss

        if split == "test" or len(self.metrics):
            result = self(batch)

            if len(self.metrics):
                vertices_ref, joints_ref = (
                    self.datamodule.feats2joints(batch["motion"])
                )
                vertices_rst, joints_rst = (
                    self.datamodule.feats2joints(result["feats_rst"])
                )

                self.metrics["TM2TMetrics"].update(
                    feats_rst=result["feats_rst"],
                    feats_ref=batch["motion"],
                    joints_rst=joints_rst,
                    joints_ref=joints_ref,
                    vertices_rst=vertices_rst,
                    vertices_ref=vertices_ref,
                    lengths=batch["length"],
                    lengths_rst=result["lengths_rst"],
                    split=split,
                    src=batch["src"],
                    name=batch["name"],
                )

            if split == "test":
                return dict(
                    result,
                    feats_ref=batch["motion"],
                    lengths=batch["length"],
                    name=batch["name"],
                    text=batch["text"],
                )

        return loss

    def on_test_epoch_end(self):
        # The inherited implementation assumes the old vae/lm branches.
        if len(self.metrics):
            self.log_dict(
                self.metrics_log_dict(),
                sync_dist=True,
            )

        self.test_step_outputs.clear()
