# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

from math import sqrt
import math
import random
import numpy as np
import os


from einops import rearrange
import torch

from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from boltz.data import const
import boltz.model.layers.initialize as init
from boltz.model.loss.diffusion import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.model.modules.encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    FourierEmbedding,
    PairwiseConditioning,
    SingleConditioning,
)
from boltz.model.modules.transformers import (
    ConditionedTransitionBlock,
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    default,
    log,
)

# Import your existing symmetry library
from boltz.data.module import symmetry_awareness as symmetry
import inspect # Add this import
print(f"DEBUG: Loaded symmetry module from: {inspect.getfile(symmetry)}")
print(f"DEBUG: Functions in symmetry: {dir(symmetry)}")

import matplotlib.pyplot as plt



class DiffusionModule(Module):
    """Diffusion module"""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        token_s : int
            The single representation dimension.
        token_z : int
            The pair representation dimension.
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        atoms_per_window_queries : int, optional
            The number of atoms per window for queries, by default 32.
        atoms_per_window_keys : int, optional
            The number of atoms per window for keys, by default 128.
        sigma_data : int, optional
            The standard deviation of the data distribution, by default 16.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.
        atom_encoder_depth : int, optional
            The depth of the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_depth : int, optional
            The depth of the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 8.
        atom_decoder_depth : int, optional
            The depth of the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        atom_feature_dim : int, optional
            The atom feature dimension, by default 128.
        conditioning_transition_layers : int, optional
            The number of transition layers for conditioning, by default 2.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.
        offload_to_cpu : bool, optional
            Whether to offload the activations to CPU, by default False.

        """

        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data

        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        r_noisy,
        times,
        relative_position_encoding,
        feats,
        multiplicity=1,
        model_cache=None,
    ):
        s, normed_fourier = self.single_conditioner(
            times=times,
            s_trunk=s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs=s_inputs.repeat_interleave(multiplicity, 0),
        )

        if model_cache is None or len(model_cache) == 0:
            z = self.pairwise_conditioner(
                z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
            )
        else:
            z = None

        # Compute Atom Attention Encoder and aggregation to coarse-grained tokens
        a, q_skip, c_skip, p_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            s_trunk=s_trunk,
            z=z,
            r=r_noisy,
            multiplicity=multiplicity,
            model_cache=model_cache,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            z=z,  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
            model_cache=model_cache,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            p=p_skip,
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=model_cache,
        )

        return {"r_update": r_update, "token_a": a}


class OutTokenFeatUpdate(Module):
    """Output token feature update"""

    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
    ):
        """Initialize the Output token feature update for confidence model.

        Parameters
        ----------
        sigma_data : float
            The standard deviation of the data distribution.
        token_s : int, optional
            The token dimension, by default 384.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.

        """

        super().__init__()
        self.sigma_data = sigma_data

        self.norm_next = nn.LayerNorm(2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier
        )

    def forward(
        self,
        times,
        acc_a,
        next_a,
    ):
        next_a = self.norm_next(next_a)
        fourier_embed = self.fourier_embed(times)
        normed_fourier = (
            self.norm_fourier(fourier_embed)
            .unsqueeze(1)
            .expand(-1, next_a.shape[1], -1)
        )
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a


# ================================
#  The high-level AtomDiffusion
# ================================
class AtomDiffusion(nn.Module):
    """
    The wrapper that:
      - samples noise
      - calls DiffusionModule => denoise
      - enforces symmetrical noising & symmetrical denoising
      - forcibly re-rotates subunits so each subunit's COM is exactly 
        at the correct position/orientation about the origin every iteration.
    """

    def __init__(
        self,
        score_model_args,
        num_sampling_steps=5,
        sigma_min=0.0004,
        sigma_max=160.0,
        sigma_data=16.0,
        rho=7,
        P_mean=-1.2,
        P_std=1.5,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        coordinate_augmentation=True,
        compile_score=False,
        alignment_reverse_diff=False,
        synchronize_sigmas=False,
        use_inference_model_cache=False,
        accumulate_token_repr=False,
        # Symmetry
        symmetry_type="C_30",
        chain_symmetry_groups=None,
        radius: float = None,
        ring_push_strength=5.0,
        ring_push_fraction=0.0,
        **kwargs,
    ):
        super().__init__()

        # 1) The U-Net
        self.score_model = DiffusionModule(**score_model_args)
        if compile_score:
            self.score_model = torch.compile(self.score_model)

        # 2) Noise schedule stuff
        self.num_sampling_steps = num_sampling_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.coordinate_augmentation = coordinate_augmentation
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.use_inference_model_cache = use_inference_model_cache

        # 3) Accumulate token reps
        self.accumulate_token_repr = accumulate_token_repr
        if accumulate_token_repr:
            self.out_token_feat_update = OutTokenFeatUpdate(
                sigma_data=sigma_data,
                token_s=score_model_args["token_s"],
                dim_fourier=score_model_args["dim_fourier"],
            )

        # 4) Symmetry info
        self.symmetry_type = symmetry_type
        self.chain_symmetry_groups = chain_symmetry_groups or {}
        self.radius = radius
        print("Radius in diffusion.py is: ", radius)
        self.ring_push_strength = ring_push_strength
        self.ring_start_fraction = ring_push_fraction

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)
        n_subunits = len(next(iter(self.chain_symmetry_groups.values()))[0])
        # If there is only one chain, skip rotation map creation.
        if n_subunits == 1:
            self.rot_mats_noI = None
        else:
            # ---------------------------------------------------
            # Remove the identity rotation BEFORE calling reorder_point_group
            # ---------------------------------------------------
            device = self.device  # uses @property device from the module
            rot_mats = symmetry.get_point_group(self.symmetry_type, n_subunits).to(device)

            eye = torch.eye(3, device=device)

            def is_identity(R, atol=1e-5):
                return bool(torch.allclose(R, eye, atol=atol))

            # find which index is truly identity, if any
            identity_idx = None
            for i in range(rot_mats.shape[0]):
                if is_identity(rot_mats[i]):
                    identity_idx = i
                    break

            # Separate identity matrix and the rest
            if identity_idx is not None:
                identity_matrix = rot_mats[identity_idx]
                non_identity_ops = []
                for i in range(rot_mats.shape[0]):
                    if i != identity_idx:
                        non_identity_ops.append(rot_mats[i])
                rot_mats_noI = torch.stack(non_identity_ops)
            else:
                identity_matrix = eye
                rot_mats_noI = rot_mats

            # Fixed code: use the same logic as in sample() to set the reference point
            if self.radius is not None:
                if self.symmetry_type.startswith("C") or self.symmetry_type.startswith("D"):
                    ref_pt = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)
                    ref_pt = ref_pt / torch.norm(ref_pt) * self.radius
                elif self.symmetry_type in {"T", "O", "I"}:
                    ref_pt = torch.tensor([1.0, 0.3, 0.5], device=device, dtype=torch.float32)
                    ref_pt = ref_pt / torch.norm(ref_pt) * self.radius
                else:
                    ref_pt = torch.tensor([self.radius, 0.0, 0.0], device=device, dtype=torch.float32)
            else:
                ref_pt = torch.tensor([100.0, 0.0, 0.0], device=device, dtype=torch.float32)

            rot_mats_noI = symmetry.reorder_point_group(
                rot_mats_noI,
                identity_matrix,
                group_name=self.symmetry_type,
                reference_point=ref_pt
            )

            # (Optionally) remove reflections if you only want chiral subgroup:
            # dets = torch.linalg.det(rot_mats_noI)
            # keep = (dets > 0.9999) & (dets < 1.0001)
            # rot_mats_noI = rot_mats_noI[keep]

            # Store the final set of transformations WITHOUT the identity
            self.rot_mats_noI = rot_mats_noI.to(self.device)




    @property
    def device(self):
        return next(self.score_model.parameters()).device

    # -------------------------
    # Noise schedule functions
    # -------------------------
    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def sample_schedule(self, num_sampling_steps=None):
        """
        e.g. Karras sampling schedule
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho
        steps = torch.arange(num_sampling_steps, device=self.device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        sigmas *= self.sigma_data
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data)**2)

    def noise_distribution(self, batch_size):
        """
        lognormal distribution for training
        """
        return (
            self.sigma_data
            * (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()
        )

    def preconditioned_network_forward(
        self,
        noised_atom_coords,
        sigma,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * net_out["r_update"]
        )
        return denoised_coords, net_out["token_a"]



    # ------------------------------------------------------
    #  Symmetrical Denoising (preconditioned forward)
    # ------------------------------------------------------
    def preconditioned_network_forward_symmetry(
        self,
        coords_noisy: torch.Tensor,  # (B, A, 3)
        sigma: torch.Tensor or float,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        """ Symmetrical Denoising using precomputed/dynamic self.rot_mats_noI """
        B = coords_noisy.shape[0]
        device = coords_noisy.device

        if isinstance(sigma, float):
            sigma = torch.full((B,), float(sigma), device=device, dtype=coords_noisy.dtype)
        elif isinstance(sigma, torch.Tensor) and sigma.ndim == 0:
            sigma = sigma.reshape(1).expand(B)
        elif isinstance(sigma, torch.Tensor) and sigma.ndim == 1 and sigma.shape[0] == 1:
            sigma = sigma.expand(B)
        elif isinstance(sigma, torch.Tensor) and sigma.ndim == 1 and sigma.shape[0] != B:
             raise ValueError(f"Sigma tensor has incorrect size {sigma.shape[0]}, expected {B}")
        # Ensure sigma has correct dtype
        sigma = sigma.to(dtype=coords_noisy.dtype)


        c_in_val = self.c_in(sigma)[:, None, None]
        times_val = self.c_noise(sigma)

        # Filter out 'input_coords' if present, as r_noisy is the input now
        filtered_kwargs = {k: v for k, v in network_condition_kwargs.items() if k != 'input_coords'}

        # *** Crucial: Pass the current noisy coordinates ***
        net_out = self.score_model(
            r_noisy=c_in_val * coords_noisy,
            times=times_val,
            **filtered_kwargs,
        )
        # r_update is the predicted noise * c_out / sigma ? No, based on AF3, it's F_theta(c_in*x_t, c_noise)
        # And x_pred = c_skip * x_t + c_out * F_theta
        # So r_update is F_theta in this context.
        r_update = net_out["r_update"]  # (B, A, 3) # This is the model's estimate of the score/noise term scaled appropriately
        token_a = net_out["token_a"]

        # Symmetrical denoising/update propagation
        if self.symmetry_type and self.rot_mats_noI is not None:
            # Get subunits - important that this is consistent with how rot_mats_noI was derived
            subunits = symmetry.get_subunit_atom_indices(
                self.symmetry_type,
                self.chain_symmetry_groups,
                network_condition_kwargs["feats"], # Use feats from kwargs
                device,
            )
            if len(subunits) > 1:
                # rot_mats should be on the correct device already if set dynamically
                rot_mats = self.rot_mats_noI.to(device, dtype=coords_noisy.dtype) # Ensure dtype and device
                # Mapping assumes atom i in subunit 0 corresponds to atom i in subunit k
                mapping = self.get_symmetrical_atom_mapping(network_condition_kwargs["feats"])

                # Ensure r_update is contiguous for indexing safety if needed, though usually not required
                # r_update = r_update.contiguous()

                for b_idx in range(B):
                    # Check if mapping is valid for this batch element
                    if not mapping: continue

                    ref_subunit_indices = subunits[0] # Indices for subunit A

                    # Iterate through corresponding atoms across subunits
                    for local_ref_idx, all_atom_global_indices in mapping.items():
                        if not all_atom_global_indices: continue

                        ref_atom_global_idx = all_atom_global_indices[0]
                        # Check if ref atom is within the bounds of r_update for safety
                        if ref_atom_global_idx >= r_update.shape[1]: continue

                        # Get the update vector predicted for the reference atom
                        ref_update_vector = r_update[b_idx, ref_atom_global_idx, :]  # (3,)

                        # Apply fixed rotations to propagate the update to symmetrical atoms
                        for s_idx in range(1, len(all_atom_global_indices)):
                            target_atom_global_idx = all_atom_global_indices[s_idx]
                            rot_idx = s_idx - 1 # Index into rot_mats (B, C, D, E, F...)

                            # Check bounds for safety
                            if target_atom_global_idx >= r_update.shape[1] or rot_idx >= rot_mats.shape[0]:
                                continue

                            R = rot_mats[rot_idx] # Get the [3, 3] rotation matrix

                            # Rotate the reference update vector
                            # rotated_update = torch.matmul(R, ref_update_vector) # R is (3,3), vec is (3,) -> (3,)
                            rotated_update = ref_update_vector @ R.T # Equivalent: vec @ R.T

                            # Assign the rotated update to the corresponding atom
                            # Use index_copy_ or index_put_ for potentially better performance/clarity?
                            # For simplicity, direct assignment works if mapping is correct.
                            r_update[b_idx, target_atom_global_idx, :] = rotated_update

        # Calculate the denoised coordinates using the (potentially symmetrized) r_update
        c_skip_val = self.c_skip(sigma)[:, None, None]
        c_out_val = self.c_out(sigma)[:, None, None]
        coords_denoised = c_skip_val * coords_noisy + c_out_val * r_update

        return coords_denoised, token_a






    # -----------------
    # forward() => training
    # -----------------
    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        relative_position_encoding,
        feats,
        multiplicity=1,
    ):
        """
        1) sample sigma
        2) add noise to coords
        3) symmetrical denoising
        4) return coords for computing loss
        """
        B = feats["coords"].shape[0]

        # 1) sample sigma
        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(B).repeat_interleave(multiplicity, 0)
        else:
            sigmas = self.noise_distribution(B * multiplicity)

        # 2) expand coords
        coords = feats["coords"]  # shape (B, N, L, 3) or (B, A, 3)
        # adapt if needed
        if coords.ndim == 4:
            B_, N_, L_, _ = coords.shape
            coords = coords.reshape(B_, N_*L_, 3)
        coords = coords.repeat_interleave(multiplicity, 0)
        feats["coords"] = coords

        # 3) random orientation
        atom_mask = feats["atom_pad_mask"].repeat_interleave(multiplicity, 0)
        coords = center_random_augmentation(coords, atom_mask, augmentation=self.coordinate_augmentation)

        # 4) add noise
        padded_sig = rearrange(sigmas, "b -> b 1 1")
        noise = torch.randn_like(coords)
        coords_noisy = coords + padded_sig * noise

        # 5) symmetrical denoising
        coords_denoised, _ = self.preconditioned_network_forward_symmetry(
            coords_noisy,
            sigmas,
            dict(
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                relative_position_encoding=relative_position_encoding,
                feats=feats,
                multiplicity=multiplicity,
            ),
            training=True,
        )

        return {
            "noised_atom_coords": coords_noisy,
            "denoised_atom_coords": coords_denoised,
            "sigmas": sigmas,
            "aligned_true_atom_coords": coords,
        }

    # -----------------
    # compute_loss
    # -----------------
    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
    ):
        coords_denoised = out_dict["denoised_atom_coords"]
        coords_noisy    = out_dict["noised_atom_coords"]
        sigmas          = out_dict["sigmas"]

        B, A, _ = coords_denoised.shape
        resolved_mask = feats["atom_resolved_mask"].repeat_interleave(multiplicity, 0)
        align_weights = coords_noisy.new_ones(B, A)

        # heavier weighting for nucleotides or ligands
        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )
        atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)
        align_weights += nucleotide_loss_weight * (
            (atom_type_mult == const.chain_type_ids["DNA"]).float()
            + (atom_type_mult == const.chain_type_ids["RNA"]).float()
        )
        align_weights += ligand_loss_weight * (
            atom_type_mult == const.chain_type_ids["NONPOLYMER"]
        ).float()

        # Weighted rigid align => MSE
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            coords_true = out_dict["aligned_true_atom_coords"]
            coords_aligned_gt = weighted_rigid_align(
                coords_true.detach().float(),
                coords_denoised.detach().float(),
                align_weights.detach().float(),
                mask=resolved_mask.detach().float(),
            )
        coords_aligned_gt = coords_aligned_gt.to(coords_denoised)

        # MSE
        mse = ((coords_denoised - coords_aligned_gt)**2).sum(dim=-1)
        mse = torch.sum(mse * align_weights * resolved_mask, dim=-1) / torch.sum(
            3 * align_weights * resolved_mask, dim=-1
        )
        w = self.loss_weight(sigmas)
        mse_loss = (mse * w).mean()
        total_loss = mse_loss

        # optional lddt
        lddt_loss = self.zero.to(coords_denoised.device)
        if add_smooth_lddt_loss:
            lddt_loss = smooth_lddt_loss(
                coords_denoised,
                feats["coords"],  # original
                (
                    (atom_type == const.chain_type_ids["DNA"]).float()
                    + (atom_type == const.chain_type_ids["RNA"]).float()
                ),
                coords_mask=feats["atom_resolved_mask"],
                multiplicity=multiplicity,
            )
            total_loss += lddt_loss

        loss_breakdown = {
            "mse_loss": mse_loss,
            "smooth_lddt_loss": lddt_loss,
        }

        return {
            "loss": total_loss,
            "loss_breakdown": loss_breakdown
        }


    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        forward_diffusion_steps=100,  # Determines start_sigma.
        multiplicity=1,
        train_accumulate_token_repr=False,
        **network_condition_kwargs,
    ):
        """
        Performs reverse diffusion sampling with rigid symmetry enforcement.
        Constraints force subunit COMs to match input reference COMs.
        Derived rotations are used ONLY for noise/denoising symmetrization.
        ** MODIFIED: Calculates I/C3 inter-trimer rotation directly. **
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        n_batches = atom_mask.shape[0] # Total batches *after* multiplicity
        n_atoms = atom_mask.shape[1]
        device = self.device
        feats = network_condition_kwargs.get("feats", {})

        # 1. Get Input Coordinates (CLEAN) - The Reference Structure
        input_coords_ref = network_condition_kwargs.get("input_coords", None)
        if input_coords_ref is None:
             input_coords_ref = feats.get("coords", None)
             if input_coords_ref is None:
                 raise ValueError("Reference coordinates must be provided either via 'input_coords' kwarg or in 'feats'.")

        # Determine original batch size (B_ref) before applying multiplicity
        if input_coords_ref.ndim == 4: # Reshape if needed (B, N_res, L_atom, 3) -> (B, A, 3)
             B_ref, N_, L_, _ = input_coords_ref.shape # Assign B_ref here
             print(f"DEBUG: Input coords ndim=4. Reshaping from {input_coords_ref.shape}...")
             input_coords_ref = input_coords_ref.reshape(B_ref, N_*L_, 3)
             print(f"DEBUG: Reshaped input coords to {input_coords_ref.shape}.")
        elif input_coords_ref.ndim == 3: # Assume shape (B, A, 3)
             B_ref = input_coords_ref.shape[0] # ***** FIX: Assign B_ref here *****
             print(f"DEBUG: Input coords ndim=3. Shape: {input_coords_ref.shape}. Original B_ref={B_ref}.")
        else:
            raise ValueError(f"Unexpected input_coords dimension: {input_coords_ref.ndim}. Expected 3 or 4.")

        # Check batch size consistency before repeat_interleave
        if B_ref * multiplicity != n_batches:
             raise ValueError(f"Input coords batch size {B_ref} * multiplicity {multiplicity} != atom_mask total batch size {n_batches}.")

        # Apply multiplicity AFTER determining B_ref
        input_coords_ref = input_coords_ref.repeat_interleave(multiplicity, 0)
        print(f"DEBUG: Input coords after repeat_interleave({multiplicity}): {input_coords_ref.shape}")


        # Final shape check after repeat_interleave
        if input_coords_ref.shape != (n_batches, n_atoms, 3):
             try:
                 input_coords_ref = input_coords_ref.view(n_batches, n_atoms, 3)
             except RuntimeError as e:
                 raise ValueError(f"Input coordinates shape {input_coords_ref.shape} is incompatible with expected ({n_batches}, {n_atoms}, 3) after multiplicity. Error: {e}")

        coords_ref = input_coords_ref.clone().to(device).float() # Use this for deriving constraints
        network_condition_kwargs["feats"]["coords"] = coords_ref # Update feats with the reference coords
        feats = network_condition_kwargs["feats"] # Update local feats reference

        # Initialize current_coords (start with reference for noise addition)
        current_coords = coords_ref.clone()

        # Get atom indices for each subunit. For 'I', expects A, B, C, D, E, F order.
        subunits = symmetry.get_subunit_atom_indices(
            self.symmetry_type, self.chain_symmetry_groups, feats, device
        )
        n_subunits_found = len(subunits)

        # ------- Calculate Initial & **TARGET** COMs from Reference -------
        print("\n" + "="*20 + " DIAGNOSTIC: Initial/Target COMs (Input Structure) " + "="*20)
        b_idx_diag = 0 # Assuming Batch=1 for detailed printout clarity
        initial_coms = {}
        target_coms_list = [] # To store target COMs for the constraint function (B, 3) tensors

        if subunits:
             for i_sub, sub_inds in enumerate(subunits):
                 sub_name = chr(ord('A') + i_sub) # Simple naming A, B, C...
                 if sub_inds.numel() > 0:
                      # Calculate COM for all batch elements -> Shape (B, 3), B=n_batches here
                      com_ref_sub = calculate_com(coords_ref[:, sub_inds, :])
                      target_coms_list.append(com_ref_sub) # Add the (n_batches, 3) tensor

                      # Diagnostics for batch 0
                      if b_idx_diag < n_batches:
                           com_ref_sub_diag = com_ref_sub[b_idx_diag].cpu().numpy()
                           initial_coms[sub_name] = com_ref_sub_diag # Store numpy array for diagnostics
                           print(f"Batch {b_idx_diag} - Input/Target COM {sub_name}: {com_ref_sub_diag}")

                           # Add Trimer COMs for diagnostics if applicable (only for batch 0)
                           if sub_name == 'C' and i_sub == 2 and len(target_coms_list) == 3:
                                trimer1_inds_diag = torch.cat(subunits[:3])
                                if trimer1_inds_diag.numel() > 0:
                                     com_T1_diag = calculate_com(coords_ref[b_idx_diag:b_idx_diag+1, trimer1_inds_diag, :]).squeeze(0).cpu().numpy()
                                     initial_coms['Trimer1'] = com_T1_diag
                                     print(f"Batch {b_idx_diag} - Input COM Trimer1 (ABC): {com_T1_diag}")
                                else: print(f"Batch {b_idx_diag} - Cannot calculate Trimer1 COM (empty subunits).")
                           elif sub_name == 'F' and i_sub == 5 and len(target_coms_list) == 6:
                                trimer2_inds_diag = torch.cat(subunits[3:6])
                                if trimer2_inds_diag.numel() > 0:
                                     com_T2_diag = calculate_com(coords_ref[b_idx_diag:b_idx_diag+1, trimer2_inds_diag, :]).squeeze(0).cpu().numpy()
                                     initial_coms['Trimer2'] = com_T2_diag
                                     print(f"Batch {b_idx_diag} - Input COM Trimer2 (DEF): {com_T2_diag}")
                                else: print(f"Batch {b_idx_diag} - Cannot calculate Trimer2 COM (empty subunits).")
                      else:
                           # Just calculate without printing diagnostic if b_idx_diag is out of bounds (shouldn't happen if B=1)
                           pass

                 else:
                     print(f"Batch {b_idx_diag} - Warning: Subunit {sub_name} (index {i_sub}) is empty.")
                     # Add placeholder if needed for tensor stacking consistency
                     target_coms_list.append(torch.zeros((n_batches, 3), device=device, dtype=coords_ref.dtype))

             if not target_coms_list or all(sub_inds.numel()==0 for sub_inds in subunits): # Check if list is empty or all subunits were empty
                  print(f"Batch {b_idx_diag} - No non-empty subunits found. Cannot define target COMs from subunits.")
                  # Handle global centering case if desired
                  com_global_ref = calculate_com(coords_ref) # (n_batches, 3)
                  target_coms_all = com_global_ref.unsqueeze(1) # Set target as input global COM (n_batches, 1, 3)
                  if b_idx_diag < n_batches:
                       initial_coms['Global'] = com_global_ref[b_idx_diag].cpu().numpy()
                       print(f"Batch {b_idx_diag} - Using Input Global COM as target: {initial_coms['Global']}")
                  subunits = [torch.arange(n_atoms, device=device)] # Treat as one global subunit
                  n_subunits_found = 1 # Override previous value
             else:
                  # Stack the calculated reference COMs into the target tensor for the constraint function
                  target_coms_all = torch.stack(target_coms_list, dim=1) # Shape: (n_batches, N_subunits, 3)
                  print(f"DEBUG: Stacked target COMs from input reference. Shape: {target_coms_all.shape}")

        else: # No subunits found by symmetry helper
             print(f"Batch {b_idx_diag} - No subunits found by get_subunit_atom_indices.")
             com_global_ref = calculate_com(coords_ref) # (n_batches, 3)
             # Treat as a single entity, target COM is its input COM
             target_coms_all = com_global_ref.unsqueeze(1) # Shape (n_batches, 1, 3)
             if b_idx_diag < n_batches:
                  initial_coms['Global'] = com_global_ref[b_idx_diag].cpu().numpy()
                  print(f"Batch {b_idx_diag} - Input/Target Global COM: {initial_coms['Global']}")
             # Define subunits as the whole structure for the constraint function
             subunits = [torch.arange(n_atoms, device=device)]
             n_subunits_found = 1 # Override previous value

        print("="*60)
        # ------- END DIAGNOSTIC/TARGET COM Calculation -------


        # ======= Calculate Rotations for Symmetrical Noise/Denoising =======
        # === MODIFICATION: Calculate I/C3 inter-trimer rotation directly ===
        self.rot_mats_noI = None # Reset instance variable

        if self.symmetry_type == 'I' and n_subunits_found >= 6:
            # print("DEBUG: Calculating I/C3 rotations for noise/denoising symmetry (Direct Calculation)") # OLD
            print("DEBUG: Calculating I/C3 rotations for noise/denoising symmetry (Direct Calculation)") # NEW NAME

            if target_coms_all is None or target_coms_all.shape[1] < 6:
                 print("Warning: Insufficient target COMs found for I symmetry rotation calculation. Symmetrical noise/denoising might be incorrect.")
                 self.rot_mats_noI = None
            else:
                # Select COMs for rotation calculation (all shape: n_batches, 3)
                com_A_ref = target_coms_all[:, 0, :]
                com_B_ref = target_coms_all[:, 1, :]
                com_C_ref = target_coms_all[:, 2, :]
                com_D_ref = target_coms_all[:, 3, :] # Needed for direct A->D vector
                com_E_ref = target_coms_all[:, 4, :] # Potentially needed if we calculate A->E directly too
                com_F_ref = target_coms_all[:, 5, :] # Potentially needed if we calculate A->F directly too

                # Ensure subunits list is correct length before indexing
                if len(subunits) < 6:
                    raise IndexError(f"Subunits list has length {len(subunits)}, expected at least 6 for I symmetry rotation calculation.")

                trimer1_indices = torch.cat(subunits[:3])
                trimer2_indices = torch.cat(subunits[3:6])
                # Recalculate Trimer COMs (using all batch elements)
                com_ABC_ref = calculate_com(coords_ref[:, trimer1_indices, :]) # (n_batches, 3)
                com_DEF_ref = calculate_com(coords_ref[:, trimer2_indices, :]) # (n_batches, 3)

                # --- Calculate Rotations WITHIN Trimer ABC (Direct) ---
                vec_A_rel = com_A_ref - com_ABC_ref # (n_batches, 3)
                vec_B_rel = com_B_ref - com_ABC_ref # (n_batches, 3)
                vec_C_rel = com_C_ref - com_ABC_ref # (n_batches, 3)

                R_B_batch = compute_rotation_matrix_from_vectors(vec_A_rel, vec_B_rel, device=device, dtype=coords_ref.dtype)
                R_C_batch = compute_rotation_matrix_from_vectors(vec_A_rel, vec_C_rel, device=device, dtype=coords_ref.dtype)

                # --- Calculate Rotation BETWEEN Trimers (A->D, Direct) ---
                # Calculate the corresponding vector within Trimer DEF
                vec_D_rel = com_D_ref - com_DEF_ref # (n_batches, 3)

                # Directly compute the rotation matrix aligning vec_A_rel to vec_D_rel
                R_I_sub_C3_batch = compute_rotation_matrix_from_vectors(
                    vec_A_rel,
                    vec_D_rel,
                    device=device,
                    dtype=coords_ref.dtype
                ) # Returns (n_batches, 3, 3)
                print(f"DEBUG: Calculated inter-trimer rotation R_I_sub_C3 directly. Shape: {R_I_sub_C3_batch.shape}")

                # --- NO AVERAGING ---
                # Select the rotation for the first batch item -> (3, 3) for stacking
                if n_batches != 1:
                     print(f"WARNING: Batch size is {n_batches}, but extracting only the first rotation for self.rot_mats_noI used in noise/denoising.")
                R_B = R_B_batch[0] # Extract (3, 3) from (n_batches, 3, 3)
                R_C = R_C_batch[0] # Extract (3, 3) from (n_batches, 3, 3)
                R_I_sub_C3 = R_I_sub_C3_batch[0] # Extract (3, 3) from (n_batches, 3, 3) - Now directly calculated A->D rotation

                # --- Combine rotations for A->E, A->F (using compositions) ---
                # While R_I_sub_C3 is now direct A->D, keep the composition logic
                # for A->E and A->F, assuming the relationships hold.
                # R_E = R_I_sub_C3 @ R_B should now more accurately reflect A->E
                # R_F = R_I_sub_C3 @ R_C should now more accurately reflect A->F
                R_E = R_I_sub_C3 @ R_B
                R_F = R_I_sub_C3 @ R_C

                effective_rots = [
                    R_B,        # A -> B (Direct intra)
                    R_C,        # A -> C (Direct intra)
                    R_I_sub_C3, # A -> D (Direct inter via A_rel -> D_rel)
                    R_E,        # A -> E (Composition using direct R_I_sub_C3)
                    R_F         # A -> F (Composition using direct R_I_sub_C3)
                ]
                self.rot_mats_noI = torch.stack(effective_rots, dim=0).to(device, coords_ref.dtype) # Shape [5, 3, 3]
                print(f"DEBUG: Calculated effective rotations for noise/denoising (Direct I/C3). Shape: {self.rot_mats_noI.shape}")

        elif self.symmetry_type and n_subunits_found > 1:
             print(f"DEBUG: Using precomputed {self.symmetry_type} rotations from __init__ for noise/denoising.")
             if self.rot_mats_noI is None:
                  print(f"Warning: Symmetry {self.symmetry_type} requested, but self.rot_mats_noI is None (from __init__). Symmetrical noise/denoising might not work correctly.")
             else:
                  self.rot_mats_noI = self.rot_mats_noI.to(device, coords_ref.dtype)
                  print(f"DEBUG: Using precomputed rot_mats_noI. Shape: {self.rot_mats_noI.shape}")
        else: # Monomer or no symmetry specified
            print("DEBUG: No symmetry or monomer case. No rotations needed for noise/denoising.")
            self.rot_mats_noI = None # Explicitly set to None
        # ======= END Rotation Calculation =======


        # ------- START DIAGNOSTIC: Calculate Desired COMs (Now just print targets) -------
        print("\n" + "="*20 + " DIAGNOSTIC: Desired COMs (Are the Input COMs) " + "="*20)
        if target_coms_all is not None:
            desired_coms_diag = {}
            num_subunits_in_target = target_coms_all.shape[1]

            if b_idx_diag < n_batches: # Print diagnostics only for the first batch element
                for i_sub in range(num_subunits_in_target):
                    if n_subunits_found > 1 and num_subunits_in_target == n_subunits_found :
                        sub_name = chr(ord('A') + i_sub)
                    elif num_subunits_in_target == 1:
                        sub_name = 'Global'
                    else:
                        sub_name = f"Subunit_{i_sub}"

                    com_desired_sub_diag = target_coms_all[b_idx_diag, i_sub, :].cpu().numpy()
                    desired_coms_diag[sub_name] = com_desired_sub_diag
                    print(f"Batch {b_idx_diag} - Desired COM {sub_name}: {com_desired_sub_diag}")

                if 'Trimer1' in initial_coms:
                    print(f"Batch {b_idx_diag} - Desired COM Trimer1 (ABC): {initial_coms['Trimer1']} (matches initial)")
                if 'Trimer2' in initial_coms:
                    print(f"Batch {b_idx_diag} - Desired COM Trimer2 (DEF): {initial_coms['Trimer2']} (matches initial)")

                print(f"\nBatch {b_idx_diag} - COM Comparison (Initial vs Desired):")
                for key in initial_coms:
                    initial_np = initial_coms[key]
                    lookup_key = key
                    if key == 'Global' and 'Global' not in desired_coms_diag and num_subunits_in_target==1:
                         lookup_key = 'Subunit_0'
                    desired_np = desired_coms_diag.get(lookup_key)

                    if desired_np is not None:
                         diff = np.linalg.norm(initial_np - desired_np)
                         print(f"  COM {key}: Initial={initial_np}, Desired={desired_np}, Diff Norm={diff:.4f}")
                    elif key in ['Trimer1', 'Trimer2']:
                         pass
                    else:
                         print(f"  COM {key}: Initial={initial_np}, Desired= N/A (Key '{lookup_key}' not found in desired_coms_diag)")
            else:
                 print(f"Cannot print diagnostics for batch {b_idx_diag} (out of range for n_batches={n_batches}).")

        else:
            print(f"Batch {b_idx_diag} - Cannot print Desired COMs (target_coms_all is None).")

        print("="*60 + "\n")
        # ------- END DIAGNOSTIC: Desired COMs -------


        # 2. Build Diffusion Schedule
        sigmas = self.sample_schedule(num_sampling_steps)

        # 3. Determine Start Point (Forward Diffusion part)
        start_index = max(0, num_sampling_steps - forward_diffusion_steps)
        start_sigma = sigmas[start_index]
        t_hat_initial = start_sigma
        sigma_tm_val_initial = sigmas[start_index + 1].item() if start_index + 1 < len(sigmas) else 0.0

        # 4. Apply Initial Symmetrical Noise to Current Coords
        # Uses self.rot_mats_noI (calculated above) if symmetry requires it
        initial_sym_noise = self._symmetrical_noise(
            current_coords, feats, subunits, self.rot_mats_noI, t_hat_initial, sigma_tm_val_initial
        )
        current_coords = current_coords + initial_sym_noise
        print(f"DEBUG: Applied initial symmetrical noise. Start sigma={start_sigma:.4f}")


        # 5. Apply Initial Rigid Constraints using **ALL TARGET COMS**
        if target_coms_all is None:
            raise RuntimeError("Target COMs tensor is None before applying initial constraints.")

        current_coords = self.apply_symmetry_constraints_rigid(
             coords=current_coords,
             subunits=subunits,
             desired_coms_all_subunits=target_coms_all
        )
        print(f"DEBUG: Applied initial rigid constraints using target COMs.")

        # 6. Build Reverse Schedule (Karras/EDM style)
        reverse_schedule = []
        for i in range(start_index, len(sigmas) - 1):
            reverse_schedule.append((sigmas[i], sigmas[i+1])) # (sigma_cur, sigma_next)

        # 7. Prepare for Denoising Loop
        token_repr = None
        model_cache = {}

        # --- 8. Reverse Diffusion Loop ---
        print(f"DEBUG: Starting reverse diffusion loop for {len(reverse_schedule)} steps.")
        for step_i, (sigma_cur, sigma_next) in enumerate(reverse_schedule):
            sigma_cur_val = sigma_cur.item()
            sigma_next_val = sigma_next.item()

            # --- EDM Sampler Step (Heun's 2nd Order) ---
            # a) Denoise at current sigma (uses self.rot_mats_noI for update symmetrization)
            denoised_coords_pred, token_a = self.preconditioned_network_forward_symmetry(
                coords_noisy=current_coords,
                sigma=sigma_cur_val,
                network_condition_kwargs=dict(
                    multiplicity=multiplicity, # Pass multiplicity here
                    model_cache=model_cache if self.use_inference_model_cache else None,
                     **network_condition_kwargs
                ),
                training=False,
            )

            # b) Apply Rigid Constraint to Denoised Prediction using **ALL TARGET COMS**
            denoised_coords_pred = self.apply_symmetry_constraints_rigid(
                coords=denoised_coords_pred,
                subunits=subunits,
                desired_coms_all_subunits=target_coms_all
            )

            # c) Calculate derivative (d) at sigma_cur
            d = (current_coords - denoised_coords_pred) / max(sigma_cur_val, 1e-9) # Avoid division by zero
            dt = sigma_next_val - sigma_cur_val # Change in sigma (negative)

            # d) Euler step to intermediate point
            coords_intermediate = current_coords + d * dt

            # e) Apply Rigid Constraint to Intermediate Point using **ALL TARGET COMS**
            coords_intermediate = self.apply_symmetry_constraints_rigid(
                 coords=coords_intermediate,
                 subunits=subunits,
                 desired_coms_all_subunits=target_coms_all
            )

            # f) Denoise at intermediate sigma (sigma_next) if not last step
            if sigma_next_val > 1e-5 : # Check if sigma_next is not effectively zero
                 denoised_coords_prime, _ = self.preconditioned_network_forward_symmetry(
                     coords_noisy=coords_intermediate,
                     sigma=sigma_next_val,
                     network_condition_kwargs=dict(
                         multiplicity=multiplicity, # Pass multiplicity here
                         model_cache=model_cache if self.use_inference_model_cache else None,
                         **network_condition_kwargs
                     ),
                     training=False,
                 )

                 # g) Apply Rigid Constraint to Second Denoised Prediction using **ALL TARGET COMS**
                 denoised_coords_prime = self.apply_symmetry_constraints_rigid(
                    coords=denoised_coords_prime,
                    subunits=subunits,
                    desired_coms_all_subunits=target_coms_all
                 )

                 # h) Calculate derivative (d_prime) at sigma_next
                 d_prime = (coords_intermediate - denoised_coords_prime) / max(sigma_next_val, 1e-9) # Avoid division by zero

                 # i) Heun's second-order update
                 current_coords = current_coords + 0.5 * (d + d_prime) * dt

            else: # Last step or sigma_next is zero, use Euler update
                 current_coords = coords_intermediate

            # j) Apply Final Rigid Constraint for the step using **ALL TARGET COMS**
            current_coords = self.apply_symmetry_constraints_rigid(
                coords=current_coords,
                subunits=subunits,
                desired_coms_all_subunits=target_coms_all
            )
            # --- End Sampler Step ---


            # Token accumulation logic
            if self.accumulate_token_repr:
                if token_repr is None:
                    token_repr = torch.zeros_like(token_a)
                with torch.set_grad_enabled(train_accumulate_token_repr):
                    # Use sigma_cur for time conditioning as it corresponds to the input 'current_coords'
                    t_tensor = torch.full((current_coords.shape[0],), sigma_cur_val, device=device, dtype=coords_ref.dtype)
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(t_tensor),
                        acc_a=token_repr,
                        next_a=token_a, # Use token_a from the *first* prediction in the step
                    )
            elif step_i == len(reverse_schedule) - 1: # Get final token repr if not accumulating
                # If the last step used Euler, need to run denoiser one last time to get token_a corresponding to final state
                if sigma_next_val <= 1e-5:
                    _, token_a = self.preconditioned_network_forward_symmetry(
                        coords_noisy=current_coords, # Use the final constrained coords
                        sigma=max(sigma_next_val, 1e-5), # Use a tiny sigma instead of 0 for stability if needed
                        network_condition_kwargs=dict(
                            multiplicity=multiplicity, # Pass multiplicity here
                            model_cache=None,
                            **network_condition_kwargs),
                        training=False
                    )
                # Otherwise, token_a from the first prediction of the Heun step is the last one calculated
                token_repr = token_a


        # 9. Final Output
        print(f"DEBUG: Reverse diffusion finished.")
        # Final constraint application just to be safe (should be minimal change now)
        final_coords = self.apply_symmetry_constraints_rigid(
             coords=current_coords,
             subunits=subunits,
             desired_coms_all_subunits=target_coms_all
        )

        return {"sample_atom_coords": final_coords, "diff_token_repr": token_repr}








    def coords_to_pdb(self, coords: torch.Tensor, feats: dict, step: int) -> str:
        """Converts coordinates and features to a PDB string, focusing ONLY on coordinates."""
        batch_idx = 0  # Assuming single batch for simplicity
        coords = coords[batch_idx].cpu().numpy()  # (A, 3)
        atom_mask = feats["atom_pad_mask"][batch_idx].cpu().numpy()  # (A,)

        pdb_lines = []
        atom_idx = 1

        for i in range(coords.shape[0]):
            if atom_mask[i] == 0:
                continue  # Skip masked atoms

            x, y, z = coords[i]

            # Basic PDB line, using "CA" as a placeholder.  You can change this.
            pdb_lines.append(
                f"ATOM  {atom_idx:5d}  CA  MOL     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
            )
            atom_idx += 1

        return "".join(pdb_lines)



    # -----------------
    # symmetrical noise
    # -----------------
    def _symmetrical_noise(
        self,
        coords: torch.Tensor,
        feats: dict,
        subunits: list[torch.Tensor],
        rot_mats: torch.Tensor, # Should be self.rot_mats_noI passed from sample
        t_hat: float,
        sigma_tm_val: float # Represents the sigma level at the *end* of the step for variance calculation
    ) -> torch.Tensor:
        """
        Generate symmetrical noise about the origin using self.rot_mats_noI.
        """
        B, A, _ = coords.shape
        device = coords.device
        dtype = coords.dtype

        # Calculate the standard deviation for the noise to add in this step
        variance_diff = max(t_hat**2 - sigma_tm_val**2, 0.0) # Ensure non-negative
        std_dev_diff = math.sqrt(variance_diff)
        scale_ = self.noise_scale * std_dev_diff

        # If no symmetry/rotations defined, or only one subunit, apply standard Gaussian noise
        if (not self.symmetry_type) or not subunits or len(subunits) < 2 or rot_mats is None:
            return scale_ * torch.randn_like(coords)

        # Ensure rot_mats is on the correct device and dtype
        rot_mats = rot_mats.to(device, dtype)

        # Proceed with symmetrical noise generation
        mapping = self.get_symmetrical_atom_mapping(feats)
        eps = torch.zeros_like(coords) # Initialize noise tensor

        for b_idx in range(B):
            if not mapping: continue # Check if mapping exists

            for local_ref_idx, all_atom_global_indices in mapping.items():
                if not all_atom_global_indices: continue

                ref_atom_global_idx = all_atom_global_indices[0]
                if ref_atom_global_idx >= A: continue # Safety check

                # Sample a random noise vector for the reference atom
                v = scale_ * torch.randn(3, device=device, dtype=dtype)
                eps[b_idx, ref_atom_global_idx, :] = v

                # Rotate and assign noise to corresponding atoms in other subunits
                for s_idx in range(1, len(all_atom_global_indices)):
                    target_atom_global_idx = all_atom_global_indices[s_idx]
                    rot_idx = s_idx - 1 # Index for B, C, D, E, F...

                    # Safety checks
                    if target_atom_global_idx >= A or rot_idx >= rot_mats.shape[0]:
                        continue

                    R = rot_mats[rot_idx] # Get the rotation matrix (3, 3)

                    # Rotate the reference noise vector: v' = R @ v
                    v_rot = v @ R.T # Equivalent to R @ v

                    eps[b_idx, target_atom_global_idx, :] = v_rot
        return eps


    # ------------------------------------------------------
    # Hard-Constraint: re-rotate each subunit => R_i * reference
    # ------------------------------------------------------
    # ------------------------------------------------------
    # Hard-Constraint: re-rotate each subunit => R_i * reference
    # ------------------------------------------------------
# ------------------------------------------------------
    # Hard-Constraint: Translate each subunit COM to its target COM
    # ------------------------------------------------------
    def apply_symmetry_constraints_rigid(
        self,
        coords: torch.Tensor,      # (B, A, 3) Current coordinates to be constrained
        subunits: list[torch.Tensor], # List of index tensors per subunit (A, B, C, D, E, F...)
        desired_coms_all_subunits: torch.Tensor # Target COMs for *all* subunits - Shape (B, N_subunits, 3)
                                                # Derived directly from the input reference structure COMs.
        # rot_mats is NO LONGER USED for rigid placement constraint
    ) -> torch.Tensor:
        """
        Applies rigid body symmetry constraints by translating EACH subunit
        so its Center of Mass (COM) matches its corresponding target COM derived
        from the input reference structure.

        Args:
            coords: Current coordinates (B, A, 3).
            subunits: List of tensors, each holding atom indices for a subunit (A, B, C...).
            desired_coms_all_subunits: Target COM for EACH subunit (B, N_subunits, 3),
                                       ordered consistently with 'subunits'.

        Returns:
            Constrained coordinates (B, A, 3).
        """
        device = coords.device
        dtype = coords.dtype
        B = coords.shape[0]

        # Handle cases with no subunits or mismatched desired COMs tensor
        if not subunits:
             print("Warning in apply_symmetry_constraints_rigid: No subunits provided. Returning unconstrained coords.")
             # Optional: Implement global centering if desired as a fallback
             # com_current = calculate_com(coords).unsqueeze(1)
             # desired_com_global = torch.zeros_like(com_current) # Center at origin example
             # shift = desired_com_global - com_current
             # return coords + shift
             return coords

        n_subunits_found = len(subunits)
        if desired_coms_all_subunits is None or desired_coms_all_subunits.shape[1] != n_subunits_found:
            raise ValueError(f"Mismatch or missing desired_coms_all_subunits. Expected shape (B, {n_subunits_found}, 3), "
                             f"got {desired_coms_all_subunits.shape if desired_coms_all_subunits is not None else 'None'}")

        # Ensure desired_coms tensor has correct device/dtype
        desired_coms_all_subunits = desired_coms_all_subunits.to(device, dtype)

        out_coords = coords.clone()

        # Iterate through each subunit (A, B, C, ...)
        for i_sub in range(n_subunits_found):
            sub_inds = subunits[i_sub]
            if sub_inds.numel() == 0:
                # print(f"Debug: Skipping empty subunit {i_sub} in constraint.")
                continue # Skip empty subunits

            # Get the target COM for this specific subunit
            target_com_i = desired_coms_all_subunits[:, i_sub, :] # Shape (B, 3)

            # Calculate current COM of this subunit
            # Ensure calculate_com handles potential empty subunits gracefully if mask isn't perfect,
            # though the numel check above should prevent issues here.
            current_com_i = calculate_com(out_coords[:, sub_inds, :]) # Shape (B, 3)

            # Calculate the shift needed for this subunit: Target COM - Current COM
            shift_i = target_com_i - current_com_i # Shape (B, 3)

            # Apply the shift to all atoms of this subunit
            # Expand shift for broadcasting: (B, 1, 3)
            out_coords[:, sub_inds, :] = out_coords[:, sub_inds, :] + shift_i.unsqueeze(1)

        # The rotation-based reconstruction part is now entirely removed.
        # Each subunit is independently translated to its target COM.

        return out_coords






    # ------------------------------------------------------
    # symmetrical atom mapping
    # ------------------------------------------------------
    def get_symmetrical_atom_mapping(self, feats: dict[str, torch.Tensor]) -> dict[int, list[int]]:
        """
        Generates a mapping from reference subunit local atom index to global atom indices
        across all symmetrical subunits.
        e.g. {0: [atom0_sub0, atom0_sub1, ...], 1: [atom1_sub0, atom1_sub1, ...], ...}

        Relies on get_subunit_atom_indices returning subunits in the correct order
        (e.g., A, B, C, D, E, F for 'I') and assumes a 1-to-1 correspondence
        in the number and order of atoms between symmetrical subunits.
        """
        device = self.device # Use the module's device property
        # Get subunits based on current type and features
        subunits = symmetry.get_subunit_atom_indices(
            self.symmetry_type,
            self.chain_symmetry_groups,
            feats,
            device,
        )

        # Handle cases with no subunits or only one subunit
        if not subunits:
            return {}
        if len(subunits) == 1:
            ref_sub_indices = subunits[0].tolist()
            # Map each global index to itself in a list
            return {global_idx: [global_idx] for global_idx in ref_sub_indices}

        # Reference subunit is the first one (e.g., A)
        ref_sub_indices = subunits[0]
        n_atoms_ref = len(ref_sub_indices)

        # Initialize mapping: key is the *local* index within the reference subunit (0 to n_atoms_ref-1)
        # Value is a list starting with the global index of that atom in the reference subunit.
        mapping = {i: [ref_sub_indices[i].item()] for i in range(n_atoms_ref)}

        # Add corresponding atoms from other subunits
        for s_idx in range(1, len(subunits)):
            current_sub_indices = subunits[s_idx]
            n_atoms_current = len(current_sub_indices)

            # Basic check: ensure the symmetrical subunit has the same number of atoms
            if n_atoms_current != n_atoms_ref:
                print(f"Warning in get_symmetrical_atom_mapping: Subunit {s_idx} has {n_atoms_current} atoms, "
                      f"but reference subunit 0 has {n_atoms_ref} atoms. Skipping this subunit for mapping.")
                # If counts don't match, we cannot guarantee a 1-to-1 mapping,
                # so we don't add this subunit's atoms to the mapping.
                # Alternatively, could try to establish partial mapping if needed.
                continue # Skip to the next subunit

            # Assuming atom order corresponds 1-to-1
            for local_idx in range(n_atoms_ref):
                # Append the global index of the corresponding atom from the current subunit
                mapping[local_idx].append(current_sub_indices[local_idx].item())

        return mapping


def plot_COM_distance_line(iterations: list[int], pre_denoising: list[float], post_denoising: list[float], filename: str, title: str = "Chain 0 COM Distance"):
    plt.figure(figsize=(8,6))
    plt.plot(iterations, pre_denoising, marker='o', color='blue', label='Before Denoising')
    plt.plot(iterations, post_denoising, marker='o', color='red', label='After Denoising')
    plt.xlabel("Iteration")
    plt.ylabel("Distance from Origin")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def compute_rotation_matrix_from_vectors(vec1, vec2):
    """
    Compute the rotation matrix that rotates vec1 to vec2.
    Both vec1 and vec2 should be 1D tensors of shape (3,).
    """
    # Normalize the input vectors.
    a = vec1 / torch.norm(vec1)
    b = vec2 / torch.norm(vec2)
    # Compute the cross product and sine of the angle.
    v = torch.cross(a, b)
    s = torch.norm(v)
    # Compute the cosine of the angle.
    c = torch.dot(a, b)
    # If the vectors are already aligned, return the identity.
    if s < 1e-6:
        return torch.eye(3, device=vec1.device, dtype=vec1.dtype)
    # Skew-symmetric cross-product matrix of v.
    vx = torch.tensor([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]], device=vec1.device, dtype=vec1.dtype)
    # Rodrigues' rotation formula.
    R = torch.eye(3, device=vec1.device, dtype=vec1.dtype) + vx + vx @ vx * ((1 - c) / (s**2))
    return R



# ================================
# Helper: rotate about origin
# ================================
def rotate_coords_about_origin(coords: torch.Tensor, rotation_mat: torch.Tensor):
    """
    Single-step rotation about the origin => p' = R * p

    coords: (N, 3)
    rotation_mat: (3, 3)
    Returns: (N, 3)
    """
    return coords @ rotation_mat.T


def calculate_com(coords: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculates the Center of Mass (COM).

    Args:
        coords (torch.Tensor): Coordinates tensor of shape (B, N, 3) or (N, 3).
        mask (torch.Tensor, optional): Boolean mask of shape (B, N) or (N,)
                                        indicating which atoms to include.
                                        Defaults to None (include all).

    Returns:
        torch.Tensor: COM coordinates tensor of shape (B, 3) or (3,).
    """
    if coords.ndim == 2:
        coords = coords.unsqueeze(0) # Add batch dimension if missing
        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)

    B, N, _ = coords.shape

    if mask is None:
        # If no mask, include all atoms
        com = torch.mean(coords, dim=1) # Shape: (B, 3)
    else:
        # Ensure mask has the same batch size
        if mask.shape[0] != B:
             if mask.shape[0] == 1 :
                 mask = mask.expand(B, -1)
             else:
                raise ValueError(f"Mask batch size {mask.shape[0]} doesn't match coords batch size {B}")
        
        # Apply mask: coords * mask -> set masked coords to zero
        # Need mask shape (B, N, 1) for broadcasting
        mask_expanded = mask.unsqueeze(-1).float()
        masked_coords = coords * mask_expanded

        # Sum coordinates and count atoms per batch element
        sum_coords = torch.sum(masked_coords, dim=1) # Shape: (B, 3)
        num_atoms = torch.sum(mask_expanded, dim=1)   # Shape: (B, 1)

        # Avoid division by zero if a subunit has no atoms in the mask
        num_atoms = torch.clamp(num_atoms, min=1e-6)

        com = sum_coords / num_atoms # Shape: (B, 3)

    # Squeeze batch dim if it was added
    if coords.ndim == 2:
         com = com.squeeze(0)

    return com


def compute_rotation_matrix_from_vectors(vec1, vec2, device='cpu', dtype=torch.float32):
    """
    Compute the rotation matrix that rotates vec1 to vec2. Handles batches.
    vec1, vec2: Tensors of shape (B, 3).
    Returns: Tensor of shape (B, 3, 3).
    """
    B = vec1.shape[0]
    a = F.normalize(vec1, p=2, dim=1)
    b = F.normalize(vec2, p=2, dim=1)

    v = torch.cross(a, b, dim=1)
    s = torch.linalg.norm(v, dim=1) # Sine of the angle (B,)
    c = torch.sum(a * b, dim=1)    # Cosine of the angle (B,)

    # Handle cases where vectors are nearly parallel or anti-parallel
    identity = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    # parallel case (a == b)
    parallel_mask = s < 1e-6
    # anti-parallel case (a == -b) -> rotate 180 deg around an arbitrary axis perp to a
    antiparallel_mask = (s < 1e-6) & (c < -0.99999)

    # Skew-symmetric cross-product matrix [v]x
    vx = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    vx[:, 0, 1] = -v[:, 2]
    vx[:, 0, 2] = v[:, 1]
    vx[:, 1, 0] = v[:, 2]
    vx[:, 1, 2] = -v[:, 0]
    vx[:, 2, 0] = -v[:, 1]
    vx[:, 2, 1] = v[:, 0]

    # Rodrigues' rotation formula component: (1 - c) / s^2
    # Avoid division by zero when s is small
    s_squared = s**2
    term_factor = torch.where(parallel_mask, torch.zeros_like(c), (1 - c) / torch.clamp(s_squared, min=1e-12))

    R = identity + vx + torch.bmm(vx, vx) * term_factor.unsqueeze(-1).unsqueeze(-1)

    # Handle anti-parallel case: find an axis perp to 'a' and rotate 180 deg
    # Find a vector not parallel to 'a'
    not_parallel = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).expand(B,-1)
    alt_vec = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).unsqueeze(0).expand(B,-1)
    # Check if [1,0,0] is parallel to 'a'
    dot_prod = torch.abs(torch.sum(a * not_parallel, dim=1))
    axis_candidate = torch.where(dot_prod.unsqueeze(-1) > 0.999, alt_vec, not_parallel)
    # Axis is cross product of 'a' and the non-parallel vector
    rot_axis = F.normalize(torch.cross(a, axis_candidate, dim=1), p=2, dim=1)
    # 180 degree rotation matrix: 2 * axis * axis^T - I
    axis_outer = torch.einsum('bi,bj->bij', rot_axis, rot_axis)
    R_180 = 2 * axis_outer - identity

    # Apply corrections
    R = torch.where(parallel_mask.unsqueeze(-1).unsqueeze(-1), identity, R)
    R = torch.where(antiparallel_mask.unsqueeze(-1).unsqueeze(-1), R_180, R)


    return R


def find_best_rotation_point_cloud(target_point: torch.Tensor, # (B, 3)
                                   candidate_rots: torch.Tensor, # (N_rots, 3, 3)
                                   ref_point: torch.Tensor # (B, 3)
                                   ) -> torch.Tensor:
    """
    Finds the best rotation matrix from candidates that maps ref_point closest to target_point.

    Args:
        target_point: Target coordinates (B, 3).
        candidate_rots: Candidate rotation matrices (N_rots, 3, 3).
        ref_point: Reference coordinates to be rotated (B, 3).

    Returns:
        torch.Tensor: The best rotation matrix for each batch element (B, 3, 3).
    """
    B = target_point.shape[0]
    N_rots = candidate_rots.shape[0]
    device = target_point.device
    dtype = target_point.dtype

    # Expand dims for broadcasting:
    # target: (B, 1, 3)
    # ref:    (B, 1, 3)
    # rots:   (1, N_rots, 3, 3)
    target_exp = target_point.unsqueeze(1)
    ref_exp = ref_point.unsqueeze(1)
    rots_exp = candidate_rots.unsqueeze(0).to(device, dtype)

    # Apply all rotations to the reference point: R @ ref.T -> (B, N_rots, 3, 3) @ (B, 1, 3, 1) -> needs einsum
    # rotated_ref = torch.einsum('rji,bzi->brj', rots_exp.squeeze(0), ref_exp) # Incorrect shape handling
    # R is (N, 3, 3), ref is (B, 3). Output should be (B, N, 3)
    rotated_ref = torch.einsum('nij,bj->bni', candidate_rots.to(device, dtype), ref_point) # Shape: (B, N_rots, 3)

    # Calculate squared distances: || target - rotated_ref ||^2
    distances_sq = torch.sum((target_exp - rotated_ref)**2, dim=2) # Shape: (B, N_rots)

    # Find the index of the minimum distance for each batch element
    best_rot_indices = torch.argmin(distances_sq, dim=1) # Shape: (B,)

    # Select the best rotation matrix for each batch element
    best_rots = candidate_rots[best_rot_indices].to(device, dtype) # Shape: (B, 3, 3)

    return best_rots


