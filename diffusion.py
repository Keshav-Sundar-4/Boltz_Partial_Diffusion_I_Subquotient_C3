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
        B = coords_noisy.shape[0]
        device = coords_noisy.device

        if isinstance(sigma, float):
            sigma = torch.full((B,), float(sigma), device=device)
        elif sigma.ndim == 0:
            sigma = sigma.reshape(1).expand(B)

        c_in_val = self.c_in(sigma)[:, None, None]
        times_val = self.c_noise(sigma)

        # Filter out 'input_coords' from network_condition_kwargs
        filtered_kwargs = {k: v for k, v in network_condition_kwargs.items() if k != 'input_coords'}

        net_out = self.score_model(
            r_noisy=c_in_val * coords_noisy,
            times=times_val,
            **filtered_kwargs,  # Use filtered kwargs here
        )
        r_update = net_out["r_update"]  # (B, A, 3)
        token_a = net_out["token_a"]

        # Symmetrical denoising (rest of the function remains unchanged)
        if self.symmetry_type:
            subunits = symmetry.get_subunit_atom_indices(
                self.symmetry_type,
                self.chain_symmetry_groups,
                network_condition_kwargs["feats"],
                device,
            )
            if len(subunits) > 1:
                # Use the precomputed, reordered rotations on the proper device.
                rot_mats = self.rot_mats_noI.to(device)
                mapping = self.get_symmetrical_atom_mapping(network_condition_kwargs["feats"])

                for b_idx in range(B):
                    for local_ref_idx, all_atoms in mapping.items():
                        ref_atom_id = all_atoms[0]
                        ref_shift = r_update[b_idx, ref_atom_id, :]  # (3,)
                        # For each neighbor chain, assign a fixed rotation:
                        for s_idx in range(1, len(all_atoms)):
                            target_atom_id = all_atoms[s_idx]
                            R = rot_mats[s_idx - 1].to(
                                device
                            )  # Fixed mapping: neighbor i gets rot_mats[i-1]
                            rotated_shift = rotate_coords_about_origin(ref_shift.unsqueeze(0), R)[0]
                            r_update[b_idx, target_atom_id, :] = rotated_shift

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
        forward_diffusion_steps=100,
        multiplicity=1,
        train_accumulate_token_repr=False,
        output_dir: str = ".", # Add output_dir for saving temp file
        **network_condition_kwargs,
    ):
        """
        Performs reverse diffusion sampling with rigid symmetry enforcement.
        MODIFIED for Icosahedral ('I'): Uses the IDEAL A->D rotation derived
        by matching input subunits A and D to a generated template based on input A.
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        n_batches = atom_mask.shape[0] # Total batches *after* multiplicity
        n_atoms = atom_mask.shape[1]
        device = self.device
        dtype = next(self.score_model.parameters()).dtype # Get dtype from model
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
                input_coords_ref = input_coords_ref.reshape(B_ref, N_*L_, 3)
        elif input_coords_ref.ndim == 3: # Assume shape (B, A, 3)
                B_ref = input_coords_ref.shape[0]
        else:
            raise ValueError(f"Unexpected input_coords dimension: {input_coords_ref.ndim}. Expected 3 or 4.")

        if B_ref * multiplicity != n_batches:
                raise ValueError(f"Input coords batch size {B_ref} * multiplicity {multiplicity} != atom_mask total batch size {n_batches}.")

        input_coords_ref = input_coords_ref.repeat_interleave(multiplicity, 0)
        try:
            input_coords_ref = input_coords_ref.view(n_batches, n_atoms, 3)
        except RuntimeError as e:
            raise ValueError(f"Input coordinates shape {input_coords_ref.shape} is incompatible with expected ({n_batches}, {n_atoms}, 3) after multiplicity. Error: {e}")

        coords_ref = input_coords_ref.clone().to(device, dtype) # Use this for deriving constraints
        network_condition_kwargs["feats"]["coords"] = coords_ref # Update feats with the reference coords
        feats = network_condition_kwargs["feats"] # Update local feats reference

        # 2. Get Atom Indices for Each Subunit
        subunits = symmetry.get_subunit_atom_indices(
            self.symmetry_type, self.chain_symmetry_groups, feats, device
        )
        n_subunits_found = len(subunits)
        print(f"DEBUG: Found {n_subunits_found} subunits based on symmetry type '{self.symmetry_type}' and chain groups.")

        # 3. Calculate TARGET COMs from INPUT Reference (coords_ref)
        target_coms_list = []
        if subunits:
                for i_sub, sub_inds in enumerate(subunits):
                    if sub_inds.numel() > 0:
                            com_ref_sub = calculate_com(coords_ref[:, sub_inds, :])
                            target_coms_list.append(com_ref_sub)
                    else:
                        print(f"Warning: Subunit {i_sub} has 0 atoms.")
                        target_coms_list.append(torch.zeros((n_batches, 3), device=device, dtype=dtype)) # Placeholder COM

                if not target_coms_list or all(sub_inds.numel()==0 for sub_inds in subunits):
                        print("Warning: All subunits are empty or no subunits found. Calculating global COM as target.")
                        com_global_ref = calculate_com(coords_ref) # Use overall COM
                        target_coms_all = com_global_ref.unsqueeze(1).expand(-1, n_subunits_found if n_subunits_found > 0 else 1, -1) # Expand to expected shape
                        if n_subunits_found == 0: # Handle case where get_subunit_atom_indices returned empty list
                            subunits = [torch.arange(n_atoms, device=device)] # Treat as one subunit
                            n_subunits_found = 1
                else:
                        target_coms_all = torch.stack(target_coms_list, dim=1) # Shape: (n_batches, N_subunits, 3)
        else: # No subunits found (e.g., monomer or incorrect setup)
                print("DEBUG: No subunits found by get_subunit_atom_indices. Treating as monomer.")
                com_global_ref = calculate_com(coords_ref)
                target_coms_all = com_global_ref.unsqueeze(1)
                subunits = [torch.arange(n_atoms, device=device)]
                n_subunits_found = 1
        print(f"DEBUG: Calculated TARGET COMs from INPUT reference. Shape: {target_coms_all.shape}")


        # ======= 4. Calculate Rotations for Symmetrical Noise/Denoising =======
        self.rot_mats_noI = None # Reset instance variable

        if self.symmetry_type == 'I' and n_subunits_found >= 6:
            print("DEBUG: Setting up IDEAL I/C3 rotations based on matching input A/D to template...")

            # --- 4a. Extract Input A and D Coordinates ---
            # Assume subunit order is A, B', C', D, E, F... returned by get_subunit_atom_indices
            if not subunits or subunits[0].numel() == 0:
                raise ValueError("Cannot find reference chain A indices in input for Icosahedral symmetry setup.")
            if n_subunits_found < 4 or subunits[3].numel() == 0:
                 raise ValueError(f"Cannot find chain D (index 3) indices or not enough subunits ({n_subunits_found}) found for Icosahedral A->D mapping.")

            # Use batch 0's A and D as representative for finding the ideal rotation map
            ref_chain_A_indices = subunits[0]
            ref_chain_D_indices = subunits[3] # Assuming D is the 4th subunit (index 3)
            coords_chain_A_input = coords_ref[0, ref_chain_A_indices, :] # (N_atoms_A, 3)
            coords_chain_D_input = coords_ref[0, ref_chain_D_indices, :] # (N_atoms_D, 3)

            # --- 4b. Call the function to get the IDEAL A->D rotation ---
            # This function also saves the temp CIF files internally
            R_AD_ideal = self._generate_ideal_icosahedron_and_find_ad_map(
                coords_chain_A = coords_chain_A_input,
                coords_chain_D = coords_chain_D_input,
                device = device,
                dtype = dtype, # Request final rotation matrix in model's dtype
                output_dir = output_dir
            ) # Returns the IDEAL (3, 3) A->D rotation matrix

            # --- 4c. Calculate other required rotations using the IDEAL A->D map ---
            # We need the rotations from reference A to B', C', D, E, F...
            # A->B' is +120 deg Z rot, A->C' is +240 deg Z rot
            rot_120_np = _build_rot_matrix_np(2 * math.pi / 3, 'z')
            rot_240_np = _build_rot_matrix_np(4 * math.pi / 3, 'z')
            R_B = torch.from_numpy(rot_120_np).to(device, dtype) # Ideal A->B'
            R_C = torch.from_numpy(rot_240_np).to(device, dtype) # Ideal A->C'
            R_D = R_AD_ideal # This IS the IDEAL A->D rotation we just found
            R_E = torch.matmul(R_AD_ideal, R_B) # Ideal A -> E = (A->D)_ideal @ (A->B')_ideal
            R_F = torch.matmul(R_AD_ideal, R_C) # Ideal A -> F = (A->D)_ideal @ (A->C')_ideal

            # --- 4d. Assemble the rotation matrices for noise/denoising ---
            # Order matters: corresponds to subunits[1] to subunits[N-1] relative to subunits[0].
            # Assuming subunit order is A, B', C', D, E, F...
            effective_rots = [ R_B, R_C, R_D, R_E, R_F ] # Rotations for B',C',D,E,F relative to A

            # Extend this list if more than 6 subunits are defined by get_subunit_atom_indices
            expected_rots = n_subunits_found - 1
            if len(effective_rots) < expected_rots:
                 print(f"[Warning] Only defined rotations for first {len(effective_rots)+1} subunits (A-F...), but found {n_subunits_found}. Noise/denoising symmetrization might be incomplete for later subunits.")
                 # Add identity matrices for the remaining expected subunits
                 num_missing_rots = expected_rots - len(effective_rots)
                 print(f"[Warning] Appending {num_missing_rots} identity matrices.")
                 identity_mat = torch.eye(3, device=device, dtype=dtype)
                 effective_rots.extend([identity_mat] * num_missing_rots)
            elif len(effective_rots) > expected_rots:
                 print(f"[Warning] Defined more rotations ({len(effective_rots)}) than needed ({expected_rots}). Truncating.")
                 effective_rots = effective_rots[:expected_rots]

            # Store the final set of transformations WITHOUT the identity (implicit A->A)
            self.rot_mats_noI = torch.stack(effective_rots, dim=0).to(device, dtype)
            print(f"DEBUG: Using IDEALIZED rotations derived from A/D matching for noise/denoising. Final Shape for {n_subunits_found} subunits: {self.rot_mats_noI.shape}")

        elif self.symmetry_type and n_subunits_found > 1:
                # Use precomputed from __init__ if available (for non-Icosahedral cases)
                if hasattr(self, 'rot_mats_noI') and self.rot_mats_noI is not None:
                    # Ensure the number of matrices matches the number of non-identity subunits
                    expected_rots = n_subunits_found - 1
                    num_precomputed = self.rot_mats_noI.shape[0]

                    if num_precomputed < expected_rots:
                            print(f"[Warning] Precomputed rot_mats_noI has {num_precomputed} matrices, but expected {expected_rots} for {n_subunits_found} subunits. Appending identities.")
                            self.rot_mats_noI = self.rot_mats_noI.to(device, dtype) # Ensure device/dtype first
                            identity_mat = torch.eye(3, device=device, dtype=dtype)
                            missing_rots = [identity_mat] * (expected_rots - num_precomputed)
                            self.rot_mats_noI = torch.cat([self.rot_mats_noI] + missing_rots, dim=0)

                    elif num_precomputed > expected_rots:
                            print(f"[Warning] Precomputed rot_mats_noI has {num_precomputed} matrices, but expected {expected_rots}. Truncating.")
                            self.rot_mats_noI = self.rot_mats_noI[:expected_rots].to(device, dtype)
                    else:
                            self.rot_mats_noI = self.rot_mats_noI.to(device, dtype) # Ensure device/dtype

                    print(f"DEBUG: Using precomputed {self.symmetry_type} rot_mats_noI. Final Shape: {self.rot_mats_noI.shape}")
                else:
                    print(f"Warning: Symmetry {self.symmetry_type} requested, but no precomputed rot_mats_noI found or generated. Symmetrical noise/denoising will be basic Gaussian noise.")
                    self.rot_mats_noI = None # Ensure it's None if no rotations available
        else: # Monomer or no symmetry specified
            print("DEBUG: No symmetry or monomer case. No rotations needed for noise/denoising.")
            self.rot_mats_noI = None

        # Ensure self.rot_mats_noI is tensor or None before proceeding
        if self.rot_mats_noI is not None:
                if not isinstance(self.rot_mats_noI, torch.Tensor):
                    raise TypeError(f"self.rot_mats_noI should be a Tensor or None, but got {type(self.rot_mats_noI)}")
                # Final check on number of rotations vs subunits
                if self.rot_mats_noI.shape[0] != (n_subunits_found - 1):
                     raise ValueError(f"Mismatch between final number of rotation matrices ({self.rot_mats_noI.shape[0]}) and expected ({n_subunits_found - 1})")


        # ======= START: Reverse Diffusion Loop =======

        # 5. Build Diffusion Schedule
        sigmas = self.sample_schedule(num_sampling_steps)

        # 6. Determine Start Point in Schedule
        start_index = max(0, num_sampling_steps - forward_diffusion_steps)
        start_sigma = sigmas[start_index]

        # 7. Initialize Coordinates and Apply Initial Noise
        # Start with the clean reference coordinates
        current_coords = coords_ref.clone()

        # Calculate initial noise level
        t_hat_initial = start_sigma * self.noise_scale
        sigma_tm_val_initial = sigmas[start_index + 1].item() if start_index + 1 < len(sigmas) else 0.0

        # Apply Initial Symmetrical Noise using the derived self.rot_mats_noI
        initial_sym_noise = self._symmetrical_noise(
            coords=current_coords,
            feats=feats, # Pass feats for get_symmetrical_atom_mapping inside _symmetrical_noise
            subunits=subunits,
            rot_mats=self.rot_mats_noI, # Use the derived/loaded rotations
            t_hat=t_hat_initial.item(),
            sigma_tm_val=sigma_tm_val_initial # sigma at end of this (conceptual) step
        )
        current_coords = current_coords + initial_sym_noise
        print(f"DEBUG: Applied initial symmetrical noise. Start t_hat={t_hat_initial:.4f}, sigma_end={sigma_tm_val_initial:.4f}")

        # 8. Apply Initial Rigid Constraints (Translate subunits to match INPUT reference COMs)
        if target_coms_all is None:
            raise RuntimeError("Target COMs tensor is None before applying initial constraints.")
        current_coords = self.apply_symmetry_constraints_rigid(
            coords=current_coords,
            subunits=subunits,
            desired_coms_all_subunits=target_coms_all # Target COMs from INPUT ref
        )
        print(f"DEBUG: Applied initial rigid constraints using target COMs from INPUT.")

        # 9. Build Reverse Schedule Steps
        reverse_schedule = []
        for i in range(start_index, len(sigmas) - 1):
            sigma_current = sigmas[i]
            sigma_next = sigmas[i + 1]
            # Determine gamma (using instance attributes like self.gamma_0, self.gamma_min)
            gamma_val = self.gamma_0 if sigma_current > self.gamma_min else 0.0
            reverse_schedule.append((sigma_current, sigma_next, gamma_val))

        # 10. Prepare for Denoising Loop
        token_repr = None
        model_cache = {} # Initialize cache if used

        # --- 11. Reverse Diffusion Loop ---
        print(f"DEBUG: Starting reverse diffusion loop for {len(reverse_schedule)} steps.")
        for step_i, (sigma_tm, sigma_t, gamma_val) in enumerate(reverse_schedule):
            sigma_tm_val = sigma_tm.item()
            sigma_t_val = sigma_t.item()
            gamma_val_val = float(gamma_val) # Ensure gamma is float
            t_hat = sigma_tm_val * (1 + gamma_val_val) # Noise level for this step

            # Add Symmetrical Noise *for this step* using self.rot_mats_noI
            step_sym_noise = self._symmetrical_noise(
                    coords=current_coords,
                    feats=feats, # Pass feats again
                    subunits=subunits,
                    rot_mats=self.rot_mats_noI,
                    t_hat=t_hat,            # Current noise level
                    sigma_tm_val=sigma_t_val # Sigma at end of step
            )
            coords_noisy_step = current_coords + step_sym_noise

            # Denoise using preconditioned_network_forward_symmetry
            # This function internally uses self.rot_mats_noI if applicable for denoising step
            denoised_coords, token_a = self.preconditioned_network_forward_symmetry(
                coords_noisy=coords_noisy_step,
                sigma=t_hat, # Pass the noise level t_hat as sigma input to network
                network_condition_kwargs=dict(
                    # Ensure all necessary kwargs are passed
                    s_inputs=network_condition_kwargs['s_inputs'],
                    s_trunk=network_condition_kwargs['s_trunk'],
                    z_trunk=network_condition_kwargs['z_trunk'],
                    relative_position_encoding=network_condition_kwargs['relative_position_encoding'],
                    feats=feats, # Pass updated feats dict
                    multiplicity=multiplicity,
                    model_cache=model_cache if self.use_inference_model_cache else None
                ),
                training=False, # Ensure model is in eval mode conceptually
            )

            # Apply Rigid Constraint after denoising (Translate subunits to match INPUT reference COMs)
            denoised_coords = self.apply_symmetry_constraints_rigid(
                coords=denoised_coords,
                subunits=subunits,
                desired_coms_all_subunits=target_coms_all # Use INPUT reference COMs
            )

            # Update current coordinates for the next step
            current_coords = denoised_coords

            # Token accumulation logic (optional)
            if self.accumulate_token_repr:
                if token_repr is None: token_repr = torch.zeros_like(token_a)
                # Ensure gradients are handled correctly if training accumulator
                with torch.set_grad_enabled(train_accumulate_token_repr):
                    t_tensor = torch.full((current_coords.shape[0],), t_hat, device=device, dtype=dtype)
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(t_tensor), acc_a=token_repr, next_a=token_a
                    )
            elif step_i == len(reverse_schedule) - 1: # Get final token repr if not accumulating
                    token_repr = token_a


        # 12. Final Output
        print(f"DEBUG: Reverse diffusion finished.")
        # Apply final constraint application using target_coms_all from INPUT
        final_coords = self.apply_symmetry_constraints_rigid(
                coords=current_coords,
                subunits=subunits,
                desired_coms_all_subunits=target_coms_all # Final enforcement
        )

        # Return the final coordinates and accumulated/final token representation
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
        rot_mats: torch.Tensor,
        t_hat: float,
        sigma_tm_val: float # Represents the sigma level at the *end* of the step for variance calculation
    ) -> torch.Tensor:
        """
        Generate symmetrical noise about the origin:
         - For each atom in the reference subunit, sample a random vector.
         - Rotate that vector for each neighbor using the fixed rotation mapping.

        Args:
            coords (torch.Tensor): (B, A, 3) coordinates.
            feats (dict): Feature dictionary.
            subunits (list[torch.Tensor]): List of index tensors for each subunit.
            rot_mats (torch.Tensor): Precomputed, reordered rotation matrices. Unused if 1 subunit.
            t_hat (float): Current noise level (sigma * (1 + gamma)).
            sigma_tm_val (float): Sigma level at the end of the current step (sigma_t).

        Returns:
            torch.Tensor: Noise tensor of shape (B, A, 3) with symmetrical noise applied.
        """
        B, A, _ = coords.shape
        device = coords.device

        # --- MODIFICATION START ---
        # Calculate the difference in variance, ensuring it's non-negative
        variance_diff = t_hat**2 - sigma_tm_val**2
        variance_diff = max(variance_diff, 0.0) # Clamp to zero if negative due to numerical issues
        std_dev_diff = math.sqrt(variance_diff)
        scale_ = self.noise_scale * std_dev_diff
        # --- MODIFICATION END ---


        if (not self.symmetry_type) or (not subunits) or (len(subunits) < 2) or (rot_mats is None):
            # If no symmetry, only one subunit, or no rot_mats, apply standard Gaussian noise
            return scale_ * torch.randn_like(coords)

        # Proceed with symmetrical noise generation if applicable
        mapping = self.get_symmetrical_atom_mapping(feats)
        eps = torch.zeros_like(coords)

        # Ensure rot_mats is on the correct device if it's not None
        if rot_mats is not None:
            rot_mats = rot_mats.to(device)

        for b_idx in range(B):
            # Check if mapping is valid for this batch element (necessary if feats vary per batch)
            if not mapping: continue

            for local_ref_idx, all_atoms in mapping.items():
                if not all_atoms: continue # Skip if no atoms mapped

                # Sample a random vector for the reference atom.
                ref_atom = all_atoms[0]
                v = scale_ * torch.randn(3, device=device)
                eps[b_idx, ref_atom, :] = v

                # For each neighbor chain, assign the fixed rotation:
                # Ensure rot_mats exists before trying to index it
                if rot_mats is not None and len(all_atoms) > 1:
                    for i_sub in range(1, len(all_atoms)):
                        # Check if index is valid for rot_mats
                        rot_idx = i_sub - 1
                        if rot_idx < len(rot_mats):
                            a_idx = all_atoms[i_sub]
                            R = rot_mats[rot_idx] # Already on correct device
                            v_rot = rotate_coords_about_origin(v.unsqueeze(0), R)[0]
                            eps[b_idx, a_idx, :] = v_rot
                        # else: # Optional: Handle cases where #subunits > #rot_mats (shouldn't happen with correct logic)
                        #     print(f"Warning: Skipping rotation for subunit {i_sub}, not enough rotation matrices.")
        return eps


    # Inside the AtomDiffusion class in diffusion.py

    def _generate_ideal_icosahedron_and_find_ad_map(
        self,
        coords_chain_A: torch.Tensor, # Coords for INPUT Chain A (N_atoms_A, 3)
        coords_chain_D: torch.Tensor, # Coords for INPUT Chain D (N_atoms_D, 3)
        device: torch.device,
        dtype: torch.dtype,           # The final desired output dtype (e.g., float32)
        output_dir: str = "."         # Directory to save the temp CIF files
    ) -> torch.Tensor:
        """
        Generates a temporary "ideal" icosahedron template using input subunit A
        as the generator. It identifies which subunits in this ideal template
        best correspond (by COM distance) to the input subunits A and D.
        It then calculates the *ideal rotation matrix* that transforms the
        A-matching ideal subunit to the D-matching ideal subunit within the template.

        Includes robust index checking and NumPy fallback for argmin. Corrected distance calc.

        Saves:
        1. The full ideal icosahedron template coordinates.
        2. The coordinates of the two ideal subunits that best match input A and D.

        Args:
            coords_chain_A: Coordinates of the input reference chain A (N_atoms_A, 3).
            coords_chain_D: Coordinates of the input target chain D (N_atoms_D, 3).
            device: Target torch device.
            dtype: Target torch dtype for the *final* output R_AD matrix.
                   Internal calculations may use float64 for precision.
            output_dir: Directory to save the temporary CIF files.

        Returns:
            torch.Tensor: The IDEAL rotation matrix (3, 3) that maps the position
                          corresponding to input A to the position corresponding
                          to input D within the generated ideal template.
                          The matrix will have the specified output `dtype`.
        """
        #print("[Debug] Generating IDEAL icosahedron template to find A->D map...")
        internal_dtype = torch.float64 # Use float64 for internal coordinate/op manipulations

        # --- 0. Input Validation and Preparation ---
        if coords_chain_A.shape[0] == 0: raise ValueError("Input coordinates for Chain A are empty.")
        if coords_chain_D.shape[0] == 0: raise ValueError("Input coordinates for Chain D are empty.")
        n_atoms_gen = coords_chain_A.shape[0]

        coords_A_gen = coords_chain_A.to(device=device, dtype=internal_dtype)
        coords_D_input = coords_chain_D.to(device=device, dtype=internal_dtype)

        # Calculate COMs. Assuming calculate_com returns (B, 3) or (3,).
        # We need it effectively as (1, 3) for broadcasting against (60, 3).
        com_input_A = calculate_com(coords_A_gen)
        if com_input_A.ndim == 1: com_input_A = com_input_A.unsqueeze(0) # Ensure (1, 3)
        if torch.isnan(com_input_A).any() or torch.isinf(com_input_A).any(): raise ValueError(f"Input A COM calculation resulted in NaN/Inf: {com_input_A}")

        com_input_D = calculate_com(coords_D_input)
        if com_input_D.ndim == 1: com_input_D = com_input_D.unsqueeze(0) # Ensure (1, 3)
        if torch.isnan(com_input_D).any() or torch.isinf(com_input_D).any(): raise ValueError(f"Input D COM calculation resulted in NaN/Inf: {com_input_D}")

        # Make sure shape is exactly (1, 3) after checks
        if com_input_A.shape != (1, 3): raise ValueError(f"com_input_A unexpected shape: {com_input_A.shape}")
        if com_input_D.shape != (1, 3): raise ValueError(f"com_input_D unexpected shape: {com_input_D.shape}")

        #print(f"[Debug] Input A COM: {com_input_A.cpu().numpy()}")
        #print(f"[Debug] Input D COM: {com_input_D.cpu().numpy()}")

        # --- 1. Generate C3 Rotations for initial ASU (using float64) ---
        C3_GENERATION_ANGLE_1 = 2 * math.pi / 3
        C3_GENERATION_ANGLE_2 = 4 * math.pi / 3
        def _build_rot_matrix_pt_f64(angle: float, axis: str = 'z') -> torch.Tensor:
            cos_a = math.cos(angle); sin_a = math.sin(angle)
            if axis.lower() == 'z': mat_np = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            else: raise ValueError("Only Z axis needed for initial trimer")
            return torch.from_numpy(mat_np).to(device=device, dtype=internal_dtype)
        rot_120, rot_240 = _build_rot_matrix_pt_f64(C3_GENERATION_ANGLE_1, 'z'), _build_rot_matrix_pt_f64(C3_GENERATION_ANGLE_2, 'z')
        identity_mat = torch.eye(3, device=device, dtype=internal_dtype)
        component_rots = [identity_mat, rot_120, rot_240]

        # --- 2. Create Ideal Trimer ASU {A, B', C'} ---
        coords_B_prime_ideal = torch.matmul(coords_A_gen, rot_120.T)
        coords_C_prime_ideal = torch.matmul(coords_A_gen, rot_240.T)
        trimer_asu_coords_ideal = torch.cat([coords_A_gen, coords_B_prime_ideal, coords_C_prime_ideal], dim=0)

        # --- 3. Get Canonical I/C3 Pseudo-Quotient Operators (g'_i) ---
        #print("[Debug] Getting canonical I/C3 operators (g'_i)...")
        try:
            candidate_I_ops = symmetry.get_pseudoquotient_operators_transformed_numpy_style(device=device, dtype=internal_dtype)
            if candidate_I_ops is None or candidate_I_ops.shape != (20, 3, 3): raise RuntimeError(f"Failed to get 20 I/C3 operators. Shape: {candidate_I_ops.shape if candidate_I_ops is not None else 'None'}")
            print(f"[Debug] Obtained {candidate_I_ops.shape[0]} canonical I/C3 operators (g'_i) with dtype {candidate_I_ops.dtype}.")
        except Exception as e: raise RuntimeError(f"Failed to get I/C3 operators: {e}") from e

        # --- 4. Generate Full Ideal Icosahedron Coordinates & Store Subunits ---
        all_coords_list_ideal, ideal_subunit_coords_list, ideal_subunit_labels, ideal_subunit_coms_list = [], [], [], []
        #print("[Debug] Applying canonical operators to the ASU and storing subunits...")
        for k, op_matrix in enumerate(candidate_I_ops):
            rotated_trimer_ideal = torch.matmul(trimer_asu_coords_ideal, op_matrix.T)
            all_coords_list_ideal.append(rotated_trimer_ideal)
            subunits_in_k = [rotated_trimer_ideal[j * n_atoms_gen : (j + 1) * n_atoms_gen] for j in range(3)]
            for j, sub_coords in enumerate(subunits_in_k):
                ideal_subunit_coords_list.append(sub_coords)
                ideal_subunit_labels.append((k, j))
                com_sub = calculate_com(sub_coords)
                # Ensure COM is (3,) before appending
                if com_sub.ndim == 2: com_sub = com_sub.squeeze(0)
                if com_sub.shape != (3,): raise ValueError(f"Ideal subunit COM shape error: Expected (3,), got {com_sub.shape}")
                ideal_subunit_coms_list.append(com_sub)

        full_coords_ideal = torch.cat(all_coords_list_ideal, dim=0)
        ideal_subunit_coms = torch.stack(ideal_subunit_coms_list, dim=0) # Shape: (60, 3)
        if ideal_subunit_coms.shape != (60, 3): raise ValueError(f"Expected ideal_subunit_coms shape (60, 3), got {ideal_subunit_coms.shape}")
        if torch.isnan(ideal_subunit_coms).any() or torch.isinf(ideal_subunit_coms).any():
            problematic_indices = torch.where(torch.isnan(ideal_subunit_coms).any(dim=1) | torch.isinf(ideal_subunit_coms).any(dim=1))[0]
            raise ValueError(f"Ideal subunit COMs contain NaN or Inf at indices: {problematic_indices}.")
        #print(f"[Debug] Full IDEAL icosahedron generated. Shape: {full_coords_ideal.shape}, Dtype: {full_coords_ideal.dtype}")
        #print(f"[Debug] Calculated COMs for {ideal_subunit_coms.shape[0]} ideal subunits. Shape: {ideal_subunit_coms.shape}")

        # --- 5. Find Ideal Subunits Matching Input A and D by COM Distance (Robust Check) ---
        def find_min_idx_robust(distances, label=""):
            """Finds min index robustly, checking torch.argmin output and using numpy fallback."""
            dev = distances.device
            dt = distances.dtype
            expected_len = 60
            # <<< Shape check moved here >>>
            if not isinstance(distances, torch.Tensor) or distances.shape != (expected_len,):
                raise ValueError(f"[{label}] Internal Error: Expected distances shape ({expected_len},), got {distances.shape if isinstance(distances, torch.Tensor) else type(distances)}")

            valid_mask = ~torch.isnan(distances) & ~torch.isinf(distances)
            num_valid = valid_mask.sum().item()
            print(f"[{label}] Number of valid (non-NaN/Inf) distances: {num_valid} / {expected_len}")
            if num_valid == 0:
                print(f"[{label}] Error: No valid distances found.")
                return None

            temp_distances = torch.where(valid_mask, distances, torch.tensor(float('inf'), device=dev, dtype=dt))
            min_idx_tensor = None
            min_idx_val = -1

            # --- Try PyTorch ---
            try:
                argmin_result = torch.argmin(temp_distances)
                print(f"[{label}] torch.argmin direct result: {argmin_result}")
                if not isinstance(argmin_result, torch.Tensor) or argmin_result.ndim != 0:
                     print(f"[{label}] Error: torch.argmin did not return a scalar tensor. Type: {type(argmin_result)}, Shape: {argmin_result.shape if isinstance(argmin_result, torch.Tensor) else 'N/A'}")
                else:
                     min_idx_val = argmin_result.item()
                     if 0 <= min_idx_val < expected_len:
                          if torch.isinf(temp_distances[min_idx_val]): print(f"[{label}] Warning: torch.argmin yielded valid index {min_idx_val}, but value is Inf.")
                          else:
                               print(f"[{label}] torch.argmin SUCCEEDED with valid index: {min_idx_val}")
                               min_idx_tensor = argmin_result
                     else: print(f"[{label}] !!! Error: torch.argmin RETURNED INVALID INDEX: {min_idx_val} for expected range [0, {expected_len-1}] !!!")
            except Exception as e: print(f"[{label}] Exception during torch.argmin or index check: {e}")

            # --- Try NumPy Fallback ---
            if min_idx_tensor is None:
                print(f"[{label}] Attempting NumPy fallback...")
                try:
                    temp_distances_np = temp_distances.cpu().numpy()
                    if temp_distances_np.shape != (expected_len,): print(f"[{label}] Error: NumPy array shape mismatch: {temp_distances_np.shape}"); return None
                    if np.any(np.isnan(temp_distances_np)): print(f"[{label}] Warning: NaNs found in NumPy array after torch.where!?")
                    min_idx_np = np.argmin(temp_distances_np)
                    print(f"[{label}] numpy.argmin result: {min_idx_np}")
                    if 0 <= min_idx_np < expected_len:
                        if np.isinf(temp_distances_np[min_idx_np]): print(f"[{label}] Warning: numpy.argmin yielded valid index {min_idx_np}, but value is Inf.")
                        else:
                            print(f"[{label}] numpy.argmin SUCCEEDED with valid index: {min_idx_np}")
                            min_idx_tensor = torch.tensor(min_idx_np, device=dev, dtype=torch.long)
                    else: print(f"[{label}] !!! Error: numpy.argmin RETURNED INVALID INDEX: {min_idx_np} for expected range [0, {expected_len-1}] !!!")
                except Exception as e_np: print(f"[{label}] Exception during numpy fallback: {e_np}")

            if min_idx_tensor is None: print(f"[{label}] Error: Both PyTorch and NumPy argmin failed."); return None
            else:
                if min_idx_tensor.ndim == 0 and (0 <= min_idx_tensor.item() < expected_len): return min_idx_tensor
                else: print(f"[{label}] Error: Final index tensor invalid: {min_idx_tensor}"); return None


        # --- Calculate Distances CORRECTLY ---
        # Subtract single input COM (1, 3) from all ideal COMs (60, 3) -> shape (60, 3)
        # Calculate L2 norm along dim=1 -> shape (60,)
        vector_diff_A = ideal_subunit_coms - com_input_A
        dist_A_to_ideal = torch.linalg.norm(vector_diff_A, dim=1)
        #print(f"[Debug] Calculated dist_A_to_ideal. Shape: {dist_A_to_ideal.shape}") # Add shape print

        vector_diff_D = ideal_subunit_coms - com_input_D
        dist_D_to_ideal = torch.linalg.norm(vector_diff_D, dim=1)
        #print(f"[Debug] Calculated dist_D_to_ideal. Shape: {dist_D_to_ideal.shape}") # Add shape print


        # Match Input A
        idx_A_match_tensor = find_min_idx_robust(dist_A_to_ideal, label="MatchA")
        if idx_A_match_tensor is None: raise ValueError("Failed to find valid index for Input A match.")
        idx_A_match = idx_A_match_tensor.item()

        # Match Input D
        idx_D_match_tensor = find_min_idx_robust(dist_D_to_ideal, label="MatchD")
        if idx_D_match_tensor is None: raise ValueError("Failed to find valid index for Input D match.")
        idx_D_match = idx_D_match_tensor.item()

        # --- Rest of the function remains the same ---

        # Labels and Coords
        min_dist_A = dist_A_to_ideal[idx_A_match]
        label_A_match = ideal_subunit_labels[idx_A_match]
        coords_A_ideal_match = ideal_subunit_coords_list[idx_A_match]
        min_dist_D = dist_D_to_ideal[idx_D_match]
        label_D_match = ideal_subunit_labels[idx_D_match]
        coords_D_ideal_match = ideal_subunit_coords_list[idx_D_match]
        #print(f"[Debug] Input A COM matches Ideal Subunit Index {idx_A_match} label {label_A_match} (Dist: {min_dist_A:.3f})")
        #print(f"[Debug] Input D COM matches Ideal Subunit Index {idx_D_match} label {label_D_match} (Dist: {min_dist_D:.3f})")

        # --- 6. Calculate the IDEAL Rotation ---
        k_A, j_A = label_A_match
        k_D, j_D = label_D_match
        R_ref_A_match = candidate_I_ops[k_A] @ component_rots[j_A]
        R_ref_D_match = candidate_I_ops[k_D] @ component_rots[j_D]
        R_ideal_AD = R_ref_D_match @ R_ref_A_match.T
        #print(f"[Debug] Calculated IDEAL rotation matrix R_ideal_AD.")

        """
        # --- 7. Save CIF Files ---
        cif_filename_full = os.path.join(output_dir, "temp_ideal_icosahedron_full.cif")
        try:
            self._save_icosahedron_to_cif(coords=full_coords_ideal, filename=cif_filename_full)
            print(f"[Debug] Saved FULL ideal template to {cif_filename_full}")
        except Exception as e: print(f"[Warning] Failed to save full ideal CIF: {e}")
        cif_filename_ad_match = os.path.join(output_dir, "temp_ideal_subunits_AD_matched.cif")
        try:
            coords_AD_matched_combined = torch.cat([coords_A_ideal_match, coords_D_ideal_match], dim=0)
            self._save_icosahedron_to_cif(coords=coords_AD_matched_combined, filename=cif_filename_ad_match)
            print(f"[Debug] Saved matched ideal subunits (A-like, D-like) to {cif_filename_ad_match}")
        except Exception as e: print(f"[Warning] Failed to save matched A/D CIF: {e}")
        """

        # --- 8. Return the IDEAL rotation matrix ---
        R_ideal_AD = R_ideal_AD.to(dtype)
        #print(f"DEBUG: Found IDEAL A->D rotation matrix. Shape: {R_ideal_AD.shape}, Output Dtype: {R_ideal_AD.dtype}")
        return R_ideal_AD









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


    def _save_icosahedron_to_cif(
        self,
        coords: torch.Tensor, # Shape (B, N_total_atoms, 3) or (N_total_atoms, 3)
        filename: str = "temp_icosahedron.cif",
        batch_idx: int = 0
    ) -> None:
        """
        Saves the Cartesian coordinates of a structure (like the generated icosahedron)
        to a barebones CIF file. Uses Cartesian coordinates directly.

        Args:
            coords: Tensor containing the coordinates. Expected shape (B, N, 3) or (N, 3).
            filename: Path to save the CIF file.
            batch_idx: Index of the batch element to save if coords has a batch dim.
        """
        if coords.dim() == 3:
            if batch_idx >= coords.shape[0]:
                print(f"Warning: batch_idx {batch_idx} out of range for coords shape {coords.shape}. Saving batch 0.")
                batch_idx = 0
            coords_to_save = coords[batch_idx].detach().cpu().numpy() # Select batch and move to CPU
        elif coords.dim() == 2:
            coords_to_save = coords.detach().cpu().numpy() # Assume single structure
        else:
            raise ValueError(f"Unexpected coordinates shape: {coords.shape}. Expected (B, N, 3) or (N, 3).")

        n_atoms = coords_to_save.shape[0]

        # Create CIF content
        cif_lines = []
        cif_lines.append("data_icosahedron_vis")
        cif_lines.append("#")
        # Add dummy cell parameters - large enough to contain the structure
        cif_lines.append("_cell_length_a     200.0")
        cif_lines.append("_cell_length_b     200.0")
        cif_lines.append("_cell_length_c     200.0")
        cif_lines.append("_cell_angle_alpha  90.0")
        cif_lines.append("_cell_angle_beta   90.0")
        cif_lines.append("_cell_angle_gamma  90.0")
        cif_lines.append("#")
        cif_lines.append("loop_")
        cif_lines.append("_atom_site_group_PDB") # Often 'ATOM' or 'HETATM'
        cif_lines.append("_atom_site_type_symbol") # Element
        cif_lines.append("_atom_site_label_atom_id") # Unique atom name
        cif_lines.append("_atom_site_label_comp_id") # Residue/Molecule ID
        cif_lines.append("_atom_site_label_asym_id") # Chain ID
        cif_lines.append("_atom_site_label_seq_id") # Residue number (optional)
        cif_lines.append("_atom_site_Cartn_x") # Cartesian X
        cif_lines.append("_atom_site_Cartn_y") # Cartesian Y
        cif_lines.append("_atom_site_Cartn_z") # Cartesian Z
        cif_lines.append("_atom_site_occupancy")
        cif_lines.append("_atom_site_B_iso_or_equiv")

        # Determine atoms per original subunit (assuming A, B', C' had same size)
        atoms_per_orig_subunit = coords_to_save.shape[0] // 60 # 20 triangles * 3 subunits per triangle
        if coords_to_save.shape[0] % 60 != 0:
             print(f"Warning: Total atoms ({coords_to_save.shape[0]}) not divisible by 60. Subunit assignment might be approximate.")
             atoms_per_orig_subunit = max(1, coords_to_save.shape[0] // 60) # Avoid division by zero

        chain_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


        for i in range(n_atoms):
            x, y, z = coords_to_save[i]

            # Assign Chain ID based on which triangle it belongs to (0-19)
            # Assign Residue ID based on which original subunit (A=1, B'=2, C'=3) within the triangle
            triangle_index = i // (atoms_per_orig_subunit * 3)
            subunit_in_triangle = (i % (atoms_per_orig_subunit * 3)) // atoms_per_orig_subunit
            
            # Assign Chain ID for the triangle
            chain_id = chain_labels[triangle_index % len(chain_labels)] # Cycle through labels if > 62 triangles

            # Assign Molecule ID (e.g., TRI for triangle) and residue number (subunit within triangle)
            comp_id = "TRI"
            seq_id = str(subunit_in_triangle + 1) # 1, 2, or 3

            pdb_group = "ATOM"
            atom_symbol = "C" # Use Carbon as placeholder
            atom_label = f"C{i+1}" # Unique atom label


            cif_lines.append(
                f"{pdb_group:<6} {atom_symbol:<2} {atom_label:<4} {comp_id:<3} {chain_id:<1} {seq_id:<3} "
                f"{x:8.3f} {y:8.3f} {z:8.3f} 1.00 20.00"
            )

        # Write to file
        try:
            with open(filename, 'w') as f:
                f.write("\n".join(cif_lines) + "\n")
            print(f"DEBUG: Saved generated icosahedron coordinates to {filename}")
        except IOError as e:
            print(f"Error saving CIF file {filename}: {e}")

    def find_best_rotation_point_cloud(
        self,
        target_point: torch.Tensor,      # Target vector (B, 3) or (3,)
        candidate_rots: torch.Tensor,    # Candidate rotation matrices (N_rots, 3, 3)
        ref_point: torch.Tensor          # Reference vector to rotate (B, 3) or (3,)
    ) -> torch.Tensor:
        """
        Finds the candidate rotation matrix that maps ref_point closest to target_point.

        Args:
            target_point: The target vector(s).
            candidate_rots: Pool of rotation matrices to test.
            ref_point: The starting reference vector(s).

        Returns:
            torch.Tensor: The best rotation matrix found (3, 3) or (B, 3, 3).
                          Returns (3,3) if input points are (3,), (B,3,3) if input points are (B,3).
        """
        n_rots = candidate_rots.shape[0]
        device = target_point.device
        dtype = target_point.dtype
        candidate_rots = candidate_rots.to(device, dtype) # Ensure correct device/dtype

        is_batched = target_point.ndim == 2

        if not is_batched:
            # Add batch dimension for unified processing
            target_point = target_point.unsqueeze(0) # (1, 3)
            ref_point = ref_point.unsqueeze(0)       # (1, 3)

        B = target_point.shape[0]

        # Prepare tensors for broadcasting
        # ref_point:      (B, 1, 3)
        # candidate_rots: (1, N_rots, 3, 3)
        # target_point:   (B, 1, 3)
        ref_point_exp = ref_point.unsqueeze(1)             # (B, 1, 3)
        cand_rots_exp = candidate_rots.unsqueeze(0)        # (1, N_rots, 3, 3)
        target_point_exp = target_point.unsqueeze(1)       # (B, 1, 3)

        # Apply all rotations to the reference point(s) via batch matrix multiplication
        # Result shape: (B, N_rots, 3)
        # rotated_points = torch.matmul(ref_point_exp.unsqueeze(2), cand_rots_exp.permute(0, 1, 3, 2)).squeeze(2) # Older PyTorch/incorrect matmul
        # Correct approach: einsum or matmul with proper dimensions
        # rotated_points[b, r, k] = sum_j ref_point_exp[b, 0, j] * cand_rots_exp[0, r, k, j] (R^T * p)
        # OR rotated_points[b, r, k] = sum_j cand_rots_exp[0, r, k, j] * ref_point_exp[b, 0, j] (R * p) need R to be (3,3)

        # Simpler: Expand ref_point and apply matmul N_rots times
        # ref_point_exp: (B, 1, 1, 3)
        # cand_rots_exp: (1, N_rots, 3, 3) -> transpose -> (1, N_rots, 3, 3)
        # Target: R @ p^T -> (3,3) @ (3,1) -> (3,1) -> transpose -> (1,3)
        # Target: p @ R^T -> (1,3) @ (3,3) -> (1,3)
        # Let's use p @ R^T
        ref_point_exp = ref_point.unsqueeze(1) # (B, 1, 3)
        cand_rots_transposed = candidate_rots.permute(0, 2, 1).unsqueeze(0) # (1, N_rots, 3, 3)
        rotated_points = torch.matmul(ref_point_exp, cand_rots_transposed) # (B, N_rots, 3)


        # Calculate squared distances between rotated points and target point
        # dist_sq shape: (B, N_rots)
        dist_sq = torch.sum((rotated_points - target_point_exp)**2, dim=-1)

        # Find the index of the minimum distance for each batch element
        best_rot_indices = torch.argmin(dist_sq, dim=1) # Shape: (B,)

        # Select the best rotation matrix for each batch element
        best_rots = candidate_rots[best_rot_indices] # Shape: (B, 3, 3)

        if not is_batched:
            return best_rots.squeeze(0) # Return (3, 3)
        else:
            return best_rots # Return (B, 3, 3)


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


def find_best_rotation_point_cloud(
    target_point: torch.Tensor,      # Target vector (B, 3) or (3,)
    candidate_rots: torch.Tensor,    # Candidate rotation matrices (N_rots, 3, 3)
    ref_point: torch.Tensor          # Reference vector to rotate (B, 3) or (3,)
) -> torch.Tensor:
    """
    Finds the candidate rotation matrix that maps ref_point closest to target_point.

    Args:
        target_point: The target vector(s).
        candidate_rots: Pool of rotation matrices to test.
        ref_point: The starting reference vector(s).

    Returns:
        torch.Tensor: The best rotation matrix found (3, 3) or (B, 3, 3).
                      Returns (3,3) if input points are (3,), (B,3,3) if input points are (B,3).
    """
    n_rots = candidate_rots.shape[0]
    device = target_point.device
    dtype = target_point.dtype
    candidate_rots = candidate_rots.to(device, dtype) # Ensure correct device/dtype

    is_batched = target_point.ndim == 2

    if not is_batched:
        # Add batch dimension for unified processing
        target_point = target_point.unsqueeze(0) # (1, 3)
        ref_point = ref_point.unsqueeze(0)       # (1, 3)

    B = target_point.shape[0]

    # Prepare tensors for broadcasting
    # ref_point:      (B, 1, 3)
    # candidate_rots: (1, N_rots, 3, 3)
    # target_point:   (B, 1, 3)
    ref_point_exp = ref_point.unsqueeze(1)             # (B, 1, 3)
    cand_rots_exp = candidate_rots.unsqueeze(0)        # (1, N_rots, 3, 3)
    target_point_exp = target_point.unsqueeze(1)       # (B, 1, 3)

    # Apply all rotations to the reference point(s) via batch matrix multiplication
    # Result shape: (B, N_rots, 3)
    # rotated_points = torch.matmul(ref_point_exp.unsqueeze(2), cand_rots_exp.permute(0, 1, 3, 2)).squeeze(2) # Older PyTorch/incorrect matmul
    # Correct approach: einsum or matmul with proper dimensions
    # rotated_points[b, r, k] = sum_j ref_point_exp[b, 0, j] * cand_rots_exp[0, r, k, j] (R^T * p)
    # OR rotated_points[b, r, k] = sum_j cand_rots_exp[0, r, k, j] * ref_point_exp[b, 0, j] (R * p) need R to be (3,3)

    # Simpler: Expand ref_point and apply matmul N_rots times
    # ref_point_exp: (B, 1, 1, 3)
    # cand_rots_exp: (1, N_rots, 3, 3) -> transpose -> (1, N_rots, 3, 3)
    # Target: R @ p^T -> (3,3) @ (3,1) -> (3,1) -> transpose -> (1,3)
    # Target: p @ R^T -> (1,3) @ (3,3) -> (1,3)
    # Let's use p @ R^T
    ref_point_exp = ref_point.unsqueeze(1) # (B, 1, 3)
    cand_rots_transposed = candidate_rots.permute(0, 2, 1).unsqueeze(0) # (1, N_rots, 3, 3)
    rotated_points = torch.matmul(ref_point_exp, cand_rots_transposed) # (B, N_rots, 3)


    # Calculate squared distances between rotated points and target point
    # dist_sq shape: (B, N_rots)
    dist_sq = torch.sum((rotated_points - target_point_exp)**2, dim=-1)

    # Find the index of the minimum distance for each batch element
    best_rot_indices = torch.argmin(dist_sq, dim=1) # Shape: (B,)

    # Select the best rotation matrix for each batch element
    best_rots = candidate_rots[best_rot_indices] # Shape: (B, 3, 3)

    if not is_batched:
        return best_rots.squeeze(0) # Return (3, 3)
    else:
        return best_rots # Return (B, 3, 3)


def _build_rot_matrix_np(angle: float, axis: str = 'z') -> np.ndarray:
    # (Copy definition from post_process or above)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    if axis.lower() == 'z':
        return np.array([[cos_a, -sin_a, 0.0],
                         [sin_a, cos_a, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    elif axis.lower() == 'y':
        return np.array([[cos_a, 0.0, sin_a],
                         [0.0, 1.0, 0.0],
                         [-sin_a, 0.0, cos_a]], dtype=np.float64)
    elif axis.lower() == 'x':
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, cos_a, -sin_a],
                         [0.0, sin_a, cos_a]], dtype=np.float64)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
