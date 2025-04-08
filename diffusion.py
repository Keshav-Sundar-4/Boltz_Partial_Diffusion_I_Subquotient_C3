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
            rot_mats = symmetry.get_point_group(self.symmetry_type).to(device)

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
        forward_diffusion_steps=50,  # Determines start_sigma.
        multiplicity=1,
        train_accumulate_token_repr=False,
        **network_condition_kwargs,
    ):
        """
        Performs reverse diffusion sampling.
        Includes special handling for Icosahedral symmetry via C3 subquotient logic
        when symmetry_type is "I" and 6 subunits are detected.
        Finds the single R_I mapping A->D, B->E, C->F.
        Applies constraints sequentially: A->(B,C), A->D, B_constrained->E, C_constrained->F.
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        n_batches = atom_mask.shape[0]
        n_atoms = atom_mask.shape[1]
        device = self.device
        feats = network_condition_kwargs.get("feats", {})

        # 1. Get Input Coordinates (CLEAN)
        input_coords = network_condition_kwargs.get("input_coords", None)
        if input_coords is None:
            input_coords = feats.get("coords", None)
            if input_coords is None:
                raise ValueError("Coordinates must be provided either via 'input_coords' kwarg or in 'feats'.")
            if input_coords.ndim == 4:
                B_, N_, L_, _ = input_coords.shape
                input_coords = input_coords.reshape(B_, N_*L_, 3)
            input_coords = input_coords.repeat_interleave(multiplicity, 0)

        coords_initial = input_coords.clone().to(device)
        if coords_initial.ndim != 3 or coords_initial.shape[0] != n_batches or coords_initial.shape[1] != n_atoms:
            try:
                coords_initial = coords_initial.view(n_batches, n_atoms, 3)
            except RuntimeError as e:
                raise ValueError(f"Input coordinates shape {input_coords.shape} is incompatible with expected ({n_batches}, {n_atoms}, 3). Error: {e}")

        current_feats = feats.copy()
        current_feats["coords"] = coords_initial
        network_condition_kwargs["feats"] = current_feats

        # ======= Identify Subunits and Check for Special I/C3 Case =======
        subunits = symmetry.get_subunit_atom_indices(
            self.symmetry_type, self.chain_symmetry_groups, current_feats, device
        )
        n_subunits = len(subunits)
        is_I_C3_case = (self.symmetry_type == "I" and n_subunits == 6)

        # Precompute transformations and mappings if it's the special case
        R_C3_1, R_C3_2 = None, None
        R_I_found = None
        composite_transforms_for_noise = None # [Id, R_C3_1, R_C3_2, R_I, R_I@R_C3_1, R_I@R_C3_2]
        atom_map_I_C3 = None
        desired_com_standard = None # For standard symmetry case

        if is_I_C3_case:
            print("Detected Icosahedral symmetry with 6 subunits. Applying I/C3 logic.")
            # Assume subunits[0]=A, [1]=B, [2]=C, [3]=D, [4]=E, [5]=F

            # Calculate COMs from INITIAL coordinates
            coms = []
            for sub_indices in subunits:
                 if sub_indices.numel() > 0:
                      com = coords_initial[:, sub_indices, :].mean(dim=1) # Shape: [B, 3]
                 else:
                      # Handle empty subunit case gracefully
                      print(f"Warning: Subunit has {sub_indices.numel()} atoms. Assigning zero COM.")
                      com = torch.zeros((n_batches, 3), device=device, dtype=coords_initial.dtype)
                 coms.append(com)
            coms_all = torch.stack(coms, dim=1) # Shape [B, 6, 3]
            com_A, com_B, com_C = coms_all[:, 0, :], coms_all[:, 1, :], coms_all[:, 2, :]
            com_D, com_E, com_F = coms_all[:, 3, :], coms_all[:, 4, :], coms_all[:, 5, :]

            # Get theoretical C3 rotations (excluding identity)
            C3_rots_all = symmetry.get_Cn_groups(3).to(device).type_as(coords_initial)
            eye = torch.eye(3, device=device, dtype=C3_rots_all.dtype)
            C3_rots_noI = [R for R in C3_rots_all if not torch.allclose(R, eye, atol=1e-5)]
            if len(C3_rots_noI) != 2:
                 raise RuntimeError(f"Expected 2 non-identity C3 rotations, found {len(C3_rots_noI)}.")
            # TODO: Determine which C3 rot maps A->B vs A->C if needed, assume order for now
            R_C3_1 = C3_rots_noI[0]
            R_C3_2 = C3_rots_noI[1]

            # Get theoretical I/C3 rotations
            I_div_C3_rots = symmetry.get_pseudoquotient("I", "C_3").to(device).type_as(coords_initial) # Shape [20, 3, 3]

            # Find the best single R_I_found based on A->D, B->E, C->F mappings (averaged over batch)
            min_avg_total_dist_sq = float('inf')
            best_R_I_found = None
            with torch.no_grad():
                 # Precompute necessary tensors outside the loop
                 com_ABC = coms_all[:, 0:3, :] # Shape [B, 3, 3] (Subunits A, B, C)
                 com_DEF = coms_all[:, 3:6, :] # Shape [B, 3, 3] (Subunits D, E, F)

                 # Loop over candidate rotations R from I/C3 set
                 for R_candidate in I_div_C3_rots: # R_candidate shape [3, 3]
                     # Apply R_candidate to COMs of A, B, C
                     # einsum: 'ij, bkj -> bik' applies R (ij) to each vector k in each subunit k for each batch b
                     transformed_com_ABC = torch.einsum('ij, bki -> bkj', R_candidate, com_ABC) # Shape [B, 3, 3]

                     # Calculate squared distances to D, E, F
                     # || R(A)-D ||^2 + || R(B)-E ||^2 + || R(C)-F ||^2
                     dist_sq = torch.sum((transformed_com_ABC - com_DEF)**2, dim=(1, 2)) # Sum over subunits and coords -> Shape [B]

                     # Average total squared distance across batch
                     avg_total_dist_sq = dist_sq.mean()

                     if avg_total_dist_sq < min_avg_total_dist_sq:
                          min_avg_total_dist_sq = avg_total_dist_sq
                          best_R_I_found = R_candidate

            if best_R_I_found is None:
                 raise RuntimeError("Could not find a best I/C3 rotation.")

            R_I_found = best_R_I_found
            print(f"Found best I/C3 rotation with avg total distance sq {min_avg_total_dist_sq:.4f}")

            # Pre-calculate composite rotations needed *only for noise generation*
            R_I_C3_1 = R_I_found @ R_C3_1
            R_I_C3_2 = R_I_found @ R_C3_2
            composite_transforms_for_noise = [eye, R_C3_1, R_C3_2, R_I_found, R_I_C3_1, R_I_C3_2]

            # Get the atom mapping
            atom_map_I_C3 = self._get_I_C3_atom_mapping(subunits)
            if not atom_map_I_C3:
                 print("Warning: Failed to generate atom mapping for I/C3 case. Disabling special I/C3 handling.")
                 is_I_C3_case = False # Fallback to standard or no symmetry

        elif self.symmetry_type and n_subunits > 1:
             # Handle standard symmetry case (logic from previous implementation)
             print(f"Applying standard symmetry logic for {self.symmetry_type} with {n_subunits} subunits.")
             actual_coms = []
             for indices in subunits:
                  com = coords_initial[:, indices, :].mean(dim=1) if indices.numel() > 0 else torch.zeros((n_batches, 3), device=device, dtype=coords_initial.dtype)
                  actual_coms.append(com)
             actual_coms = torch.stack(actual_coms, dim=1) # Shape: [B, n_subunits, 3]

             ref_com = actual_coms[:, 0:1, :] # Shape: [B, 1, 3]
             desired_com_standard = ref_com.clone() # Use ref COM as the target for constraints

             theoretical_rot_mats_all = symmetry.get_point_group(self.symmetry_type).to(device).type_as(coords_initial)
             eye = torch.eye(3, device=device, dtype=coords_initial.dtype)
             non_identity_indices = [i for i, R in enumerate(theoretical_rot_mats_all) if not torch.allclose(R, eye, atol=1e-5)]
             theoretical_rot_mats_noI = theoretical_rot_mats_all[non_identity_indices]
             n_theory = theoretical_rot_mats_noI.shape[0]

             assigned_rot_mats_list = []
             used_theoretical_indices = set()
             for target_idx in range(1, n_subunits):
                 target_com = actual_coms[:, target_idx:target_idx+1, :]
                 predicted_coms_all = torch.einsum('njk,bik->bnj', theoretical_rot_mats_noI, ref_com)
                 distances = torch.norm(predicted_coms_all - target_com, dim=2)
                 avg_distances = distances.mean(dim=0)

                 best_dist_for_target = float('inf')
                 best_idx_for_target = -1
                 for theory_idx in range(n_theory):
                      if theory_idx not in used_theoretical_indices:
                           if avg_distances[theory_idx] < best_dist_for_target:
                                best_dist_for_target = avg_distances[theory_idx]
                                best_idx_for_target = theory_idx
                 if best_idx_for_target != -1:
                      assigned_rot_mats_list.append(theoretical_rot_mats_noI[best_idx_for_target])
                      used_theoretical_indices.add(best_idx_for_target)
                 else:
                      assigned_rot_mats_list.append(eye) # Fallback

             if assigned_rot_mats_list:
                  self.rot_mats_noI = torch.stack(assigned_rot_mats_list, dim=0)
             else:
                  self.rot_mats_noI = torch.empty((0, 3, 3), device=device, dtype=coords_initial.dtype)

        elif n_subunits <= 1 :
             print("No symmetry or only one subunit detected.")
             self.rot_mats_noI = None
             if n_subunits == 1 and subunits[0].numel() > 0:
                 desired_com_standard = coords_initial[:, subunits[0], :].mean(dim=1, keepdim=True)
             elif n_atoms > 0:
                 desired_com_standard = coords_initial.mean(dim=1, keepdim=True)
             else:
                 desired_com_standard = torch.zeros((n_batches, 1, 3), device=device, dtype=coords_initial.dtype)
        # ======= END SYMMETRY SETUP =======

        # 2. Build Diffusion Schedule
        sigmas = self.sample_schedule(num_sampling_steps)

        # 3. Determine Start Point (Forward Diffusion part)
        start_index = max(0, num_sampling_steps - forward_diffusion_steps)
        start_sigma = sigmas[start_index]
        initial_noise_level = start_sigma * self.noise_scale

        # 4. Apply Initial Symmetrical Noise
        current_coords = coords_initial.clone()
        eps_initial = torch.zeros_like(current_coords)

        if is_I_C3_case and atom_map_I_C3:
            # I/C3 specific noise using composite transforms relative to A
            ref_subunit_indices = subunits[0]
            n_atoms_ref = len(ref_subunit_indices)
            if n_atoms_ref > 0: # Only proceed if ref subunit not empty
                 noise_A = initial_noise_level * torch.randn(n_batches, n_atoms_ref, 3, device=device, dtype=current_coords.dtype)
                 for b_idx in range(n_batches):
                     for i, atom_idx_A in enumerate(ref_subunit_indices):
                          atom_idx_A = atom_idx_A.item()
                          if atom_idx_A in atom_map_I_C3:
                               all_indices = atom_map_I_C3[atom_idx_A]
                               v = noise_A[b_idx, i, :] # Noise vector for this atom in ref subunit
                               for sub_i, atom_idx_sub in enumerate(all_indices):
                                    R = composite_transforms_for_noise[sub_i] # Use precomputed composite R
                                    rotated_v = v @ R.T
                                    eps_initial[b_idx, atom_idx_sub, :] = rotated_v
        else:
            # Standard symmetrical noise OR simple Gaussian noise
             if self.symmetry_type and n_subunits > 1 and hasattr(self, 'rot_mats_noI') and self.rot_mats_noI is not None:
                 sigma_tm_val_initial = sigmas[start_index + 1] if start_index + 1 < len(sigmas) else 0.0
                 gamma_initial = self.gamma_0 if start_sigma > self.gamma_min else 0.0
                 t_hat_initial = start_sigma * (1 + gamma_initial)
                 eps_initial = self._symmetrical_noise(
                       current_coords, current_feats, subunits, self.rot_mats_noI,
                       t_hat_initial, sigma_tm_val_initial
                 )
             else:
                 eps_initial = initial_noise_level * torch.randn_like(current_coords)

        coords_noisy_start = current_coords + eps_initial
        current_coords = coords_noisy_start

        # 5. Apply Initial Rigid Constraints
        if is_I_C3_case:
            # Apply I/C3 constraints sequentially
            temp_coords = current_coords.clone() # Work on a copy
            for b_idx in range(n_batches):
                 coords_A = temp_coords[b_idx, subunits[0], :]
                 # Constrain B, C based on A
                 coords_B_c = rotate_coords_about_origin(coords_A, R_C3_1)
                 coords_C_c = rotate_coords_about_origin(coords_A, R_C3_2)
                 # Constrain D based on A
                 coords_D_c = rotate_coords_about_origin(coords_A, R_I_found)
                 # Constrain E based on B_constrained
                 coords_E_c = rotate_coords_about_origin(coords_B_c, R_I_found)
                 # Constrain F based on C_constrained
                 coords_F_c = rotate_coords_about_origin(coords_C_c, R_I_found)
                 # Update the main tensor
                 current_coords[b_idx, subunits[1], :] = coords_B_c
                 current_coords[b_idx, subunits[2], :] = coords_C_c
                 current_coords[b_idx, subunits[3], :] = coords_D_c
                 current_coords[b_idx, subunits[4], :] = coords_E_c
                 current_coords[b_idx, subunits[5], :] = coords_F_c
            # Note: We don't explicitly recenter A here, assuming initial coords are centered or handled elsewhere.
        elif hasattr(self, 'rot_mats_noI') and self.rot_mats_noI is not None:
             current_coords = self.apply_symmetry_constraints_rigid(
                  current_coords, subunits, self.rot_mats_noI, desired_com_standard
             )
        # Else: No constraints

        # 6. Build Reverse Schedule
        reverse_schedule = []
        for i in range(start_index, len(sigmas) - 1):
            sigma_current, sigma_next = sigmas[i], sigmas[i+1]
            gamma_val = self.gamma_0 if sigma_current > self.gamma_min else 0.0
            reverse_schedule.append((sigma_current, sigma_next, gamma_val))

        # 7. Prepare for Denoising Loop
        token_repr = None
        model_cache = {}

        # --- 8. Reverse Diffusion Loop ---
        for step_i, (sigma_tm, sigma_t, gamma_val) in enumerate(reverse_schedule):
            sigma_tm_val, sigma_t_val = sigma_tm.item(), sigma_t.item()
            gamma_val_val = float(gamma_val)
            t_hat = sigma_tm_val * (1 + gamma_val_val)

            # --- Calculate Symmetrical Noise for this step ---
            variance_diff = max(0.0, t_hat**2 - sigma_t_val**2)
            scale_ = self.noise_scale * math.sqrt(variance_diff)
            step_sym_noise = torch.zeros_like(current_coords)

            if is_I_C3_case and atom_map_I_C3:
                 # I/C3 specific noise generation using composite transforms
                 ref_subunit_indices = subunits[0]
                 n_atoms_ref = len(ref_subunit_indices)
                 if n_atoms_ref > 0:
                      noise_A = scale_ * torch.randn(n_batches, n_atoms_ref, 3, device=device, dtype=current_coords.dtype)
                      for b_idx in range(n_batches):
                           for i, atom_idx_A in enumerate(ref_subunit_indices):
                                atom_idx_A = atom_idx_A.item()
                                if atom_idx_A in atom_map_I_C3:
                                     all_indices = atom_map_I_C3[atom_idx_A]
                                     v = noise_A[b_idx, i, :]
                                     for sub_i, atom_idx_sub in enumerate(all_indices):
                                          R = composite_transforms_for_noise[sub_i]
                                          rotated_v = v @ R.T
                                          step_sym_noise[b_idx, atom_idx_sub, :] = rotated_v
            else:
                 # Standard symmetrical noise OR simple Gaussian noise
                 if self.symmetry_type and n_subunits > 1 and hasattr(self, 'rot_mats_noI') and self.rot_mats_noI is not None:
                      step_sym_noise = self._symmetrical_noise(
                           current_coords, current_feats, subunits, self.rot_mats_noI,
                           t_hat, sigma_t_val
                      )
                 else:
                      step_sym_noise = scale_ * torch.randn_like(current_coords)

            coords_noisy_step = current_coords + step_sym_noise

            # --- Denoising Step ---
            temp_feats = network_condition_kwargs["feats"].copy()
            temp_feats["coords"] = coords_noisy_step # Network sees input for this step

            original_rot_mats_backup = getattr(self, 'rot_mats_noI', None) # Backup standard rot mats
            if is_I_C3_case:
                 self.rot_mats_noI = None # Prevent standard symm logic inside forward_symmetry

            denoised_coords_pred, token_a = self.preconditioned_network_forward_symmetry(
                 coords_noisy=coords_noisy_step,
                 sigma=t_hat,
                 network_condition_kwargs=dict(
                     multiplicity=multiplicity,
                     model_cache=model_cache if self.use_inference_model_cache else None,
                     **{k:v for k,v in network_condition_kwargs.items() if k!='feats'},
                     feats=temp_feats
                 ),
                 training=False,
             )

            if is_I_C3_case:
                 self.rot_mats_noI = original_rot_mats_backup # Restore

            # --- Apply Rigid Symmetry Constraints ---
            if is_I_C3_case:
                 # Apply I/C3 constraints sequentially A->BC, A->D, B->E, C->F
                 constrained_coords = denoised_coords_pred.clone()
                 for b_idx in range(n_batches):
                      # Get predicted coords for A
                      coords_A = constrained_coords[b_idx, subunits[0], :]
                      # Calculate and apply constraints
                      coords_B_c = rotate_coords_about_origin(coords_A, R_C3_1)
                      coords_C_c = rotate_coords_about_origin(coords_A, R_C3_2)
                      coords_D_c = rotate_coords_about_origin(coords_A, R_I_found)
                      coords_E_c = rotate_coords_about_origin(coords_B_c, R_I_found) # Use constrained B
                      coords_F_c = rotate_coords_about_origin(coords_C_c, R_I_found) # Use constrained C

                      # Update the tensor (leave A untouched, update B-F)
                      constrained_coords[b_idx, subunits[1], :] = coords_B_c
                      constrained_coords[b_idx, subunits[2], :] = coords_C_c
                      constrained_coords[b_idx, subunits[3], :] = coords_D_c
                      constrained_coords[b_idx, subunits[4], :] = coords_E_c
                      constrained_coords[b_idx, subunits[5], :] = coords_F_c
                 current_coords = constrained_coords # Update current coords

            elif hasattr(self, 'rot_mats_noI') and self.rot_mats_noI is not None:
                 # Apply standard constraints
                 current_coords = self.apply_symmetry_constraints_rigid(
                      denoised_coords_pred, subunits, self.rot_mats_noI, desired_com_standard
                 )
            else:
                 # No constraints
                 current_coords = denoised_coords_pred


            # --- Accumulate Token Representations (Optional) ---
            if self.accumulate_token_repr:
                if token_repr is None: token_repr = torch.zeros_like(token_a)
                with torch.set_grad_enabled(train_accumulate_token_repr):
                    t_tensor = torch.full((current_coords.shape[0],), t_hat, device=device)
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(t_tensor), acc_a=token_repr, next_a=token_a,
                    )
            elif step_i == len(reverse_schedule) - 1:
                token_repr = token_a

        # --- End Reverse Diffusion Loop ---

        return {"sample_atom_coords": current_coords, "diff_token_repr": token_repr}




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



    # ------------------------------------------------------
    # Hard-Constraint: re-rotate each subunit => R_i * reference
    # ------------------------------------------------------
    def apply_symmetry_constraints_rigid(
        self,
        coords: torch.Tensor,
        subunits: list[torch.Tensor],
        rot_mats: torch.Tensor,  # still available for legacy reasons
        desired_com: torch.Tensor,
    ) -> torch.Tensor:
        device = coords.device
        B = coords.shape[0]
        if not subunits:  # Handle the case of no subunits.
            return coords

        out_coords = coords.clone()
        # Re-center the reference subunit (subunits[0]) to desired_com.
        ref_inds = subunits[0]
        for b_idx in range(B):
            ref_coords = out_coords[b_idx, ref_inds, :]
            com_ref = ref_coords.mean(dim=0, keepdim=True)
            shift = desired_com[b_idx] - com_ref
            ref_coords = ref_coords + shift
            out_coords[b_idx, ref_inds, :] = ref_coords

            # Now for each additional subunit, rotate the re-centered reference.
            for i_sub in range(1, len(subunits)):
                target_inds = subunits[i_sub]
                #Crucial Change
                if self.rot_mats_noI is not None:
                    rot_idx = i_sub - 1  # we assume the identity is excluded for subunit 0
                    R = self.rot_mats_noI[rot_idx].to(device) #crucial change
                else:
                    R = torch.eye(3, device=device)
                rotated = rotate_coords_about_origin(ref_coords, R)
                out_coords[b_idx, target_inds, :] = rotated

        return out_coords





    # ------------------------------------------------------
    # symmetrical atom mapping
    # ------------------------------------------------------
    def get_symmetrical_atom_mapping(self, feats: dict[str, torch.Tensor]) -> dict[int, list[int]]:
        """
        e.g. {0: [atom0_sub0, atom0_sub1, ...], 1: [atom1_sub0, atom1_sub1, ...], ...}
        so we can rotate the reference subunit's vectors => others
        """
        device = self.device
        subunits = symmetry.get_subunit_atom_indices(
            self.symmetry_type,
            self.chain_symmetry_groups,
            feats,
            device,
        )
        if not subunits: #crucial change. 
            return {}

        if len(subunits) < 2:
            # If only one subunit, map each atom index to itself in a list.
            return {i: [i] for i in range(len(subunits[0]))} #crucial change

        ref_sub = subunits[0]
        n_atoms_ref = len(ref_sub)
        mapping = {i: [ref_sub[i].item()] for i in range(n_atoms_ref)}

        for s_idx in range(1, len(subunits)):
            s_ = subunits[s_idx]
            if len(s_) != n_atoms_ref:
                continue
            for i in range(n_atoms_ref):
                mapping[i].append(s_[i].item())

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


def _get_I_C3_atom_mapping(self, subunits: list[torch.Tensor]) -> dict[int, list[int]]:
    """
    Creates a mapping from atom indices in the first subunit (A) to the
    corresponding atom indices in all six subunits (A, B, C, D, E, F).
    Assumes all subunits have the same number of atoms and are ordered A, B, C, D, E, F.

    Args:
        subunits: List of 6 tensors, where each tensor contains the atom indices for one subunit.

    Returns:
        A dictionary {atom_idx_A: [atom_idx_A, atom_idx_B, ..., atom_idx_F]}.
    """
    if len(subunits) != 6:
        raise ValueError(f"Expected 6 subunits for I/C3 mapping, got {len(subunits)}")

    ref_subunit_indices = subunits[0]
    n_atoms_per_subunit = len(ref_subunit_indices)

    # Basic check: ensure all subunits have the same size
    for i in range(1, 6):
        if len(subunits[i]) != n_atoms_per_subunit:
            # Allow for slight variations if masking is involved, but warn.
            # Strict equality might be too brittle if terminal residues differ slightly.
            # Let's assume for now they MUST be equal for this mapping.
            raise ValueError(f"Subunits must have the same number of atoms for I/C3 mapping. "
                             f"Subunit 0 has {n_atoms_per_subunit}, Subunit {i} has {len(subunits[i])}")

    mapping = {}
    for i in range(n_atoms_per_subunit):
        atom_idx_A = ref_subunit_indices[i].item()
        corresponding_atoms = [atom_idx_A] # Start with A
        for sub_idx in range(1, 6):
            corresponding_atoms.append(subunits[sub_idx][i].item())
        mapping[atom_idx_A] = corresponding_atoms

    return mapping
