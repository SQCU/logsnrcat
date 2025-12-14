
class DynamicSpatialBuffer(nn.Module):
    def __init__(self, max_grid_size=16, head_dim=64, device='cuda'):
        super().__init__()
        self.max_grid_size = max_grid_size
        y = torch.arange(max_grid_size, device=device)
        x = torch.arange(max_grid_size, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
        d = torch.cdist(self.coords, self.coords, p=2)
        self.register_buffer('dist_matrix', d)
        self.dim_half = head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim_half, 2, device=device).float() / self.dim_half))
        self.register_buffer('inv_freq', inv_freq)

    def get_rope(self, B, grid_size, base_grid_size=8):
        """
        RoPE with semantic distance invariance for 2D spatial grids.
        
        Principle:
            Two patches that span the same *fraction* of the image
            should have similar relative position encodings, regardless
            of the total image resolution.
        
        Example:
            - At 8×8 grid: Patches 2 units apart span 2/8 = 25% of image
            - At 32×32 grid: Patches 8 units apart span 8/32 = 25% of image
            → Both pairs should have the same RoPE rotation angle
        
        Implementation:
            rotation_angle = θ · distance
            
            For semantic invariance:
            θ_effective = θ_base · (grid_size / base_grid_size)
            
            This makes rotation_angle invariant when distance scales
            proportionally with grid_size.
        
        Why This Matters:
            - Checkerboards: A tile at 8×8 spans ~2 patches; at 32×32 spans ~8 patches.
            With scaling, both are recognized as "same tile" by attention.
            - Torii: Curvature features at 8×8 span ~4 patches; at 32×32 span ~16 patches.
            With scaling, the model learns shape geometry consistently.
        """
        scale = grid_size / base_grid_size
        scaled_base = self.base_freq * scale
        
        inv_freq = 1.0 / (scaled_base ** (
            torch.arange(0, self.dim_half, 2, device=self.inv_freq.device).float() 
            / self.dim_half
        ))
        
        # Get active coordinates
        if grid_size == self.max_grid_size:
            active_coords = self.coords
        else:
            y = torch.arange(grid_size, device=self.inv_freq.device)
            grid_y, grid_x = torch.meshgrid(y, y, indexing='ij')
            active_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
        
        y_pos, x_pos = active_coords[:, 0], active_coords[:, 1]
        freqs_y = torch.einsum('i, j -> i j', y_pos, inv_freq)
        freqs_x = torch.einsum('i, j -> i j', x_pos, inv_freq)
        emb = torch.cat((torch.cat([freqs_y, freqs_x], dim=-1), 
                        torch.cat([freqs_y, freqs_x], dim=-1)), dim=-1)
        return emb.cos()[None], emb.sin()[None]
    def get_rope_spherical(self, angular_distance, bandwidth_L, base_bandwidth_L=16):
        """
        RoPE for SO(3)-equivariant spherical convolutions.
        
        Principle:
            Angular distance on a sphere is resolution-independent.
            However, the *discretization* (spherical harmonic bandwidth L)
            determines the Nyquist frequency.
        
        Example:
            - At L=16: Can resolve features with ~16 lobes
            - At L=64: Can resolve features with ~64 lobes
            → A 4-lobe pattern at L=16 is analogous to a 16-lobe pattern at L=64
        
        Implementation:
            For a feature with k lobes:
            θ_effective = θ_base · (bandwidth_L / base_bandwidth_L)
            
            This makes the rotation period match the *perceptual frequency*
            of features (lobes/harmonics) rather than raw angular distance.
        
        Note:
            Unlike 2D grids, spherical distance is already normalized [0, π].
            The scaling applies to the *frequency content* we can represent,
            not the coordinate system itself.
        """
        scale = bandwidth_L / base_bandwidth_L
        scaled_base = self.base_freq * scale
        
        inv_freq = 1.0 / (scaled_base ** (
            torch.arange(0, self.dim_half, 2).float() / self.dim_half
        ))
        
        # angular_distance ∈ [0, π] (geodesic on sphere)
        freqs = torch.einsum('i, j -> i j', angular_distance, inv_freq)
        return freqs.cos(), freqs.sin()
    def get_rope_graph(self, geodesic_dist, diameter, base_diameter=10):
        """
        RoPE for graph-structured data using geodesic distance.
        
        Principle:
            Nodes that are X% of the graph's diameter apart should
            have similar relative encodings, regardless of graph size.
        
        Example:
            - Small molecule (diameter=5): Nodes 2 hops apart are "distant" (40%)
            - Protein (diameter=50): Nodes 20 hops apart are "distant" (40%)
            → Both pairs should have similar attention affinity
        
        Implementation:
            Normalize geodesic distance by diameter:
            d_semantic = geodesic_dist / diameter
            
            Then scale RoPE:
            θ_effective = θ_base · (diameter / base_diameter)
            
            This makes attention depend on *relative position in the graph's
            structure*, not absolute hop count.
        
        Alternative (For Irregular Graphs):
            Use diffusion distance instead of geodesic:
            d_diffusion = -log(P_t(i → j))
            
            where P_t is the t-step random walk transition matrix.
            This captures "functional distance" (how information flows)
            rather than topological distance (shortest path).
        """
        scale = diameter / base_diameter
        scaled_base = self.base_freq * scale
        
        inv_freq = 1.0 / (scaled_base ** (
            torch.arange(0, self.dim_half, 2).float() / self.dim_half
        ))
        
        # geodesic_dist: [num_nodes, num_nodes] matrix of shortest path lengths
        freqs = torch.einsum('ij, k -> ijk', geodesic_dist.float(), inv_freq)
        return freqs.cos(), freqs.sin()
    def get_rope_char_lm(self, position, seq_length, base_seq_length=512):
        """
        RoPE for character-level language models (no BPE tokenization).
        
        Principle:
            Character-level models lack the "compression" of BPE, which
            naturally adapts to content density. A 10-character word is
            semantically similar whether embedded in a 100-char or 10,000-char
            document.
        
        Problem:
            Standard RoPE uses absolute position → "character 50" has the same
            encoding in a tweet (100 chars) and a novel (100k chars), even though
            its *semantic role* is very different.
        
        Solution:
            Scale RoPE by sequence length:
            θ_effective = θ_base · (seq_length / base_seq_length)
            
            This makes position encoding reflect *relative progress through
            the document*, not raw character index.
        
        Caveat:
            This assumes documents of different lengths have comparable
            *density* of information per character. For highly heterogeneous
            corpora (tweets + books), consider adaptive scaling based on
            local compression ratio (entropy per character).
        
        Why BPE Models Don't Need This:
            BPE tokens are already "semantically normalized" — common patterns
            (words, subwords) become single tokens regardless of context.
            A 100-token BPE sequence and a 10,000-token sequence are directly
            comparable (both represent ~100 and ~10,000 semantic units).
            Character-level models lack this normalization.
        """
        scale = seq_length / base_seq_length
        scaled_base = self.base_freq * scale
        
        inv_freq = 1.0 / (scaled_base ** (
            torch.arange(0, self.dim_half, 2).float() / self.dim_half
        ))
        
        # position: [seq_length] (0, 1, 2, ..., seq_length-1)
        freqs = torch.einsum('i, j -> i j', position.float(), inv_freq)
        return freqs.cos(), freqs.sin()
    """
    def get_mask(self, grid_size, radius=2.5):
        if grid_size == self.max_grid_size: d = self.dist_matrix
        else:
            y = torch.arange(grid_size, device=self.dist_matrix.device)
            grid_y, grid_x = torch.meshgrid(y, y, indexing='ij')
            active_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
            d = torch.cdist(active_coords, active_coords, p=2)
        valid_mask = d < radius
        num_tokens = grid_size * grid_size
        def mask_mod(b, h, q_idx, kv_idx):
            q_safe, k_safe = q_idx.clamp(0, num_tokens - 1), kv_idx.clamp(0, num_tokens - 1)
            return valid_mask[q_safe, k_safe] & (q_idx < num_tokens) & (kv_idx < num_tokens)
        return create_block_mask(mask_mod, B=1, H=1, Q_LEN=num_tokens, KV_LEN=num_tokens)
    """
    def get_mask(self, grid_size, base_radius=2.5, base_grid=8.0):
        """
        Compute attention mask with logarithmically-scaled radius.
        
        Scaling Law:
            radius(G) = base_radius + log2(G / G_base) * scale_factor
        
        Intuition:
            At 8×8:   radius = 2.5 (covers ~20% of spatial extent)
            At 16×16: radius = 4.0 (covers ~12.5% of spatial extent)
            At 32×32: radius = 5.5 (covers ~8.6% of spatial extent)
            
            In absolute terms, the radius grows to capture
            finer details. In relative terms, it shrinks to
            maintain local inductive bias.
        
        Why Logarithmic (not linear, not constant):
            - Linear scaling: O(N²) attention cost, loses local bias
            - Constant radius: Fails to capture multi-tile patterns at high-res
            - Logarithmic: Balances detail capture with computational efficiency
        
        Why This Preserves Perceptual Consistency:
            The number of patches WITHIN a semantic structure
            (e.g., one checkerboard tile) grows linearly with resolution.
            But the number of ADJACENT structures we need to attend to
            grows logarithmically (due to hierarchical spatial decomposition).
        
        Connection to CNNs:
            This mimics how CNN receptive fields grow:
            - Early layers: Small, constant receptive field
            - Deep layers: Exponentially growing receptive field
            
            We implement this spatially (across resolution) rather than
            temporally (across layers).
        """
        # Logarithmic scaling coefficient
        # Tuned so that radius grows ~1.5 patches per doubling of resolution
        log_scale_factor = 1.5
        
        if grid_size <= base_grid:
            effective_radius = base_radius
        else:
            log_ratio = torch.log2(torch.tensor(grid_size / base_grid, dtype=torch.float32))
            effective_radius = base_radius + log_scale_factor * log_ratio.item()
        
        # Compute distance matrix
        if grid_size == self.max_grid_size:
            d = self.dist_matrix
        else:
            y = torch.arange(grid_size, device=self.dist_matrix.device)
            grid_y, grid_x = torch.meshgrid(y, y, indexing='ij')
            active_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
            d = torch.cdist(active_coords, active_coords, p=2)
        
        valid_mask = d < effective_radius
        num_tokens = grid_size * grid_size
        
        def mask_mod(b, h, q_idx, kv_idx):
            """
            Block mask predicate for flex_attention.
            
            Ensures queries only attend to keys within the spatial neighborhood
            defined by effective_radius. This creates a spatially-localized
            attention pattern that scales gracefully across resolutions.
            """
            q_safe = q_idx.clamp(0, num_tokens - 1)
            k_safe = kv_idx.clamp(0, num_tokens - 1)
            return valid_mask[q_safe, k_safe] & (q_idx < num_tokens) & (kv_idx < num_tokens)
        
        return create_block_mask(
            mask_mod, 
            B=1, H=1, 
            Q_LEN=num_tokens, 
            KV_LEN=num_tokens
        )
