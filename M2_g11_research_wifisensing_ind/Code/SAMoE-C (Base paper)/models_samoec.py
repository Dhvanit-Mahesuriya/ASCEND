"""
Model definitions for SAMoE-C (Scene-Adaptive Mixture of Experts with
Clustered Specialists), following Sec. III and Table I of the paper.

ENHANCED with:
  - SSLInformedRouter: hybrid gate + SSL prototype-based routing (NOVEL)
  - get_intermediate_features: specialist feature extraction for SSL-anchored
    knowledge distillation (NOVEL)

Data flow (per Table I):
  X (B,C,T,S) --reshape--> (B*T,C,S) --Shared Backbone--> Z (B,T,256,29)
  Z --GAP over S--> (B,T,256) --Temporal Attention--> c (B,256)
  c --Semantic Router (SSLInformedRouter)--> p (B,N) --argmax--> k*
  Z --Specialist_{k*} (conv + BiGRU)--> logits (B,K)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Basic building block: residual Conv1D block (used both in the shared
# backbone -- stride 1 -- and in the specialists -- stride 2 for downsampling)
# --------------------------------------------------------------------------- #
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


# --------------------------------------------------------------------------- #
# Part A: Shared feature extraction backbone Phi_shared (Table I, rows 1-7)
# --------------------------------------------------------------------------- #
class SharedBackbone(nn.Module):
    """
    Input : X of shape (B, C, T, S)   [C=3, T=10, S=114 by default]
    Output: Z of shape (B, T, 256, S') with S' ~ 29 for the default config.
    The same Conv1D stack is applied independently (weight-shared) to each of
    the T timesteps, mirroring the paper's reshape to (T, C, S) in Table I.
    """
    def __init__(self, in_channels: int, channels=(64, 96, 144, 256)):
        super().__init__()
        stem_out = channels[0]
        # Stem: Conv1D(7x7) & MaxPool1D  ->  (C,S) -> (stem_out, S/4)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_out, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stem_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        blocks = []
        in_ch = stem_out
        for ch in channels:
            blocks.append(ConvBlock1D(in_ch, ch, stride=1))
            in_ch = ch
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels[-1]

    def forward(self, x):
        B, C, T, S = x.shape
        # (B,C,T,S) -> (B,T,C,S) -> (B*T,C,S)   [Table I row 2 reshape]
        x = x.permute(0, 2, 1, 3).contiguous().view(B * T, C, S)
        x = self.stem(x)
        x = self.blocks(x)                      # (B*T, 256, S')
        S_out = x.shape[-1]
        Z = x.view(B, T, self.out_channels, S_out)  # (B,T,256,S')
        return Z


# --------------------------------------------------------------------------- #
# Temporal Additive Attention (Sec. III-A, Eqs. 4-5)
# --------------------------------------------------------------------------- #
class TemporalAttention(nn.Module):
    """
    Input : Z_pool of shape (B, T, d)   [spatially global-avg-pooled features]
    Output: context vector c of shape (B, d), attention weights (B, T)
    """
    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.W = nn.Linear(d, hidden, bias=True)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, z):
        # e_t = v^T tanh(W z_t + b)
        e = self.v(torch.tanh(self.W(z))).squeeze(-1)   # (B,T)
        alpha = F.softmax(e, dim=-1)                      # (B,T)
        c = torch.einsum("bt,btd->bd", alpha, z)          # (B,d)
        return c, alpha


# --------------------------------------------------------------------------- #
# Part B: Semantic Router -- Residual Gating Network (ResGateNet), Eqs. 6-7
# --------------------------------------------------------------------------- #
class GateBlock(nn.Module):
    """A single residual-gating unit: y = x + gate(x) * transform(x)."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        f = self.fc2(self.act(self.fc1(x)))
        g = torch.sigmoid(self.gate(x))
        return x + g * f


class ResGateNet(nn.Module):
    """Semantic router G(.): context vector c -> logits over N domains."""
    def __init__(self, dim: int, n_domains: int, n_blocks: int = 3):
        super().__init__()
        self.blocks = nn.Sequential(*[GateBlock(dim) for _ in range(n_blocks)])
        self.classifier = nn.Linear(dim, n_domains)

    def forward(self, c):
        h = self.blocks(c)
        return self.classifier(h)   # raw logits; apply softmax outside if needed

    def add_domain(self):
        """Expand the router's output layer by one when a new domain arrives."""
        old = self.classifier
        new = nn.Linear(old.in_features, old.out_features + 1).to(old.weight.device)
        with torch.no_grad():
            new.weight[: old.out_features] = old.weight
            new.bias[: old.out_features] = old.bias
        self.classifier = new


# --------------------------------------------------------------------------- #
# SSL-Informed Router: combines learned gate routing with SSL prototype-based
# cosine similarity routing (NOVEL COMPONENT — not in SAMoE-C or CIG-MAE)
# --------------------------------------------------------------------------- #
class SSLInformedRouter(nn.Module):
    """
    Hybrid domain router that blends:
      1. Learned gate-based routing (ResGateNet from SAMoE-C)
      2. SSL prototype-based routing (cosine similarity in SSL feature space)

    The blend weight alpha is LEARNED, allowing the model to dynamically
    decide how much to trust each routing signal. SSL prototypes are computed
    as the mean backbone representation of each domain's data and registered
    as non-gradient buffers.

    This creates a direct link between SSL representations and the routing
    mechanism — neither SAMoE-C nor CIG-MAE proposes this.
    """
    def __init__(self, dim: int, n_blocks: int = 3):
        super().__init__()
        self.gate_router = ResGateNet(dim, n_domains=0, n_blocks=n_blocks)
        self.prototypes = nn.ParameterList()  # SSL-space domain prototypes
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learned blend weight
        self.dim = dim

    def add_domain(self):
        """Expand gate router + add a zero-initialized prototype."""
        self.gate_router.add_domain()
        # Prototype will be set later via register_prototype()
        proto = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
        self.prototypes.append(proto)

    @torch.no_grad()
    def register_prototype(self, domain_idx, prototype_vector):
        """Set the SSL prototype for a domain (computed from backbone representations)."""
        self.prototypes[domain_idx].data = prototype_vector.detach().clone()

    def forward(self, context, ssl_features=None):
        """
        Args:
            context: (B, dim) — context vector from temporal attention
            ssl_features: (B, dim) — GAP'd backbone features (optional)

        Returns:
            logits: (B, N_domains) — blended routing logits
        """
        gate_logits = self.gate_router(context)  # (B, N)

        if ssl_features is not None and len(self.prototypes) > 0:
            # Compute cosine similarity to each domain prototype
            proto_stack = torch.stack([p.data for p in self.prototypes], dim=0)  # (N, dim)
            proto_stack = proto_stack.to(ssl_features.device)
            # Normalize for cosine similarity
            ssl_norm = F.normalize(ssl_features, dim=-1)  # (B, dim)
            proto_norm = F.normalize(proto_stack, dim=-1)  # (N, dim)
            proto_sims = ssl_norm @ proto_norm.T  # (B, N)
            # Scale similarities to match gate logit magnitude
            proto_sims = proto_sims * 5.0  # temperature scaling

            # Learned blend
            alpha = torch.sigmoid(self.alpha)
            logits = alpha * gate_logits + (1 - alpha) * proto_sims
        else:
            logits = gate_logits

        return logits


# --------------------------------------------------------------------------- #
# Part C: Activity classification specialist E_k (Table I, rows 9-12)
# --------------------------------------------------------------------------- #
class Specialist(nn.Module):
    """
    Input : Z of shape (B, T, 256, S')  [raw, un-pooled shared features]
    Output: logits of shape (B, K)
    """
    def __init__(self, in_channels: int, num_classes: int,
                 channels=(128, 96, 64), gru_hidden=128, gru_layers=2):
        super().__init__()
        blocks = []
        in_ch = in_channels
        for ch in channels:
            blocks.append(ConvBlock1D(in_ch, ch, stride=2))
            in_ch = ch
        self.conv_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gru = nn.GRU(
            input_size=channels[-1], hidden_size=gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True,
        )
        self.head = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, Z):
        B, T, C, S = Z.shape
        x = Z.view(B * T, C, S)
        x = self.conv_blocks(x)              # (B*T, 64, S'')
        x = self.pool(x).squeeze(-1)         # (B*T, 64)
        x = x.view(B, T, -1)                 # (B, T, 64)
        _, h_n = self.gru(x)                 # h_n: (num_layers*2, B, hidden)
        h_fwd, h_bwd = h_n[-2], h_n[-1]       # last layer, both directions
        z_last = torch.cat([h_fwd, h_bwd], dim=-1)   # (B, 2*hidden)
        return self.head(z_last)

    def get_intermediate_features(self, Z):
        """
        Extract intermediate features BEFORE the classification head.
        Used for SSL-anchored knowledge distillation (NOVEL).

        Returns: (B, 2*gru_hidden) feature vector.
        """
        B, T, C, S = Z.shape
        x = Z.view(B * T, C, S)
        x = self.conv_blocks(x)
        x = self.pool(x).squeeze(-1)
        x = x.view(B, T, -1)
        _, h_n = self.gru(x)
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        return torch.cat([h_fwd, h_bwd], dim=-1)

# --------------------------------------------------------------------------- #
# Domain-Specific Adapter (NOVEL — not in SAMoE-C or CIG-MAE)
# Lightweight bottleneck module that provides domain-specific feature
# refinement WITHOUT modifying the frozen backbone. Each domain gets its
# own adapter, eliminating feature space drift entirely.
# --------------------------------------------------------------------------- #
class DomainAdapter(nn.Module):
    """
    Bottleneck adapter with residual connection.
    Transforms backbone features Z with a small domain-specific perturbation.

    Zero-initialized output projection means the adapter starts as a perfect
    identity function — training gradually moves it away from identity.

    Parameters: channels × bottleneck × 2 ≈ 16K (negligible vs backbone's ~500K)
    """
    def __init__(self, channels=256, bottleneck_dim=32):
        super().__init__()
        self.down = nn.Linear(channels, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, channels)
        # Zero-init so adapter starts as identity (output = input)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, Z):
        """
        Z: (B, T, channels, S') — backbone feature tensor
        Returns: Z_adapted of same shape, with small residual perturbation
        """
        B, T, C, S_prime = Z.shape
        Z_flat = Z.permute(0, 1, 3, 2).contiguous().reshape(-1, C)  # (B*T*S', C)
        delta = self.up(self.act(self.down(Z_flat)))                 # (B*T*S', C)
        Z_adapted = Z_flat + delta                                   # residual
        return Z_adapted.reshape(B, T, S_prime, C).permute(0, 1, 3, 2).contiguous()


# --------------------------------------------------------------------------- #
# Part D: Domain Adversarial Training / Subject Discriminator
# --------------------------------------------------------------------------- #
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class SubjectDiscriminator(nn.Module):
    def __init__(self, in_dim, n_subjects, hidden=128):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_subjects)
        )

    def forward(self, pooled_features, lambda_=1.0):
        self.grl.lambda_ = lambda_
        return self.net(self.grl(pooled_features))


# --------------------------------------------------------------------------- #
# Full SAMoE-C framework (enhanced with SSL-informed routing)
# --------------------------------------------------------------------------- #
class SAMoEC(nn.Module):
    """
    Wraps the shared backbone, temporal attention, semantic router and a
    growing list of specialists. Supports three forward modes:
      - mode="route" : normal deployment - router argmax picks one specialist
      - mode="force" : force a specific specialist index (used for training a
                       given specialist and for evaluating per-domain val acc)
      - mode="all"   : run every specialist and average logits (MEMO baseline)

    Enhanced with SSLInformedRouter for hybrid gate+prototype routing.
    """
    def __init__(self, cfg, use_ssl_router=False, n_subjects=40):
        super().__init__()
        self.cfg = cfg
        self.use_ssl_router = use_ssl_router
        self.backbone = SharedBackbone(cfg.C, cfg.backbone_channels)
        self.attn = TemporalAttention(cfg.backbone_channels[-1], cfg.attn_hidden)

        # Subject Discriminator for DANN (scrubs subject identity from backbone)
        self.subj_discriminator = SubjectDiscriminator(
            in_dim=cfg.backbone_channels[-1], 
            n_subjects=n_subjects
        )

        if use_ssl_router:
            self.router = SSLInformedRouter(cfg.router_dim, n_blocks=cfg.router_gate_blocks)
        else:
            self.router = ResGateNet(cfg.router_dim, n_domains=0, n_blocks=cfg.router_gate_blocks)

        self.specialists = nn.ModuleList()
        self.adapters = nn.ModuleList()  # one adapter per domain

    def add_specialist(self):
        spec = Specialist(
            in_channels=self.cfg.backbone_channels[-1],
            num_classes=self.cfg.K,
            channels=self.cfg.specialist_channels,
            gru_hidden=self.cfg.gru_hidden,
            gru_layers=self.cfg.gru_layers,
        )
        adapter = DomainAdapter(
            channels=self.cfg.backbone_channels[-1],
            bottleneck_dim=self.cfg.adapter_bottleneck_dim,
        )
        device = next(self.parameters()).device
        spec = spec.to(device)
        adapter = adapter.to(device)
        self.specialists.append(spec)
        self.adapters.append(adapter)
        self.router.add_domain()
        self.router.to(device)
        return len(self.specialists) - 1  # index of new specialist

    def shared_forward(self, X):
        """Returns Z (raw, for specialists) and c (context, for router)."""
        Z = self.backbone(X)                       # (B,T,256,S')
        Z_pool = Z.mean(dim=-1)                     # GAP over subcarrier axis -> (B,T,256)
        c, alpha = self.attn(Z_pool)                # (B,256)
        return Z, c, alpha

    def route_logits(self, c, ssl_features=None):
        if self.use_ssl_router and ssl_features is not None:
            return self.router(c, ssl_features)
        elif self.use_ssl_router:
            return self.router(c)
        else:
            return self.router(c)  # (B, N) raw logits

    def forward(self, X, mode="route", specialist_idx=None):
        Z, c, alpha = self.shared_forward(X)
        # Compute SSL features for routing (GAP over T and S')
        ssl_features = Z.mean(dim=(1, 3))  # (B, 256)
        router_logits = self.route_logits(c, ssl_features)

        if mode == "force":
            assert specialist_idx is not None
            Z_adapted = self.adapters[specialist_idx](Z)
            out = self.specialists[specialist_idx](Z_adapted)
            return out, router_logits, alpha

        if mode == "route":
            k_star = router_logits.argmax(dim=-1)   # (B,)
            # Since each sample in a batch may route to a different expert,
            # we compute per-expert outputs only for the experts actually
            # selected in this batch (keeps cost close to "single expert").
            B = X.shape[0]
            out = torch.zeros(B, self.cfg.K, device=X.device, dtype=Z.dtype)
            for k in k_star.unique():
                mask = k_star == k
                Z_adapted = self.adapters[int(k)](Z[mask])
                out[mask] = self.specialists[int(k)](Z_adapted)
            return out, router_logits, k_star

        if mode == "all":
            outs = []
            for adapter, spec in zip(self.adapters, self.specialists):
                Z_adapted = adapter(Z)
                outs.append(spec(Z_adapted))
            out = torch.stack(outs, dim=0).mean(dim=0)  # (N,B,K) -> (B,K)
            return out, router_logits, alpha

        raise ValueError(f"Unknown mode {mode}")

    def forward_adversarial(self, X, lambda_=1.0):
        """
        Forward pass specifically for Domain-Adversarial Training.
        Extracts pooled backbone features and passes them through the GRL and Subject Discriminator.
        """
        Z, c, alpha = self.shared_forward(X)
        # We can use the temporal attention context vector (c) or GAP features
        # as the representation from which we want to scrub subject identity.
        # c has shape (B, 256), perfectly matching the discriminator input.
        subj_logits = self.subj_discriminator(c, lambda_=lambda_)
        return subj_logits

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def freeze_specialist(self, idx):
        for p in self.specialists[idx].parameters():
            p.requires_grad = False

    def get_specialist_features(self, X, specialist_idx):
        """Extract intermediate specialist features for SSL-anchored distillation."""
        Z = self.backbone(X)
        Z_adapted = self.adapters[specialist_idx](Z)
        return self.specialists[specialist_idx].get_intermediate_features(Z_adapted)

    def trainable_params(self, module_names):
        """module_names: subset of {'backbone','attn','router','specialist_i','subj_discriminator'}"""
        params = []
        for name in module_names:
            if name == "backbone":
                params += list(self.backbone.parameters())
            elif name == "attn":
                params += list(self.attn.parameters())
            elif name == "router":
                params += list(self.router.parameters())
            elif name == "subj_discriminator":
                params += list(self.subj_discriminator.parameters())
            elif name.startswith("specialist_"):
                idx = int(name.split("_")[1])
                params += list(self.specialists[idx].parameters())
        return params


# --------------------------------------------------------------------------- #
# Baseline: Basic CL (single specialist, retrained from scratch each domain,
# no router) -- Sec. V-A
# --------------------------------------------------------------------------- #
class BasicCL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = SharedBackbone(cfg.C, cfg.backbone_channels)
        self.specialist = Specialist(
            in_channels=cfg.backbone_channels[-1],
            num_classes=cfg.K,
            channels=cfg.specialist_channels,
            gru_hidden=cfg.gru_hidden,
            gru_layers=cfg.gru_layers,
        )

    def forward(self, X):
        Z = self.backbone(X)
        return self.specialist(Z)
