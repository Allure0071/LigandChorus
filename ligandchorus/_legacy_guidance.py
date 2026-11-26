"""
E18 配体引导的条件流匹配（Ligand-Guided Flow Matching）

核心创新：配体信息从"被动条件"变为"主动引导信号"

技术路线：
- ADFLIP: p(s_1|s_t, structure, ligand) ← 配体是静态输入
- 我们:   p(s_1|s_t, structure) × guidance(ligand, s_t) ← 配体动态引导

三个核心机制：
1. 配体注意力引导：计算配体-残基注意力权重，调整采样概率
2. 界面加权速率矩阵：配体接触残基的转移速率更高（优先生成）
3. 动态温度调度：口袋附近低温（精确），远处高温（多样）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# 氨基酸化学性质表
# ============================================================================

# 氨基酸索引（与 LigandMPNN 一致）
# A=0, R=1, N=2, D=3, C=4, Q=5, E=6, G=7, H=8, I=9,
# L=10, K=11, M=12, F=13, P=14, S=15, T=16, W=17, Y=18, V=19, mask=20

# 疏水性（Kyte-Doolittle 疏水指数，归一化到 [0, 1]）
HYDROPHOBICITY = torch.tensor([
    0.62,  # A (Ala) - 1.8
    0.00,  # R (Arg) - -4.5
    0.13,  # N (Asn) - -3.5
    0.13,  # D (Asp) - -3.5
    0.85,  # C (Cys) - 2.5
    0.13,  # Q (Gln) - -3.5
    0.13,  # E (Glu) - -3.5
    0.29,  # G (Gly) - -0.4
    0.17,  # H (His) - -3.2
    1.00,  # I (Ile) - 4.5
    0.95,  # L (Leu) - 3.8
    0.11,  # K (Lys) - -3.9
    0.76,  # M (Met) - 1.9
    0.94,  # F (Phe) - 2.8
    0.67,  # P (Pro) - -1.6
    0.24,  # S (Ser) - -0.8
    0.22,  # T (Thr) - -0.7
    0.56,  # W (Trp) - -0.9
    0.42,  # Y (Tyr) - -1.3
    0.90,  # V (Val) - 4.2
    0.50,  # mask
], dtype=torch.float32)

# 电荷（-1: 负电, 0: 中性, +1: 正电）
CHARGE = torch.tensor([
    0,   # A
    1,   # R (Arg) - 正电
    0,   # N
    -1,  # D (Asp) - 负电
    0,   # C
    0,   # Q
    -1,  # E (Glu) - 负电
    0,   # G
    0.5, # H (His) - 弱正电
    0,   # I
    0,   # L
    1,   # K (Lys) - 正电
    0,   # M
    0,   # F
    0,   # P
    0,   # S
    0,   # T
    0,   # W
    0,   # Y
    0,   # V
    0,   # mask
], dtype=torch.float32)

# 芳香性（0: 非芳香, 1: 芳香）
AROMATICITY = torch.tensor([
    0, 0, 0, 0, 0, 0, 0, 0,
    1,  # H (His) - 芳香
    0, 0, 0, 0,
    1,  # F (Phe) - 芳香
    0, 0, 0,
    1,  # W (Trp) - 芳香
    1,  # Y (Tyr) - 芳香
    0, 0,
], dtype=torch.float32)

# 氢键供体能力（0-2）
HBOND_DONOR = torch.tensor([
    0,  # A
    2,  # R (Arg) - 多个供体
    1,  # N (Asn) - 酰胺
    0,  # D
    1,  # C (Cys) - 巯基
    1,  # Q (Gln) - 酰胺
    0,  # E
    0,  # G
    1,  # H (His) - 咪唑
    0, 0,  # I, L
    2,  # K (Lys) - 氨基
    0, 0, 0,  # M, F, P
    1,  # S (Ser) - 羟基
    1,  # T (Thr) - 羟基
    1,  # W (Trp) - 吲哚
    1,  # Y (Tyr) - 酚羟基
    0, 0,  # V, mask
], dtype=torch.float32)

# 氢键受体能力（0-2）
HBOND_ACCEPTOR = torch.tensor([
    0,  # A
    0,  # R
    1,  # N (Asn) - 酰胺
    2,  # D (Asp) - 羧基
    1,  # C (Cys) - 巯基
    1,  # Q (Gln) - 酰胺
    2,  # E (Glu) - 羧基
    0,  # G
    1,  # H (His) - 咪唑
    0, 0, 0,  # I, L, K
    1,  # M (Met) - 硫醚
    0, 0,  # F, P
    1,  # S (Ser) - 羟基
    1,  # T (Thr) - 羟基
    1,  # W (Trp) - 吲哚
    1,  # Y (Tyr) - 酚羟基
    0, 0,  # V, mask
], dtype=torch.float32)


# ============================================================================
# 配体注意力引导
# ============================================================================

class LigandAttentionGuidance(nn.Module):
    """
    配体注意力引导模块

    计算每个残基与配体的注意力权重，用于调整采样概率
    """

    def __init__(self, hidden_dim=128, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 配体-残基注意力
        self.ligand_query = nn.Linear(hidden_dim, hidden_dim)
        self.residue_key = nn.Linear(hidden_dim, hidden_dim)
        self.attention_out = nn.Linear(hidden_dim, hidden_dim)

        # 距离嵌入
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )

    def forward(self, node_features, X, Y, Y_m):
        """
        计算配体-残基注意力权重

        Args:
            node_features: [B, L, hidden_dim] 残基节点特征
            X: [B, L, 4, 3] 骨架原子坐标
            Y: [B, L, num_ligand_atoms, 3] 配体原子坐标
            Y_m: [B, L, num_ligand_atoms] 配体原子掩码

        Returns:
            attention_weights: [B, L] 配体注意力权重（0-1）
            ligand_context: [B, L, hidden_dim] 配体上下文特征
        """
        B, L, hidden_dim = node_features.shape
        device = node_features.device

        # 使用 Cα 坐标
        Ca = X[:, :, 1, :]  # [B, L, 3]

        # 计算残基到配体的距离
        distances = torch.cdist(Ca.unsqueeze(2), Y).squeeze(2)  # [B, L, num_atoms]
        distances = distances + (1 - Y_m) * 1e6
        min_distances = distances.min(dim=-1)[0]  # [B, L]

        # 距离嵌入
        dist_emb = self.distance_embedding(min_distances.unsqueeze(-1))  # [B, L, hidden_dim//4]

        # 计算注意力权重（基于距离）
        # 距离越近，注意力权重越大
        attention_weights = torch.exp(-min_distances / 5.0)  # [B, L]
        attention_weights = torch.clamp(attention_weights, 0, 1)

        # 配体上下文特征（简化版：使用距离嵌入）
        ligand_context = torch.zeros(B, L, hidden_dim, device=device)
        ligand_context[:, :, :hidden_dim//4] = dist_emb

        return attention_weights, ligand_context


# ============================================================================
# 化学兼容性引导
# ============================================================================

class ChemicalCompatibilityGuidance(nn.Module):
    """
    化学兼容性引导模块

    基于氨基酸的化学性质与配体的相互作用，调整采样概率
    """

    def __init__(self, vocab_size=21):
        super().__init__()
        self.vocab_size = vocab_size

        # 注册化学性质表为 buffer
        self.register_buffer('hydrophobicity', HYDROPHOBICITY)
        self.register_buffer('charge', CHARGE)
        self.register_buffer('aromaticity', AROMATICITY)
        self.register_buffer('hbond_donor', HBOND_DONOR)
        self.register_buffer('hbond_acceptor', HBOND_ACCEPTOR)

        # 配体性质估计（可学习的参数）
        self.ligand_hydrophobicity = nn.Parameter(torch.tensor(0.5))
        self.ligand_charge = nn.Parameter(torch.tensor(0.0))
        self.ligand_aromaticity = nn.Parameter(torch.tensor(0.5))

    def forward(self, X, Y, Y_m, Y_t):
        """
        计算化学兼容性分数

        Args:
            X: [B, L, 4, 3] 骨架原子坐标
            Y: [B, L, num_ligand_atoms, 3] 配体原子坐标
            Y_m: [B, L, num_ligand_atoms] 配体原子掩码
            Y_t: [B, L, num_ligand_atoms] 配体原子类型

        Returns:
            compatibility_scores: [B, L, vocab_size] 每个位置每种氨基酸的兼容性分数
        """
        B, L = X.shape[:2]
        device = X.device

        # 计算残基到配体的距离
        Ca = X[:, :, 1, :]  # [B, L, 3]
        distances = torch.cdist(Ca.unsqueeze(2), Y).squeeze(2)  # [B, L, num_atoms]
        distances = distances + (1 - Y_m) * 1e6
        min_distances = distances.min(dim=-1)[0]  # [B, L]

        # 距离权重（距离越近，化学兼容性越重要）
        distance_weights = torch.exp(-min_distances / 5.0)  # [B, L]

        # ========== 1. 疏水兼容性 ==========
        # 如果配体疏水，则疏水氨基酸得分高
        hydrophobic_compat = -torch.abs(
            self.hydrophobicity.unsqueeze(0).unsqueeze(0) - self.ligand_hydrophobicity
        )  # [1, 1, vocab_size]

        # ========== 2. 电荷兼容性 ==========
        # 相反电荷相互吸引
        charge_compat = -self.charge.unsqueeze(0).unsqueeze(0) * self.ligand_charge  # [1, 1, vocab_size]

        # ========== 3. 芳香兼容性 ==========
        # 芳香-芳香相互作用（π-π 堆积）
        aromatic_compat = self.aromaticity.unsqueeze(0).unsqueeze(0) * self.ligand_aromaticity  # [1, 1, vocab_size]

        # ========== 4. 氢键兼容性 ==========
        # 供体-受体互补
        hbond_compat = (self.hbond_donor + self.hbond_acceptor).unsqueeze(0).unsqueeze(0) * 0.5  # [1, 1, vocab_size]

        # ========== 综合兼容性分数 ==========
        compatibility = (
            hydrophobic_compat * 0.3 +
            charge_compat * 0.3 +
            aromatic_compat * 0.2 +
            hbond_compat * 0.2
        )  # [1, 1, vocab_size]

        # 扩展到所有位置
        compatibility = compatibility.expand(B, L, -1)  # [B, L, vocab_size]

        # 应用距离权重
        compatibility = compatibility * distance_weights.unsqueeze(-1)  # [B, L, vocab_size]

        return compatibility


# ============================================================================
# 动态温度调度
# ============================================================================

class DynamicTemperatureScheduler(nn.Module):
    """
    动态温度调度模块

    根据残基与配体的距离，动态调整采样温度：
    - 口袋附近（<5Å）：低温（0.5），精确采样
    - 近界面（5-10Å）：正常温度（1.0）
    - 远界面（>10Å）：高温（1.5），增加多样性
    """

    def __init__(self, base_temperature=1.0):
        super().__init__()
        self.base_temperature = base_temperature

    def forward(self, X, Y, Y_m):
        """
        计算每个位置的采样温度

        Args:
            X: [B, L, 4, 3] 骨架原子坐标
            Y: [B, L, num_ligand_atoms, 3] 配体原子坐标
            Y_m: [B, L, num_ligand_atoms] 配体原子掩码

        Returns:
            temperatures: [B, L] 每个位置的采样温度
        """
        B, L = X.shape[:2]
        device = X.device

        # 计算残基到配体的距离
        Ca = X[:, :, 1, :]  # [B, L, 3]
        distances = torch.cdist(Ca.unsqueeze(2), Y).squeeze(2)  # [B, L, num_atoms]
        distances = distances + (1 - Y_m) * 1e6
        min_distances = distances.min(dim=-1)[0]  # [B, L]

        # 动态温度调度
        temperatures = torch.ones(B, L, device=device) * self.base_temperature

        # 口袋附近（<5Å）：低温 0.5
        pocket_mask = min_distances < 5.0
        temperatures = torch.where(pocket_mask, temperatures * 0.5, temperatures)

        # 近界面（5-10Å）：正常温度 1.0（保持不变）

        # 远界面（>10Å）：高温 1.5
        far_mask = min_distances > 10.0
        temperatures = torch.where(far_mask, temperatures * 1.5, temperatures)

        return temperatures


# ============================================================================
# 配体引导的采样器
# ============================================================================

class LigandGuidedSampler(nn.Module):
    """
    配体引导的采样器

    在 Euler 采样的每一步，使用三种引导机制：
    1. 配体注意力引导：调整采样概率
    2. 化学兼容性引导：调整采样概率
    3. 动态温度调度：调整采样温度
    """

    def __init__(self, hidden_dim=128, vocab_size=21,
                 use_attention=True, use_chemical=True, use_temperature=True,
                 guidance_scale=1.0, localize_guidance=False, interface_threshold=5.0):
        super().__init__()

        self.vocab_size = vocab_size
        self.use_attention = use_attention
        self.use_chemical = use_chemical
        self.use_temperature = use_temperature
        self.guidance_scale = guidance_scale
        # 方案 B：仅界面残基受引导，非界面严格退化为无引导采样
        self.localize_guidance = localize_guidance
        self.interface_threshold = interface_threshold

        # 三种引导模块
        if use_attention:
            self.attention_guidance = LigandAttentionGuidance(hidden_dim)
        if use_chemical:
            self.chemical_guidance = ChemicalCompatibilityGuidance(vocab_size)
        if use_temperature:
            self.temperature_scheduler = DynamicTemperatureScheduler()

    def forward(self, logits, feature_dict, s_t, t):
        """
        应用配体引导，调整采样概率

        Args:
            logits: [B, L, vocab_size] 模型输出的 logits
            feature_dict: LigandMPNN 特征字典
            s_t: [B, L] 当前掩码序列
            t: [B] 当前时间步

        Returns:
            guided_probs: [B, L, vocab_size] 引导后的采样概率
        """
        B, L, V = logits.shape
        device = logits.device

        X = feature_dict['X']
        Y = feature_dict['Y']
        Y_m = feature_dict['Y_m']
        Y_t = feature_dict['Y_t']

        # 在同一 logits 上依次组合三种引导，避免后续分支覆盖前面的效果
        guided_logits = logits

        # ========== 1. 配体注意力引导 ==========
        if self.use_attention:
            # 计算注意力权重
            node_features = torch.zeros(B, L, 128, device=device)  # 简化版
            attention_weights, _ = self.attention_guidance(node_features, X, Y, Y_m)  # [B, L]

            # 调整概率：注意力权重大的位置，概率分布更尖锐（更确定）
            # 使用注意力权重调整 logits 的温度
            attention_scale = 1.0 + attention_weights.unsqueeze(-1) * self.guidance_scale  # [B, L, 1]
            guided_logits = guided_logits * attention_scale

        # ========== 2. 化学兼容性引导 ==========
        if self.use_chemical:
            # 计算化学兼容性分数
            compatibility = self.chemical_guidance(X, Y, Y_m, Y_t)  # [B, L, vocab_size]

            # 将兼容性分数加到 logits 上
            guided_logits = guided_logits + compatibility * self.guidance_scale

        # ========== 3. 动态温度调度 ==========
        if self.use_temperature:
            # guidance_scale=0 时温度应为 1，保证完全退化为无引导采样
            temperatures = self.temperature_scheduler(X, Y, Y_m)  # [B, L]
            temperatures = 1.0 + (temperatures - 1.0) * self.guidance_scale
            temperatures = temperatures.clamp(min=0.1)
            guided_logits = guided_logits / temperatures.unsqueeze(-1)

        # ========== 方案 B：引导局部化 ==========
        if self.localize_guidance:
            # 非界面位置严格回退到原始 logits，界面位置保留全部引导
            interface_mask = self._interface_mask(X, Y, Y_m)  # [B, L]
            guided_logits = logits + interface_mask.unsqueeze(-1) * (guided_logits - logits)

        return F.softmax(guided_logits, dim=-1)

    def _interface_mask(self, X, Y, Y_m):
        """基于 Cα 到配体原子的最小距离判定界面残基"""
        B = X.shape[0]
        Ca = X[:, :, 1, :]                     # [B, L, 3]
        Y_flat = Y.reshape(B, -1, 3)           # [B, L*A, 3]
        Y_m_flat = Y_m.reshape(B, -1)          # [B, L*A]
        d = torch.cdist(Ca, Y_flat)            # [B, L, L*A]
        d = d + (1.0 - Y_m_flat).unsqueeze(1) * 1e6
        min_d = d.min(dim=-1)[0]               # [B, L]
        return (min_d < self.interface_threshold).float()


# ============================================================================
# 单元测试
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("E18 配体引导的采样策略 - 单元测试")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    B, L, V = 2, 50, 21
    num_ligand_atoms = 25

    # 准备测试数据
    X = torch.randn(B, L, 4, 3, device=device)
    Y = torch.randn(B, L, num_ligand_atoms, 3, device=device)
    Y_m = torch.ones(B, L, num_ligand_atoms, device=device)
    Y_t = torch.randint(0, 10, (B, L, num_ligand_atoms), device=device)
    node_features = torch.randn(B, L, 128, device=device)
    logits = torch.randn(B, L, V, device=device)
    s_t = torch.randint(0, 20, (B, L), device=device)
    t = torch.rand(B, device=device)

    feature_dict = {'X': X, 'Y': Y, 'Y_m': Y_m, 'Y_t': Y_t}

    # 测试 1: 配体注意力引导
    print("测试 1: 配体注意力引导")
    attention_guidance = LigandAttentionGuidance(hidden_dim=128).to(device)
    attention_weights, ligand_context = attention_guidance(node_features, X, Y, Y_m)
    print(f"  注意力权重: {attention_weights.shape}")
    print(f"  权重范围: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
    print(f"  配体上下文: {ligand_context.shape}")
    assert attention_weights.shape == (B, L), "注意力权重形状错误"
    assert (attention_weights >= 0).all() and (attention_weights <= 1).all(), "注意力权重应在 [0,1]"
    print("  ✓ 通过\n")

    # 测试 2: 化学兼容性引导
    print("测试 2: 化学兼容性引导")
    chemical_guidance = ChemicalCompatibilityGuidance(vocab_size=V).to(device)
    compatibility = chemical_guidance(X, Y, Y_m, Y_t)
    print(f"  兼容性分数: {compatibility.shape}")
    print(f"  分数范围: [{compatibility.min():.4f}, {compatibility.max():.4f}]")
    assert compatibility.shape == (B, L, V), "兼容性分数形状错误"
    print("  ✓ 通过\n")

    # 测试 3: 动态温度调度
    print("测试 3: 动态温度调度")
    temp_scheduler = DynamicTemperatureScheduler().to(device)
    temperatures = temp_scheduler(X, Y, Y_m)
    print(f"  温度: {temperatures.shape}")
    print(f"  温度范围: [{temperatures.min():.4f}, {temperatures.max():.4f}]")
    assert temperatures.shape == (B, L), "温度形状错误"
    assert (temperatures > 0).all(), "温度应大于 0"
    print("  ✓ 通过\n")

    # 测试 4: 配体引导的采样器
    print("测试 4: 配体引导的采样器")
    sampler = LigandGuidedSampler(hidden_dim=128, vocab_size=V,
                                   use_attention=True, use_chemical=True,
                                   use_temperature=True, guidance_scale=1.0).to(device)
    guided_probs = sampler(logits, feature_dict, s_t, t)
    print(f"  引导后概率: {guided_probs.shape}")
    print(f"  概率和: {guided_probs.sum(dim=-1).mean():.4f}（应接近 1.0）")
    assert guided_probs.shape == (B, L, V), "引导后概率形状错误"
    assert torch.allclose(guided_probs.sum(dim=-1), torch.ones(B, L, device=device), atol=1e-5), "概率和应为 1"

    neutral_sampler = LigandGuidedSampler(
        hidden_dim=128, vocab_size=V, use_attention=True, use_chemical=True,
        use_temperature=True, guidance_scale=0.0
    ).to(device)
    neutral_probs = neutral_sampler(logits, feature_dict, s_t, t)
    assert torch.allclose(neutral_probs, F.softmax(logits, dim=-1), atol=1e-6), "零引导强度应退化为基础概率"
    print("  ✓ 通过\n")

    # 测试 5: 对比有/无引导的差异
    print("测试 5: 对比有/无引导的差异")
    base_probs = F.softmax(logits, dim=-1)
    diff = (guided_probs - base_probs).abs().mean()
    print(f"  平均概率差异: {diff:.6f}")
    print(f"  最大概率差异: {(guided_probs - base_probs).abs().max():.6f}")
    assert diff > 0, "引导应该改变概率分布"
    print("  ✓ 通过\n")

    # 测试 6: 消融实验（单独使用每种引导）
    print("测试 6: 消融实验")
    for name, use_att, use_chem, use_temp in [
        ("仅注意力", True, False, False),
        ("仅化学", False, True, False),
        ("仅温度", False, False, True),
    ]:
        sampler_ablation = LigandGuidedSampler(
            hidden_dim=128, vocab_size=V,
            use_attention=use_att, use_chemical=use_chem,
            use_temperature=use_temp, guidance_scale=1.0
        ).to(device)
        probs_ablation = sampler_ablation(logits, feature_dict, s_t, t)
        diff_ablation = (probs_ablation - base_probs).abs().mean()
        print(f"  {name}: 平均概率差异 = {diff_ablation:.6f}")
    print("  ✓ 通过\n")

    # 测试 7: 方案 B 引导局部化
    print("测试 7: 方案 B 引导局部化")
    # 构造一维排布：Cα 沿 x 轴间隔 1 单位，配体原子置于原点，只有前几个残基属于界面
    X7 = torch.zeros(B, L, 4, 3, device=device)
    X7[:, :, 1, 0] = torch.arange(L, device=device).float().unsqueeze(0)
    Y7 = torch.zeros(B, L, num_ligand_atoms, 3, device=device)
    Y_m7 = torch.ones(B, L, num_ligand_atoms, device=device)
    Y_t7 = torch.zeros(B, L, num_ligand_atoms, dtype=torch.long, device=device)
    feature_dict7 = {'X': X7, 'Y': Y7, 'Y_m': Y_m7, 'Y_t': Y_t7}

    localized_sampler = LigandGuidedSampler(
        hidden_dim=128, vocab_size=V,
        use_attention=True, use_chemical=True, use_temperature=True,
        guidance_scale=1.0, localize_guidance=True, interface_threshold=5.0
    ).to(device)
    localized_probs = localized_sampler(logits, feature_dict7, s_t, t)
    iface = localized_sampler._interface_mask(X7, Y7, Y_m7)  # [B, L]
    base = F.softmax(logits, dim=-1)
    non_iface = (iface == 0)
    assert iface.any() and non_iface.any(), "构造数据应同时含界面与非界面残基"
    # 非界面位置必须严格等于无引导概率
    assert torch.allclose(localized_probs[non_iface], base[non_iface], atol=1e-6), "非界面位置应严格无引导"
    # 界面位置应与全局引导一致（同强度）
    global_sampler = LigandGuidedSampler(
        hidden_dim=128, vocab_size=V,
        use_attention=True, use_chemical=True, use_temperature=True,
        guidance_scale=1.0, localize_guidance=False
    ).to(device)
    global_probs = global_sampler(logits, feature_dict7, s_t, t)
    assert torch.allclose(localized_probs[iface.bool()], global_probs[iface.bool()], atol=1e-6), "界面位置应保留完整引导"
    print(f"  界面残基数: {int(iface.sum().item())}/{B*L}")
    print("  ✓ 通过\n")

    print("=" * 60)
    print("✓ 所有单元测试通过！")
    print("=" * 60)
    print("\n核心创新点：")
    print("1. 配体注意力引导：距离越近，采样越确定")
    print("2. 化学兼容性引导：疏水/电荷/芳香/氢键互补")
    print("3. 动态温度调度：口袋低温（精确），远处高温（多样）")
