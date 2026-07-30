import torch
from torch import nn, Tensor


class TabMModel(nn.Module):
    """TabM 表格模型的模块化封装"""

    def __init__(self, d_in: int, d_out: int, train_x: Tensor):
        super().__init__()
        # 1. 依赖检查
        try:
            import rtdl_num_embeddings
            import tabm
        except Exception as exc:
            raise RuntimeError(
                "TabM dependencies are missing. Install tabm and rtdl-num-embeddings, "
                "or remove 'tabm' from --models for a first run."
            ) from exc

        # 2. 计算分箱边界（仅依赖训练数据）
        bins = rtdl_num_embeddings.compute_bins(train_x, n_bins=512)

        # 3. 构建数值嵌入层（作为子模块注册）
        self.num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins,
            d_embedding=32,
            activation=False,
            version="B",
        )

        # 4. 构建 TabM 核心网络（作为子模块注册）
        self.tabm = tabm.TabM.make(
            n_num_features=d_in,
            cat_cardinalities=[],
            d_out=d_out,
            num_embeddings=self.num_embeddings,
            n_blocks=3,
            d_block=640,
            dropout=0.0,
            k=8,
        )

        # 5. 保存超参数（方便后续恢复/记录）
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x: Tensor) -> Tensor:
        return self.tabm(x)