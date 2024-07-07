from transformers import PretrainedConfig

class VisionMambaConfig(PretrainedConfig):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=192,
        depth=24,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        use_final_norm=True,
        final_pool_type='mean',
        if_abs_pos_embed=True,
        if_cls_token=True,
        use_middle_cls_token=True,
        drop_rate=0.,
        drop_path_rate=0.1,
        if_rope=False,
        if_rope_residual=False,
        flip_img_sequences_ratio=0.0,
        pt_hw_seq_len=14,
        bimamba_type="none",
        if_use_residual=True,
        residual_in_fp32=False,
        fused_add_norm=False,
        use_fast_path=True,
        if_devide_out=False,
        init_layer_scale=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.dt_rank = dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias
        self.use_final_norm = use_final_norm
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_cls_token = if_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.pt_hw_seq_len = pt_hw_seq_len
        self.bimamba_type = bimamba_type
        self.if_use_residual = if_use_residual
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_fast_path = use_fast_path
        self.if_devide_out = if_devide_out
        self.init_layer_scale = init_layer_scale