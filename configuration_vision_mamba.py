class VisionMambaConfig:
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
        dt_scale="random",
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        use_norm=True,
        use_rot_emb=True,
        kernel_size=4,
        use_final_norm=True,
        final_pool_type='mean',
        if_abs_pos_embed=True,
        if_cls_token=True,
        use_middle_cls_token=True,
        **kwargs
    ):
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
        self.use_norm = use_norm
        self.use_rot_emb = use_rot_emb
        self.kernel_size = kernel_size
        self.use_final_norm = use_final_norm
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_cls_token = if_cls_token
        self.use_middle_cls_token = use_middle_cls_token