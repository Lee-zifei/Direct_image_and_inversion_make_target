from .dt import DeblendingTransformer
from .hydranetv2 import Hydranetv2
from .wudtnet import WUDTnet
from .Dn_CNN import DnCNN
from .restormer import Restormer
from .unet import Unet
from .nakaunet import NakaUnet
from .network_swinir import SwinIR
from .wudt_STAnet import WUDT_STAnet

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'DnCNN':
        model = DnCNN()
    elif model_type == 'Unet':
        model = Unet()
    elif model_type == 'Hydranetv2':
        model = Hydranetv2(img_size=config.DATA.IMG_SIZE,
                           in_chans=config.MODEL.DT.IN_CHANS,
                           num_classes=config.MODEL.NUM_CLASSES,
                           embed_dim=config.MODEL.DT.EMBED_DIM,
                           depths=config.MODEL.DT.DEPTHS,
                           num_heads=config.MODEL.DT.NUM_HEADS,
                           window_size=config.MODEL.DT.WINDOW_SIZE,
                           mlp_ratio=2,
                           qkv_bias=config.MODEL.DT.QKV_BIAS,
                           qk_scale=config.MODEL.DT.QK_SCALE,
                           drop_rate=0.,
                           drop_path_rate=0.1,
                           patch_norm=config.MODEL.DT.PATCH_NORM, config=config)
    elif model_type == 'NakaUnet':
        model = NakaUnet()

    elif model_type == 'WUDTnet':
        model = WUDTnet(img_size=config.DATA.IMG_SIZE,
                        in_chans=config.MODEL.DT.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.DT.EMBED_DIM,
                        depths=config.MODEL.DT.DEPTHS,
                        num_heads=config.MODEL.DT.NUM_HEADS,
                        window_size=config.MODEL.DT.WINDOW_SIZE,
                        mlp_ratio=2,
                        qkv_bias=config.MODEL.DT.QKV_BIAS,
                        qk_scale=config.MODEL.DT.QK_SCALE,
                        drop_rate=0.,
                        drop_path_rate=0.1,
                        patch_norm=config.MODEL.DT.PATCH_NORM, config=config)

    elif model_type == 'DT':
        model = DeblendingTransformer(img_size=config.DATA.IMG_SIZE,
                                      in_chans=config.MODEL.DT.IN_CHANS,
                                      num_classes=config.MODEL.NUM_CLASSES,
                                      embed_dim=config.MODEL.DT.EMBED_DIM,
                                      depths=config.MODEL.DT.DEPTHS,
                                      num_heads=config.MODEL.DT.NUM_HEADS,
                                      window_size=config.MODEL.DT.WINDOW_SIZE,
                                      mlp_ratio=config.MODEL.DT.MLP_RATIO,
                                      qkv_bias=config.MODEL.DT.QKV_BIAS,
                                      qk_scale=config.MODEL.DT.QK_SCALE,
                                      drop_rate=config.MODEL.DROP_RATE,
                                      drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                      patch_norm=config.MODEL.DT.PATCH_NORM,
                                      use_checkpoint=config.TRAIN.USE_CHECKPOINT)


    elif model_type == 'swir':
        model = SwinIR(upscale=1, in_chans=1, img_size=64, window_size=8,
                       img_range=1.0, depths=[6, 6, 6, 6], embed_dim=60,
                       num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=None, resi_connection="3conv",
                       talking_heads=False, use_attn_fn="softmax", head_scale=False, on_attn=False,
                       use_mask=True, mask_ratio1=75, mask_ratio2=75, mask_is_diff=False, type="stand",
                       )

    elif model_type == 'restormer':
        model = Restormer(inp_channels=1,
                          out_channels=1,
                          dim=32,
                          num_blocks=[2, 3, 3, 4],
                          num_refinement_blocks=2,
                          ffn_expansion_factor=2)
    elif model_type == 'WUDT_STAnet':
        model = WUDT_STAnet(
                    # embed_dim=[96, 192, 448, 640], # 95M, 15.6G, 269 FPS
                    img_size=config.DATA.IMG_SIZE,
                    in_chans=config.MODEL.DT2.IN_CHANS,
                    num_classes=config.MODEL.NUM_CLASSES,
                    embed_dim=config.MODEL.DT2.EMBED_DIM,
                    depths=config.MODEL.DT2.DEPTHS ,
                    num_heads=config.MODEL.DT2.NUM_HEADS ,
                    n_iter=config.MODEL.DT2.NITER, 
                    stoken_size=config.MODEL.DT2.STOKEN_SIZE, # for 224/384
                    projection=1024,
                    mlp_ratio=config.MODEL.DT2.MLP_RATIO,
                    qkv_bias=config.MODEL.DT2.QKV_BIAS,
                    qk_scale=config.MODEL.DT2.QK_SCALE,
                    drop_rate=0,
                    drop_path_rate=0.6 , 
                    layerscale=[False, False, True],
                    init_values=1e-6,config=config)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
