########################################################################################################################
##                                                                                                                    ##
##  lib/config.py — central configuration for FaceAnonyMixer.                                                         ##
##                                                                                                                    ##
##  Update DATASETS to point to your local dataset directories before running any script.                             ##
##                                                                                                                    ##
########################################################################################################################

########################################################################################################################
##                                                 [ Datasets ]                                                       ##
########################################################################################################################
# Map a short dataset key (passed via --dataset) to the root directory on your filesystem.
# The root must contain a `train/` sub-folder organised as one sub-folder per identity.
#
# Example layout:
#   /data/VGGFace2/
#   └── train/
#       ├── n000001/
#       │   ├── 0001.jpg
#       │   └── 0002.jpg
#       └── n000002/
#           └── 0001.jpg
#
DATASETS = {
    'sample_IJB-C': '/path/to/your/dataset',   # <-- UPDATE THIS
    # 'vggface2':    '/path/to/vggface2',
    # 'celebahq':    '/path/to/celebahq',
}

CelebA_classes = (
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
    'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
    'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
    'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young',
)

########################################################################################################################
##                                                    [ FaRL ]                                                        ##
########################################################################################################################
FARL_EP = 64
FARL_PRETRAIN_MODEL = 'FaRL-Base-Patch16-LAIONFace20M-ep{}.pth'.format(FARL_EP)

FARL = (
    'https://www.dropbox.com/s/xxhmvo3q7avlcac/farl.tar?dl=1',
    '1d67cc6fd3cea9fdd7ec6af812a32e6b02374162d02137dd80827283d496b2d8',
)

########################################################################################################################
##                                                   [ ArcFace ]                                                      ##
########################################################################################################################
ARCFACE = (
    'https://www.dropbox.com/s/idulblr8pdrmbq1/arcface.tar?dl=1',
    'edd5854cacd86c17a78a11f70ab8c49bceffefb90ee070754288fa7ceadcdfb2',
)

########################################################################################################################
##                                                     [ E4E ]                                                        ##
########################################################################################################################
E4E = (
    'https://www.dropbox.com/s/1jujsdr6ytzilym/e4e.tar?dl=1',
    'b4a95155f2bebbb229b7dfc914fe937753b9a5b8de9a837875f9fbcacf8bb287',
)

########################################################################################################################
##                                                     [ SFD ]                                                        ##
########################################################################################################################
SFD = (
    'https://www.dropbox.com/scl/fi/eo6c8prmvuhpvpx7sh4u8/sfd.tar?rlkey=7bpo0kxennilgz2kglwpgozy5&dl=1',
    '2bea5f1c10110e356eef3f4efd45169100b9c7704eb6e6abd309df58f34452d4',
)

########################################################################################################################
##                                           [ GenForce GAN Generators ]                                              ##
########################################################################################################################
GENFORCE = (
    'https://www.dropbox.com/scl/fi/yec5wgg8j388fc0saigbj/genforce.tar?rlkey=kkhkkxvnc985746ichtdyct3f&dl=1',
    '63284b4f4ffeac38037061fd175c462afff82bbe570ed80092720a724a67a6dc',
)

GENFORCE_MODELS = {
    'stylegan2_ffhq1024': ('stylegan2_ffhq1024.pth', 1024),
    'stylegan2_ffhq512':  ('stylegan2_ffhq512.pth',  512),
}

STYLEGAN_LAYERS = {
    'stylegan2_ffhq1024': 18,
    'stylegan2_ffhq512':  16,
}

GAN_BASE_LATENT_DIM = {
    'stylegan2_ffhq1024': 512,
    'stylegan2_ffhq512':  512,
}

STYLEGAN2_STYLE_SPACE_LAYERS = {
    'stylegan2_ffhq1024': {
        'style00': 512, 'style01': 512, 'style02': 512, 'style03': 512, 'style04': 512,
        'style05': 512, 'style06': 512, 'style07': 512, 'style08': 512, 'style09': 512,
        'style10': 256, 'style11': 256, 'style12': 128, 'style13': 128, 'style14': 64,
        'style15': 64,  'style16': 32,
    },
    'stylegan2_ffhq512': {
        'style00': 512, 'style01': 512, 'style02': 512, 'style03': 512, 'style04': 512,
        'style05': 512, 'style06': 512, 'style07': 512, 'style08': 512, 'style09': 512,
        'style10': 256, 'style11': 256, 'style12': 128, 'style13': 128, 'style14': 64,
    },
}

STYLEGAN2_STYLE_SPACE_TARGET_LAYERS = {
    'stylegan2_ffhq1024': {
        'style00': 512, 'style01': 512, 'style02': 512, 'style03': 512, 'style04': 512,
        'style05': 512, 'style06': 512, 'style07': 512,
    },
    'stylegan2_ffhq512': {
        'style00': 512, 'style01': 512, 'style02': 512, 'style03': 512, 'style04': 512,
        'style05': 512, 'style06': 512, 'style07': 512,
    },
}
