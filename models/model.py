import sys
sys.path.append("..\Github_commit\models")
import segmentation_models_pytorch as smp
import torch
import Unet_Segmentation_Pytorch_Nest_of_Unets.Models as spm
import Unet_Segmentation_Pytorch_Nest_of_Unets.AtteffUnet as atteffunet


def build_model(CFG):
    if (CFG['model_name'] == 'Unet'):
        model = smp.Unet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        decoder_use_DIA = False,
                    ).to(CFG['device'])
        print('model is vgg16-Unet')
        # print('model is efficientnet-b0-Unet')
    elif (CFG['model_name'] == 'UnetPlusPlus'):
        model = smp.UnetPlusPlus(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        decoder_use_DIA = False,
                    ).to(CFG['device'])
        # print('model is vgg16-Unet++')
        print('model is efficientnet-b0-Unet')
    elif (CFG['model_name'] == 'SCSE_Unet'):
        model = smp.Unet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        decoder_attention_type=CFG['attention'],
                        decoder_use_DIA=False,
                    ).to(CFG['device'])
        # print('model is vgg16-SCSE_Unet')
        print('model is efficientnet-b0-SCSE_Unet')
    elif (CFG['model_name'] == 'SCSE_Unet'):
        model = smp.UnetPlusPlus(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                        decoder_use_DIA = False,
                    ).to(CFG['device'])
        # print('model is vgg16-UnetPlusPlus')
        print('model is efficientnet-b0-UnetPlusPlus')
    elif (CFG['model_name'] == 'AttentionUnet'):
        if (CFG['backbone'] == 'efficientnet-b0'):
            model = atteffunet.get_efficientunet_b0(out_channels=3, concat_input=True, pretrained=False).to(CFG['device'])
            print('model is efficientnet-b0-AttentionUnet')
        else:
            model = spm.AttU_Net(img_ch=3, output_ch=3).to(CFG['device'])
            print('model is vgg16-AttentionUnet')
    elif (CFG['model_name'] == 'SIAUnet'):
        model = smp.Unet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                        decoder_use_DIA = CFG['use_channel_attention']
                    ).to(CFG['device'])
        # print('model is vgg16-DIAUnet')
        print('model is efficientnet-b0-DIAUnet')
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

