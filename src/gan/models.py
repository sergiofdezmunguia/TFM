import torch
import torch.nn as nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, features=32):
        super(UNetGenerator, self).__init__()

        # --- Encoder ---
        self.enc1 = self._unet_block(in_channels, features, batch_norm=False, activation="leaky") 
        self.enc2 = self._unet_block(features,     features * 2, batch_norm=True, activation="leaky")
        self.enc3 = self._unet_block(features * 2, features * 4, batch_norm=True, activation="leaky") 
        self.enc4 = self._unet_block(features * 4, features * 8, batch_norm=True, activation="leaky") 
        self.enc5 = self._unet_block(features * 8, features * 8, batch_norm=True, activation="leaky") 
        self.enc6 = self._unet_block(features * 8, features * 8, batch_norm=True, activation="leaky") 
        self.enc7 = self._unet_block(features * 8, features * 8, batch_norm=True, activation="leaky")
        
        bottleneck_stride = 2 
        self.bottleneck = self._unet_block(features * 8, features * 8, batch_norm=False, activation="leaky", stride_override=bottleneck_stride) # Out: features*8
                                                                                                                        
        # --- Decoder ---

        self.dec1 = self._unet_block(features * 8,         features * 8, batch_norm=True, activation="relu", upsample=True, dropout=True)

        self.dec2 = self._unet_block(features * 8 * 2,     features * 8, batch_norm=True, activation="relu", upsample=True, dropout=True) 

        self.dec3 = self._unet_block(features * 8 * 2,     features * 8, batch_norm=True, activation="relu", upsample=True, dropout=True) 

        self.dec4 = self._unet_block(features * 8 * 2,     features * 8, batch_norm=True, activation="relu", upsample=True)                
                                                                                                                                        
        self.dec5 = self._unet_block(features * 8 * 2,     features * 4, batch_norm=True, activation="relu", upsample=True)                

        self.dec6 = self._unet_block(features * 4 * 2,     features * 2, batch_norm=True, activation="relu", upsample=True)                

        self.dec7_final_before_output = self._unet_block(features * 2 * 2, features, batch_norm=True, activation="relu", upsample=True)     
        
        # Capa final de salida
        self.final_conv_out = nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1)
        self.final_activation = nn.Sigmoid()

    def _unet_block(self, in_channels, out_channels, batch_norm=True, activation="relu", 
                   upsample=False, dropout=False, stride_override=None):
        layers = []
        use_bias = not batch_norm 
        if upsample:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias))
        else:
            stride = stride_override if stride_override is not None else 2
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=use_bias))

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if dropout:
            layers.append(nn.Dropout(0.5))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        bottleneck_out = self.bottleneck(e7)

        # Decoder
        d1_out = self.dec1(bottleneck_out)
        d2_out = self.dec2(torch.cat([d1_out, e7], dim=1))
        d3_out = self.dec3(torch.cat([d2_out, e6], dim=1))
        d4_out = self.dec4(torch.cat([d3_out, e5], dim=1))
        d5_out = self.dec5(torch.cat([d4_out, e4], dim=1))
        d6_out = self.dec6(torch.cat([d5_out, e3], dim=1))
        d7_out = self.dec7_final_before_output(torch.cat([d6_out, e2], dim=1))
        
        final_input_to_output_layer = torch.cat([d7_out, e1], dim=1)
        out = self.final_conv_out(final_input_to_output_layer)
        out = self.final_activation(out)
        
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels_condition=5, in_channels_target=1, features=64):
        super(PatchDiscriminator, self).__init__()
        total_in_channels = in_channels_condition + in_channels_target

        self.conv1 = self._disc_block(total_in_channels, features, batch_norm=False)
        self.conv2 = self._disc_block(features, features * 2, batch_norm=True)
        self.conv3 = self._disc_block(features * 2, features * 4, batch_norm=True)
        self.conv4 = self._disc_block(features * 4, features * 8, stride=1, batch_norm=True)
        self.final_conv = nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)

    def _disc_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)]
        if batch_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x_condition, y_image):
        combined_input = torch.cat([x_condition, y_image], dim=1)
        out = self.conv1(combined_input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.final_conv(out)
        return out