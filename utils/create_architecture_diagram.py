import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Create complete GAN architecture diagram
def create_gan_architecture_diagram():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14))
    
    # Colors
    input_color = '#4CAF50'  # Green
    encoder_color = '#2196F3'  # Blue
    bottleneck_color = '#FF5722'  # Red-orange
    decoder_color = '#9C27B0'  # Purple
    output_color = '#FF9800'  # Orange
    skip_color = '#607D8B'  # Blue-grey
    disc_color = '#795548'  # Brown
    
    # === GENERATOR (Top subplot) ===
    ax1.text(22, 13, 'Generator: U-Net Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Generator layers
    gen_layers = [
        # Input
        (1, 3, 1.5, 6, 5, "Input\n(5 channels)"),
        
        # Encoder
        (4, 3.5, 1.2, 5, 64, "enc1\n(64)"),
        (6.5, 3.8, 1, 4.4, 128, "enc2\n(128)"),
        (9, 4.1, 0.8, 3.8, 256, "enc3\n(256)"),
        (11.5, 4.4, 0.6, 3.2, 512, "enc4\n(512)"),
        (14, 4.7, 0.5, 2.6, 512, "enc5\n(512)"),
        (16.5, 5, 0.4, 2, 512, "enc6\n(512)"),
        (19, 5.3, 0.3, 1.4, 512, "enc7\n(512)"),
        
        # Bottleneck
        (21.5, 5.5, 0.3, 1, 512, "bottleneck\n(512)"),
        
        # Decoder
        (24, 5.3, 0.3, 1.4, 512, "dec1\n(512)"),
        (26.5, 5, 0.4, 2, 512, "dec2\n(512)"),
        (29, 4.7, 0.5, 2.6, 512, "dec3\n(512)"),
        (31.5, 4.4, 0.6, 3.2, 256, "dec4\n(256)"),
        (34, 4.1, 0.8, 3.8, 128, "dec5\n(128)"),
        (36.5, 3.8, 1, 4.4, 64, "dec6\n(64)"),
        (39, 3.5, 1.2, 5, 64, "dec7\n(64)"),
        
        # Output
        (42, 3, 1.5, 6, 1, "Generated\nPath\n(1 channel)")
    ]
    
    # Draw generator layers
    for i, (x, y, width, height, channels, name) in enumerate(gen_layers):
        if i == 0:  # Input
            color = input_color
        elif i <= 7:  # Encoder
            color = encoder_color
        elif i == 8:  # Bottleneck
            color = bottleneck_color
        elif i <= 15:  # Decoder
            color = decoder_color
        else:  # Output
            color = output_color
            
        # Draw rectangle
        rect = patches.Rectangle((x, y), width, height, 
                               linewidth=2, edgecolor='black', 
                               facecolor=color, alpha=0.7)
        ax1.add_patch(rect)
        
        # Add text
        ax1.text(x + width/2, y + height/2, name, 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw generator forward connections
    for i in range(len(gen_layers)-1):
        x1 = gen_layers[i][0] + gen_layers[i][2]
        y1 = gen_layers[i][1] + gen_layers[i][3]/2
        x2 = gen_layers[i+1][0]
        y2 = gen_layers[i+1][1] + gen_layers[i+1][3]/2
        
        ax1.arrow(x1, y1, x2-x1-0.1, y2-y1, head_width=0.1, 
                head_length=0.08, fc='black', ec='black')
    
    # Draw skip connections for generator
    skip_connections = [
        (1, 16),  # e1 to dec7
        (2, 15),  # e2 to dec6  
        (3, 14),  # e3 to dec5
        (4, 13),  # e4 to dec4
        (5, 12),  # e5 to dec3
        (6, 11),  # e6 to dec2
        (7, 9),   # e7 to dec1
    ]
    
    for enc_idx, dec_idx in skip_connections:
        x1 = gen_layers[enc_idx][0] + gen_layers[enc_idx][2]/2
        y1 = gen_layers[enc_idx][1] + gen_layers[enc_idx][3]
        x2 = gen_layers[dec_idx][0] + gen_layers[dec_idx][2]/2
        y2 = gen_layers[dec_idx][1] + gen_layers[dec_idx][3]
        
        # Draw curved skip connection
        mid_x = (x1 + x2) / 2
        mid_y = max(y1, y2) + 0.8
        
        # Create curved path
        t = np.linspace(0, 1, 50)
        curve_x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
        curve_y = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2
        
        ax1.plot(curve_x, curve_y, color=skip_color, linewidth=1.5, 
                linestyle='--', alpha=0.8)
    
    # Generator labels
    ax1.text(12, 0.5, 'Encoder\n(Downsampling)', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=encoder_color, alpha=0.3))
    
    ax1.text(32, 0.5, 'Decoder\n(Upsampling)', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=decoder_color, alpha=0.3))
    
    ax1.text(22, 11, 'Skip Connections (Concatenation)', 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=skip_color, alpha=0.3))
    
    ax1.set_xlim(0, 44)
    ax1.set_ylim(0, 14)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # === DISCRIMINATOR (Bottom subplot) ===
    ax2.text(14, 10, 'Discriminator: PatchGAN Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Discriminator layers
    disc_layers = [
        # Inputs (condition + target/generated)
        (2, 3, 2, 4, "5+1", "Condition\n+\nTarget/Generated\n(6 channels)"),
        
        # Discriminator layers
        (6, 3.5, 2.5, 3, 64, "conv1\n(64)\nNo BatchNorm"),
        (10.5, 4, 2.5, 2, 128, "conv2\n(128)\nInstanceNorm"),
        (15, 4.2, 2.5, 1.6, 256, "conv3\n(256)\nInstanceNorm"),
        (19.5, 4.4, 2.5, 1.2, 512, "conv4\n(512)\nInstanceNorm\nstride=1"),
        
        # Final output
        (24, 4.5, 2, 1, 1, "final_conv\n(1)\nPatch Score")
    ]
    
    # Draw discriminator layers
    for i, (x, y, width, height, channels, name) in enumerate(disc_layers):
        if i == 0:  # Input
            color = input_color
        elif i == len(disc_layers)-1:  # Output
            color = output_color
        else:  # Discriminator layers
            color = disc_color
            
        # Draw rectangle
        rect = patches.Rectangle((x, y), width, height, 
                               linewidth=2, edgecolor='black', 
                               facecolor=color, alpha=0.7)
        ax2.add_patch(rect)
        
        # Add text
        ax2.text(x + width/2, y + height/2, name, 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw discriminator forward connections
    for i in range(len(disc_layers)-1):
        x1 = disc_layers[i][0] + disc_layers[i][2]
        y1 = disc_layers[i][1] + disc_layers[i][3]/2
        x2 = disc_layers[i+1][0]
        y2 = disc_layers[i+1][1] + disc_layers[i+1][3]/2
        
        ax2.arrow(x1, y1, x2-x1-0.1, y2-y1, head_width=0.1, 
                head_length=0.08, fc='black', ec='black')
    
    # Add discriminator annotations
    ax2.text(14, 1.5, 'Convolutional Layers\n(Progressive Downsampling)', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=disc_color, alpha=0.3))
    
    ax2.text(25, 2, 'Real/Fake\nClassification\nper Patch', 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=output_color, alpha=0.3))
    
    ax2.set_xlim(0, 28)
    ax2.set_ylim(0, 11)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Add main title
    fig.suptitle('Complete GAN Architecture\nGenerator + Discriminator', 
                fontsize=20, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('../../tfm_memoria/img', exist_ok=True)
    plt.savefig('../../tfm_memoria/img/gan_complete_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('gan_complete_architecture.png', dpi=300, bbox_inches='tight')
    
    print("✓ Complete GAN architecture diagram saved as gan_complete_architecture.png")
    plt.show()

if __name__ == "__main__":
    create_gan_architecture_diagram()
