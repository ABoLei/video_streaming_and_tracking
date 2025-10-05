from model import ClassificationModel, LightweightModel, create_model

if __name__ == "__main__":
    print("=== Model Parameter Analysis ===")
    
    # Test efficient model with RGB input
    print("\n1. Efficient Model (RGB):")
    model_rgb = create_model('efficient', num_classes=100, input_channels=3)
    total_params_rgb = sum(p.numel() for p in model_rgb.parameters())
    print(f"   Parameters: {total_params_rgb:,}")
    
    # Test efficient model with grayscale input
    print("\n2. Efficient Model (Grayscale):")
    model_gray = create_model('efficient', num_classes=100, input_channels=1)
    total_params_gray = sum(p.numel() for p in model_gray.parameters())
    print(f"   Parameters: {total_params_gray:,}")
    
    # Test lightweight model with RGB input
    print("\n3. Lightweight Model (RGB):")
    model_light_rgb = create_model('lightweight', num_classes=100, input_channels=3)
    total_params_light_rgb = sum(p.numel() for p in model_light_rgb.parameters())
    print(f"   Parameters: {total_params_light_rgb:,}")
    
    # Test lightweight model with grayscale input
    print("\n4. Lightweight Model (Grayscale):")
    model_light_gray = create_model('lightweight', num_classes=100, input_channels=1)
    total_params_light_gray = sum(p.numel() for p in model_light_gray.parameters())
    print(f"   Parameters: {total_params_light_gray:,}")
    
    print("\n=== Recommendation ===")
    print("For parameter efficiency ranking:")
    print(f"1. Lightweight Grayscale: {total_params_light_gray:,} params")
    print(f"2. Lightweight RGB: {total_params_light_rgb:,} params")
    print(f"3. Efficient Grayscale: {total_params_gray:,} params")
    print(f"4. Efficient RGB: {total_params_rgb:,} params")
