from PIL import Image
import os

def convert_and_compress(image_path, output_path, max_size_mb=4):
    """Convert a JPG image to PNG and compress it to a maximum size."""
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    
    # Open the image
    with Image.open(image_path) as img:
        # Convert to PNG format
        img = img.convert("RGBA")
        
        # Save the image with compression
        quality = 95  # Start with high quality
        while True:
            img.save(output_path, format="PNG", optimize=True, quality=quality)
            if os.path.getsize(output_path) <= max_size_bytes or quality <= 10:
                break
            quality -= 5  # Reduce quality to compress further
    
    print(f"Image saved to {output_path} with size {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

# Example usage
input_image = "test.jpg"
output_image = "tests.png"
convert_and_compress(input_image, output_image)