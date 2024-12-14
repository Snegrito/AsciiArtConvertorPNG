import sys
import os

# Update paths so we can import from ascii and vector directories
sys.path.append(os.path.join(os.path.dirname(__file__), 'ascii'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'vector'))

from ascii.ascii_conv_logic import convert_to_ascii
from vector.vector_conv_logic import image_to_vector

if __name__ == "__main__":
    # Simple menu
    print("What output do you want?")
    print("1) ASCII art")
    print("2) Vector art")

    choice = input("Enter your choice (1/2): ")

    image_path = "bear.png"  # Using the provided image
    if choice == '1':
        output_path = "bear_ascii.txt"
        convert_to_ascii(image_path, output_path)
        print(f"ASCII art saved to {output_path}")
    elif choice == '2':
        output_path = "bear_vector.svg"
        image_to_vector(image_path, output_path)
        print(f"Vector art saved to {output_path}")
    else:
        print("Invalid choice. Exiting.")
