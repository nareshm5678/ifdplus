import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Import project modules
from model_architecture import ForgeryDetectionNet
from inference import load_model, preprocess_image, predict, visualize_prediction, overlay_segmentation

class ForgeryDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Forgery Detection")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model path
        self.model_path = os.path.join(os.getcwd(), 'forgery_detection_model.pth')
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"Model file not found at {self.model_path}")
            self.root.destroy()
            return
        
        # Load model
        try:
            self.model = load_model(self.model_path, self.device)
            print("Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
            return
        
        # Create UI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        title_label = tk.Label(main_frame, text="Image Forgery Detection", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Description
        desc_label = tk.Label(main_frame, text="Select an image to detect if it has been tampered with.")
        desc_label.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        # Browse button
        browse_button = tk.Button(button_frame, text="Browse Image", command=self.browse_image, width=15)
        browse_button.pack(side=tk.LEFT, padx=5)
        
        # Status frame
        self.status_frame = tk.Frame(main_frame)
        self.status_frame.pack(fill=tk.X, pady=10)
        
        # Status label
        self.status_label = tk.Label(self.status_frame, text="No image selected", anchor="w")
        self.status_label.pack(fill=tk.X)
        
        # Result frame
        self.result_frame = tk.Frame(main_frame)
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Result label
        self.result_label = tk.Label(self.result_frame, text="")
        self.result_label.pack()
    
    def browse_image(self):
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        
        if not file_path:
            return
        
        self.status_label.config(text=f"Selected: {os.path.basename(file_path)}")
        self.process_image(file_path)
    
    def process_image(self, image_path):
        try:
            # Update status
            self.status_label.config(text=f"Processing: {os.path.basename(image_path)}...")
            self.root.update()
            
            # Preprocess image
            image_tensor, original_image = preprocess_image(image_path)
            
            # Run inference
            results = predict(self.model, image_tensor, self.device)
            
            # Get prediction
            prediction = results['prediction']
            prob_authentic = results['prob_authentic']
            prob_tampered = results['prob_tampered']
            
            # Display result
            if prediction == 0:
                result_text = f"Result: Authentic (Confidence: {prob_authentic:.2f})"
                result_color = "green"
            else:
                result_text = f"Result: Tampered (Confidence: {prob_tampered:.2f})"
                result_color = "red"
            
            self.result_label.config(text=result_text, fg=result_color, font=("Arial", 12, "bold"))
            
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.getcwd(), "results")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save visualization
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{name}_result.png")
            
            # Visualize results
            visualize_prediction(original_image, results, output_path)
            
            # Create overlay image with segmentation mask if tampered
            if prediction == 1:  # If tampered
                overlay = overlay_segmentation(
                    original_image, 
                    cv2.resize(results['segmentation_mask'].squeeze(), (original_image.shape[1], original_image.shape[0]))
                )
                
                overlay_path = os.path.join(output_dir, f"{name}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Show the visualization
            plt.close('all')  # Close any existing plots
            img = Image.open(output_path)
            img.show()
            
            # Update status
            self.status_label.config(text=f"Processed: {os.path.basename(image_path)} | Results saved to {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            self.status_label.config(text=f"Error processing: {os.path.basename(image_path)}")

def main():
    # Create tkinter root
    root = tk.Tk()
    app = ForgeryDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()