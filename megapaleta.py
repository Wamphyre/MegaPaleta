import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import time

class ColorConverter:
    @staticmethod
    def rgb_to_yuv(r, g, b):
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        return y, u, v

    @staticmethod
    def color_distance(color1, color2, weights=(1.0, 2.0, 2.0)):
        """
        Calculate weighted YUV distance between two RGB colors
        Gives more importance to chrominance (UV) than luminance (Y)
        """
        if isinstance(color1[0], np.uint8):
            color1 = tuple(int(c) for c in color1)
        if isinstance(color2[0], np.uint8):
            color2 = tuple(int(c) for c in color2)
            
        y1, u1, v1 = ColorConverter.rgb_to_yuv(*color1)
        y2, u2, v2 = ColorConverter.rgb_to_yuv(*color2)
        
        wy, wu, wv = weights
        dy = (y1 - y2) * wy
        du = (u1 - u2) * wu
        dv = (v1 - v2) * wv
        
        return np.sqrt(dy*dy + du*du + dv*dv)

    @staticmethod
    def calculate_color_frequency(img_data):
        """Calculate frequency of each color in the image"""
        color_counts = defaultdict(int)
        total_pixels = img_data.shape[0] * img_data.shape[1]
        
        for pixel in img_data.reshape(-1, 4):
            if pixel[3] < 128:  # Skip transparent pixels
                continue
            color = tuple(pixel[:3])
            color_counts[color] += 1
        
        return {color: count/total_pixels for color, count in color_counts.items()}

    @staticmethod
    def rgb_to_megadrive(rgba):
        """Convert RGBA color to Mega Drive format"""
        r, g, b, a = rgba
        if a < 128:  # Transparent pixel
            return None
        return (round((r / 255) * 15), round((g / 255) * 15), round((b / 255) * 15))

    @staticmethod
    def megadrive_to_rgb(md_color):
        """Convert Mega Drive color to RGB format"""
        if md_color is None:
            return (0, 0, 0, 0)  # Completely transparent
        r, g, b = md_color
        return (round((r / 15) * 255), round((g / 15) * 255), round((b / 15) * 255), 255)

class PaletteManager:
    def __init__(self, max_colors=16, max_palettes=4):
        self.max_colors = max_colors
        self.max_palettes = max_palettes
        self.palettes = [[] for _ in range(max_palettes)]

    def create_indexed_image(self, image_data, palette):
        """Create indexed image with improved color mapping"""
        height, width = image_data.shape[:2]
        indexed_data = np.zeros((height, width), dtype=np.uint8)
        palette_lookup = {}
        
        # Create lookup table for colors
        for idx, color in enumerate(palette):
            if color is None:
                palette_lookup[None] = 0
            else:
                rgb = ColorConverter.megadrive_to_rgb(color)[:3]
                palette_lookup[rgb] = idx
                
        # Process each pixel
        for y in range(height):
            for x in range(width):
                pixel = tuple(int(c) for c in image_data[y, x][:3])
                alpha = image_data[y, x][3]
                
                if alpha < 128:
                    indexed_data[y, x] = 0  # Transparent
                else:
                    # Find exact match or closest color
                    if pixel not in palette_lookup:
                        min_distance = float('inf')
                        best_idx = 0
                        for pal_color, idx in palette_lookup.items():
                            if pal_color is not None:
                                dist = ColorConverter.color_distance(pixel, pal_color)
                                if dist < min_distance:
                                    min_distance = dist
                                    best_idx = idx
                        indexed_data[y, x] = best_idx
                    else:
                        indexed_data[y, x] = palette_lookup[pixel]
        
        return indexed_data

    def cluster_colors_to_palettes(self, unique_colors):
        """Cluster colors into multiple palettes"""
        colors_array = np.array([list(color) for color in unique_colors if color is not None])
        
        if len(colors_array) <= self.max_colors - 1:
            return [[None] + list(map(tuple, colors_array.tolist()))] + [[] for _ in range(self.max_palettes-1)]
        
        kmeans = KMeans(
            n_clusters=min(self.max_palettes, len(colors_array)), 
            random_state=42,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(colors_array)
        
        palette_colors = defaultdict(list)
        for color, label in zip(colors_array, cluster_labels):
            palette_colors[label].append(tuple(color))
        
        palettes = []
        for i in range(self.max_palettes):
            if i in palette_colors:
                colors = sorted(palette_colors[i], key=lambda c: sum(c))[:self.max_colors-1]
                palettes.append([None] + colors)  # Add transparent color at index 0
            else:
                palettes.append([])
        
        return palettes

class ImageProcessor:
    def __init__(self):
        self.color_converter = ColorConverter()
        self.palette_manager = PaletteManager()

    def process_image(self, image, reduce_colors=False):
        """Process image and return converted result"""
        if image is None:
            raise ValueError("No image provided")

        # Ensure dimensions are multiples of 8
        width, height = image.size
        new_width = ((width + 7) // 8) * 8
        new_height = ((height + 7) // 8) * 8
        
        if width != new_width or height != new_height:
            image = image.resize((new_width, new_height), Image.Resampling.NEAREST)

        img_data = np.array(image.convert('RGBA'))
        
        if reduce_colors:
            return self._reduce_to_single_palette(img_data)
        else:
            return self._convert_to_megadrive(img_data)

    def _convert_to_megadrive(self, img_data):
        """Convert image to Mega Drive format"""
        unique_colors = set()
        for pixel in img_data.reshape(-1, 4):
            if pixel[3] >= 128:  # Non-transparent pixel
                md_color = ColorConverter.rgb_to_megadrive(tuple(pixel))
                if md_color is not None:
                    unique_colors.add(md_color)
        
        self.palette_manager.palettes = self.palette_manager.cluster_colors_to_palettes(unique_colors)
        
        # Create the converted image
        height, width = img_data.shape[:2]
        converted_data = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Create color mapping
        color_map = {}
        for palette in self.palette_manager.palettes:
            for color in palette:
                if color not in color_map:
                    color_map[color] = ColorConverter.megadrive_to_rgb(color)

        # Convert pixels
        for y in range(height):
            for x in range(width):
                pixel = tuple(img_data[y, x])
                alpha = pixel[3]
                if alpha < 128:
                    converted_data[y, x] = [0, 0, 0, 0]
                else:
                    md_color = ColorConverter.rgb_to_megadrive(pixel)
                    if md_color is not None and md_color in color_map:
                        rgba_color = color_map[md_color]
                        converted_data[y, x] = [*rgba_color[:3], alpha]
                    else:
                        # Find closest color if exact match not found
                        min_distance = float('inf')
                        closest_color = None
                        pixel_rgb = pixel[:3]
                        for pal_color in color_map:
                            if pal_color is not None:
                                rgb_color = ColorConverter.megadrive_to_rgb(pal_color)[:3]
                                distance = ColorConverter.color_distance(pixel_rgb, rgb_color)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_color = rgb_color
                        if closest_color:
                            converted_data[y, x] = [*closest_color, alpha]

        return Image.fromarray(converted_data, 'RGBA'), self.palette_manager.palettes

    def _reduce_to_single_palette(self, img_data):
        """Reduce image to a single 16-color palette"""
        color_frequencies = ColorConverter.calculate_color_frequency(img_data)
        unique_colors = list(color_frequencies.keys())
        
        if len(unique_colors) <= 15:
            reduced_palette = [None] + [ColorConverter.rgb_to_megadrive((*c, 255)) for c in unique_colors]
        else:
            reduced_palette = self._cluster_colors(unique_colors, color_frequencies)

        # Create converted image
        height, width = img_data.shape[:2]
        converted_data = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Create color mapping
        color_mapping = {}
        for color in reduced_palette[1:]:  # Skip transparent color
            if color is not None:
                rgb = ColorConverter.megadrive_to_rgb(color)[:3]
                color_mapping[rgb] = rgb

        # Convert pixels
        for y in range(height):
            for x in range(width):
                pixel = img_data[y, x]
                if pixel[3] < 128:
                    converted_data[y, x] = [0, 0, 0, 0]
                else:
                    original_color = tuple(pixel[:3])
                    # Find nearest color in palette
                    min_distance = float('inf')
                    nearest_color = None
                    for palette_rgb in color_mapping.values():
                        distance = ColorConverter.color_distance(original_color, palette_rgb)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_color = palette_rgb
                    
                    if nearest_color:
                        converted_data[y, x] = [*nearest_color, pixel[3]]

        # Update palettes
        self.palette_manager.palettes = [reduced_palette] + [[] for _ in range(self.palette_manager.max_palettes - 1)]
        
        return Image.fromarray(converted_data, 'RGBA'), self.palette_manager.palettes

    def _cluster_colors(self, unique_colors, color_frequencies):
        """Perform color clustering with improved weights"""
        colors_array = np.array(unique_colors)
        weights = np.array([color_frequencies[tuple(color)] for color in unique_colors])
        weights = weights.reshape(-1, 1) * 1000
        
        # First clustering stage
        initial_clusters = min(30, len(unique_colors))
        kmeans1 = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10)
        labels1 = kmeans1.fit_predict(colors_array)
        
        # Group colors by initial clusters
        cluster_groups = defaultdict(list)
        cluster_weights = defaultdict(list)
        for color, label, weight in zip(colors_array, labels1, weights):
            cluster_groups[label].append(color)
            cluster_weights[label].append(weight[0])
        
        # Select representative colors
        final_colors = []
        for label in range(initial_clusters):
            if label in cluster_groups:
                cluster_colors = np.array(cluster_groups[label])
                cluster_w = np.array(cluster_weights[label])
                weighted_avg = np.average(cluster_colors, weights=cluster_w, axis=0)
                closest_idx = np.argmin([
                    ColorConverter.color_distance(weighted_avg, color)
                    for color in cluster_colors
                ])
                final_colors.append(tuple(cluster_colors[closest_idx]))
        
        # Second stage clustering if needed
        if len(final_colors) > 15:
            colors_array = np.array(final_colors)
            kmeans2 = KMeans(n_clusters=15, random_state=42, n_init=10)
            centers = kmeans2.fit(colors_array).cluster_centers_
            
            reduced_palette = [None]  # Start with transparency
            for center in centers:
                md_color = ColorConverter.rgb_to_megadrive((*center, 255))
                if md_color not in reduced_palette:
                    reduced_palette.append(md_color)
        else:
            reduced_palette = [None] + [ColorConverter.rgb_to_megadrive((*c, 255)) for c in final_colors]
        
        return reduced_palette

class MegaDrivePaletteConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("MegaDrive Palette Converter")
        
        # Initialize components
        self.processor = ImageProcessor()
        self.current_image = None
        self.original_image = None
        self.simplified_image = None
        
        # Set up UI
        self.setup_ui()

    def setup_ui(self):
        # Set window icon
        try:
            icon_path = "icon.ico"
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except tk.TclError:
            print("Warning: Could not load icon")

        # Main frame
        self.main_frame = tk.Frame(self.root, padx=10, pady=10)
        self.main_frame.pack(expand=True, fill='both')
        
        # Buttons frame
        buttons_frame = tk.Frame(self.main_frame)
        buttons_frame.pack(fill='x', pady=5)
        
        buttons = [
            ("Load Image", self.load_image),
            ("Convert", lambda: self.process_image(False)),
            ("Reduce to Single Palette", lambda: self.process_image(True)),
            ("Save Image", self.save_image),
            ("Export Palettes", self.export_palettes)
        ]
        
        for text, command in buttons:
            tk.Button(buttons_frame, text=text, command=command).pack(side='left', padx=5)
        
        # Image preview frame
        preview_frame = tk.Frame(self.main_frame)
        preview_frame.pack(fill='both', expand=True, pady=10)
        
        self.original_label = tk.Label(preview_frame, text="Original")
        self.original_label.pack(side='left', padx=5)
        
        self.converted_label = tk.Label(preview_frame, text="Converted")
        self.converted_label.pack(side='right', padx=5)
        
        # Log frame
        log_frame = tk.Frame(self.main_frame, relief=tk.GROOVE, borderwidth=2)
        log_frame.pack(fill='x', pady=5)
        
        # Log title
        tk.Label(log_frame, text="Process Log", font=('TkDefaultFont', 9, 'bold')).pack(pady=2)
        
        # Log text area with scrollbar
        log_container = tk.Frame(log_frame)
        log_container.pack(fill='x', padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(log_container)
        scrollbar.pack(side='right', fill='y')
        
        self.log_text = tk.Text(log_container, height=6, wrap=tk.WORD, 
                               font=('Consolas', 9), 
                               bg='#f5f5f5')
        self.log_text.pack(fill='x')
        
        scrollbar.config(command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Make log readonly
        self.log_text.config(state='disabled')

    def add_to_log(self, message):
        """Add a message to the log with timestamp"""
        timestamp = time.strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state='normal')
        self.log_text.insert('end', log_message)
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def display_image(self, image, label):
        """Display an image in the specified label"""
        if image:
            # Get screen dimensions
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            # Calculate maximum dimensions
            max_width = (screen_width - 100) // 2
            max_height = screen_height - 200
            
            # Calculate scaling
            width, height = image.size
            width_ratio = max_width / width
            height_ratio = max_height / height
            scale_factor = min(width_ratio, height_ratio, 2.0)  # Max zoom 2x
            
            # Resize image for display
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            display_image = image.resize((new_width, new_height), Image.Resampling.NEAREST)
            
            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(display_image)
            label.config(image=photo)
            label.image = photo  # Keep reference!

    def process_image(self, reduce_colors=False):
        """Process the current image"""
        if self.current_image is None:
            messagebox.showwarning("Error", "Please load an image first")
            return

        try:
            start_time = time.time()
            self.add_to_log(f"Starting image processing {'(single palette)' if reduce_colors else ''}")
            self.add_to_log(f"Image size: {self.current_image.size[0]}x{self.current_image.size[1]} pixels")
            
            # Process image
            self.simplified_image, palettes = self.processor.process_image(
                self.current_image, reduce_colors)
            
            # Update display
            self.display_image(self.simplified_image, self.converted_label)
            
            # Log results
            process_time = time.time() - start_time
            if reduce_colors:
                palette = palettes[0]
                self.add_to_log(f"Reduced to {len(palette)} colors")
            else:
                total_colors = sum(len(p) for p in palettes if p)
                palettes_used = sum(1 for p in palettes if p)
                self.add_to_log(f"Total colors: {total_colors}")
                self.add_to_log(f"Palettes used: {palettes_used}")
            
            self.add_to_log(f"Processing completed in {process_time:.2f} seconds")
                
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            self.add_to_log(f"ERROR: {error_msg}")
            messagebox.showerror("Error", error_msg)

    def export_palettes(self):
        """Export the current palettes as an indexed image"""
        if not any(self.processor.palette_manager.palettes):
            messagebox.showwarning("Error", "No palettes to export")
            return
        
        try:
            start_time = time.time()
            self.add_to_log("Starting palette export")
            
            # Count used palettes
            used_palettes = sum(1 for p in self.processor.palette_manager.palettes if p)
            
            # Create palette visualization
            CELL_SIZE = 32
            palette_img = Image.new('RGBA', 
                (16 * CELL_SIZE, used_palettes * CELL_SIZE), 
                (0, 0, 0, 0)
            )
            draw = ImageDraw.Draw(palette_img)
            
            # Create indexed data and palette
            indexed_data = np.zeros((used_palettes * CELL_SIZE, 16 * CELL_SIZE), dtype=np.uint8)
            palette_data = []
            color_index = 0
            color_map = {}

            # Draw palette colors and build indexed data
            current_row = 0
            for palette in self.processor.palette_manager.palettes:
                if not palette:
                    continue
                
                for col_idx, color in enumerate(palette):
                    x = col_idx * CELL_SIZE
                    y = current_row * CELL_SIZE
                    
                    if color is None:
                        # Handle transparent color
                        color_map[None] = 0
                        # Fill transparent area in indexed data
                        indexed_data[y:y + CELL_SIZE, x:x + CELL_SIZE] = 0
                        
                        # Draw checkerboard pattern for preview
                        for i in range(CELL_SIZE):
                            for j in range(CELL_SIZE):
                                if (i + j) % 2 == 0:
                                    draw.point((x + i, y + j), fill=(255, 255, 255, 255))
                                else:
                                    draw.point((x + i, y + j), fill=(192, 192, 192, 255))
                    else:
                        # Handle regular color
                        if color not in color_map:
                            color_index += 1
                            color_map[color] = color_index
                            rgb_color = ColorConverter.megadrive_to_rgb(color)
                            palette_data.extend(rgb_color[:3])
                        
                        # Fill area with color index
                        indexed_data[y:y + CELL_SIZE, x:x + CELL_SIZE] = color_map[color]
                        
                        # Draw color for preview
                        rgb_color = ColorConverter.megadrive_to_rgb(color)
                        draw.rectangle(
                            [x, y, x + CELL_SIZE - 1, y + CELL_SIZE - 1], 
                            fill=rgb_color
                        )
                
                current_row += 1
            
            # Add grid lines for preview
            for i in range(17):
                x = i * CELL_SIZE
                draw.line(
                    [(x, 0), (x, used_palettes * CELL_SIZE)], 
                    fill=(64, 64, 64, 255), 
                    width=1
                )
            
            for i in range(used_palettes + 1):
                y = i * CELL_SIZE
                draw.line(
                    [(0, y), (16 * CELL_SIZE, y)], 
                    fill=(64, 64, 64, 255), 
                    width=1
                )
            
            # Save files
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")]
            )
            
            if file_path:
                # Save preview image
                preview_path = file_path[:-4] + "_preview.png"
                palette_img.save(preview_path)
                
                # Create and save indexed image
                indexed_image = Image.fromarray(indexed_data, mode='P')
                indexed_image.putpalette(palette_data * 16)  # Repeat palette to fill 256 colors
                indexed_image.save(file_path, optimize=True)
                
                process_time = time.time() - start_time
                self.add_to_log(f"Palettes exported successfully:")
                self.add_to_log(f"- Preview: {os.path.basename(preview_path)}")
                self.add_to_log(f"- Indexed: {os.path.basename(file_path)}")
                self.add_to_log(f"Total colors: {color_index}")
                self.add_to_log(f"Export completed in {process_time:.2f} seconds")
                
                messagebox.showinfo("Success", "Palettes exported successfully")
                
        except Exception as e:
            error_msg = f"Error exporting palettes: {str(e)}"
            self.add_to_log(f"ERROR: {error_msg}")
            messagebox.showerror("Error", error_msg)

    def save_image(self):
        """Save the converted image"""
        if self.simplified_image is None:
            messagebox.showwarning("Error", "No converted image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")]
        )
        
        if not file_path:
            return
        
        try:
            start_time = time.time()
            self.add_to_log(f"Starting image save: {os.path.basename(file_path)}")
            
            # Save original RGBA image with transparency
            self.simplified_image.save(file_path, optimize=True)
            
            process_time = time.time() - start_time
            self.add_to_log(f"Image saved successfully: {os.path.basename(file_path)}")
            self.add_to_log(f"Size: {self.simplified_image.size[0]}x{self.simplified_image.size[1]} pixels")
            self.add_to_log(f"Save completed in {process_time:.2f} seconds")
            
            messagebox.showinfo("Success", "Image saved successfully")
            
        except Exception as e:
            error_msg = f"Error saving image: {str(e)}"
            self.add_to_log(f"ERROR: {error_msg}")
            messagebox.showerror("Error", error_msg)

    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            try:
                self.add_to_log(f"Loading image: {os.path.basename(file_path)}")
                self.original_image = Image.open(file_path).convert('RGBA')
                self.current_image = self.original_image.copy()
                self.display_image(self.original_image, self.original_label)
                
                # Log image details
                self.add_to_log(f"Image loaded successfully")
                self.add_to_log(f"Size: {self.original_image.size[0]}x{self.original_image.size[1]} pixels")
                self.add_to_log(f"Format: {self.original_image.format}")
                
            except Exception as e:
                error_msg = f"Error loading image: {str(e)}"
                self.add_to_log(f"ERROR: {error_msg}")
                messagebox.showerror("Error", error_msg)

def main():
    root = tk.Tk()
    # Set window size to 80% of screen size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    app = MegaDrivePaletteConverter(root)
    root.mainloop()

if __name__ == "__main__":
    main()