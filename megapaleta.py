import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

def rgb_to_yuv(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return y, u, v

def color_distance(color1, color2, weights=(1.0, 2.0, 2.0)):
    """
    Calculate weighted YUV distance between two RGB colors
    Gives more importance to chrominance (UV) than luminance (Y)
    """
    y1, u1, v1 = rgb_to_yuv(*color1)
    y2, u2, v2 = rgb_to_yuv(*color2)
    
    wy, wu, wv = weights
    dy = (y1 - y2) * wy
    du = (u1 - u2) * wu
    dv = (v1 - v2) * wv
    
    return np.sqrt(dy*dy + du*du + dv*dv)

def calculate_color_frequency(img_data):
    """
    Calculate frequency of each color in the image
    """
    color_counts = defaultdict(int)
    total_pixels = img_data.shape[0] * img_data.shape[1]
    
    for pixel in img_data.reshape(-1, 4):
        if pixel[3] < 128:  # Skip transparent pixels
            continue
        color = tuple(pixel[:3])
        color_counts[color] += 1
    
    # Convert counts to frequencies
    color_frequencies = {color: count/total_pixels 
                        for color, count in color_counts.items()}
    
    return color_frequencies

class MegaDrivePaletteConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("MegaDrive Palette Converter")
        
        # Get the directory where the script is located
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set icon using absolute path
        icon_path = os.path.join(self.app_dir, "icon.ico")
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except tk.TclError:
                print(f"Warning: Could not load icon from {icon_path}")
        
        # Constants
        self.COLORS_PER_PALETTE = 16
        self.MAX_PALETTES = 4
        self.TILE_SIZE = 8
        self.ZOOM_FACTOR = 2
        
        # State variables
        self.current_image = None
        self.original_image = None
        self.simplified_image = None
        self.palettes = [[] for _ in range(self.MAX_PALETTES)]
        self.tiles_data = {}
        self.tiles_to_index = {}
        self.index_to_tiles = {}
        self.tile_map = []
        self.tile_palette_map = {}
        
        self.setup_ui()

    def setup_ui(self):
        # Main frame
        self.main_frame = tk.Frame(self.root, padx=10, pady=10)
        self.main_frame.pack(expand=True, fill='both')
        
        # Buttons frame
        buttons_frame = tk.Frame(self.main_frame)
        buttons_frame.pack(fill='x', pady=5)
        
        buttons = [
            ("Load Image", self.load_image),
            ("Convert", self.process_image),
            ("Reduce to Single Palette", self.reduce_to_single_palette),
            ("Save Image", self.save_image),
            ("Export Palettes", self.export_palettes)
        ]
        
        for text, command in buttons:
            tk.Button(buttons_frame, text=text, command=command).pack(side='left', padx=5)
        
        # Preview frame
        preview_frame = tk.Frame(self.main_frame)
        preview_frame.pack(fill='both', expand=True, pady=10)
        
        self.original_label = tk.Label(preview_frame, text="Original")
        self.original_label.pack(side='left', padx=5)
        
        self.converted_label = tk.Label(preview_frame, text="Converted")
        self.converted_label.pack(side='right', padx=5)
        
        # Info panel
        self.info_label = tk.Label(self.main_frame, text="", justify=tk.LEFT)
        self.info_label.pack(pady=5)

    def rgb_to_megadrive(self, rgba):
        r, g, b, a = rgba
        if a < 128:  # Transparent pixel
            return None
        return (round((r / 255) * 15), round((g / 255) * 15), round((b / 255) * 15))

    def megadrive_to_rgb(self, md_color):
        if md_color is None:
            return (0, 0, 0, 0)  # Completely transparent
        r, g, b = md_color
        return (round((r / 15) * 255), round((g / 15) * 255), round((b / 15) * 255), 255)

    def cluster_colors_to_palettes(self, unique_colors):
        colors_array = np.array([list(color) for color in unique_colors if color is not None])
        
        if len(colors_array) <= self.COLORS_PER_PALETTE - 1:
            return [[None] + list(map(tuple, colors_array.tolist()))] + [[] for _ in range(self.MAX_PALETTES-1)]
        
        kmeans = KMeans(
            n_clusters=min(self.MAX_PALETTES, len(colors_array)), 
            random_state=42
        )
        cluster_labels = kmeans.fit_predict(colors_array)
        
        palette_colors = defaultdict(list)
        for color, label in zip(colors_array, cluster_labels):
            palette_colors[label].append(tuple(color))
        
        palettes = []
        for i in range(self.MAX_PALETTES):
            if i in palette_colors:
                colors = sorted(palette_colors[i], key=lambda c: sum(c))[:self.COLORS_PER_PALETTE-1]
                palettes.append([None] + colors)
            else:
                palettes.append([])
        
        return palettes

    def extract_tiles(self, img_data):
        height, width = img_data.shape[:2]
        height_tiles = height // self.TILE_SIZE
        width_tiles = width // self.TILE_SIZE
        
        self.tiles_data.clear()
        self.tiles_to_index.clear()
        self.index_to_tiles.clear()
        tile_map = []
        current_index = 0
        
        for y in range(height_tiles):
            tile_row = []
            for x in range(width_tiles):
                tile = img_data[y*8:(y+1)*8, x*8:(x+1)*8].copy()
                tile_tuple = tuple(map(tuple, tile.reshape(-1, 4)))
                
                if tile_tuple not in self.tiles_to_index:
                    self.tiles_to_index[tile_tuple] = current_index
                    self.index_to_tiles[current_index] = tile_tuple
                    self.tiles_data[current_index] = tile_tuple
                    current_index += 1
                
                tile_row.append(self.tiles_to_index[tile_tuple])
            tile_map.append(tile_row)
        
        return tile_map

    def validate_and_assign_tile_palettes(self):
        tile_palette_map = {}
        invalid_tiles = []
        
        for tile_idx, tile_data in self.tiles_data.items():
            tile_colors = set(color for color in tile_data if color is not None)
            
            valid_palette = False
            for pal_idx, palette in enumerate(self.palettes):
                if all(color in palette for color in tile_colors):
                    tile_palette_map[tile_idx] = pal_idx
                    valid_palette = True
                    break
            
            if not valid_palette:
                invalid_tiles.append(tile_idx)
        
        return tile_palette_map, invalid_tiles

    def optimize_tile_palettes(self, invalid_tiles):
        for tile_idx in invalid_tiles:
            tile_data = self.tiles_data[tile_idx]
            tile_colors = set(color for color in tile_data if color is not None)
            
            best_palette_idx = 0
            best_match_count = 0
            
            for pal_idx, palette in enumerate(self.palettes):
                if not palette:
                    continue
                match_count = sum(1 for color in tile_colors if color in palette)
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_palette_idx = pal_idx
            
            self.tile_palette_map[tile_idx] = best_palette_idx

    def reduce_to_single_palette(self):
        if self.current_image is None:
            messagebox.showwarning("Error", "Please load an image first")
            return

        # Convert image to RGBA if it isn't already
        img_data = np.array(self.current_image.convert('RGBA'))
        
        # Calculate color frequencies
        color_frequencies = calculate_color_frequency(img_data)
        unique_colors = list(color_frequencies.keys())
        
        # If we have 15 or fewer colors (excluding transparente), no need for clustering
        if len(unique_colors) <= 15:
            reduced_palette = [None] + [self.rgb_to_megadrive((*c, 255)) for c in unique_colors]
        else:
            # Convert colors to array for clustering
            colors_array = np.array(unique_colors)
            
            # First clustering stage: create more clusters initially
            initial_clusters = min(30, len(unique_colors))
            kmeans1 = KMeans(n_clusters=initial_clusters, random_state=42)
            
            # Weight the colors by their frequency
            weights = np.array([color_frequencies[tuple(color)] for color in unique_colors])
            weights = weights.reshape(-1, 1) * 1000  # Scale up for numerical stability
            
            # Fit first stage clustering
            labels1 = kmeans1.fit_predict(colors_array)
            
            # Group colors by their initial clusters
            cluster_groups = defaultdict(list)
            cluster_weights = defaultdict(list)
            for color, label, weight in zip(colors_array, labels1, weights):
                cluster_groups[label].append(color)
                cluster_weights[label].append(weight[0])
            
            # Select representative colors from each initial cluster
            final_colors = []
            for label in range(initial_clusters):
                if label in cluster_groups:
                    cluster_colors = np.array(cluster_groups[label])
                    cluster_w = np.array(cluster_weights[label])
                    
                    # Weight by both frequency and distance to cluster center
                    center = np.average(cluster_colors, weights=cluster_w, axis=0)
                    
                    # Find the actual color closest to the weighted center
                    distances = [color_distance(center, color) for color in cluster_colors]
                    best_idx = np.argmin(distances)
                    final_colors.append(tuple(cluster_colors[best_idx]))
            
            # Second stage: reduce to final palette size if necessary
            if len(final_colors) > 15:
                colors_array = np.array(final_colors)
                kmeans2 = KMeans(n_clusters=15, random_state=42)
                centers = kmeans2.fit(colors_array).cluster_centers_
                
                # Convert centers to valid Mega Drive colors
                reduced_palette = [None]  # Start with transparency
                for center in centers:
                    md_color = self.rgb_to_megadrive((*center, 255))
                    if md_color not in reduced_palette:
                        reduced_palette.append(md_color)
            else:
                reduced_palette = [None] + [self.rgb_to_megadrive((*c, 255)) for c in final_colors]

                # Create the converted image with transparency
        width, height = self.current_image.size
        converted_data = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Extraemos el canal alfa
        alpha_channel = img_data[..., 3]
        
        # Create color mapping using perceptual distance
        color_mapping = {}
        for y in range(height):
            for x in range(width):
                if alpha_channel[y, x] < 128:  # Pixel transparente
                    converted_data[y, x] = [0, 0, 0, 0]
                    continue
                    
                original_color = tuple(img_data[y, x][:3])
                if original_color not in color_mapping:
                    # Find the perceptually nearest color in the reduced palette
                    min_distance = float('inf')
                    nearest_color = None
                    for palette_color in reduced_palette[1:]:  # Skip transparent color
                        if palette_color is None:
                            continue
                        palette_rgb = self.megadrive_to_rgb(palette_color)[:3]
                        distance = color_distance(original_color, palette_rgb)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_color = palette_rgb
                    color_mapping[original_color] = nearest_color
                
                mapped_color = color_mapping[original_color]
                converted_data[y, x] = [*mapped_color, alpha_channel[y, x]]
        
        # Create final image
        self.simplified_image = Image.fromarray(converted_data, 'RGBA')
        self.display_image(self.simplified_image, self.converted_label)
        
        # Update palettes
        self.palettes = [reduced_palette] + [[] for _ in range(self.MAX_PALETTES - 1)]
        
        # Update info label
        self.info_label.config(
            text=f"Colors reduced to {len(reduced_palette)} colors\n"
                 f"Original unique colors: {len(unique_colors)}\n"
                 f"Single palette mode with perceptual color mapping"
        )

    def process_image(self):
        if self.current_image is None:
            messagebox.showwarning("Error", "Please load an image first")
            return

        width, height = self.current_image.size
        new_width = ((width + 7) // 8) * 8
        new_height = ((height + 7) // 8) * 8
        if width != new_width or height != new_height:
            self.current_image = self.current_image.resize((new_width, new_height), Image.Resampling.NEAREST)

        img_data = np.array(self.current_image.convert('RGBA'))
        unique_colors = {self.rgb_to_megadrive(tuple(pixel)) 
                        for pixel in img_data.reshape(-1, 4)}
        unique_colors.discard(None)  # Remove None (transparent) from the set
        
        self.palettes = self.cluster_colors_to_palettes(unique_colors)
        self.tile_map = self.extract_tiles(img_data)
        self.tile_palette_map, invalid_tiles = self.validate_and_assign_tile_palettes()
        
        if invalid_tiles:
            self.optimize_tile_palettes(invalid_tiles)

        # Create the converted image with transparency
        converted_data = np.zeros((new_height, new_width, 4), dtype=np.uint8)
        
        # Extract the alpha channel
        alpha_channel = img_data[..., 3]
        
        # Create a mapping of MD colors to RGB
        color_map = {}
        for pal in self.palettes:
            for color in pal:
                if color not in color_map:
                    rgb_color = self.megadrive_to_rgb(color)
                    color_map[color] = rgb_color

        # Convert the image while maintaining transparency
        for y in range(new_height):
            for x in range(new_width):
                if alpha_channel[y, x] < 128:  # Transparent pixel
                    converted_data[y, x] = [0, 0, 0, 0]
                else:
                    pixel = tuple(img_data[y, x])
                    md_color = self.rgb_to_megadrive(pixel)
                    if md_color is not None:
                        rgba_color = color_map[md_color]
                        converted_data[y, x] = [*rgba_color[:3], alpha_channel[y, x]]
        
        # Create final image
        self.simplified_image = Image.fromarray(converted_data, 'RGBA')
        self.display_image(self.simplified_image, self.converted_label)
        
        total_colors = sum(len(p) for p in self.palettes if p)
        self.info_label.config(
            text=f"Unique tiles: {len(self.tiles_data)}\n"
                 f"Total colors: {total_colors}\n"
                 f"Palettes used: {sum(1 for p in self.palettes if p)}\n"
                 f"Invalid tiles optimized: {len(invalid_tiles)}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            self.original_image = Image.open(file_path).convert('RGBA')
            self.current_image = self.original_image.copy()
            self.display_image(self.original_image, self.original_label)
            self.info_label.config(text="Image loaded. Click Convert to process.")

    def display_image(self, image, label):
        if image:
            width, height = image.size
            display_image = image.resize(
                (width * self.ZOOM_FACTOR, height * self.ZOOM_FACTOR), 
                Image.Resampling.NEAREST
            )
            photo = ImageTk.PhotoImage(display_image)
            label.config(image=photo)
            label.image = photo

    def save_image(self):
        if self.simplified_image is None:
            messagebox.showwarning("Error", "No converted image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")])
        
        if file_path:
            self.simplified_image.save(file_path, "PNG")
            messagebox.showinfo("Success", "Image saved successfully")

    def export_palettes(self):
        if not any(self.palettes):
            messagebox.showwarning("Error", "No palettes to export")
            return
        
        # Contamos cuántas paletas están realmente en uso
        used_palettes = sum(1 for p in self.palettes if p)
        
        # Create a larger palette image for better visibility
        CELL_SIZE = 32
        # Ahora la altura depende del número de paletas en uso
        palette_img = Image.new('RGBA', (16 * CELL_SIZE, used_palettes * CELL_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(palette_img)
        
        current_row = 0
        for pal_idx, palette in enumerate(self.palettes):
            if not palette:  # Skip empty palettes
                continue
                
            for col_idx, color in enumerate(palette):
                x = col_idx * CELL_SIZE
                y = current_row * CELL_SIZE
                
                if color is None:
                    # Draw checkerboard pattern for transparent colors
                    for i in range(CELL_SIZE):
                        for j in range(CELL_SIZE):
                            if (i + j) % 2 == 0:
                                draw.point((x + i, y + j), fill=(255, 255, 255, 255))
                            else:
                                draw.point((x + i, y + j), fill=(192, 192, 192, 255))
                else:
                    # Convert MegaDrive color to RGB
                    r, g, b = color
                    rgb_color = (
                        int((r / 15) * 255),
                        int((g / 15) * 255),
                        int((b / 15) * 255),
                        255  # Fully opaque for non-transparent colors
                    )
                    draw.rectangle([x, y, x + CELL_SIZE - 1, y + CELL_SIZE - 1], fill=rgb_color)
            
            current_row += 1
        
        # Add grid lines
        for i in range(17):
            x = i * CELL_SIZE
            draw.line([(x, 0), (x, used_palettes * CELL_SIZE)], fill=(64, 64, 64, 255), width=1)
        
        for i in range(used_palettes + 1):
            y = i * CELL_SIZE
            draw.line([(0, y), (16 * CELL_SIZE, y)], fill=(64, 64, 64, 255), width=1)
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")])
        
        if file_path:
            palette_img.save(file_path, "PNG")
            messagebox.showinfo("Success", "Palettes exported successfully")

def main():
    root = tk.Tk()
    app = MegaDrivePaletteConverter(root)
    root.mainloop()

if __name__ == "__main__":
    main()