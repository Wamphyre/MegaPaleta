# MegaPaleta

MegaPaleta is a specialized image conversion tool designed to adapt images to the Sega Mega Drive/Genesis color palette system. It supports both multi-palette tiles and single palette conversion while preserving transparency.

![Main Interface Screenshot](main.png)

## Features

- Converts images to Mega Drive/Genesis color format (RGB444)
- Supports up to 4 palettes of 16 colors each
- Single palette mode with optimized color reduction
- Transparent PNG support
- Real-time preview
- Exports palettes for easy integration into your development workflow
- Intelligent tile-based palette assignment
- Perceptually-based color distance calculation in YUV space

## Technical Details

### Color Format
- Input: Any PNG/JPG/BMP image
- Output: RGB444 format (4 bits per channel)
- First color in each palette can be transparent
- YUV-based perceptual color distance for optimal palette matching

### Palette System
- Maximum of 4 palettes
- 16 colors per palette
- Color 0 in each palette can be transparent
- Intelligent palette clustering using K-means algorithm

### Tile System
- 8x8 pixel tiles
- Automatic tile deduplication
- Smart palette assignment per tile
- Fallback optimization for tiles using colors from multiple palettes

### Color Reduction Algorithm
1. Initial color frequency analysis
2. Two-stage K-means clustering:
   - First stage: Up to 30 clusters for color space exploration
   - Second stage: Final reduction to 15 colors + transparency
3. Perceptual weighting based on color frequency
4. YUV color space distance calculation for better visual results

## Installation

### Windows
1. Download the latest installer (MegaPaleta_Setup.exe)
2. Run the installer and follow the prompts
3. Launch MegaPaleta from the Start Menu or desktop shortcut

### Building from Source
Prerequisites:
- Python 3.8 or higher
- Required packages: See requirements.txt

```bash
# Install dependencies
pip install -r requirements.txt

# Run from source
python megapaleta.py
```

## Usage

### Basic Operation
1. Click "Load Image" to open your source image
2. Choose conversion mode:
   - "Convert" for multi-palette mode (up to 4 palettes)
   - "Reduce to Single Palette" for single palette mode
3. Click "Save Image" to save the converted image
4. Optionally, use "Export Palettes" to save the color palettes

### Advanced Features

#### Multi-Palette Mode
- Automatically segments the image into tiles
- Assigns optimal palettes to each tile
- Handles tiles that need colors from multiple palettes
- Shows statistics about tile usage and optimization

#### Single Palette Mode
- Reduces colors while maintaining visual quality
- Preserves transparency
- Uses color frequency weighting for better results
- Optimal for images that don't need tile-based palette switching

## Technical Requirements

- OS: Windows 10 or later (64-bit)
- RAM: 2GB minimum
- Storage: 300MB free space
- Display: 1024x768 minimum resolution

## Building the Installer
1. Build the executable:
```bash
pyinstaller build.spec
```
2. Create the installer using Inno Setup:
- Open setup.iss with Inno Setup Compiler
- Click Compile
- Find the installer in the Output directory
