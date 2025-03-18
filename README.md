# Minecraft-Rust

A voxel engine / Minecraft clone built with Rust and wgpu.

## Libraries

- **wgpu**: Cross-platform graphics API
- **winit**: Cross-platform window handling
- **cgmath**: Math library for 3D graphics
- **noise**: Procedural noise generation for terrain
- **imgui**: Immediate mode GUI for debug / config

## Building and Running

### Setup

1. Clone the repository:

   ```
   git clone https://github.com/bknower/minecraft-rust.git
   cd minecraft-rust
   ```

2. Build and run in debug mode:

   ```
   cargo run
   ```

3. Build and run in release mode for better performance:
   ```
   cargo run --release
   ```

## Controls

- **WASD**: Move the camera
- **Mouse**: Look around
- **Space/Shift**: Move up/down

## Implementation Details

### Meshing Techniques

The project implements several meshing algorithms:

1. **Naive**: Renders each visible face of each block
2. **Greedy**: Combines contiguous 3D volumes of the same block type to reduce
   vertex count
3. **TODO: Global Lattice**: Technique that directly uploads a type buffer
   representing the blocks in a chunk to the GPU, letting the GPU handle meshing

## Source File Organization

- `src/block.rs`: Block type definitions and properties
- `src/camera.rs`: Camera, projection, and controls
- `src/imgui_renderer.rs`: imgui rendering code (adapted from [imgui-wgpu-rs](https://github.com/Yatekii/imgui-wgpu-rs/blob/master/src/lib.rs),
  which didn't support the version of winit I'm using)

- `src/lib.rs`: Main application logic (game loop, render pipeline)
- `src/main.rs`: Entry point
- `src/model.rs`: 3D model loading and rendering
- `src/resources.rs`: Resource loading utilities
- `src/texture.rs`: Texture loading and management
- `src/utils.rs`: Utility functions and helpers
- `src/world.rs`: World generation, chunk management, and meshing

- `src/shader.wgsl`: WGSL shader code for rendering
