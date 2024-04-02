# minecraft-rust

Resources used:

- https://sotrh.github.io/learn-wgpu/beginner/tutorial1-window/#added-support-for-the-web
- https://zdgeier.com/wgpuintro.html
- https://github.com/sotrh/learn-wgpu/issues/503
- https://github.com/nanovis/Wenderer/blob/5cef5e4befa7b172a03b00f2717f546ac7b8814d/src/main.rs

information needed to render a voxel
- chunk x - u32
- chunk z - u32
- position in chunk (u4, u8, u4)
- texture position in atlas (u8, u8)