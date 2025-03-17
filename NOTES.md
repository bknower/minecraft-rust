# minecraft-rust

Resources used:

- https://sotrh.github.io/learn-wgpu/beginner/tutorial1-window/#added-support-for-the-web
- https://zdgeier.com/wgpuintro.html
- https://github.com/sotrh/learn-wgpu/issues/503
- https://github.com/nanovis/Wenderer/blob/5cef5e4befa7b172a03b00f2717f546ac7b8814d/src/main.rs

- https://www.reddit.com/r/VoxelGameDev/comments/1ceau2a/global_lattice_implementation/

information needed to render a voxel

- chunk x - u32
- chunk z - u32
- position in chunk (u4, u8, u4)
- texture position in atlas (u8, u8)

2 problems with meshing

- How can you make blocks tile properly? Vertices need to know what the atlas
  coords are -- tex coords are not enough, since vertices can be larger than
  blocks, and if they are then we will get to different blocks in the texture
  atlas when we extend past. if we do a modulus beforehand then there are other
  issues
- multiple different blocks means we have to run the meshing algorithm once per
  type of block.

Global lattice technique overview

- instead of meshing, we just create 64 \* 6 faces per chunk (one for each
  position 0 - 64 for each of the 6 possible faces )
- we upload a type buffer to the GPU, which tells it what type of block is at
  each location in the chunk
- the GPU can figure out how to render every face by just checking the type
  buffer to see what the texture is
- we can use a 2d array to store all the voxel data, with a single integer
  index that can be calculated each time
- each voxel can be represented as just a type integer index
