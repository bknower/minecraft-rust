use noise::{core::perlin, NoiseFn, Perlin, Seedable};

use crate::block::Block;

pub struct Chunk {
    pub blocks: [[[Block; 16]; 256]; 16],
}

impl Chunk {
    fn new(chunk_x: i32, chunk_z: i32, perlin: Perlin) -> Self {
        let mut blocks: [[[Block; 16]; 256]; 16] = [[[Block::Air; 16]; 256]; 16];
        let sea_level = 80.0;
        let height_variability = 20.0;
        for x in 0usize..16 {
            for z in 0usize..16 {
                let height_noise = perlin.get([
                    (x as f64 + 16.0 * chunk_x as f64) / 16.0,
                    (z as f64 + 16.0 * chunk_z as f64) / 16.0,
                ]);
                // println!("{}, {}, {}", height_noise, x, z);
                let height = (sea_level + height_noise * height_variability) as usize;
                for y in 0usize..256 {
                    if y < height {
                        blocks[x][y][z] = Block::Grass;
                    } else {
                        blocks[x][y][z] = Block::Air;
                    }
                }
            }
        }
        Self { blocks }
    }
}
pub struct World {
    // chunks: Vec<Vec<Chunk>>,
    pub chunks: Vec<Chunk>,
}

impl World {
    pub fn new(seed: u32) -> Self {
        let perlin = Perlin::new(seed);
        // let val = perlin.get([42.4, 37.7, 2.8]);
        let chunks = vec![Chunk::new(0, 0, perlin)];
        Self { chunks }
    }
}
