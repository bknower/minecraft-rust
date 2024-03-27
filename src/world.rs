use std::collections::HashMap;

use cgmath::{Point2, Point3};
use noise::{core::perlin, NoiseFn, Perlin, Seedable};

use crate::{
    block::Block,
    model::{Mesh, Model},
};

pub const CHUNK_SIZE_X: usize = 16;
pub const CHUNK_SIZE_Y: usize = 256;
pub const CHUNK_SIZE_Z: usize = 16;

pub struct Chunk {
    pub chunk_x: i32,
    pub chunk_z: i32,
    pub blocks: [[[Block; CHUNK_SIZE_Z]; CHUNK_SIZE_Y]; CHUNK_SIZE_X],
}

impl Chunk {
    fn new(chunk_x: i32, chunk_z: i32, perlin: Perlin) -> Self {
        let mut blocks = [[[Block::Air; CHUNK_SIZE_Z]; CHUNK_SIZE_Y]; CHUNK_SIZE_X];
        let sea_level = 80.0;
        let height_variability = 20.0;
        for x in 0usize..CHUNK_SIZE_X {
            for z in 0usize..CHUNK_SIZE_Z {
                let height_noise = perlin.get([
                    (x as f64 + 16.0 * chunk_x as f64) / 60000000.0,
                    (z as f64 + 16.0 * chunk_z as f64) / 60000000.0,
                ]) * 1000000.0;
                // println!("{}, {}, {}", height_noise, x, z);
                let height = (sea_level + height_noise * height_variability) as usize;
                for y in 0usize..CHUNK_SIZE_Y {
                    if y < height {
                        blocks[x][y][z] = Block::Grass;
                    } else {
                        blocks[x][y][z] = Block::Air;
                    }
                }
            }
        }
        Self {
            chunk_x,
            chunk_z,
            blocks,
        }
    }

    fn to_model(self) -> Model {
        let meshes = vec![];
        let materials = vec![];
        let material_map: HashMap<Block, usize> = HashMap::new();
        let blocks = self.blocks;
        for x in 0..blocks.len() {
            for y in 0..blocks[0].len() {
                for z in 0..blocks[0][0].len() {
                    let block = blocks[x][y][z];
                    use Block::*;
                    let left = x == 0 || blocks[x - 1][y][z] == Air;
                    let right = x == CHUNK_SIZE_X - 1 || blocks[x + 1][y][z] == Air;
                    let down = y == 0 || blocks[x][y - 1][z] == Air;
                    let up = y == CHUNK_SIZE_X - 1 || blocks[x][y + 1][z] == Air;
                    let back = z == 0 || blocks[x][y][z - 1] == Air;
                    let front = z == CHUNK_SIZE_X - 1 || blocks[x][y][z + 1] == Air;

                    let material_id = match material_map.get(&block) {
                        Some(id) => id
                    }
                }
            }
        }
        Model { meshes, materials }
    }
}
pub struct World {
    // chunks: Vec<Vec<Chunk>>,
    pub chunks: Vec<Chunk>,
    pub render_distance: i32,
    pub perlin: Perlin,
    pub position: Point3<f32>,
}

fn position_to_chunk_position(p: Point3<f32>) -> Point2<i32> {
    [
        (p.x / CHUNK_SIZE_X as f32) as i32,
        (p.z / CHUNK_SIZE_Z as f32) as i32,
    ]
    .into()
}
impl World {
    pub fn new(seed: u32) -> Self {
        let perlin = Perlin::new(seed);
        // let val = perlin.get([42.4, 37.7, 2.8]);
        let render_distance = 5;
        let chunks = vec![];
        Self {
            chunks,
            render_distance,
            perlin,
            position: (0.0, 0.0, 0.0).into(),
        }
    }

    pub fn update(&mut self, position: Point3<f32>) -> bool {
        let old_chunk_position = position_to_chunk_position(self.position);
        let chunk_position = position_to_chunk_position(position);
        if old_chunk_position != chunk_position || self.chunks.len() == 0 {
            self.position = position;
            println!(
                "old_chunk_position: {:?}, chunk_position: {:?}",
                old_chunk_position, chunk_position
            );
            let max_distance = 2 * self.render_distance * self.render_distance;
            let mut chunks_in_render_distance: Vec<Point2<i32>> = (-self.render_distance
                ..=self.render_distance)
                .flat_map(|x| {
                    (-self.render_distance..=self.render_distance).filter_map(move |z| {
                        let distance = x * x + z * z;
                        let point: Option<Point2<i32>> = match distance < max_distance {
                            true => Some((x + chunk_position.x, z + chunk_position.y).into()),
                            false => None,
                        };
                        point
                    })
                })
                .collect();
            // chunks_in_render_distance
            //     .iter()
            //     .for_each(|chunk| println!("chunk: {:?}", chunk));

            let mut new_chunks: Vec<Chunk> = vec![];

            // copy the already existing chunks to the new chunk array
            while let Some(chunk) = self.chunks.pop() {
                if let Some(index) = chunks_in_render_distance.iter().position(|chunk_coord| {
                    chunk.chunk_x == chunk_coord.x && chunk.chunk_z == chunk_coord.y
                }) {
                    chunks_in_render_distance.swap_remove(index);
                    println!(
                        "copied chunk: Point2 [{:?}, {:?}]",
                        chunk.chunk_x, chunk.chunk_z
                    );
                    new_chunks.push(chunk);
                }
            }

            // iterate over the remaining chunks (should be the newly rendered ones)
            chunks_in_render_distance.iter().for_each(|chunk_coord| {
                let new_chunk = Chunk::new(chunk_coord.x, chunk_coord.y, self.perlin);
                new_chunks.push(new_chunk);
                println!("new_chunk: {:?}", chunk_coord);
            });
            self.chunks = new_chunks;
            // for chunk in self.chunks {
            //     if chunks_in_render_distance.iter().any(|chunk_coord| {
            //         chunk.chunk_x == chunk_coord.x && chunk.chunk_z == chunk_coord.y
            //     }) {
            //         new_chunks.push(chunk);
            //     }
            // }
            // add any new chunks
            // chunks_in_render_distance.iter().for_each(|chunk_coord| {
            //     let chunk = self
            //         .chunks
            //         .iter()
            //         .find(|c| c.chunk_x == chunk_coord.x && c.chunk_z == chunk_coord.y);
            //     match chunk {
            //         Some(chunk) => new_chunks.push(),
            //         None => new_chunks.push(Chunk::new(chunk_coord.x, chunk_coord.y, self.perlin)),
            //     }
            // });
            // self.chunks = chunks_in_render_distance
            //     .iter()
            //     .map(|point| Chunk::new(point.x, point.y, self.perlin))
            //     .collect();
            return true;
        }
        return false;
    }
}
