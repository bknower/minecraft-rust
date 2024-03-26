use cgmath::{Point2, Point3};
use noise::{core::perlin, NoiseFn, Perlin, Seedable};

use crate::block::Block;

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
        let render_distance = 2;
        let chunks = vec![];
        Self {
            chunks,
            render_distance,
            perlin,
            position: (0.0, 0.0, 0.0).into(),
        }
    }

    pub fn update(&mut self, position: Point3<f32>) {
        let old_chunk_position = position_to_chunk_position(self.position);
        let chunk_position = position_to_chunk_position(position);
        if old_chunk_position != chunk_position || self.chunks.len() == 0 {
            self.position = position;
            println!(
                "position: {:?}, self.position: {:?}",
                position, self.position
            );
            let max_distance = self.render_distance * self.render_distance;
            let chunks_in_render_distance: Vec<Point2<i32>> = (-self.render_distance
                ..self.render_distance)
                .flat_map(|x| {
                    (-self.render_distance..self.render_distance).filter_map(move |z| {
                        let distance = x * x + z * z;
                        let point: Option<Point2<i32>> = match distance <= max_distance {
                            true => Some((x + chunk_position.x, z + chunk_position.y).into()),
                            false => None,
                        };
                        point
                    })
                })
                .collect();
            chunks_in_render_distance
                .iter()
                .for_each(|chunk| println!("chunk: {:?}", chunk));
            self.chunks = chunks_in_render_distance
                .iter()
                .map(|point| Chunk::new(point.x, point.y, self.perlin))
                .collect();
        }
    }
}
