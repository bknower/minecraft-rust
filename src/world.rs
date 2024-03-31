use std::collections::HashMap;

use cgmath::{Point2, Point3, Vector2, Vector3};
use noise::{core::perlin, NoiseFn, Perlin, Seedable};
use tobj::Material;

use crate::{
    block::Block,
    model::{Mesh, Model, ModelVertex},
};
use wgpu::{util::DeviceExt, Device};

pub const CHUNK_SIZE_X: usize = 16;
pub const CHUNK_SIZE_Y: usize = 256;
pub const CHUNK_SIZE_Z: usize = 16;

pub const ATLAS_SIZE: usize = 32;

pub struct Chunk {
    pub chunk_x: i32,
    pub chunk_z: i32,
    pub blocks: [[[Block; CHUNK_SIZE_Z]; CHUNK_SIZE_Y]; CHUNK_SIZE_X],
    pub mesh: Option<Mesh>,
}

impl Chunk {
    fn new(chunk_x: i32, chunk_z: i32, perlin: Perlin) -> Self {
        let mut blocks = [[[Block::Air; CHUNK_SIZE_Z]; CHUNK_SIZE_Y]; CHUNK_SIZE_X];
        let sea_level = 80.0;
        let height_variability = 20.0;
        for x in 0usize..CHUNK_SIZE_X {
            for z in 0usize..CHUNK_SIZE_Z {
                let height_noise = perlin.get([
                    (x as f64 + 0.5 + 16.0 * chunk_x as f64),
                    (z as f64 + 0.5 + 16.0 * chunk_z as f64),
                ]);
                // println!("{}, {}, {}", height_noise, x, z);
                let height = (sea_level + height_noise * height_variability) as usize;
                for y in 0usize..CHUNK_SIZE_Y {
                    if y < height {
                        blocks[x][y][z] = Block::Stone;
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
            mesh: None,
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mesh = self.to_mesh(&device, &queue);
        self.mesh = Some(mesh);
    }

    pub fn to_mesh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        // layout: &wgpu::BindGroupLayout,
    ) -> Mesh {
        let blocks = self.blocks;

        let normal: [f32; 3] = [0.0, 0.0, 0.0];
        const V0: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);
        const V1: Vector3<f32> = Vector3::new(0.0, 0.0, 1.0);
        const V2: Vector3<f32> = Vector3::new(0.0, 1.0, 0.0);
        const V3: Vector3<f32> = Vector3::new(0.0, 1.0, 1.0);
        const V4: Vector3<f32> = Vector3::new(1.0, 0.0, 0.0);
        const V5: Vector3<f32> = Vector3::new(1.0, 0.0, 1.0);
        const V6: Vector3<f32> = Vector3::new(1.0, 1.0, 0.0);
        const V7: Vector3<f32> = Vector3::new(1.0, 1.0, 1.0);

        let left_vertices = vec![
            ModelVertex {
                position: V0.into(),
                tex_coords: [1.0, 0.0],
                normal,
            },
            ModelVertex {
                position: V2.into(),
                tex_coords: [1.0, 1.0],
                normal,
            },
            ModelVertex {
                position: V4.into(),
                tex_coords: [0.0, 0.0],
                normal,
            },
            ModelVertex {
                position: V6.into(),
                tex_coords: [0.0, 1.0],
                normal,
            },
        ];

        let right_vertices = vec![
            ModelVertex {
                position: V1.into(),
                tex_coords: [0.0, 0.0],
                normal,
            },
            ModelVertex {
                position: V3.into(),
                tex_coords: [0.0, 1.0],
                normal,
            },
            ModelVertex {
                position: V5.into(),
                tex_coords: [1.0, 0.0],
                normal,
            },
            ModelVertex {
                position: V7.into(),
                tex_coords: [1.0, 1.0],
                normal,
            },
        ];

        let down_vertices = vec![
            ModelVertex {
                position: V0.into(),
                tex_coords: [0.0, 1.0],
                normal,
            },
            ModelVertex {
                position: V1.into(),
                tex_coords: [1.0, 1.0],
                normal,
            },
            ModelVertex {
                position: V4.into(),
                tex_coords: [0.0, 0.0],
                normal,
            },
            ModelVertex {
                position: V5.into(),
                tex_coords: [1.0, 0.0],
                normal,
            },
        ];

        let up_vertices = vec![
            ModelVertex {
                position: V2.into(),
                tex_coords: [0.0, 0.0],
                normal,
            },
            ModelVertex {
                position: V3.into(),
                tex_coords: [0.0, 1.0],
                normal,
            },
            ModelVertex {
                position: V6.into(),
                tex_coords: [1.0, 0.0],
                normal,
            },
            ModelVertex {
                position: V7.into(),
                tex_coords: [1.0, 1.0],
                normal,
            },
        ];

        let back_vertices = vec![
            ModelVertex {
                position: V4.into(),
                tex_coords: [1.0, 0.0],
                normal,
            },
            ModelVertex {
                position: V5.into(),
                tex_coords: [0.0, 0.0],
                normal,
            },
            ModelVertex {
                position: V6.into(),
                tex_coords: [1.0, 1.0],
                normal,
            },
            ModelVertex {
                position: V7.into(),
                tex_coords: [0.0, 1.0],
                normal,
            },
        ];

        let front_vertices = vec![
            ModelVertex {
                position: V0.into(),
                tex_coords: [0.0, 0.0],
                normal,
            },
            ModelVertex {
                position: V1.into(),
                tex_coords: [1.0, 0.0],
                normal,
            },
            ModelVertex {
                position: V2.into(),
                tex_coords: [0.0, 1.0],
                normal,
            },
            ModelVertex {
                position: V3.into(),
                tex_coords: [1.0, 1.0],
                normal,
            },
        ];

        fn add_position_to_vertices(
            verts: &[ModelVertex],
            p: Vector3<f32>,
            atlas_coords: Vector2<f32>,
        ) -> Vec<ModelVertex> {
            verts
                .iter()
                .map(|vertex| {
                    let ModelVertex {
                        position,
                        tex_coords,
                        normal,
                    } = vertex;
                    ModelVertex {
                        position: (p + Vector3::<f32>::from(*position)).into(),
                        tex_coords: ((Vector2::<f32>::from(*tex_coords) + atlas_coords)
                            / ATLAS_SIZE as f32)
                            .into(),
                        // tex_coords: *tex_coords,
                        normal: *normal,
                    }
                })
                .collect()
        }

        fn add_indices(vert_indices: Vec<u32>, starting_length: usize, indices: &mut Vec<u32>) {
            indices.append(
                &mut vert_indices
                    .iter()
                    .map(|i| (i + starting_length as u32) - 1)
                    .collect(),
            );
        }

        let mut vertices: Vec<ModelVertex> = vec![];
        let mut indices: Vec<u32> = vec![];

        for x in 0..blocks.len() {
            for y in 0..blocks[0].len() {
                for z in 0..blocks[0][0].len() {
                    let block = blocks[x][y][z];
                    use Block::*;
                    let left = x == 0 || blocks[x - 1][y][z] == Air;
                    let right = x == CHUNK_SIZE_X - 1 || blocks[x + 1][y][z] == Air;
                    let down = y == 0 || blocks[x][y - 1][z] == Air;
                    let up = y == CHUNK_SIZE_Y - 1 || blocks[x][y + 1][z] == Air;
                    let front = z == 0 || blocks[x][y][z - 1] == Air;
                    let back = z == CHUNK_SIZE_Z - 1 || blocks[x][y][z + 1] == Air;

                    let atlas_coords = block.get_atlas_coords();

                    if let Some(atlas_coords) = atlas_coords {
                        let position = Vector3::new(x as f32, y as f32, z as f32);
                        let texture_length = atlas_coords.len();
                        let face_tuples = vec![
                            (left, left_vertices.as_slice(), vec![0, 1, 2, 2, 1, 3]),
                            (right, right_vertices.as_slice(), vec![0, 2, 1, 1, 2, 3]),
                            (down, down_vertices.as_slice(), vec![0, 2, 1, 1, 2, 3]),
                            (up, up_vertices.as_slice(), vec![0, 1, 2, 2, 1, 3]),
                            (back, back_vertices.as_slice(), vec![1, 0, 3, 3, 0, 2]),
                            (front, front_vertices.as_slice(), vec![0, 1, 2, 2, 1, 3]),
                        ];

                        for (i, (direction, direction_vertices, direction_indices)) in
                            face_tuples.into_iter().enumerate()
                        {
                            // let face_tuple = face_tuples.get(i).unwrap();
                            // let (direction, direction_vertices, direction_indices) = face_tuple;
                            let atlas_coords = if texture_length == 1 {
                                atlas_coords.first().unwrap()
                            } else {
                                atlas_coords.get(i).unwrap()
                            };
                            if direction {
                                let starting_length = vertices.len() + 1;
                                vertices.append(&mut add_position_to_vertices(
                                    direction_vertices,
                                    position,
                                    *atlas_coords,
                                ));
                                add_indices(direction_indices, starting_length, &mut indices);
                            }
                        }
                    }
                }
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Vertex Buffer", "block_name")),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Index Buffer", "block_name")),
            contents: bytemuck::cast_slice(indices.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mesh = Mesh {
            name: "".to_string(),
            vertex_buffer,
            index_buffer,
            num_elements: indices.len() as u32,
            material: 0,
        };

        mesh
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
        let render_distance = 1;
        let chunks = vec![];
        Self {
            chunks,
            render_distance,
            perlin,
            position: (0.0, 0.0, 0.0).into(),
        }
    }

    pub fn update(
        &mut self,
        position: Point3<f32>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> bool {
        let old_chunk_position = position_to_chunk_position(self.position);
        let chunk_position = position_to_chunk_position(position);
        if old_chunk_position != chunk_position || self.chunks.is_empty() {
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
            for chunk in &mut self.chunks {
                chunk.update(device, queue);
            }
            return true;
        }
        false
    }
}
