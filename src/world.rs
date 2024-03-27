use std::collections::HashMap;

use cgmath::{Point2, Point3, Vector3};
use noise::{core::perlin, NoiseFn, Perlin, Seedable};

use crate::{
    block::Block,
    model::{Mesh, Model, ModelVertex},
};
use wgpu::util::DeviceExt;

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

    fn to_model(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> Model {
        let meshes = vec![];
        let materials = vec![];
        let material_map: HashMap<Block, usize> = HashMap::new();
        let blocks = self.blocks;

        let normal: [f32; 3] = [0.0, 0.0, 0.0];
        const V0: Vector3<f32> = [0.0, 0.0, 0.0].into();
        const V1: Vector3<f32> = [0.0, 0.0, 1.0].into();
        const V2: Vector3<f32> = [0.0, 1.0, 0.0].into();
        const V3: Vector3<f32> = [0.0, 1.0, 1.0].into();
        const V4: Vector3<f32> = [1.0, 0.0, 0.0].into();
        const V5: Vector3<f32> = [1.0, 0.0, 1.0].into();
        const V6: Vector3<f32> = [1.0, 1.0, 0.0].into();
        const V7: Vector3<f32> = [1.0, 1.0, 1.0].into();

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

        let down_vertices = vec![
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

        let up_vertices = vec![
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

        let back_vertices = vec![
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

        let front_vertices = vec![
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

        fn add_position_to_vertices(verts: Vec<ModelVertex>, p: Vector3<f32>) -> Vec<ModelVertex> {
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
                        tex_coords: *tex_coords,
                        normal: *normal,
                    }
                })
                .collect()
        }

        fn add_indices(vert_indices: Vec<u32>, starting_length: usize, indices: &mut Vec<u32>) {
            indices.append(
                &mut vert_indices
                    .iter()
                    .map(|i| (i + starting_length as u32) as u32)
                    .collect(),
            );
        }

        let mut meshes = vec![];

        for x in 0..blocks.len() {
            for y in 0..blocks[0].len() {
                for z in 0..blocks[0][0].len() {
                    let block = blocks[x][y][z];
                    use Block::*;
                    let left = x == 0 || blocks[x - 1][y][z] == Block::Air;
                    let right = x == CHUNK_SIZE_X - 1 || blocks[x + 1][y][z] == Block::Air;
                    let down = y == 0 || blocks[x][y - 1][z] == Block::Air;
                    let up = y == CHUNK_SIZE_X - 1 || blocks[x][y + 1][z] == Block::Air;
                    let back = z == 0 || blocks[x][y][z - 1] == Block::Air;
                    let front = z == CHUNK_SIZE_X - 1 || blocks[x][y][z + 1] == Block::Air;

                    let material_id = match material_map.get(&block) {
                        Some(id) => id,
                        None => {
                            let material = block.material(device, queue, layout);
                            materials.push(material);
                            &(materials.len() - 1)
                        }
                    };

                    let mut vertices: Vec<ModelVertex> = vec![];
                    let mut indices: Vec<u32> = vec![];

                    let position = Vector3::new(x as f32, y as f32, z as f32);

                    let face_tuples = vec![
                        (left, left_vertices, vec![0, 1, 2, 2, 1, 3]),
                        (right, right_vertices, vec![0, 2, 1, 1, 2, 3]),
                        (down, down_vertices, vec![0, 2, 1, 1, 2, 3]),
                        (up, up_vertices, vec![0, 1, 2, 2, 1, 3]),
                        (back, back_vertices, vec![1, 0, 3, 3, 0, 2]),
                        (front, front_vertices, vec![0, 1, 2, 2, 1, 3]),
                    ];

                    for face_tuple in face_tuples {
                        match face_tuple {
                            (direction, direction_vertices, direction_indices) => {
                                let starting_length = vertices.len() + 1;
                                vertices.append(&mut add_position_to_vertices(
                                    direction_vertices,
                                    position,
                                ));
                                add_indices(direction_indices, starting_length, &mut indices);
                            }
                        }
                    }
                    let vertex_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("{:?} Vertex Buffer", "block_name")),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let index_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

                    meshes.push(mesh);

                    // let add_indices = |starting_length: usize, added_indices: Vec<u32>| {
                    //     indices.append(added_indices.iter().map(|i| (i + starting_length as u32) as u32).collect());
                    // };
                    // fn add_indices(starting_length: usize, added_indices: &mut Vec<u32>, indices: &mut Vec<u32>) {
                    //     indices.append(added_indices.iter().map(|i| (i + starting_length as u32) as u32).collect());
                    // }
                    // if left {
                    //     let starting_length = vertices.len() + 1;
                    //     vertices.append(&mut add_position_to_vertices(left_vertices, position));
                    //     add_indices(vec![0, 1, 2, 2, 1, 3], starting_length, &mut indices);

                    //     // indices.append(&mut vec![0, 1, 2, 2, 1, 3].iter().map(|i| (i + starting_length) as u32).collect());
                    // }
                    // if right {
                    //     let starting_length = vertices.len() + 1;
                    //     vertices.append(&mut add_position_to_vertices(right_vertices, position));
                    //     add_indices(vec![0, 2, 1, 1, 2, 3], starting_length, &mut indices);
                    // }
                    // if down {
                    //     let starting_length = vertices.len() + 1;
                    //     vertices.append(&mut add_position_to_vertices(down_vertices, position));
                    //     add_indices(vec![0, 2, 1, 1, 2, 3], starting_length, &mut indices);
                    // }
                    // if up {
                    //     let starting_length = vertices.len() + 1;
                    //     vertices.append(&mut add_position_to_vertices(up_vertices, position));
                    //     add_indices(vec![0, 1, 2, 2, 1, 3], starting_length, &mut indices);
                    // }
                    // if back {
                    //     let starting_length = vertices.len() + 1;
                    //     vertices.append(&mut add_position_to_vertices(back_vertices, position));
                    //     add_indices(vec![1, 0, 3, 3, 0, 2], starting_length, &mut indices);
                    // }
                    // if front {
                    //     let starting_length = vertices.len() + 1;
                    //     vertices.append(&mut add_position_to_vertices(front_vertices, position));
                    //     add_indices(vec![0, 1, 2, 2, 1, 3], starting_length, &mut indices);
                    // }
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
