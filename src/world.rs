use std::{
    collections::{HashMap, HashSet, VecDeque},
    hash::Hash,
    mem::size_of,
};

use cgmath::{InnerSpace, Point2, Point3, Vector2, Vector3};
use instant::{now, Duration};
use noise::{core::perlin, NoiseFn, Perlin, Seedable};
use tobj::Material;

use crate::{
    block::Block,
    model::{Mesh, Model, ModelVertex},
};
use wgpu::{naga::FastHashSet, util::DeviceExt, Device};

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

enum Side {
    LEFT,
    RIGHT,
    DOWN,
    UP,
    BACK,
    FRONT,
}

const V0: [f32; 3] = [0.0, 0.0, 0.0];
const V1: [f32; 3] = [0.0, 0.0, 1.0];
const V2: [f32; 3] = [0.0, 1.0, 0.0];
const V3: [f32; 3] = [0.0, 1.0, 1.0];
const V4: [f32; 3] = [1.0, 0.0, 0.0];
const V5: [f32; 3] = [1.0, 0.0, 1.0];
const V6: [f32; 3] = [1.0, 1.0, 0.0];
const V7: [f32; 3] = [1.0, 1.0, 1.0];

const BOTTOM_LEFT: [f32; 2] = [0.0, 0.0];
const TOP_LEFT: [f32; 2] = [0.0, 1.0];
const BOTTOM_RIGHT: [f32; 2] = [1.0, 0.0];
const TOP_RIGHT: [f32; 2] = [1.0, 1.0];

const LEFT_VERTICES: [ModelVertex; 4] = [
    ModelVertex {
        position: V0,
        tex_coords: BOTTOM_RIGHT,
    },
    ModelVertex {
        position: V2,
        tex_coords: TOP_RIGHT,
    },
    ModelVertex {
        position: V4,
        tex_coords: BOTTOM_LEFT,
    },
    ModelVertex {
        position: V6,
        tex_coords: TOP_LEFT,
    },
];

const RIGHT_VERTICES: [ModelVertex; 4] = [
    ModelVertex {
        position: V1,
        tex_coords: BOTTOM_LEFT,
    },
    ModelVertex {
        position: V3,
        tex_coords: TOP_LEFT,
    },
    ModelVertex {
        position: V5,
        tex_coords: BOTTOM_RIGHT,
    },
    ModelVertex {
        position: V7,
        tex_coords: TOP_RIGHT,
    },
];

const DOWN_VERTICES: [ModelVertex; 4] = [
    ModelVertex {
        position: V0,
        tex_coords: TOP_LEFT,
    },
    ModelVertex {
        position: V1,
        tex_coords: TOP_RIGHT,
    },
    ModelVertex {
        position: V4,
        tex_coords: BOTTOM_LEFT,
    },
    ModelVertex {
        position: V5,
        tex_coords: BOTTOM_RIGHT,
    },
];

const UP_VERTICES: [ModelVertex; 4] = [
    ModelVertex {
        position: V2,
        tex_coords: BOTTOM_LEFT,
    },
    ModelVertex {
        position: V3,
        tex_coords: TOP_LEFT,
    },
    ModelVertex {
        position: V6,
        tex_coords: BOTTOM_RIGHT,
    },
    ModelVertex {
        position: V7,
        tex_coords: TOP_RIGHT,
    },
];

const BACK_VERTICES: [ModelVertex; 4] = [
    ModelVertex {
        position: V4,
        tex_coords: BOTTOM_RIGHT,
    },
    ModelVertex {
        position: V5,
        tex_coords: BOTTOM_LEFT,
    },
    ModelVertex {
        position: V6,
        tex_coords: TOP_RIGHT,
    },
    ModelVertex {
        position: V7,
        tex_coords: TOP_LEFT,
    },
];

const FRONT_VERTICES: [ModelVertex; 4] = [
    ModelVertex {
        position: V0,
        tex_coords: BOTTOM_LEFT,
    },
    ModelVertex {
        position: V1,
        tex_coords: BOTTOM_RIGHT,
    },
    ModelVertex {
        position: V2,
        tex_coords: TOP_LEFT,
    },
    ModelVertex {
        position: V3,
        tex_coords: TOP_RIGHT,
    },
];

const LEFT_INDICES: [u32; 6] = [0, 1, 2, 2, 1, 3];
const RIGHT_INDICES: [u32; 6] = [0, 2, 1, 1, 2, 3];
const DOWN_INDICES: [u32; 6] = [0, 2, 1, 1, 2, 3];
const UP_INDICES: [u32; 6] = [0, 1, 2, 2, 1, 3];
const FRONT_INDICES: [u32; 6] = [0, 1, 2, 2, 1, 3];
const BACK_INDICES: [u32; 6] = [1, 0, 3, 3, 0, 2];

impl Chunk {
    fn new(chunk_x: i32, chunk_z: i32, perlin: Perlin) -> Self {
        let mut blocks = [[[Block::Air; CHUNK_SIZE_Z]; CHUNK_SIZE_Y]; CHUNK_SIZE_X];
        let sea_level = 80.0;
        let height_variability = 20.0;
        for x in 0usize..CHUNK_SIZE_X {
            for z in 0usize..CHUNK_SIZE_Z {
                let height_noise = perlin.get([
                    (x as f64 + 0.5 + 16.0 * chunk_x as f64) / 64.0,
                    (z as f64 + 0.5 + 16.0 * chunk_z as f64) / 64.0,
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
        if let None = self.mesh {
            let mesh = self.to_mesh(device, queue);
            self.mesh = Some(mesh);
        }
    }

    pub fn to_mesh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        // layout: &wgpu::BindGroupLayout,
    ) -> Mesh {
        let blocks = self.blocks;

        fn add_position_and_scale_to_vertices(
            verts: [ModelVertex; 4],
            start_coords: [f32; 3],
            scale: [f32; 3],
            atlas_coords: [f32; 2],
        ) -> Vec<ModelVertex> {
            verts
                .iter()
                .map(|vertex| {
                    let ModelVertex {
                        position,
                        tex_coords,
                        // normal,
                    } = vertex;
                    ModelVertex {
                        position: [
                            start_coords[0] + position[0] * scale[0],
                            start_coords[1] + position[1] * scale[1],
                            start_coords[2] + position[2] * scale[2],
                        ],
                        tex_coords: [
                            (atlas_coords[0] + tex_coords[0]) / ATLAS_SIZE as f32,
                            (atlas_coords[1] + tex_coords[1]) / ATLAS_SIZE as f32,
                        ],
                    }
                })
                .collect()
        }

        fn add_indices(vert_indices: [u32; 6], starting_length: usize, indices: &mut Vec<u32>) {
            indices.append(
                &mut vert_indices
                    .iter()
                    .map(|i| (i + starting_length as u32))
                    .collect(),
            );
        }

        fn add_combined_mesh(
            block: Block,
            start_coords: (usize, usize, usize),
            scale: (usize, usize, usize),
            indices: &mut Vec<u32>,
            vertices: &mut Vec<ModelVertex>,
        ) {
            // println!(
            //     "adding combined mesh from {:?} to {:?} with scale: {:?}",
            //     start_coords,
            //     (
            //         start_coords.0 + scale.0,
            //         start_coords.1 + scale.1,
            //         start_coords.2 + scale.2
            //     ),
            //     scale
            // );
            let atlas_coords = block.get_atlas_coords();
            let (start_x, start_y, start_z) = start_coords;
            let (scale_x, scale_y, scale_z) = scale;

            if let Some(atlas_coords) = atlas_coords {
                let position = [start_x as f32, start_y as f32, start_z as f32];
                let texture_length = atlas_coords.len();
                let face_tuples = [
                    (LEFT_VERTICES, LEFT_INDICES),
                    (RIGHT_VERTICES, RIGHT_INDICES),
                    (DOWN_VERTICES, DOWN_INDICES),
                    (UP_VERTICES, UP_INDICES),
                    (FRONT_VERTICES, FRONT_INDICES),
                    (BACK_VERTICES, BACK_INDICES),
                ];

                for (i, (direction_vertices, direction_indices)) in
                    face_tuples.into_iter().enumerate()
                {
                    let atlas_coords = atlas_coords.get(i).unwrap();

                    let starting_length = vertices.len();
                    vertices.append(&mut add_position_and_scale_to_vertices(
                        direction_vertices,
                        position,
                        [scale_x as f32, scale_y as f32, scale_z as f32],
                        *atlas_coords,
                    ));
                    add_indices(direction_indices, starting_length, indices);
                }
            }
        }
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
                        // normal,
                    } = vertex;
                    ModelVertex {
                        position: (p + Vector3::<f32>::from(*position)).into(),
                        tex_coords: ((Vector2::<f32>::from(*tex_coords) + atlas_coords)
                            / ATLAS_SIZE as f32)
                            .into(),
                        // tex_coords: *tex_coords,
                        // normal: *normal,
                    }
                })
                .collect()
        }

        // a chunk is 16  * 16 * 256 blocks
        let mut vertices: Vec<ModelVertex> = vec![];
        let mut indices: Vec<u32> = vec![];

        // greedy meshing

        // the problem is that we don't recheck the ranges while growing, so
        // they could maybe grow into the other ranges

        let coord_vec: Vec<(usize, usize, usize)> = (0..CHUNK_SIZE_X)
            .flat_map(|x| {
                (0..CHUNK_SIZE_Y).flat_map(move |y| (0..CHUNK_SIZE_Z).map(move |z| (x, y, z)))
            })
            .collect();

        let mut ranges: Vec<(usize, usize, usize, usize, usize, usize)> = vec![];

        for coords in coord_vec {
            use Block::*;
            let (start_x, start_y, start_z) = coords;
            let (mut end_x, mut end_y, mut end_z) = (CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z);

            // if this block is not in any of the existing ranges
            if !ranges.iter().any(|range| {
                let &(
                    range_x_start,
                    range_y_start,
                    range_z_start,
                    range_x_size,
                    range_y_size,
                    range_z_size,
                ) = range;

                let (range_x_end, range_y_end, range_z_end) = (
                    range_x_start + range_x_size,
                    range_y_start + range_y_size,
                    range_z_start + range_z_size,
                );
                let (after_start_x, after_start_y, after_start_z) = (
                    start_x >= range_x_start,
                    start_y >= range_y_start,
                    start_z >= range_z_start,
                );
                if !after_start_x {
                    end_x = end_x.min(range_x_start);
                }
                if !after_start_y {
                    end_y = end_y.min(range_y_start);
                }
                if !after_start_z {
                    end_z = end_z.min(range_z_start);
                }
                after_start_x
                    && start_x < range_x_end
                    && after_start_y
                    && start_y < range_y_end
                    && after_start_z
                    && start_z < range_z_end
            }) {
                let start_block = blocks[start_x][start_y][start_z];
                if start_block != Air {
                    for x in (start_x + 1)..end_x {
                        let curr_block = blocks[x][start_y][start_z];
                        if start_block != curr_block {
                            end_x = x;
                            break;
                        }
                    }

                    'outer: for z in (start_z + 1)..end_z {
                        for x in start_x..end_x {
                            let curr_block = blocks[x][start_y][z];
                            if start_block != curr_block {
                                end_z = z;
                                break 'outer;
                            }
                        }
                    }

                    'outer: for y in (start_y + 1)..end_y {
                        for x in start_x..end_x {
                            for z in start_z..end_z {
                                let curr_block = blocks[x][y][z];
                                if start_block != curr_block {
                                    end_y = y;
                                    break 'outer;
                                }
                            }
                        }
                    }

                    let scale = (end_x - start_x, end_y - start_y, end_z - start_z);
                    // add adjusted mesh
                    add_combined_mesh(start_block, coords, scale, &mut indices, &mut vertices);
                    ranges.push((start_x, start_y, start_z, scale.0, scale.1, scale.2));
                }
            }
        }

        // naive meshing
        // for x in 0..CHUNK_SIZE_X {
        //     for y in 0..CHUNK_SIZE_Y {
        //         for z in 0..CHUNK_SIZE_Z {
        //             let block = blocks[x][y][z];
        //             use Block::*;
        //             let front = x == 0 || blocks[x - 1][y][z] == Air;
        //             let back = x == CHUNK_SIZE_X - 1 || blocks[x + 1][y][z] == Air;
        //             let down = y == 0 || blocks[x][y - 1][z] == Air;
        //             let up = y == CHUNK_SIZE_Y - 1 || blocks[x][y + 1][z] == Air;
        //             let left = z == 0 || blocks[x][y][z - 1] == Air;
        //             let right = z == CHUNK_SIZE_Z - 1 || blocks[x][y][z + 1] == Air;
        //             // let (front, back, down, up, left, right) = (true, true, true, true, true, true);
        //             let atlas_coords = block.get_atlas_coords();

        //             if let Some(atlas_coords) = atlas_coords {
        //                 // let chunk_position =
        //                 // Vector3::new(self.chunk_x as f32, 0.0, self.chunk_z as f32);
        //                 let position = [x as f32, y as f32, z as f32];
        //                 let texture_length = atlas_coords.len();
        //                 let face_tuples = [
        //                     (left, LEFT_VERTICES, LEFT_INDICES),
        //                     (right, RIGHT_VERTICES, RIGHT_INDICES),
        //                     (down, DOWN_VERTICES, DOWN_INDICES),
        //                     (up, UP_VERTICES, UP_INDICES),
        //                     (front, FRONT_VERTICES, FRONT_INDICES),
        //                     (back, BACK_VERTICES, BACK_INDICES),
        //                 ];

        //                 for (i, (direction, direction_vertices, direction_indices)) in
        //                     face_tuples.into_iter().enumerate()
        //                 {
        //                     // let face_tuple = face_tuples.get(i).unwrap();
        //                     // let (direction, direction_vertices, direction_indices) = face_tuple;
        //                     let atlas_coords = if texture_length == 1 {
        //                         atlas_coords.first().unwrap()
        //                     } else {
        //                         atlas_coords.get(i).unwrap()
        //                     };
        //                     if direction {
        //                         let starting_length = vertices.len();
        //                         vertices.append(&mut add_position_to_vertices(
        //                             &direction_vertices,
        //                             position.into(),
        //                             (*atlas_coords).into(),
        //                         ));
        //                         add_indices(direction_indices, starting_length, &mut indices);
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        // println!("indices: {:?}", indices);

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

        // println!(
        //     "vertices: {:?}, indices: {:?}",
        //     vertices.len(),
        //     indices.len()
        // );

        mesh
    }
}

pub struct World {
    // chunks: Vec<Vec<Chunk>>,
    pub chunks: Vec<Chunk>,
    pub render_distance: i32,
    pub perlin: Perlin,
    pub position: Point3<f32>,
    pub chunks_to_generate: VecDeque<Point2<i32>>,
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
        let render_distance = 10;
        let chunks = vec![];
        Self {
            chunks,
            render_distance,
            perlin,
            position: (0.0, 0.0, 0.0).into(),
            chunks_to_generate: VecDeque::new(),
        }
    }

    pub fn update(
        &mut self,
        position: Point3<f32>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> bool {
        let mut updated = false;
        let old_chunk_position = position_to_chunk_position(self.position);
        let chunk_position = position_to_chunk_position(position);
        if old_chunk_position != chunk_position || self.chunks.is_empty() {
            self.position = position;
            self.chunks_to_generate.clear();
            println!(
                "old_chunk_position: {:?}, chunk_position: {:?}",
                old_chunk_position, chunk_position
            );
            let max_distance = 2 * self.render_distance * self.render_distance;
            let mut chunks_in_render_distance: HashSet<Point2<i32>> = (-self.render_distance
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

            // remove chunks that are no longer in render distance
            let mut i = 0;
            while i < self.chunks.len() {
                let chunk = self.chunks.get(i).unwrap();
                let chunk_coord = Point2::new(chunk.chunk_x, chunk.chunk_z);
                if !chunks_in_render_distance.contains(&chunk_coord) {
                    self.chunks.swap_remove(i);
                } else {
                    i += 1;
                    // we want to remove the already existing chunks from the
                    // set, so we don't have to create them again
                    chunks_in_render_distance.remove(&chunk_coord);
                }
            }

            // chunks_in_render_distance
            //     .iter()
            //     .for_each(|chunk| println!("chunk: {:?}", chunk));

            // let mut new_chunks: Vec<Chunk> = vec![];

            // copy the already existing chunks to the new chunk array
            // while let Some(chunk) = self.chunks.pop() {
            //     if let Some(index) = chunks_in_render_distance.iter().position(|chunk_coord| {
            //         chunk.chunk_x == chunk_coord.x && chunk.chunk_z == chunk_coord.y
            //     }) {
            //         chunks_in_render_distance.swap_remove(index);
            //         println!(
            //             "copied chunk: Point2 [{:?}, {:?}]",
            //             chunk.chunk_x, chunk.chunk_z
            //         );
            //         new_chunks.push(chunk);
            //     }
            // }

            // iterate over the remaining chunks (should be the newly rendered
            // ones)
            let mut chunks_vec: Vec<Point2<i32>> = chunks_in_render_distance.into_iter().collect();
            chunks_vec.sort_by(|a, b| {
                let da = (a - chunk_position).magnitude2();
                let db = (b - chunk_position).magnitude2();
                da.cmp(&db)
            });
            self.chunks_to_generate.extend(chunks_vec);

            // let now = instant::Instant::now();
            // let dt = now - last_render_time;
            // last_render_time = now;
            // chunks_in_render_distance
            //     .into_iter()
            //     .for_each(|chunk_coord| {
            //         let new_chunk = Chunk::new(chunk_coord.x, chunk_coord.y, self.perlin);
            //         new_chunks.push(new_chunk);
            //         println!("new_chunk: {:?}", chunk_coord);
            //     });
            // for chunk in &mut self.chunks {
            //     chunk.update(device, queue);
            // }
            updated = true;
        }
        let start = instant::Instant::now();
        let desired_fps = 120;
        let max_time: Duration = Duration::from_millis(1000 / desired_fps);

        while !self.chunks_to_generate.is_empty() && instant::Instant::now() < start + max_time {
            // println!("chunks to generate: {:?}", self.chunks_to_generate);
            let chunk_coord = self.chunks_to_generate.pop_front().unwrap();
            let chunk_start = now();
            let mut new_chunk = Chunk::new(chunk_coord.x, chunk_coord.y, self.perlin);
            let chunk_end = now();

            let mesh_start = now();
            new_chunk.update(device, queue);
            let mesh_end = now();
            self.chunks.push(new_chunk);
            println!("new_chunk: {:?}", chunk_coord);
            println!(
                "chunk time: {:?}ms, mesh time: {:?}ms",
                chunk_end - chunk_start,
                mesh_end - mesh_start,
            );
        }
        updated
    }
}
