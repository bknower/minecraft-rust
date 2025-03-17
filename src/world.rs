use derive_more::Display;
use paste::paste;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
    hash::Hash,
    mem::size_of,
};
use strum::IntoEnumIterator;

use cgmath::{InnerSpace, Point2, Point3, Vector2, Vector3};
use instant::{now, Duration};
use noise::{core::perlin, NoiseFn, Perlin, Seedable};
use tobj::Material;

use crate::{
    block::Block,
    model::{Mesh, Model, ModelVertex},
    printy,
};
use wgpu::{naga::FastHashSet, util::DeviceExt, Device};

pub const CHUNK_SIZE_X: usize = 64;
pub const CHUNK_SIZE_Y: usize = 64;
pub const CHUNK_SIZE_Z: usize = 64;

pub const ATLAS_SIZE: usize = 32;

pub fn index_to_xyz(index: usize) -> (usize, usize, usize) {
    return (
        index % CHUNK_SIZE_X,
        (index / CHUNK_SIZE_Y) % CHUNK_SIZE_X,
        ((index / CHUNK_SIZE_Z) / CHUNK_SIZE_Y) % CHUNK_SIZE_X,
    );
}

pub fn xyz_to_index(x: usize, y: usize, z: usize) -> usize {
    return x + z * CHUNK_SIZE_X + y * CHUNK_SIZE_X * CHUNK_SIZE_Z;
}

pub struct Chunk {
    pub chunk_x: i32,
    pub chunk_z: i32,
    pub blocks: [Block; CHUNK_SIZE_Z * CHUNK_SIZE_Y * CHUNK_SIZE_Z],
    pub mesh: Option<Mesh>,
}

#[derive(Display, Debug)]
enum Side {
    LEFT,
    RIGHT,
    DOWN,
    UP,
    BACK,
    FRONT,
}

impl Side {
    /// Convert enum to the corresponding index for atlas_coords and face tuples.
    fn as_index(&self) -> usize {
        match self {
            Side::LEFT => 0,
            Side::RIGHT => 1,
            Side::DOWN => 2,
            Side::UP => 3,
            Side::FRONT => 4,
            Side::BACK => 5,
        }
    }

    fn from_index(index: usize) -> Side {
        match index {
            0 => Side::LEFT,
            1 => Side::RIGHT,
            2 => Side::DOWN,
            3 => Side::UP,
            4 => Side::FRONT,
            5 => Side::BACK,
            _ => unreachable!(),
        }
    }
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
        position: V1,
        tex_coords: BOTTOM_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V0,
        tex_coords: BOTTOM_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V3,
        tex_coords: TOP_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V2,
        tex_coords: TOP_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
];

const RIGHT_VERTICES: [ModelVertex; 4] = [
    ModelVertex {
        position: V4,
        tex_coords: BOTTOM_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V5,
        tex_coords: BOTTOM_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V6,
        tex_coords: TOP_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V7,
        tex_coords: TOP_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
];

const DOWN_VERTICES: [ModelVertex; 4] = [
    ModelVertex {
        position: V1,
        tex_coords: BOTTOM_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V5,
        tex_coords: BOTTOM_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V0,
        tex_coords: TOP_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V4,
        tex_coords: TOP_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
];

const UP_VERTICES: [ModelVertex; 4] = [
    ModelVertex {
        position: V2,
        tex_coords: BOTTOM_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V6,
        tex_coords: BOTTOM_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V3,
        tex_coords: TOP_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V7,
        tex_coords: TOP_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
];

const BACK_VERTICES: [ModelVertex; 4] = [
    ModelVertex {
        position: V5,
        tex_coords: BOTTOM_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V1,
        tex_coords: BOTTOM_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V7,
        tex_coords: TOP_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V3,
        tex_coords: TOP_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
];

const FRONT_VERTICES: [ModelVertex; 4] = [
    ModelVertex {
        position: V0,
        tex_coords: BOTTOM_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V4,
        tex_coords: BOTTOM_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V2,
        tex_coords: TOP_LEFT,
        atlas_coords: [0.0, 0.0],
    },
    ModelVertex {
        position: V6,
        tex_coords: TOP_RIGHT,
        atlas_coords: [0.0, 0.0],
    },
];

const FACE_INDICES: [u32; 6] = [3, 1, 0, 2, 3, 0];

impl Chunk {
    fn new(chunk_x: i32, chunk_z: i32) -> Self {
        // let mut blocks = [[[Block::Air; CHUNK_SIZE_Z]; CHUNK_SIZE_Y];
        // CHUNK_SIZE_X];
        let mut blocks = [Block::Air; CHUNK_SIZE_X * CHUNK_SIZE_Y * CHUNK_SIZE_Z];
        Self {
            chunk_x,
            chunk_z,
            blocks,
            mesh: None,
        }
    }

    fn generate(&mut self, perlin: Perlin) {
        let sea_level = 40.0;
        let height_variability = 20.0;
        for x in 0usize..CHUNK_SIZE_X {
            for z in 0usize..CHUNK_SIZE_Z {
                let height_noise = perlin.get([
                    (x as f64 + 0.5 + 16.0 * self.chunk_x as f64) / 64.0,
                    (z as f64 + 0.5 + 16.0 * self.chunk_z as f64) / 64.0,
                ]);
                // println!("{}, {}, {}", height_noise, x, z);
                let height = (sea_level + height_noise * height_variability) as usize;
                for y in 0usize..CHUNK_SIZE_Y {
                    if y < height {
                        self.set_block(x, y, z, Block::Grass);
                        // blocks[x][y][z] = Block::Stone;
                    } else {
                        self.set_block(x, y, z, Block::Air);
                        // blocks[x][y][z] = Block::Air;
                    }
                }
            }
        }
        // for x in 0usize..CHUNK_SIZE_X {
        //     for z in 0usize..CHUNK_SIZE_Z {
        //         let height_noise = perlin.get([
        //             (x as f64 + 0.5 + 16.0 * self.chunk_x as f64) / 64.0,
        //             (z as f64 + 0.5 + 16.0 * self.chunk_z as f64) / 64.0,
        //         ]);
        //         // println!("{}, {}, {}", height_noise, x, z);
        //         let height = (sea_level + height_noise * height_variability) as usize;
        //         for y in 0usize..CHUNK_SIZE_Y {
        //             if x == 0 && y == 0 && z == 0 {
        //                 self.set_block(x, y, z, Block::Stone);
        //                 // blocks[x][y][z] = Block::Stone;
        //             } else {
        //                 self.set_block(x, y, z, Block::Air);
        //                 // blocks[x][y][z] = Block::Air;
        //             }
        //         }
        //     }
        // }
    }

    pub fn get_3d<T>(array: &[T], x: usize, y: usize, z: usize) -> T
    where
        T: Default,
        T: Copy,
    {
        if x < 0 || x >= CHUNK_SIZE_X || y < 0 || y >= CHUNK_SIZE_Y || z < 0 || z >= CHUNK_SIZE_Z {
            Default::default()
        } else {
            array[xyz_to_index(x, y, z)]
        }
    }

    pub fn set_3d<T>(array: &mut [T], x: usize, y: usize, z: usize, val: T) {
        array[xyz_to_index(x, y, z)] = val
    }

    pub fn get_block(&self, x: usize, y: usize, z: usize) -> Block {
        // not sure if this works when these sizes are different
        Chunk::get_3d(&self.blocks, x, y, z)
    }
    pub fn set_block(&mut self, x: usize, y: usize, z: usize, block: Block) {
        // not sure if this works when these sizes are different
        Chunk::set_3d(&mut self.blocks, x, y, z, block);
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, meshing_algorithm: usize) {
        if let None = self.mesh {
            let mesh = self.to_mesh(device, queue, meshing_algorithm);
            self.mesh = Some(mesh);
        }
    }

    pub fn to_mesh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshing_algorithm: usize, // layout: &wgpu::BindGroupLayout,
    ) -> Mesh {
        let blocks = self.blocks;
        fn add_position_and_scale_to_vertices(
            verts: [ModelVertex; 4],
            start_coords: [f32; 3],
            scale: [f32; 3],
            atlas_offset: [f32; 2],
            face_index: usize,
        ) -> Vec<ModelVertex> {
            verts
                .iter()
                .map(|v| {
                    // Always scale all 3 dimensions for the position
                    let scaled_pos = [
                        start_coords[0] + v.position[0] * scale[0],
                        start_coords[1] + v.position[1] * scale[1],
                        start_coords[2] + v.position[2] * scale[2],
                    ];

                    let scaled_uv = match Side::from_index(face_index) {
                        Side::LEFT | Side::RIGHT => {
                            [v.tex_coords[0] * scale[2], v.tex_coords[1] * scale[1]]
                        }
                        Side::FRONT | Side::BACK => {
                            [v.tex_coords[0] * scale[0], v.tex_coords[1] * scale[1]]
                        }
                        Side::DOWN | Side::UP => {
                            [v.tex_coords[0] * scale[0], v.tex_coords[1] * scale[2]]
                        }
                    };

                    // printy!(
                    //     Side::from_index(face_index),
                    //     scaled_pos,
                    //     scaled_uv,
                    //     v.position,
                    //     v.tex_coords
                    // );

                    ModelVertex {
                        position: scaled_pos,
                        tex_coords: scaled_uv,
                        atlas_coords: atlas_offset,
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
            let atlas_coords = block.get_atlas_coords();
            let (start_x, start_y, start_z) = start_coords;
            let (scale_x, scale_y, scale_z) = scale;

            if let Some(atlas_coords) = atlas_coords {
                let position = [start_x as f32, start_y as f32, start_z as f32];
                let texture_length = atlas_coords.len();
                let face_tuples = [
                    (LEFT_VERTICES, FACE_INDICES),
                    (RIGHT_VERTICES, FACE_INDICES),
                    (DOWN_VERTICES, FACE_INDICES),
                    (UP_VERTICES, FACE_INDICES),
                    (FRONT_VERTICES, FACE_INDICES),
                    (BACK_VERTICES, FACE_INDICES),
                ];

                for (i, (direction_vertices, direction_indices)) in
                    face_tuples.into_iter().enumerate()
                {
                    // if i > 3 {
                    //     continue;
                    // }
                    let atlas_coords = atlas_coords.get(i).unwrap();

                    let starting_length = vertices.len();
                    vertices.append(&mut add_position_and_scale_to_vertices(
                        direction_vertices,
                        position,
                        [scale_x as f32, scale_y as f32, scale_z as f32],
                        *atlas_coords,
                        i,
                    ));
                    add_indices(direction_indices, starting_length, indices);
                }
            }
        }

        fn add_single_face_mesh_2d(
            block: Block,
            start_coords: (usize, usize, usize),
            scale: (usize, usize, usize),
            face: Side,
            indices: &mut Vec<u32>,
            vertices: &mut Vec<ModelVertex>,
        ) {
            // printy!(start_coords, scale);
            let atlas_coords_opt = block.get_atlas_coords();
            if atlas_coords_opt.is_none() {
                // If there's no texture data, we can't add anything.
                return;
            }

            let atlas_coords = atlas_coords_opt.unwrap();
            let (start_x, start_y, start_z) = start_coords;
            let (scale_x, scale_y, scale_z) = scale;
            let position = [start_x as f32, start_y as f32, start_z as f32];

            // All possible face definitions (vertices and indices):
            let face_tuples = [
                (LEFT_VERTICES, FACE_INDICES),
                (RIGHT_VERTICES, FACE_INDICES),
                (DOWN_VERTICES, FACE_INDICES),
                (UP_VERTICES, FACE_INDICES),
                (FRONT_VERTICES, FACE_INDICES),
                (BACK_VERTICES, FACE_INDICES),
            ];

            // Determine which face we are adding
            let i = face.as_index();
            // printy!(i);

            // Retrieve the relevant data for that face
            let (direction_vertices, direction_indices) = face_tuples[i];
            let face_atlas_coords = atlas_coords[i];

            // Add the face
            let starting_length = vertices.len();
            vertices.append(&mut add_position_and_scale_to_vertices(
                direction_vertices,
                position,
                [scale_x as f32, scale_y as f32, scale_z as f32],
                face_atlas_coords,
                i,
            ));
            add_indices(direction_indices, starting_length, indices);
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
                        ..
                    } = vertex;
                    ModelVertex {
                        position: (p + Vector3::<f32>::from(*position)).into(),
                        tex_coords: Vector2::<f32>::from(*tex_coords).into(),
                        // tex_coords: *tex_coords,
                        // normal: *normal,
                        atlas_coords: atlas_coords.into(),
                    }
                })
                .collect()
        }

        // a chunk is 16  * 16 * 256 blocks
        let mut vertices: Vec<ModelVertex> = vec![];
        let mut indices: Vec<u32> = vec![];

        // greedy meshing
        const CHUNK_SIZE: usize = CHUNK_SIZE_X;
        match *MESHING_ALGORITHMS.get(meshing_algorithm).unwrap() {
            "greedy" => {
                let mut masks: Vec<(Block, [bool; CHUNK_SIZE_X * CHUNK_SIZE_Y * CHUNK_SIZE_Z])> =
                    vec![];
                for block_type in Block::iter() {
                    masks.push((
                        block_type,
                        [false; CHUNK_SIZE_X * CHUNK_SIZE_Y * CHUNK_SIZE_Z],
                    ));
                }
                for (i, block) in self.blocks.iter().enumerate() {
                    masks.get_mut(*block as usize).unwrap().1[i] = true;
                }

                for block_type in Block::iter() {
                    if block_type == Block::Air {
                        continue;
                    }
                    let mut mask: [bool; CHUNK_SIZE_X * CHUNK_SIZE_Y * CHUNK_SIZE_Z] =
                        masks.get_mut(block_type as usize).unwrap().1;
                    for start_x in 0..CHUNK_SIZE_X {
                        for start_y in 0..CHUNK_SIZE_Y {
                            for start_z in 0..CHUNK_SIZE_Z {
                                use Block::*;
                                let (mut end_x, mut end_y, mut end_z) =
                                    (CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z);

                                if Chunk::get_3d(&mut mask, start_x, start_y, start_z) {
                                    for x in (start_x + 1)..end_x {
                                        if !Chunk::get_3d(&mut mask, x, start_y, start_z) {
                                            end_x = x;
                                            break;
                                        }
                                    }

                                    'outer: for z in (start_z + 1)..end_z {
                                        for x in start_x..end_x {
                                            if !Chunk::get_3d(&mut mask, x, start_y, z) {
                                                end_z = z;
                                                break 'outer;
                                            }
                                        }
                                    }

                                    'outer: for y in (start_y + 1)..end_y {
                                        for x in start_x..end_x {
                                            for z in start_z..end_z {
                                                if !Chunk::get_3d(&mut mask, x, y, z) {
                                                    end_y = y;
                                                    break 'outer;
                                                }
                                            }
                                        }
                                    }

                                    let scale = (end_x - start_x, end_y - start_y, end_z - start_z);
                                    add_combined_mesh(
                                        block_type,
                                        (start_x, start_y, start_z),
                                        scale,
                                        &mut indices,
                                        &mut vertices,
                                    );
                                    for x in start_x..end_x {
                                        for y in start_y..end_y {
                                            for z in start_z..end_z {
                                                Chunk::set_3d(&mut mask, x, y, z, false);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            "naive" => {
                for x in 0..CHUNK_SIZE_X {
                    for y in 0..CHUNK_SIZE_Y {
                        for z in 0..CHUNK_SIZE_Z {
                            let block = self.get_block(x, y, z);
                            if let Some(atlas_coords) = block.get_atlas_coords() {
                                // Determine if each face is visible
                                // (i.e. there’s air or an edge on that side).
                                let is_left_face_visible =
                                    (x == 0) || self.get_block(x - 1, y, z) == Block::Air;
                                let is_right_face_visible = (x == CHUNK_SIZE_X - 1)
                                    || self.get_block(x + 1, y, z) == Block::Air;

                                let is_front_face_visible =
                                    (z == 0) || self.get_block(x, y, z - 1) == Block::Air;
                                let is_back_face_visible = (z == CHUNK_SIZE_Z - 1)
                                    || self.get_block(x, y, z + 1) == Block::Air;

                                let is_down_face_visible =
                                    (y == 0) || self.get_block(x, y - 1, z) == Block::Air;
                                let is_up_face_visible = (y == CHUNK_SIZE_Y - 1)
                                    || self.get_block(x, y + 1, z) == Block::Air;

                                let position = [x as f32, y as f32, z as f32];

                                // Pair each “visibility flag” with the correct face array.
                                // The enumerations match the indexing inside atlas_coords.
                                let face_tuples = [
                                    (is_left_face_visible, &LEFT_VERTICES, &FACE_INDICES),
                                    (is_right_face_visible, &RIGHT_VERTICES, &FACE_INDICES),
                                    (is_down_face_visible, &DOWN_VERTICES, &FACE_INDICES),
                                    (is_up_face_visible, &UP_VERTICES, &FACE_INDICES),
                                    (is_back_face_visible, &BACK_VERTICES, &FACE_INDICES),
                                    (is_front_face_visible, &FRONT_VERTICES, &FACE_INDICES),
                                ];

                                // Now add whichever faces are visible.
                                for (i, (face_visible, direction_vertices, direction_indices)) in
                                    face_tuples.iter().enumerate()
                                {
                                    if *face_visible {
                                        // Grab the correct sub‐tile offset from atlas_coords[i]
                                        let uv_offset = atlas_coords[i];

                                        let start_len = vertices.len() as u32;
                                        // Add the 4 face‐corner vertices
                                        for &base_v in direction_vertices.iter() {
                                            let mut v = base_v;
                                            // Move the face to (x,y,z)
                                            v.position = [
                                                position[0] + v.position[0],
                                                position[1] + v.position[1],
                                                position[2] + v.position[2],
                                            ];
                                            // Set the block’s atlas offset
                                            v.atlas_coords = uv_offset;
                                            vertices.push(v);
                                        }
                                        // Add the face’s 6 indices
                                        for &idx in (*direction_indices).iter() {
                                            indices.push(start_len + idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            _ => unreachable!(),
        }

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
    pub last_render_distance: i32,
    pub perlin: Perlin,
    pub position: Point3<f32>,
    pub chunks_to_generate: VecDeque<Point2<i32>>,
    pub world_stats: WorldStats,
    pub meshing_algorithm: usize,
    pub last_meshing_algorithm: usize,
}

// macro written by chatgpt to auto implement adding new times and getting
// averages from times
macro_rules! impl_stat_methods {
    ($($field:ident => $short_name:ident: $type:ty),*) => {
        impl WorldStats {
            $(
                paste! {
                    pub fn [<add_ $short_name>](&mut self, value: $type) {
                        if self.$field.len() < self.rolling_average_length {
                            self.$field.push(value);
                        } else {
                            self.$field.remove(0);
                            self.$field.push(value);
                        }
                    }
                }

                paste! {
                    pub fn [<get_ $short_name _avg>](&self) -> f64 {
                        if self.$field.is_empty() {
                            0.0
                        } else {
                            self.$field.iter().map(|&x| x as f64).sum::<f64>() / self.$field.len() as f64
                        }
                    }
                }
            )*
        }
    };
}

pub struct WorldStats {
    pub rolling_average_length: usize,
    pub chunk_gen_times: Vec<f64>,
    pub mesh_gen_times: Vec<f64>,
    pub triangles: u64,
    pub chunk_triangles: Vec<u64>,
}

impl Default for WorldStats {
    fn default() -> WorldStats {
        WorldStats {
            rolling_average_length: 20,
            chunk_gen_times: vec![],
            mesh_gen_times: vec![],
            triangles: 0,
            chunk_triangles: vec![],
        }
    }
}

impl_stat_methods!(
    chunk_gen_times => chunk_gen_time: f64,
    mesh_gen_times => mesh_gen_time: f64,
    chunk_triangles => chunk_triangles: u64
);

// impl WorldStats {
//     fn new_time(&mut self, time: f64, vec: &mut Vec<f64>) {
//         if vec.len() < self.rolling_average_length {
//             vec.push(time);
//         } else {
//             vec.remove(0);
//             vec.push(time)
//         }
//     }

//     fn get_average(vec: &Vec<f64>) -> f64 {
//         vec.iter().sum::<f64>() / vec.len() as f64
//     }
// }

fn position_to_chunk_position(p: Point3<f32>) -> Point2<i32> {
    [
        (p.x / CHUNK_SIZE_X as f32) as i32,
        (p.z / CHUNK_SIZE_Z as f32) as i32,
    ]
    .into()
}

pub const MESHING_ALGORITHMS: [&str; 5] = ["naive", "greedy", "greedy2", "greedy3", "greedy4"];

impl World {
    pub fn new(seed: u32) -> Self {
        let perlin = Perlin::new(seed);
        // let val = perlin.get([42.4, 37.7, 2.8]);
        let render_distance = 20;
        let chunks = vec![];
        Self {
            chunks,
            render_distance,
            last_render_distance: 0,
            perlin,
            position: (0.0, 0.0, 0.0).into(),
            chunks_to_generate: VecDeque::new(),
            world_stats: WorldStats::default(),
            meshing_algorithm: 1,
            last_meshing_algorithm: 0,
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
        if self.last_meshing_algorithm != self.meshing_algorithm
            || self.last_render_distance != self.render_distance
        {
            self.last_meshing_algorithm = self.meshing_algorithm;
            self.last_render_distance = self.render_distance;
            self.chunks_to_generate.clear();
            self.chunks.clear()
        }
        if old_chunk_position != chunk_position || self.chunks.is_empty() {
            self.position = position;
            self.chunks_to_generate.clear();
            println!(
                "old_chunk_position: {:?}, chunk_position: {:?}",
                old_chunk_position, chunk_position
            );
            let max_distance = self.render_distance * CHUNK_SIZE_X as i32;
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

            // iterate over the remaining chunks (should be the newly rendered
            // ones)
            let mut chunks_vec: Vec<Point2<i32>> = chunks_in_render_distance.into_iter().collect();
            chunks_vec.sort_by(|a, b| {
                let da = (a - chunk_position).magnitude2();
                let db = (b - chunk_position).magnitude2();
                da.cmp(&db)
            });
            self.chunks_to_generate.extend(chunks_vec);
            updated = true;
        }
        let start = instant::Instant::now();
        let desired_fps = 120;
        let max_time: Duration = Duration::from_millis(1000 / desired_fps);

        while !self.chunks_to_generate.is_empty() && instant::Instant::now() < start + max_time {
            // println!("chunks to generate: {:?}", self.chunks_to_generate);
            let chunk_coord = self.chunks_to_generate.pop_front().unwrap();
            let chunk_start = now();
            let mut new_chunk = Chunk::new(chunk_coord.x, chunk_coord.y);
            new_chunk.generate(self.perlin);
            let chunk_end = now();

            let mesh_start = now();
            new_chunk.update(device, queue, self.meshing_algorithm);
            let mesh_end = now();
            self.chunks.push(new_chunk);
            self.world_stats.add_chunk_gen_time(chunk_end - chunk_start);
            self.world_stats.add_mesh_gen_time(mesh_end - mesh_start);
        }
        updated
    }
}
