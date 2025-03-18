use crate::{
    model::{Material, Mesh},
    resources::load_texture,
};
use cgmath::Vector2;
use strum_macros::EnumIter;
#[derive(Copy, Clone, PartialEq, Eq, Hash, EnumIter, Default)]

pub enum Block {
    #[default]
    Air,
    Grass,
    Stone,
    Wool,
}

impl Block {
    pub fn get_atlas_coords(self) -> Option<[[f32; 2]; 6]> {
        match self {
            Block::Wool => Some([
                [22.0, 0.0],
                [22.0, 1.0],
                [22.0, 2.0],
                [22.0, 3.0],
                [22.0, 4.0],
                [22.0, 5.0],
            ]),
            Block::Grass => Some([
                [1.0, 10.0],
                [1.0, 10.0],
                [8.0, 5.0],
                [4.0, 10.0],
                [1.0, 10.0],
                [1.0, 10.0],
            ]),
            Block::Stone => Some([
                [19.0, 6.0],
                [19.0, 6.0],
                [19.0, 6.0],
                [19.0, 6.0],
                [19.0, 6.0],
                [19.0, 6.0],
            ]),
            _ => None,
        }
    }
}

pub struct Cube {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
}

impl Cube {
    fn new() {
        // let vertex_buffer
    }
}
