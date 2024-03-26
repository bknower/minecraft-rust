use crate::model::Mesh;

#[derive(Copy, Clone, PartialEq)]

pub enum Block {
    Air,
    Grass,
    Stone,
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
