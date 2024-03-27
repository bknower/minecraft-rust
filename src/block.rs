use crate::{
    model::{Material, Mesh},
    resources::load_texture,
};

#[derive(Copy, Clone, PartialEq, Eq, Hash)]

pub enum Block {
    Air,
    Grass,
    Stone,
}

impl Block {
    fn material(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> Option<Material> {
        let diffuse_texture = load_texture(t, device, queue).await.unwrap();
        match self {
            Stone
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
