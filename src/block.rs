use cgmath::Vector2;

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
	pub fn get_atlas_coords(self) -> Option<Vector2<f32>> {
		match self {
			Block::Stone => Some([19.0, 6.0].into()),
			Block::Grass => Some([1.0, 10.0].into()),
			_ => None
		}
	}
    pub async fn material(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> Option<Material> {
        let file_name_option = match self {
            Block::Stone => Some("stone.png"),
            Block::Grass => Some("grass.png"),
            Block::Air => None,
        };

        match file_name_option {
            Some(file_name) => {
                let diffuse_texture = load_texture(file_name, device, queue).await.unwrap();
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                        },
                    ],
                    label: None,
                });
                Some(Material {
                    name: file_name.to_string(),
                    diffuse_texture,
                    bind_group,
                })
            }
            None => None,
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
