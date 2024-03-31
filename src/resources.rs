use std::io::{BufReader, Cursor};

use cfg_if::cfg_if;
use wgpu::util::DeviceExt;

use crate::{
    model::{self, ModelVertex},
    texture,
};

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let mut origin = location.origin().unwrap();
    if !origin.ends_with("learn-wgpu") {
        origin = format!("{}/learn-wgpu", origin);
    }
    let base = reqwest::Url::parse(&format!("{}/", origin,)).unwrap();
    base.join(file_name).unwrap()
}

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(file_name);
            let txt = reqwest::get(url)
                .await?
                .text()
                .await?;
        } else {
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("res")
                .join(file_name);
            println!("{}", path.display());
            let txt = std::fs::read_to_string(path)?;
        }
    }

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(file_name);
            let data = reqwest::get(url)
                .await?
                .bytes()
                .await?
                .to_vec();
        } else {
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("res")
                .join(file_name);
            let data = std::fs::read(path)?;
        }
    }

    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, file_name)
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let mat_text = load_string(&p).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        let diffuse_texture = load_texture(&m.diffuse_texture.unwrap(), device, queue).await?;
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

        materials.push(model::Material {
            name: m.name,
            diffuse_texture,
            bind_group,
        })
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                })
                .collect::<Vec<_>>();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes, materials })
}

pub async fn load_block(
    block_name: &str,
    face_textures: Vec<&str>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> model::Model {
    let mut materials = Vec::new();
    for t in face_textures {
        let diffuse_texture = load_texture(t, device, queue).await.unwrap();
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

        materials.push(model::Material {
            name: t.to_string(),
            diffuse_texture,
            bind_group,
        })
    }

    let normal: [f32; 3] = [0.0, 0.0, 0.0];
    const V0: [f32; 3] = [0.0, 0.0, 0.0];
    const V1: [f32; 3] = [0.0, 0.0, 1.0];
    const V2: [f32; 3] = [0.0, 1.0, 0.0];
    const V3: [f32; 3] = [0.0, 1.0, 1.0];
    const V4: [f32; 3] = [1.0, 0.0, 0.0];
    const V5: [f32; 3] = [1.0, 0.0, 1.0];
    const V6: [f32; 3] = [1.0, 1.0, 0.0];
    const V7: [f32; 3] = [1.0, 1.0, 1.0];
    let vertices: Vec<ModelVertex> = vec![
        // front face
        ModelVertex {
            position: V0,
            tex_coords: [0.0, 0.0],
            normal,
        },
        ModelVertex {
            position: V1,
            tex_coords: [1.0, 0.0],
            normal,
        },
        ModelVertex {
            position: V2,
            tex_coords: [0.0, 1.0],
            normal,
        },
        ModelVertex {
            position: V3,
            tex_coords: [1.0, 1.0],
            normal,
        },
        // back face
        ModelVertex {
            position: V4,
            tex_coords: [1.0, 0.0],
            normal,
        },
        ModelVertex {
            position: V5,
            tex_coords: [0.0, 0.0],
            normal,
        },
        ModelVertex {
            position: V6,
            tex_coords: [1.0, 1.0],
            normal,
        },
        ModelVertex {
            position: V7,
            tex_coords: [0.0, 1.0],
            normal,
        },
        // left face
        ModelVertex {
            position: V0,
            tex_coords: [1.0, 0.0],
            normal,
        },
        ModelVertex {
            position: V2,
            tex_coords: [1.0, 1.0],
            normal,
        },
        ModelVertex {
            position: V4,
            tex_coords: [0.0, 0.0],
            normal,
        },
        ModelVertex {
            position: V6,
            tex_coords: [0.0, 1.0],
            normal,
        },
        // right face
        ModelVertex {
            position: V1,
            tex_coords: [0.0, 0.0],
            normal,
        },
        ModelVertex {
            position: V3,
            tex_coords: [0.0, 1.0],
            normal,
        },
        ModelVertex {
            position: V5,
            tex_coords: [1.0, 0.0],
            normal,
        },
        ModelVertex {
            position: V7,
            tex_coords: [1.0, 1.0],
            normal,
        },
        // up face
        ModelVertex {
            position: V2,
            tex_coords: [0.0, 0.0],
            normal,
        },
        ModelVertex {
            position: V3,
            tex_coords: [0.0, 1.0],
            normal,
        },
        ModelVertex {
            position: V6,
            tex_coords: [1.0, 0.0],
            normal,
        },
        ModelVertex {
            position: V7,
            tex_coords: [1.0, 1.0],
            normal,
        },
        // down face
        ModelVertex {
            position: V0,
            tex_coords: [0.0, 1.0],
            normal,
        },
        ModelVertex {
            position: V1,
            tex_coords: [1.0, 1.0],
            normal,
        },
        ModelVertex {
            position: V4,
            tex_coords: [0.0, 0.0],
            normal,
        },
        ModelVertex {
            position: V5,
            tex_coords: [1.0, 0.0],
            normal,
        },
    ];

    let indices: &[u32] = &[
        0, 1, 2, 2, 1, 3, // front
        5, 4, 7, 7, 4, 6, // back
        8, 9, 10, 10, 9, 11, // left
        12, 14, 13, 13, 14, 15, // right
        16, 17, 18, 18, 17, 19, // up
        20, 22, 21, 21, 22, 23, // down
    ];

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{:?} Vertex Buffer", block_name)),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{:?} Index Buffer", block_name)),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let mesh = model::Mesh {
        name: block_name.to_string(),
        vertex_buffer,
        index_buffer,
        num_elements: indices.len() as u32,
        material: 0,
    };
    model::Model {
        meshes: vec![mesh],
        materials,
    }
}
