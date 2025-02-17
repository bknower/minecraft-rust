#![allow(unused_imports, unused)]
mod block;
mod camera;
mod imgui_renderer;
mod model;
mod resources;
mod texture;
mod utils;
mod world;
use camera::{Camera, CameraController, Projection};
use cgmath::{num_traits::float, prelude::*};
use imgui::internal::DataTypeKind;
use imgui::sys::ImGuiKey_J;
use imgui::{Context, DrawData, FontSource, MouseCursor, StyleVar, Ui};
use imgui_renderer::{Renderer, RendererConfig};
// use imgui_wgpu::{Renderer, RendererConfig};
use imgui::Condition;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use model::{DrawModel, Mesh, Model, Vertex};
use num_format::{Locale, ToFormattedString};
use std::env;
use std::fmt::Display;
use std::iter::Map;
use std::sync::Arc;
use std::time::Instant;
use texture::Texture;
use utils::*;
use wgpu::util::DeviceExt;
use wgpu::PresentMode;
use winit::dpi::PhysicalPosition;
use winit::window::{self, WindowId};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::*,
    window::Window,
    // window::WindowBuilder,
    window::WindowAttributes,
};
use world::MESHING_ALGORITHMS;

use crate::model::Material;
use crate::resources::load_texture;
use crate::world::World;

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation))
            // * cgmath::Matrix4::from_scale(0.5))// note: this was just for scaling the cube
            .into(),
        }
    }
}
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}
impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in the shader.
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials, we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5, not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// unsafe impl bytemuck::Pod for Vertex {}
// unsafe impl bytemuck::Zeroable for Vertex {}

// lib.rs
// Changed
// const VERTICES: &[Vertex] = &[
//     // Changed
//     Vertex {
//         position: [-0.0868241, 0.49240386, 0.0],
//         tex_coords: [0.4131759, 0.00759614],
//     }, // A
//     Vertex {
//         position: [-0.49513406, 0.06958647, 0.0],
//         tex_coords: [0.0048659444, 0.43041354],
//     }, // B
//     Vertex {
//         position: [-0.21918549, -0.44939706, 0.0],
//         tex_coords: [0.28081453, 0.949397],
//     }, // C
//     Vertex {
//         position: [0.35966998, -0.3473291, 0.0],
//         tex_coords: [0.85967, 0.84732914],
//     }, // D
//     Vertex {
//         position: [0.44147372, 0.2347359, 0.0],
//         tex_coords: [0.9414737, 0.2652641],
//     }, // E
// ];

// const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];
// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly, so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
    }
}

struct ImguiState {
    context: imgui::Context,
    platform: WinitPlatform,
    renderer: Renderer,
    clear_color: wgpu::Color,
    last_cursor: Option<MouseCursor>,
    demo_open: bool,
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    clear_color: wgpu::Color,
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    camera: Camera,
    projection: Projection,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    frame: u32,
    depth_texture: Texture,
    atlas: Material,
    mouse_pressed: bool,
    world: World,
    imgui: Option<ImguiState>,
    window: Arc<Window>,
    hidpi_factor: f64,
    last_render_time: Instant,
}
impl State {
    // Creating some of the wgpu types requires async code
    async fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let size = window.inner_size();
        let hidpi_factor = window.scale_factor();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::default();

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web, we'll have to disable some.
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();
        // let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        // let surface_format = surface_caps.formats.iter()
        //     .copied()
        //     .filter(|f| f.is_srgb())
        //     .next()
        //     .unwrap_or(surface_caps.formats[0]);
        let mut config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        config.present_mode = PresentMode::Immediate;
        surface.configure(&device, &config);
        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").unwrap();
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view), // CHANGED!
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler), // CHANGED!
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let clear_color = wgpu::Color::BLACK;

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let camera = camera::Camera::new((0.0, 120.0, 2.0), cgmath::Deg(0.0), cgmath::Deg(-50.0));
        let projection =
            camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);
        let camera_controller = camera::CameraController::new(30.0, 0.4);
        // in new() after creating `camera`

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None, // 5.
        });

        const NUM_INSTANCES_PER_ROW: u32 = 10;
        const SPACE_BETWEEN: f32 = 3.0;
        let mut world = World::new(1);

        // let blocks = world.chunks.into_iter()
        // .flat_map(|chunk| {
        //     chunk.blocks.map()
        // });
        let mut block_instances: Vec<Instance> = Vec::new();

        let instances = block_instances;
        // let instances = (0..NUM_INSTANCES_PER_ROW)
        //     .flat_map(|z| {
        //         (0..NUM_INSTANCES_PER_ROW).map(move |x| {
        //             let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
        //             let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

        //             let position = cgmath::Vector3 { x, y: 0.0, z };

        //             let rotation = if position.is_zero() {
        //                 cgmath::Quaternion::from_axis_angle(
        //                     cgmath::Vector3::unit_z(),
        //                     cgmath::Deg(0.0),
        //                 )
        //             } else {
        //                 cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
        //             };

        //             Instance { position, rotation }
        //         })
        //     })
        //     .collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let frame = 0;

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");
        // let obj_model =
        //     resources::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
        //         .await
        //         .unwrap();
        // let obj_model =
        // resources::load_block("stone",
        // vec!["atlas.png"], &device, &queue, &texture_bind_group_layout)
        // 	.await;

        let atlas_texture = load_texture("atlas.png", &device, &queue).await.unwrap();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas_texture.sampler),
                },
            ],
            label: None,
        });
        let atlas = Material {
            name: "atlas".to_string(),
            diffuse_texture: atlas_texture,
            bind_group,
        };

        Self {
            surface,
            device,
            queue,
            config,
            size,
            clear_color,
            render_pipeline,
            diffuse_bind_group,
            diffuse_texture,
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            instances,
            instance_buffer,
            frame,
            depth_texture,
            atlas,
            mouse_pressed: false,
            world,
            imgui: None,
            window,
            hidpi_factor,
            last_render_time: Instant::now(),
        }
    }

    // pub fn window(&self) -> &Window {
    //     &self.window
    // }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.projection.resize(new_size.width, new_size.height);
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(keycode),
                        state,
                        ..
                    },
                ..
            } => self.camera_controller.process_keyboard(*keycode, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, dt: instant::Duration) {
        #[allow(unused)]
        let frame = self.frame;
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        let updated = self
            .world
            .update(self.camera.position, &self.device, &self.queue);

        // if updated {
        let mut chunk_instances = vec![];

        let mut triangles = 0;
        for chunk in self.world.chunks.as_slice() {
            if let Some(mesh) = &chunk.mesh {
                let chunk_triangles = mesh.vertex_buffer.size() / 3;
                self.world.world_stats.add_chunk_triangles(chunk_triangles);
                triangles += chunk_triangles;
                let position = cgmath::Vector3 {
                    x: ((chunk.chunk_x as f32 - 0.5) * world::CHUNK_SIZE_X as f32),
                    y: 0.0,
                    z: ((chunk.chunk_z as f32 - 0.5) * world::CHUNK_SIZE_Z as f32),
                };
                let rotation = cgmath::Quaternion::from_axis_angle(
                    cgmath::Vector3::unit_z(),
                    cgmath::Deg(0.0),
                );
                chunk_instances.push(Instance { position, rotation });
            }
        }
        self.world.world_stats.triangles = triangles;
        // let position = cgmath::Vector3 {
        // 	x: 0.0,
        // 	y: 0.0,
        // 	z: 0.0};
        // let rotation =                         cgmath::Quaternion::from_axis_angle(
        // 	cgmath::Vector3::unit_z(),
        // 	cgmath::Deg(0.0),
        // );
        // chunk_instances.push(Instance {position, rotation});
        self.instances = chunk_instances;

        let instance_data = self
            .instances
            .iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();
        self.instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            });
        // }
        self.frame += 1;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture().unwrap();
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut meshes: Vec<&Mesh> = vec![];
            for chunk in &self.world.chunks {
                let mesh = &chunk.mesh;
                if let Some(mesh) = mesh {
                    meshes.push(mesh);
                }
            }
            // let meshes: Vec<Mesh> = self.world.chunks.iter().map(|chunk| chunk.to_mesh(&self.device, &self.queue)).collect();
            // let mesh_refs: Vec<&Mesh> = meshes.iter().map(|mesh| mesh).collect();
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.clear_color),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            render_pass.set_pipeline(&self.render_pipeline);

            // // render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            // // render_pass.set_bind_group(1, &self.camera_bind_group, &[]);

            // use model::DrawModel;
            // render_pass.draw_model_instanced(
            //     &self.obj_model,
            //     0..self.instances.len() as u32,
            //     &self.camera_bind_group,
            // );
            // let meshes: Vec<Option<Mesh>> = self.world.chunks.iter().map(|chunk|  chunk.mesh).collect();
            // self.world.chunks.iter().for_each(|chunk| {
            // 	let mesh = chunk.mesh.unwrap();

            // 	// if let Some(mesh) = chunk.mesh {
            // 		render_pass.draw_mesh(&mesh, &self.atlas, &self.camera_bind_group);
            // 	// }
            // });
            // for mesh in meshes {
            // 	render_pass.draw_mesh(mesh, &self.atlas, &self.camera_bind_group);
            // }
            for i in 0..meshes.len() {
                let mesh = meshes.get(i).unwrap();
                render_pass.draw_mesh_instanced(
                    mesh,
                    &self.atlas,
                    (i as u32)..(i as u32 + 1),
                    &self.camera_bind_group,
                );
            }

            // render imgui
            let render_imgui = true;
            if render_imgui {
                let imgui = self.imgui.as_mut().unwrap();
                let mut imgui_context = &mut imgui.context;
                let delta_s = self.last_render_time.elapsed();

                // let frame = self.surface.get_current_texture().unwrap();

                imgui
                    .platform
                    .prepare_frame(imgui_context.io_mut(), &self.window)
                    .expect("Failed to prepare frame");

                // imgui.platform.prepare_render(ui, &window);
                // let draw_data = imgui_context.render();

                let ui = imgui_context.new_frame();

                fn prerender_imgui_element(ui: &Ui, label: &str, width: f32) {
                    ui.text(format!("{}: ", label));
                    ui.same_line();
                    // ui.push_item_width(width);

                    let window_width = ui.window_size()[0];
                    let cursor_y = ui.cursor_pos()[1];

                    let right_align_x = window_width - width - 10.0; // Adjust with padding
                    ui.set_cursor_pos([right_align_x, cursor_y]);

                    ui.set_next_item_width(width);
                }

                fn render_slider<T>(ui: &Ui, label: &str, value: &mut T, min: T, max: T, width: f32)
                where
                    T: DataTypeKind, // `imgui-rs` requires Numeric trait (i32, f32, etc.)
                {
                    prerender_imgui_element(ui, label, width);
                    ui.slider("##slider", min, max, value);
                }

                {
                    let window = ui.window("Debug");
                    window
                        .size([300.0, 100.0], imgui::Condition::FirstUseEver)
                        .build(|| {

                            // additional print ideas
                            // - interactable meshing algorithm selector
                            let prints = stats! {
                                "FPS" => ui.io().framerate,
                                "Position" => self.camera.display_position(),
                                "Chunks loaded" => self.world.chunks.len(),
                                "Last mesh time" => self.world.world_stats.mesh_gen_times.last().unwrap_or(&0.0),
                                "Average mesh time (last 20)" => self.world.world_stats.get_mesh_gen_time_avg(),                                
                                "Last chunk time" => self.world.world_stats.chunk_gen_times.last().unwrap_or(&0.0),
                                "Average chunk time (last 20)" => self.world.world_stats.get_chunk_gen_time_avg(),
                                "Total Tris" => (self.world.world_stats.triangles).to_formatted_string(&Locale::en).to_string(),
                                "Average Tris per chunk" => self.world.world_stats.get_chunk_triangles_avg()
                            };

                            for (name, value) in &prints {
                                let left_text = format!("{:}:", name);
                                let right_text = format!("{:.2}", value());
                                let window_width = ui.window_size()[0];
                                let cursor_y: f32 = ui.cursor_pos()[1];

                                ui.text(left_text);
                                ui.same_line();
                                // Calculate right-aligned position
                                let text_width = ui.calc_text_size(&right_text)[0]; // Get the width of the right text
                                let right_align_x = window_width - text_width - 10.0; // Adjust with padding

                                // Set cursor for right-aligned text
                                ui.set_cursor_pos([right_align_x, cursor_y]);
                                ui.text(right_text);

                            }

                            let width = 150.0;

                            render_slider(ui, "Render Distance", &mut self.world.render_distance, 0, 100, width);

                            prerender_imgui_element(ui, "Meshing Algorithm", width);
                            ui.combo_simple_string("##combo", &mut self.world.meshing_algorithm, &MESHING_ALGORITHMS);


                        });

                    // ui.show_demo_window(&mut imgui.demo_open);
                }

                if imgui.last_cursor != ui.mouse_cursor() {
                    imgui.last_cursor = ui.mouse_cursor();
                    imgui.platform.prepare_render(ui, &self.window);
                }

                imgui
                    .renderer
                    .render(
                        imgui.context.render(),
                        &self.queue,
                        &self.device,
                        &mut render_pass,
                    )
                    .expect("Rendering failed");

                // render_pass.
            }
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));

        output.present();

        Ok(())
    }

    fn setup_imgui(&mut self) {
        let mut context = Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::new(&mut context);
        platform.attach_window(
            context.io_mut(),
            &self.window,
            imgui_winit_support::HiDpiMode::Default,
        );

        let font_size = (13.0 * self.hidpi_factor) as f32;
        context.io_mut().font_global_scale = (1.0 / self.hidpi_factor) as f32;

        context.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: font_size,
                ..Default::default()
            }),
        }]);

        let clear_color = wgpu::Color {
            r: 0.1,
            g: 0.2,
            b: 0.3,
            a: 1.0,
        };

        // let font_atlas = imgui.fonts();
        // font_atlas.add_font(&[imgui::FontSource::DefaultFontData {
        //     config: Some(imgui::FontConfig::default()),
        // }]);
        // let _ = font_atlas.build_rgba32_texture();
        let renderer_config = RendererConfig {
            texture_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            ..Default::default()
        };

        let mut renderer = Renderer::new(&mut context, &self.device, &self.queue, renderer_config);

        self.imgui = Some(ImguiState {
            context,
            platform,
            renderer,
            clear_color,
            last_cursor: None,
            demo_open: true,
        });
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // letwindow = Some(State::new(event_loop));
        let rt = tokio::runtime::Runtime::new().unwrap();
        // self.state = Some(rt.block_on(State::new(self.window.as_ref().unwrap().clone())));
        self.state = Some(rt.block_on(State::new(event_loop)));
        self.state.as_mut().unwrap().setup_imgui();
    }

    // fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
    //     let state = self.state.as_mut().unwrap();
    //     let window = &state.window;
    //     let imgui = state.imgui.as_mut().unwrap();
    //     let mut imgui_context = &mut imgui.context;
    //     let now = Instant::now();
    //     imgui_context
    //         .io_mut()
    //         .update_delta_time(now - imgui.last_render_time);
    //     imgui.last_render_time = now;
    // }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let state = self.state.as_mut().unwrap();
        let window = &state.window;
        let imgui = state.imgui.as_mut().unwrap();
        let mut imgui_context = &mut imgui.context;
        imgui
            .platform
            .prepare_frame(imgui_context.io_mut(), &window)
            .expect("Failed to prepare frame");
        window.request_redraw();
        imgui.platform.handle_event(
            imgui_context.io_mut(),
            &window,
            &Event::<WindowEvent>::AboutToWait,
        );
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let state = self.state.as_mut().unwrap();
        let window = &state.window;
        let imgui = state.imgui.as_mut().unwrap();
        let mut imgui_context = &mut imgui.context;

        match event {
            DeviceEvent::MouseMotion { delta } => {
                if state.mouse_pressed {
                    state.camera_controller.process_mouse(delta.0, delta.1)
                }
            }
            DeviceEvent::Removed => {}
            _ => {}
        }
        imgui.platform.handle_event::<()>(
            imgui.context.io_mut(),
            window,
            &Event::DeviceEvent { device_id, event },
        );
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = self.state.as_mut().unwrap();
        let window = state.window.clone();
        // {
        //     let imgui = state.imgui.as_mut().unwrap();
        //     let mut imgui_context = &mut imgui.context;
        //     imgui.platform.handle_event(
        //         imgui_context.io_mut(),
        //         &window,
        //         &Event::<WindowEvent>::WindowEvent {
        //             window_id,
        //             event,
        //         }
        //     );

        // }
        // return;

        let imgui = state.imgui.as_mut().unwrap();
        let mut imgui_context = &mut imgui.context;
        imgui.platform.handle_event(
            imgui_context.io_mut(),
            &window,
            &Event::<WindowEvent>::WindowEvent {
                window_id,
                event: event.clone(),
            },
        );

        if window_id == window.id() {
            if !imgui_context.io().want_capture_mouse {
                state.input(&event);
            }

            match &event {
                #[cfg(not(target_arch = "wasm32"))]
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: winit::keyboard::PhysicalKey::Code(KeyCode::Escape),
                            ..
                        },
                    ..
                } => {
                    event_loop.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { .. } => state.resize(window.inner_size()),

                WindowEvent::RedrawRequested => {
                    let imgui = state.imgui.as_mut().unwrap();
                    let mut imgui_context = &mut imgui.context;

                    let delta_s = state.last_render_time.elapsed();
                    let now = instant::Instant::now();
                    let dt = now - state.last_render_time;
                    state.last_render_time = now;
                    imgui_context.io_mut().update_delta_time(dt);
                    // println!("{:?}", imgui_context.io_mut().framerate);

                    state.update(dt);
                    match state.render() {
                        Ok(_) => {}
                        // Reconfigure the surface if lost
                        Err(wgpu::SurfaceError::Lost) => {
                            state.resize(state.size);
                        }
                        // The system is out of memory, we should probably quit
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                        Err(e) => eprintln!("{:?}", e),
                    }

                    // let imgui = state.imgui.as_mut().unwrap();
                    // let mut imgui_context = &mut imgui.context;

                    // imgui_context.io_mut().update_delta_time(dt);
                    // state.last_render_time = now;

                    // let frame = match state.surface.get_current_texture() {
                    //     Ok(frame) => frame,
                    //     Err(e) => {
                    //         eprintln!("dropped frame: {e:?}");
                    //         return;
                    //     }
                    // };

                    // imgui
                    //     .platform
                    //     .prepare_frame(imgui_context.io_mut(), &window)
                    //     .expect("Failed to prepare frame");

                    // // imgui.platform.prepare_render(ui, &window);
                    // // let draw_data = imgui_context.render();

                    // let ui = imgui_context.new_frame();

                    // {
                    //     let window = ui.window("Hello World");
                    //     window
                    //         .size([300.0, 100.0], imgui::Condition::FirstUseEver)
                    //         .build(|| {
                    //             ui.text("Hello world!");
                    //             ui.text("This...is...imgui-rs on WGPU!");
                    //             ui.separator();
                    //             let mouse_pos = ui.io().mouse_pos;
                    //             ui.text(format!(
                    //                 "Mouse Position: ({:.1},{:.1})",
                    //                 mouse_pos[0], mouse_pos[1]
                    //             ));
                    //         });

                    //     let window = ui.window("Hello too");
                    //     window
                    //         .size([400.0, 200.0], Condition::FirstUseEver)
                    //         .position([400.0, 200.0], Condition::FirstUseEver)
                    //         .build(|| {
                    //             ui.text(format!("Frametime: {delta_s:?}"));
                    //         });

                    //     ui.show_demo_window(&mut imgui.demo_open);
                    // }

                    // let mut encoder: wgpu::CommandEncoder = state
                    //     .device
                    //     .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                    // if imgui.last_cursor != ui.mouse_cursor() {
                    //     imgui.last_cursor = ui.mouse_cursor();
                    //     imgui.platform.prepare_render(ui, &state.window);
                    // }

                    // let view = frame
                    //     .texture
                    //     .create_view(&wgpu::TextureViewDescriptor::default());

                    // let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    //     label: None,
                    //     color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    //         view: &view,
                    //         resolve_target: None,
                    //         ops: wgpu::Operations {
                    //             load: wgpu::LoadOp::Clear(imgui.clear_color),
                    //             store: wgpu::StoreOp::Store,
                    //         },
                    //     })],
                    //     depth_stencil_attachment: None,
                    //     timestamp_writes: None,
                    //     occlusion_query_set: None,
                    // });

                    // imgui
                    //     .renderer
                    //     .render(
                    //         imgui.context.render(),
                    //         &state.queue,
                    //         &state.device,
                    //         &mut rpass,
                    //     )
                    //     .expect("Rendering failed");
                    // drop(rpass);
                    // state.queue.submit(Some(encoder.finish()));
                    // frame.present();
                }
                _ => {}
            }
        }
        // event => {
        //     platform.handle_event(imgui.io_mut(), &window, &event);
        // }
        // _ => {}
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: ()) {
        let state = self.state.as_mut().unwrap();
        let imgui = state.imgui.as_mut().unwrap();
        imgui.platform.handle_event::<()>(
            imgui.context.io_mut(),
            &state.window,
            &Event::UserEvent(event),
        );
    }
}

pub async fn run() {
    // env::set_var("RUST_BACKTRACE", "1");
    env_logger::init(); // Necessary for logging within WGPU
    let event_loop = EventLoop::new().expect("Event loop exists"); // Loop provided by winit for handling window events
                                                                   // let window = WindowBuilder::new().build(&event_loop).unwrap();

    // let mut app = App::new(platform, imgui);
    event_loop.run_app(&mut App::default()).unwrap();
}
