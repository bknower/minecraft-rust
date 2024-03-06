// use winit::{
//     event::*,
//     event_loop::{ControlFlow, EventLoop},
//     keyboard::*,
//     window::WindowBuilder,
//     window::Window
// };

// struct State {
//     surface: wgpu::Surface,
//     device: wgpu::Device,
//     queue: wgpu::Queue,
//     config: wgpu::SurfaceConfiguration,
//     size: winit::dpi::PhysicalSize<u32>,
//     // The window must be declared after the surface so
//     // it gets dropped after it as the surface contains
//     // unsafe references to the window's resources.
//     window: Window,
// }

// impl State {
//     // Creating some of the wgpu types requires async code
//     async fn new(window: Window) -> Self {
//         let size = window.inner_size();

//         // The instance is a handle to our GPU
//         // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
//         let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
//             backends: wgpu::Backends::all(),
//             ..Default::default()
//         });

//         // # Safety
//         //
//         // The surface needs to live as long as the window that created it.
//         // State owns the window, so this should be safe.
//         let surface = unsafe { instance.create_surface(&window) }.unwrap();

//         let adapter = instance.request_adapter(
//             &wgpu::RequestAdapterOptions {
//                 power_preference: wgpu::PowerPreference::default(),
//                 compatible_surface: Some(&surface),
//                 force_fallback_adapter: false,
//             },
//         ).await.unwrap();

//         let (device, queue) = adapter.request_device(
//             &wgpu::DeviceDescriptor {
//                 required_features: wgpu::Features::empty(),
//                 // WebGL doesn't support all of wgpu's features, so if
//                 // we're building for the web, we'll have to disable some.
//                 required_limits: if cfg!(target_arch = "wasm32") {
//                     wgpu::Limits::downlevel_webgl2_defaults()
//                 } else {
//                     wgpu::Limits::default()
//                 },
//                 label: None,
//             },
//             None, // Trace path
//         ).await.unwrap();
//         // let surface_caps = surface.get_capabilities(&adapter);
//         // Shader code in this tutorial assumes an sRGB surface texture. Using a different
//         // one will result in all the colors coming out darker. If you want to support non
//         // sRGB surfaces, you'll need to account for that when drawing to the frame.
//         // let surface_format = surface_caps.formats.iter()
//         //     .copied()
//         //     .filter(|f| f.is_srgb())
//         //     .next()
//         //     .unwrap_or(surface_caps.formats[0]);
//         let config = surface
//         .get_default_config(&adapter, size.width, size.height)
//         .unwrap();
//         surface.configure(&device, &config);

//         Self {
//             window,
//             surface,
//             device,
//             queue,
//             config,
//             size,
//         }
//     }

//     pub fn window(&self) -> &Window {
//         &self.window
//     }

//     fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
//         todo!()
//     }

//     fn input(&mut self, event: &WindowEvent) -> bool {
//         todo!()
//     }

//     fn update(&mut self) {
//         todo!()
//     }

//     fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
//         todo!()
//     }
// }

// async fn main() {
//     env_logger::init(); // Necessary for logging within WGPU
//     let event_loop = EventLoop::new().expect("Event loop exists"); // Loop provided by winit for handling window events
//     let window = WindowBuilder::new().build(&event_loop).unwrap();

//     let mut state = State::new(window).await;

//     // let instance = wgpu::Instance::default();
//     // let surface = instance.create_surface(&window).unwrap();
//     // let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
//     //     power_preference: wgpu::PowerPreference::default(),
//     //     compatible_surface: Some(&surface),
//     //     force_fallback_adapter: false,
//     // }))
//     // .unwrap();

//     // let (device, queue) = pollster::block_on(adapter.request_device(
//     //     &wgpu::DeviceDescriptor {
//     //         label: None,
//     //         required_features: wgpu::Features::empty(),
//     //         required_limits: wgpu::Limits::default(),
//     //     },
//     //     None, // Trace path
//     // ))
//     // .unwrap();

//     // let size = window.inner_size();
//     // let config = surface
//     //     .get_default_config(&adapter, size.width, size.height)
//     //     .unwrap();
//     // surface.configure(&device, &config);

//     // let mut blue_value: f64 = 0.0; // New
//     // let mut blue_inc: f64 = 0.0; // New
//     // let red_inc: f64 = 0.0;
//     event_loop
//         .run(move |event, target| {
//             match event {
//                 Event::WindowEvent { window_id, event } => {
//                     match event {
//                         WindowEvent::RedrawRequested => {
//                             let output = surface.get_current_texture().unwrap();
//                             let view = output
//                                 .texture
//                                 .create_view(&wgpu::TextureViewDescriptor::default());
//                             let mut encoder =
//                                 device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
//                                     label: Some("Render Encoder"),
//                                 });

//                             {
//                                 let _render_pass =
//                                     encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
//                                         label: Some("Render Pass"),
//                                         color_attachments: &[Some(
//                                             wgpu::RenderPassColorAttachment {
//                                                 view: &view,
//                                                 resolve_target: None,
//                                                 ops: wgpu::Operations {
//                                                     load: wgpu::LoadOp::Clear(wgpu::Color {
//                                                         r: 0.1,
//                                                         g: 0.9,
//                                                         b: blue_value, // New
//                                                         a: 1.0,
//                                                     }),
//                                                     store: wgpu::StoreOp::Store,
//                                                 },
//                                             },
//                                         )],
//                                         depth_stencil_attachment: None,
//                                         timestamp_writes: None,
//                                         occlusion_query_set: None,
//                                     });
//                             }

//                             // submit will accept anything that implements IntoIter
//                             queue.submit(std::iter::once(encoder.finish()));
//                             output.present();
//                             blue_value += (blue_inc as f64) * 0.001;
//                             if blue_value > 1.0 {
//                                 blue_inc = -1.0;
//                                 blue_value = 1.0;
//                             } else if blue_value < 0.0 {
//                                 blue_inc = 1.0;
//                                 blue_value = 0.0;
//                             }
//                         }
//                         WindowEvent::CloseRequested => target.exit(),
//                         WindowEvent::KeyboardInput {
//                             device_id: _,
//                             event,
//                             is_synthetic: _,
//                         } => {
//                             if event.physical_key == KeyCode::Escape {
//                                 target.exit()
//                             }
//                         }
//                         _ => {}
//                     }
//                 }
//                 Event::AboutToWait => {
//                     target.
//                 }
//                 _ => {}
//             }
//         })
//         .unwrap();
// }
use minecraft_rust::run;

fn main() {
    pollster::block_on(run());
}
