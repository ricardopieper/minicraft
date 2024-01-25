#![feature(const_fn_floating_point_arithmetic)]
mod blocks;
mod camera;

use blocks::Vertex;
use blocks::*;
use camera::*;
use cgmath::*;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

/// Perspective matrix that is suitable for Vulkan.
///
/// It inverts the projected y-axis. And set the depth range to 0..1
/// instead of -1..1. Mind the vertex winding order though.

struct Minicraft {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: winit::window::Window,
    render_pipeline: wgpu::RenderPipeline,

    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    vertices_count: u32,
    indices_count: u32,

    atlas_bind_group: wgpu::BindGroup,
    movement_speed: f32,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl Minicraft {
    fn create_wgpu_instance() -> wgpu::Instance {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        })
    }

    async fn new(window: winit::window::Window) -> Self {
        let size = window.inner_size();
        let instance = Self::create_wgpu_instance();
        let surface = unsafe { instance.create_surface(&window).unwrap() };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) =
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        features: wgpu::Features::empty(),
                        limits: wgpu::Limits::default(),
                        label: None,
                    },
                    None,
                )
                .await
                .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        let png_bytes = std::include_bytes!("./textures/atlas1.png").to_vec();
        let image =
            image::load_from_memory_with_format(&png_bytes, image::ImageFormat::Png).unwrap();
        let atlas_rgb8 = image.as_rgba8().unwrap();
        use image::GenericImageView;
        let (width, height) = image.dimensions();

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("Atlas texture"),
            view_formats: &vec![],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_rgb8,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            texture_size,
        );

        let atlas_texture_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Atlas sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let atlas_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Atlas bind group layout"),
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let atlas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas bind group"),
            layout: &atlas_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        let camera = Camera {
            position: vec3(0.0, 0.0, 0.0),
            horizontal_angle: -3.14,
            vertical_angle: 0.0,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            zfar: 10000.0,
            znear: 0.1,
        };

        let camera_uniform = camera.view_proj_matrix();

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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&atlas_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
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
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        use wgpu::util::DeviceExt;

        let world = World::worldgen();
        let start = std::time::Instant::now();
        let (vertices, indices) = world.meshgen();
        let elapsed = start.elapsed();
        println!("Meshgen took {:?}", elapsed);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let minicraft = Minicraft {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            indices_count: indices.len() as u32,
            vertices_count: vertices.len() as u32,
            atlas_bind_group,
            camera,
            camera_bind_group,
            camera_buffer,
            camera_uniform,
            movement_speed: 1.0,
        };

        return minicraft;
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn input(&mut self, event: &winit::event::WindowEvent) -> bool {
        false
    }

    pub fn update(&mut self) {
        self.camera_uniform = self.camera.view_proj_matrix();
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Minicraft Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.atlas_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.indices_count, 0, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let mut game = Minicraft::new(window).await;

    let mouse_speed = 1f32;
    let mut prev = std::time::SystemTime::now();

    let mut keys_being_pressed = std::collections::hash_set::HashSet::new();
    let mut allow_camera_movement = true;
    let mut last_mouse_pos = vec2(0.0, 0.0);

    event_loop.run(move |event, _, control_flow| {
        let current = std::time::SystemTime::now();
        let delta = current.duration_since(prev).unwrap();

        match event {
            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::CloseRequested,
                ..
            } => {
                println!("Closed requested!");
                *control_flow = ControlFlow::Exit;
            }
            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::Resized(size),
                ..
            } => {
                game.resize(size);
            }
            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::ScaleFactorChanged { new_inner_size, .. },
                ..
            } => {
                game.resize(*new_inner_size);
            }

            winit::event::Event::MainEventsCleared => {
                //tracy_client::start_noncontinuous_frame!("frame");

                let movement = game.movement_speed * delta.as_secs_f32();
                for key in keys_being_pressed.iter() {
                    match key {
                        winit::event::VirtualKeyCode::W => {
                            game.camera.go_forward(movement);
                        }
                        winit::event::VirtualKeyCode::S => {
                            game.camera.go_backward(movement);
                        }
                        winit::event::VirtualKeyCode::A => {
                            game.camera.go_left(movement);
                        }
                        winit::event::VirtualKeyCode::D => {
                            game.camera.go_right(movement);
                        }
                        winit::event::VirtualKeyCode::Key1 => {
                            println!("Camera: {:?}", game.camera);
                        }
                        _ => {}
                    }
                }

                game.window.request_redraw();
            }
            winit::event::Event::RedrawEventsCleared => {}
            winit::event::Event::RedrawRequested(window_id) if window_id == game.window.id() => {
                game.update();
                match game.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => game.resize(game.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            winit::event::Event::NewEvents(_) => {}
            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                //println!("Mouse moved: {:?}", position);
                let new_mouse_pos = vec2(position.x as f32, position.y as f32);
                if allow_camera_movement {
                    game.camera.horizontal_angle +=
                        mouse_speed * delta.as_secs_f32() * (last_mouse_pos.x - new_mouse_pos.x);
                    game.camera.vertical_angle +=
                        mouse_speed * delta.as_secs_f32() * (last_mouse_pos.y - new_mouse_pos.y);
                }

                last_mouse_pos = new_mouse_pos;
            }

            winit::event::Event::WindowEvent {
                event:
                    winit::event::WindowEvent::MouseWheel {
                        delta: winit::event::MouseScrollDelta::PixelDelta(physical_position),
                        ..
                    },
                ..
            } => {
                println!("Mouse moved: {:?}", physical_position);
                let change_x = 0.1 * delta.as_secs_f32() * physical_position.x as f32;
                let change_y = 0.1 * mouse_speed * delta.as_secs_f32() * physical_position.y as f32;
                println!(
                    "Change in X, Y: {}, {}, delta secs: {}",
                    change_x,
                    change_y,
                    delta.as_secs_f32()
                );
                game.camera.horizontal_angle += change_x;
                game.camera.vertical_angle += change_y;
            }
            winit::event::Event::WindowEvent {
                event:
                    winit::event::WindowEvent::MouseInput {
                        button: winit::event::MouseButton::Middle,
                        state,
                        ..
                    },
                ..
            } => {
                // println!("Mouse moved: {:?}", position);
                allow_camera_movement = match state {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                }

                // game.camera.horizontal_angle += mouse_speed * delta.as_secs_f32() * (last_mouse_pos.x - new_mouse_pos.x);
                //  game.camera.vertical_angle += mouse_speed * delta.as_secs_f32() * (last_mouse_pos.y - new_mouse_pos.y);
            }

            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::AxisMotion { .. },
                ..
            } => {}
            winit::event::Event::WindowEvent {
                event:
                    winit::event::WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                virtual_keycode: Some(key),
                                state,
                                ..
                            },
                        ..
                    },
                ..
            } => match state {
                winit::event::ElementState::Pressed => {
                    keys_being_pressed.insert(key);
                }
                winit::event::ElementState::Released => {
                    keys_being_pressed.remove(&key);
                }
            },
            winit::event::Event::DeviceEvent { .. } => {}
            e => {
                println!("Window Event: {:?}", e)
            }
        }

        prev = current;
    });
}

fn main() {
    pollster::block_on(run());
}
