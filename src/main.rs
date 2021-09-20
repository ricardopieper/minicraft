#![feature(const_fn_floating_point_arithmetic)]
mod camera;
mod blocks;

use vulkano::instance::Instance;
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{WindowBuilder};
use vulkano::instance::debug::{DebugCallback, MessageType};
use std::sync::Arc;
use vk::sync::GpuFuture;
use cgmath::*;
use camera::*;
use blocks::*;
use blocks::Vertex;

type WindowSurface = Arc<vk::swapchain::Surface<winit::window::Window>>; 
type WindowSwapchain = Arc<vk::swapchain::Swapchain<winit::window::Window>>; 
type WindowSwapchainImage = std::sync::Arc<vulkano::image::SwapchainImage<winit::window::Window>>;

const VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_LUNARG_standard_validation"
];


#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;

#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
    if !ENABLE_VALIDATION_LAYERS {
        return None;
    }

    let msg_types = MessageType {
        general: true,
        validation: true,
        performance: true
    };

    DebugCallback::new(&instance, vulkano::instance::debug::MessageSeverity::errors_and_warnings(), msg_types, |msg| {
        println!("Vulkan validation: {:?}", msg.description)
    }).ok()
}

fn check_validation_layer_support() -> bool {
    let layers: Vec<_> = vulkano::instance::layers_list()
        .unwrap()
        .map(|x| x.name().to_owned())
        .collect();
    
    VALIDATION_LAYERS.iter().all(|layer_name| layers.contains(&layer_name.to_string()))
}

fn create_vulkan_instance() -> Arc<vulkano::instance::Instance> {

    let extensions_supported_by_core = vulkano::instance::InstanceExtensions::supported_by_core().unwrap();
    println!("Extensions supported by core: {:?}", extensions_supported_by_core);

    let app_info = vulkano::instance::ApplicationInfo {
        application_name: Some(std::borrow::Cow::from("Minicraft")),
        application_version: Some(vulkano::instance::Version {
            major: 1,
            minor: 0,
            patch: 0
        }),
        engine_name: Some(std::borrow::Cow::from("No engine")),
        engine_version: Some(vulkano::instance::Version {
            major: 1,
            minor: 0,
            patch: 0
        }),
    };

    let mut extensions = vulkano_win::required_extensions();
    if ENABLE_VALIDATION_LAYERS {
        println!("Required by vulkano_win: {:?}", extensions);
    }

    if ENABLE_VALIDATION_LAYERS && check_validation_layer_support() {
        println!("Creating debug instance");
        extensions.ext_debug_utils = true;
        let instance = vulkano::instance::Instance::new(Some(&app_info), vk::Version::V1_1, &extensions, VALIDATION_LAYERS.iter().cloned());
        return instance.expect("Failed to create Vulkan instance");
    } else {
        let instance = vulkano::instance::Instance::new(Some(&app_info), vk::Version::V1_0, &extensions, None);
        return instance.expect("Failed to create Vulkan instance");
    }
}

fn rate_device_suitability(device: vulkano::instance::PhysicalDevice) -> u32 {
    let features = device.supported_features();
    
    if !features.geometry_shader {
        return 0;
    }

    use vulkano::instance::PhysicalDeviceType;

    let mut score: u32 = 0;

    let properties: &vk::device::Properties = device.properties();

    match properties.device_type.unwrap() {
        PhysicalDeviceType::DiscreteGpu => score += 1000,
        PhysicalDeviceType::IntegratedGpu => score += 500,
        PhysicalDeviceType::VirtualGpu => score += 100,
        PhysicalDeviceType::Cpu => score += 10,
        PhysicalDeviceType::Other => score += 1,
    };

    score += properties.max_image_dimension2_d.unwrap_or(0);

    return score;
}

fn get_physical_device(instance: &Arc<Instance>) -> vk::instance::PhysicalDevice {

    let all_devices = vulkano::instance::PhysicalDevice::enumerate(instance);
    for device in all_devices.clone() {
        let device_type = device.properties().device_type;
        let device_name = device.properties().device_name.clone().unwrap();
        println!("Device found: ${:?}, type: {:?}", device_name, device_type);
    }

    let mut devices_ranked_by_suitability: Vec<_> = all_devices
        .clone()
        .map(|x| (x, rate_device_suitability(x)))
        .collect();

    devices_ranked_by_suitability.sort_by(|(_, rate1), (_, rate2)| rate1.cmp(rate2));
    devices_ranked_by_suitability.reverse();

    let (device, _) = devices_ranked_by_suitability[0];
    return device;
}


#[derive(Copy, Clone)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

use vulkano as vk;

impl QueueFamilyIndices {

    fn is_complete(&self) -> bool {
        if self.graphics_family.is_some() && self.present_family.is_some() {
            return true;
        }
        return false;
    }

    fn find_queue_families(
        device: vk::instance::PhysicalDevice, 
        surface: &Arc<vk::swapchain::Surface<winit::window::Window>>    
    ) -> QueueFamilyIndices {
        let mut queue_family_indices = QueueFamilyIndices {
            graphics_family: None,
            present_family: None
        };
        
        for (i, queue_family) in device.queue_families().enumerate() {
            if queue_family.supports_graphics() {
                queue_family_indices.graphics_family = Some(i as u32)
            }
            if surface.is_supported(queue_family).unwrap_or(false) {
                queue_family_indices.present_family = Some(i as u32)
            }

            if queue_family_indices.is_complete() {
                break
            }
        }

        return queue_family_indices;        
    }
}

fn create_logical_device(device: vk::instance::PhysicalDevice, indices: QueueFamilyIndices) -> (Arc<vk::device::Device>, Arc<vk::device::Queue>, Arc<vk::device::Queue>) {
    let queue_family = device.queue_families().nth(indices.graphics_family.unwrap() as usize).unwrap();
    let extensions_supported_by_chosen_device = vk::device::DeviceExtensions::supported_by_device(device);
    println!("Extensions supported by {}: {:?}", device.properties().device_name.clone().unwrap(), extensions_supported_by_chosen_device);
    println!("Features supported by {}: {:?}", device.properties().device_name.clone().unwrap(), device.supported_features());
    
    let device_extensions = vk::device::DeviceExtensions {
        khr_swapchain: true,
        .. vk::device::DeviceExtensions::none()
    };    
    
    let (device, mut queues) = vk::device::Device::new(
        device, 
        &vk::device::Features::none(),
        &vk::device::DeviceExtensions::required_extensions(device).union(&device_extensions),
        [(queue_family, 1.0)].iter().cloned()).unwrap();
    
    let graphics_queue = queues.next().unwrap();
    let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

    return (device, graphics_queue, present_queue)
}

struct SwapchainSupportDetails { capabilities: vk::swapchain::Capabilities }

fn get_swapchain_support_details(device: vk::instance::PhysicalDevice, surface: &WindowSurface) -> SwapchainSupportDetails {
    let capabilities = surface.capabilities(device);
    return SwapchainSupportDetails { capabilities: capabilities.unwrap() };
}

fn find_optimal_format(details: &SwapchainSupportDetails) -> (vk::format::Format, vk::swapchain::ColorSpace) {

    let optimal_format = vk::format::Format::B8G8R8A8Srgb;
    let optimal_space = vk::swapchain::ColorSpace::SrgbNonLinear;

    let optimal_found = details.capabilities.supported_formats
        .iter()
        .find(|(format, space)| *format == optimal_format && *space == optimal_space);

    match optimal_found {
        Some(_) => return (optimal_format, optimal_space),
        None => {
            let format = details.capabilities.supported_formats[0];
            println!("All formats: {:?}",  details.capabilities.supported_formats);
            println!("Found format {:?}", format);
            return format;
        }
    }
}

const TRY_MAILBOX : bool = true;
const TRY_IMMEDIATE : bool = true;
fn find_optimal_present_mode(details: &SwapchainSupportDetails) -> vk::swapchain::PresentMode {
    if details.capabilities.present_modes.mailbox && TRY_MAILBOX {
        vk::swapchain::PresentMode::Mailbox
    } else if details.capabilities.present_modes.immediate && TRY_IMMEDIATE {
        vk::swapchain::PresentMode::Immediate
    } else {
        vk::swapchain::PresentMode::Fifo
    }
}
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
fn find_swap_extent(details: &SwapchainSupportDetails) -> [u32; 2] {
    if let Some(extent) = details.capabilities.current_extent {
        return extent;
    } else {
        let mut actual_extent = [WIDTH, HEIGHT];
        actual_extent[0] = details.capabilities.min_image_extent[0].max(
            details.capabilities.max_image_extent[0].min(WIDTH)
        );
        actual_extent[1] = details.capabilities.min_image_extent[1].max(
            details.capabilities.max_image_extent[1].min(HEIGHT)
        );
        return actual_extent;
    }
}

mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "./src/shaders/vertex.glsl"
    }
}


mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "./src/shaders/fragment.glsl"
    }
}
//type AbstractRenderPass = dyn vk::framebuffer::RenderPassAbstract + Send + Sync;

type Framebuffer = dyn vk::render_pass::FramebufferAbstract + Send + Sync;
type MinicraftGraphicsPipeline = dyn vk::pipeline::GraphicsPipelineAbstract + Send + Sync;


fn create_graphics_pipeline(device: &Arc<vk::device::Device>, swapchain: &WindowSwapchain) -> 
    (Arc<MinicraftGraphicsPipeline>, Arc<vk::render_pass::RenderPass>) {
    let vs = vertex_shader::Shader::load(device.clone()).unwrap();
    let fs = fragment_shader::Shader::load(device.clone()).unwrap();

    let render_pass = create_render_pass(&device, swapchain.format());

    let pipeline_builder = 
        vk::pipeline::GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>() 
        .vertex_shader(vs.main_entry_point(), ())
        /*VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;*/
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        //.primitive_restart(false)
        /*
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;
        */
        //.viewports(vec![viewport]) //if you do this then pipe becomes non-dynamic
        .fragment_shader(fs.main_entry_point(), ())
        /*
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        */
        .polygon_mode_fill() //can we change this to render wireframe?
        .line_width(1.0) 
        //without this line depth testing is disabled by default
        .depth_stencil_simple_depth()
        
        .cull_mode_back()
        .front_face_clockwise() 
        /*
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional
        */
        .blend_pass_through()
        .render_pass(vk::render_pass::Subpass::from(render_pass.clone(), 0).unwrap());
    

    let result = pipeline_builder.build(device.clone()).unwrap();
    (Arc::new(result), render_pass.clone())
}


fn create_render_pass(device: &Arc<vk::device::Device>, color_format: vk::format::Format) -> Arc<vk::render_pass::RenderPass> {
    
    //it's unfortunate that vulkano decided a macro would be ideal. 
    //There are only a few details here to fill up, and it doesn't seem macros really give any benefit.
    //Macro stuff also isn't indexed by Rust-analyzer, the syntax isn't predictable. 
    //IMO heavy use of macros in this situation only makes things hard to figure out using only Rust tools.
    //Like, for the shaders it's ok.

    let result = Arc::new(
        vk::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: color_format,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: vk::format::Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        ).unwrap()
    );

    return result;
}

//Arc<MinicraftGraphicsPipeline>, Arc<vk::render_pass::RenderPass>
fn create_framebuffers(device: &Arc<vk::device::Device>, render_pass: &Arc<vk::render_pass::RenderPass>, images: &[WindowSwapchainImage]) 
    -> Vec<Arc<Framebuffer>> {
    let depth_buffer = vk::image::view::ImageView::new(
        vk::image::attachment::AttachmentImage::transient(device.clone(), images[0].dimensions(), vk::format::Format::D16Unorm).unwrap()
    ).unwrap();

    return images.iter()
    .map(move |image| {
        let view = vk::image::view::ImageView::new(image.clone()).unwrap();
        Arc::new(
            vk::render_pass::Framebuffer::start(render_pass.clone())
            .add(view)
            .unwrap()
            .add(depth_buffer.clone())
            .unwrap()
            .build()
            .unwrap()
        ) as Arc<Framebuffer>
    }).collect();
}



/// Perspective matrix that is suitable for Vulkan.
///
/// It inverts the projected y-axis. And set the depth range to 0..1
/// instead of -1..1. Mind the vertex winding order though.
pub fn perspective<S, F>(fovy: F, aspect: S, near: S, far: S) -> Matrix4<S>
where
    S: BaseFloat,
    F: Into<Rad<S>>,
{
    let two = S::one() + S::one();
    let f = Rad::cot(fovy.into() / two);

    let c0r0 = f / aspect;
    let c0r1 = S::zero();
    let c0r2 = S::zero();
    let c0r3 = S::zero();

    let c1r0 = S::zero();
    let c1r1 = -f;
    let c1r2 = S::zero();
    let c1r3 = S::zero();

    let c2r0 = S::zero();
    let c2r1 = S::zero();
    let c2r2 = -far / (far - near);
    let c2r3 = -S::one();

    let c3r0 = S::zero();
    let c3r1 = S::zero();
    let c3r2 = -(far * near) / (far - near);
    let c3r3 = S::zero();

    #[cfg_attr(rustfmt, rustfmt::skip)]
    Matrix4::new(
        c0r0, c0r1, c0r2, c0r3,
        c1r0, c1r1, c1r2, c1r3,
        c2r0, c2r1, c2r2, c2r3,
        c3r0, c3r1, c3r2, c3r3,
    )
}

/// Clamp `value` between `min` and `max`.
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    let value = if value > max { max } else { value };
    if value < min {
        min
    } else {
        value
    }
}

fn update_uniform_buffer(start_time: std::time::Instant, dimensions: [f32; 2], camera: &Camera) -> vertex_shader::ty::UniformBufferObject {
    let elapsed = std::time::Instant::now().duration_since(start_time);
    let model = Matrix4::from_value(1 as f32);

    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
    let mut proj = cgmath::perspective(
        Deg(45.0),
        aspect_ratio,
        0.01,
        10000.0,
    );
    proj.y.y *= -1.0;

    let view = camera.view_matrix();
    return vertex_shader::ty::UniformBufferObject {
        model: model.into(),
        view: (view).into(),
        proj: proj.into(),
    }

}


struct Minicraft {
    camera: Camera, 
    movement_speed: f32
}


struct MinicraftVulkanGraphics {
    instance: Arc<vk::instance::Instance>,
    physical_device_index: usize,
    logical_device: Arc<vk::device::Device>,
    graphics_queue: Arc<vk::device::Queue>,
    present_queue: Arc<vk::device::Queue>,
    family_indices: QueueFamilyIndices,
    vertex_buffer: Arc<dyn vk::buffer::BufferAccess + Send + Sync>,
    index_buffer: Arc<dyn vk::buffer::TypedBufferAccess<Content=[u32]> + Send + Sync>,
   // #[allow(dead_code)]
    uniform_buffers: vulkano::buffer::CpuBufferPool<vertex_shader::ty::UniformBufferObject>,
    swapchain: WindowSwapchain,
    swapchain_images: Vec<WindowSwapchainImage>,
    framebuffers: Vec<Arc<Framebuffer>>,
    #[allow(dead_code)]
    start_time: std::time::Instant,
    pipeline: Arc<MinicraftGraphicsPipeline>, 
    render_pass: Arc<vk::render_pass::RenderPass>
}

impl MinicraftVulkanGraphics {
    fn physical_device(&self) -> vk::instance::PhysicalDevice {
        return vk::instance::PhysicalDevice::from_index(&self.instance, self.physical_device_index).unwrap();
    }

    fn new() -> (MinicraftVulkanGraphics, winit::event_loop::EventLoop<()>) {
        let instance = create_vulkan_instance();
        setup_debug_callback(&instance.clone());

        let event_loop = EventLoop::new();
        let surface: WindowSurface = WindowBuilder::new()
            .with_title("Minicraft")
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        let physical_device = get_physical_device(&instance);

        println!("Chosen device: {}", physical_device.properties().device_name.as_ref().unwrap());
    
        let queue_family_indices = QueueFamilyIndices::find_queue_families(physical_device, &surface);
        let (logical_device, graphics_queue, present_queue) = create_logical_device(physical_device, queue_family_indices);

        let world = World::worldgen();
        let start = std::time::Instant::now();
        let (vertices, indices) = world.meshgen();
        let elapsed = start.elapsed();
        println!("Meshgen took {:?}", elapsed);
       
        /* let vertices = vec![
            Vertex::new([1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
            Vertex::new([2.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
            Vertex::new([2.0, 0.0, 0.0], [0.8, 0.2, 0.0]),
            Vertex::new([1.0, 0.0, 0.0], [0.8, 0.2, 0.0]),

            Vertex::new([1.0, 1.0,  -10.0], [0.0, 1.0, 0.0]),
            Vertex::new([2.0, 1.0,  -10.0], [0.0, 1.0, 0.0]),
            Vertex::new([2.0, 0.0,  -10.0], [0.8, 0.2, 0.0]),
            Vertex::new([1.0, 0.0,  -10.0], [0.8, 0.2, 0.0]),
        ];
        
        let indices = [
            //front
            0, 1, 2,  2, 3, 0,
            //right
            1,5,6,     6,2,1,    
            //left
            4,0,3,     3, 7, 4,
            //back
            5,4,7,     7,6,5,
            //top
            4,5,1,     1,0,4,
            //bottom:
            3,2,6,     6,7,3
        ];
        */
       
        let (swapchain, images) = Self::create_swapchain(&logical_device, 
            physical_device, &surface, queue_family_indices);

        let start = std::time::Instant::now();

        let (vertex_buffer, future) = vk::buffer::ImmutableBuffer::from_iter(
            vertices.iter().cloned(),
            vk::buffer::BufferUsage::vertex_buffer(), 
            graphics_queue.clone(),
        ).unwrap();
        future.flush().unwrap();
        

        let (index_buffer, future) = vk::buffer::ImmutableBuffer::from_iter(
            indices.iter().cloned(),
            vk::buffer::BufferUsage::index_buffer(), 
            graphics_queue.clone(),
        ).unwrap();
        future.flush().unwrap();


        let uniforms = vk::buffer::cpu_pool::CpuBufferPool::<vertex_shader::ty::UniformBufferObject>::new(
            logical_device.clone(), vk::buffer::BufferUsage::all()
        );


        println!("Created swapchain: {:?} with {} images", swapchain, images.len());
        let (pipeline, render_pass) = create_graphics_pipeline(&logical_device,  &swapchain);
        
        println!("Created graphics pipeline");

        let framebuffers = create_framebuffers(&logical_device, &render_pass, &images);

        let window_context = MinicraftVulkanGraphics {
            instance: instance.clone(),
            physical_device_index: physical_device.index(),
            graphics_queue: graphics_queue,
            present_queue: present_queue,
            logical_device: logical_device,
            family_indices: queue_family_indices,
            vertex_buffer: vertex_buffer,
            index_buffer: index_buffer,
            start_time: start,
            uniform_buffers: uniforms,
            pipeline: pipeline,
            render_pass: render_pass,

            //must be recreated together
            swapchain: swapchain,
            swapchain_images: images,
            framebuffers: framebuffers,
        };


        return (window_context, event_loop);
    }



    fn recreate_graphics(&mut self, width: u32, height: u32) {
        let (swapchain, images) = self.swapchain.recreate().dimensions([width, height]).build().unwrap();
        println!("Recreated swapchain: {:?} with {} images", swapchain, images.len());
        self.framebuffers = create_framebuffers(&self.logical_device, &self.render_pass, &images);
        self.swapchain = swapchain;
        self.swapchain_images = images;
    }

    fn create_swapchain(logical_device: &Arc<vk::device::Device>, 
        device: vk::instance::PhysicalDevice,
        surface: &WindowSurface,
        queue_family: QueueFamilyIndices
    ) -> (WindowSwapchain, Vec<WindowSwapchainImage>) {

        let capabilities = get_swapchain_support_details(device, &surface);
        let swapchain_adequate = !capabilities.capabilities.supported_formats.is_empty() 
        && !(capabilities.capabilities.present_modes == vk::swapchain::SupportedPresentModes::none());

        if !swapchain_adequate {
            panic!("Graphics card chosen does not have adequate swapchain capabilities");
        }

        let mut image_count = capabilities.capabilities.min_image_count + 1;
        if let Some(max) = capabilities.capabilities.max_image_count  {
            if image_count > max {
                image_count = max;
            }
        }
        println!("Will request {} images", image_count);

        let (format, color_space) = find_optimal_format(&capabilities);
        println!("Format and color space: {:?}", (format, color_space));

        let presentation = find_optimal_present_mode(&capabilities);
        println!("Presentation mode: {:?}", presentation);

        let extent = find_swap_extent(&capabilities);
        println!("Swapchain extent: {:?}", extent);

        let image_use = vk::image::ImageUsage {
        color_attachment: true, //VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        .. vk::image::ImageUsage::none()
        };

        let has_to_use_concurrent = queue_family.graphics_family != queue_family.present_family;
        let sharing_mode = if has_to_use_concurrent {
            vk::sync::SharingMode::Concurrent(vec![2])
        } else {
            vk::sync::SharingMode::Exclusive
        };


        let composite_alpha = vk::swapchain::CompositeAlpha::Opaque;//means "no alpha" which is weird that the tutorial wants us to use it
        let fullscreen_exclusive = vk::swapchain::FullscreenExclusive::Default;//means "no alpha" which is weird that the tutorial wants us to use it
        let clip_invisible_window_pixels = true; //if another window is on top of ours, then we don't render the obscured pixels

        let swapchain_builder = vk::swapchain::Swapchain::start(logical_device.clone(), surface.clone());
        let (swapchain, images) = swapchain_builder
            .num_images(image_count)
            .format(format)
            .dimensions(extent)
            .layers(1)
            .usage(image_use)
            .sharing_mode(sharing_mode)
            .transform(capabilities.capabilities.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(presentation)
            .fullscreen_exclusive(fullscreen_exclusive)
            .clipped(clip_invisible_window_pixels)
            .color_space(color_space)
            .build().expect("Could not create swapchain!");
    

        println!("Swapchain created!");

        return (swapchain, images);
    }


}



fn main() {
    let mut game = Minicraft {
        camera: Camera {
            position: vec3(0.0, 0.0, 0.0),
            horizontal_angle: -3.14,
            vertical_angle: 0.0
        },
        movement_speed: 1.0
    };
    let (mut window_context, event_loop) = MinicraftVulkanGraphics::new();
    let mouse_speed = 1f32;
    let mut recreate_swapchain = false;
    let mut prev = std::time::SystemTime::now();
    let mut frame = 0u64;
    let mut resize_width = WIDTH;
    let mut resize_height = HEIGHT;



    let mut keys_being_pressed = std::collections::hash_set::HashSet::new();
    let mut allow_camera_movement = false;
    let mut last_mouse_pos = vec2(0.0, 0.0);

    use image::*;

    let (texture, tex_future) = {
        let png_bytes = std::include_bytes!("./textures/atlas1.png").to_vec();
        let mut image = image::load_from_memory_with_format(&png_bytes, image::ImageFormat::Png).unwrap();
        let (width, height) = image.dimensions();
        let vulkan_dims = vulkano::image::ImageDimensions::Dim2d {
            width: width,
            height: height,
            array_layers: 1
        };
        let image_data = image.as_mut_rgba8().unwrap();
        let image_bytes = image_data.to_vec();
        let (img, future) = vk::image::ImmutableImage::from_iter(
            image_bytes.iter().cloned(),
            vulkan_dims,
            vk::image::MipmapsCount::One,
            vk::format::Format::R8G8B8A8Srgb,
            window_context.graphics_queue.clone()).unwrap();
        
        (vk::image::view::ImageView::new(img).unwrap(), future)
    };

    let sampler = vk::sampler::Sampler::new(
        window_context.logical_device.clone(),
        vk::sampler::Filter::Nearest,
        vk::sampler::Filter::Nearest,
        vk::sampler::MipmapMode::Nearest,
        vk::sampler::SamplerAddressMode::Repeat,
        vk::sampler::SamplerAddressMode::Repeat,
        vk::sampler::SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0).unwrap();
    
    let mut previous_frame_end = Some(tex_future.boxed());

    event_loop.run(move |event, _, control_flow| {

        let current = std::time::SystemTime::now();
        let delta = current.duration_since(prev).unwrap(); 

        match event {
            winit::event::Event::WindowEvent  { event: winit::event::WindowEvent::CloseRequested, .. } => {
                println!("Closed requested!");
                *control_flow  = ControlFlow::Exit;
            },
            winit::event::Event::WindowEvent  { event: winit::event::WindowEvent::Resized(size), .. } => {
                println!("Resized to {:?}! Will recreate swapchain", size);
                resize_width = size.width;
                resize_height = size.height;
                recreate_swapchain = true;
            },
            
            winit::event::Event::MainEventsCleared => {

                let movement = game.movement_speed * delta.as_secs_f32();
                for key in keys_being_pressed.iter() {
                    match key {
                        winit::event::VirtualKeyCode::W => {
                            game.camera.go_forward(movement);
                        },
                        winit::event::VirtualKeyCode::S => {
                            game.camera.go_backward(movement);
                        },
                        winit::event::VirtualKeyCode::A => {
                            game.camera.go_left(movement);
                        },
                        winit::event::VirtualKeyCode::D => {
                            game.camera.go_right(movement);
                        }, 
                        winit::event::VirtualKeyCode::Key1 => {
                            println!("Camera: {:?}", game.camera);
                        },
                        _ => {}
                    }
                }
              // 

                previous_frame_end.as_mut().unwrap().cleanup_finished();

              
                let milis = delta.as_secs_f64() * 1000.0f64;
                if frame % 1000 == 0 {
                    if milis == 0.0 {
                        println!("fps: infinite, elapsed: {:?}", milis);
                    } else {
                        println!("fps: {:?}, elapsed: {:?}", 1000.0 / milis, milis);
                    }
                }

                prev = current;
                frame = frame + 1;

                let uniform_buffer_subbuffer = {
                    let uniforms = update_uniform_buffer(window_context.start_time, [resize_width as f32, resize_height as f32], &game.camera);
                    window_context.uniform_buffers.next(uniforms).unwrap()
                };
                
                use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

                let pipe_layout = window_context.pipeline.layout().descriptor_set_layout(0).unwrap();
                let descriptor_set = Arc::new(
                    PersistentDescriptorSet::start(pipe_layout.clone())
                        .add_buffer(uniform_buffer_subbuffer).unwrap()
                        .add_sampled_image(texture.clone(), sampler.clone()).unwrap()
                        .build()
                        .unwrap()
                );

                //println!("Redraw Events");
                let acquire_result = vk::swapchain::acquire_next_image(window_context.swapchain.clone(), None);
        
                match acquire_result {
                    Ok((image_idx, suboptimal, acquire_future)) => {
                        if suboptimal {
                            println!("Suboptimal! will recreate swapchain");
                            recreate_swapchain = true;
                        } else {
                            
                            let image = &window_context.swapchain_images[image_idx];
                            let framebuffer = &window_context.framebuffers[image_idx];

                            let dimensions = image.dimensions();

                            let subpass_contents = vk::command_buffer::SubpassContents::Inline;
                            let mut dynamic_state = vk::command_buffer::DynamicState::none();
                            let width: f32 = dimensions[0] as f32;
                            let height: f32 = dimensions[1] as f32;
                            let viewport = vk::pipeline::viewport::Viewport {
                                origin: [0.0, 0.0],
                                dimensions: [width, height],
                                depth_range: 0.0 .. 1.0
                            };
                            dynamic_state.viewports = Some(vec![viewport]); 
                            
                            let mut command_buffer_builder = vk::command_buffer::AutoCommandBufferBuilder::primary(
                                window_context.logical_device.clone(), 
                                window_context.graphics_queue.family(), 
                                vk::command_buffer::CommandBufferUsage::OneTimeSubmit).unwrap();
                            
                            command_buffer_builder
                                .begin_render_pass(framebuffer.clone(), subpass_contents, vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()])
                                .unwrap()
                                .draw_indexed(
                                    window_context.pipeline.clone(), 
                                    &dynamic_state,
                                    vec![window_context.vertex_buffer.clone()], 
                                    window_context.index_buffer.clone(), 
                                    descriptor_set.clone(), (), vec![])
                                .unwrap()
                                .end_render_pass()
                                .unwrap();
            
                            let buffer = Arc::new(command_buffer_builder.build().unwrap());
                          
                            let future = previous_frame_end
                                    .take()
                                    .unwrap()
                                    .join(acquire_future)
                                    .then_execute(window_context.graphics_queue.clone(), buffer)
                                    .unwrap()
                                    .then_swapchain_present(window_context.present_queue.clone(), window_context.swapchain.clone(), image_idx)
                                    .then_signal_fence_and_flush();
    
                            match future {
                                Err(vk::sync::FlushError::OutOfDate) => {
                                    println!("Aquire is out of date! Maybe will recreate swapchain in the next events.");
                                    //recreate_swapchain = true
                                    previous_frame_end = Some(vk::sync::now(window_context.logical_device.clone()).boxed());
                                },
                                Err(e) => panic!("Error on acquire_future: {:?}", e),
                                Ok(f) => {
                                    previous_frame_end = Some(f.boxed());
                                }
                            }
                        }
                    },
                    Err(vulkano::swapchain::AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        println!("Command buffer out of date! will recreate swapchain");
                    }
                    Err(e) => {
                        println!("Acquire error: {:?}", e);
                    }
                  
                }
                //limit FPS to 200 to not overstress the GPU during testing,
                //but still make dragging the window smooth
                std::thread::sleep(std::time::Duration::from_millis(5));

            },
            winit::event::Event::RedrawEventsCleared => { 
            },
            winit::event::Event::RedrawRequested(_) => {
                

            },
            winit::event::Event::NewEvents(_) => {},
            winit::event::Event::WindowEvent { event : winit::event::WindowEvent::CursorMoved{ position, ..}, .. } => {
                //println!("Mouse moved: {:?}", position);
                let new_mouse_pos = vec2(position.x as f32, position.y as f32);
                if allow_camera_movement {
                    game.camera.horizontal_angle += mouse_speed * delta.as_secs_f32() * (last_mouse_pos.x - new_mouse_pos.x);
                    game.camera.vertical_angle += mouse_speed * delta.as_secs_f32() * (last_mouse_pos.y - new_mouse_pos.y);
                }
               
                last_mouse_pos = new_mouse_pos;
            },
            winit::event::Event::WindowEvent { event : winit::event::WindowEvent::MouseWheel { delta : winit::event::MouseScrollDelta::PixelDelta(
                physical_position
            ) , ..}, .. } => {
                println!("Mouse moved: {:?}", physical_position);
                let change_x = 0.1 * delta.as_secs_f32() * physical_position.x as f32;
                let change_y = 0.1 * mouse_speed * delta.as_secs_f32() * physical_position.y as f32;
                println!("Change in X, Y: {}, {}, delta secs: {}", change_x, change_y, delta.as_secs_f32());
                game.camera.horizontal_angle += change_x;
                game.camera.vertical_angle += change_y;
            },
            winit::event::Event::WindowEvent { event : winit::event::WindowEvent::MouseInput{ 
                button: winit::event::MouseButton::Middle, state, ..}, .. } => {
                // println!("Mouse moved: {:?}", position);
                allow_camera_movement = match state {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                }
 
                // game.camera.horizontal_angle += mouse_speed * delta.as_secs_f32() * (last_mouse_pos.x - new_mouse_pos.x);
               //  game.camera.vertical_angle += mouse_speed * delta.as_secs_f32() * (last_mouse_pos.y - new_mouse_pos.y);
               
             },
            
            winit::event::Event::WindowEvent { event : winit::event::WindowEvent::AxisMotion{ ..}, .. } => { },
            winit::event::Event::WindowEvent { event : winit::event::WindowEvent::KeyboardInput {
                input: winit::event::KeyboardInput { virtual_keycode: Some(key), state, .. }, .. }, ..} => {

                match state {
                    winit::event::ElementState::Pressed => {
                        keys_being_pressed.insert(key);
                    },
                    winit::event::ElementState::Released => {
                        keys_being_pressed.remove(&key);
                    }
                }
             },
            winit::event::Event::DeviceEvent { .. } => { },
            e => {
                println!("Window Event: {:?}", e)
            }
        }

 
       

        if recreate_swapchain {
            window_context.recreate_graphics(resize_width, resize_height);
            recreate_swapchain = false;
        }
      
    });
}