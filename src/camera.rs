use cgmath::*;
use std::f32::consts::FRAC_PI_2;
use winit::event::{ElementState, MouseScrollDelta};
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

pub struct Projection {
    pub aspect: f32,
    pub fovy: Rad<f32>,
    pub znear: f32,
    pub zfar: f32,
}

impl Projection {
    pub fn new(
        width: u32,
        height: u32,
        fovy: impl Into<Rad<f32>>,
        znear: f32,
        zfar: f32,
    ) -> Projection {
        Projection {
            aspect: width as f32 / height as f32,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }

    pub fn to_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * cgmath::perspective(self.fovy, self.aspect, self.znear, self.zfar)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> CameraUniform {
        CameraUniform {
            model: Matrix4::identity().into(),
            view: Matrix4::identity().into(),
            proj: Matrix4::identity().into(),
        }
    }

    pub fn update(&mut self, projection: &Projection, camera: &Camera) {
        self.model = Matrix4::from_value(1.0 as f32).into();
        self.proj = projection.to_matrix().into();
        self.view = camera.view_matrix().into();
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Camera {
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
    pub position: Point3<f32>,
}

impl Camera {
    pub fn new(
        position: Point3<f32>,
        yaw: impl Into<Rad<f32>>,
        pitch: impl Into<Rad<f32>>,
    ) -> Self {
        Self {
            yaw: yaw.into(),
            pitch: pitch.into(),
            position,
        }
    }

    pub fn direction(&self) -> Vector3<f32> {
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        Vector3 {
            x: cos_pitch * sin_yaw,
            y: sin_pitch,
            z: cos_pitch * cos_yaw,
        }
        .normalize()
    }

    pub fn right(&self) -> Vector3<f32> {
        Vector3 {
            x: (self.yaw.0 - 3.14f32 / 2.0f32).sin(),
            y: 0.0,
            z: (self.yaw.0 - 3.14f32 / 2.0f32).cos(),
        }
        .normalize()
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        let right = self.right();
        let direction = self.direction();
        let up = right.cross(direction);
        Matrix4::look_at_rh(self.position, self.position + direction, up)
    }
}

pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    speed: f32,
    sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn process_keyboard(
        &mut self,
        key: winit::event::VirtualKeyCode,
        state: ElementState,
    ) -> bool {
        let amount = match state {
            ElementState::Pressed => 1.0,
            ElementState::Released => 0.0,
        };
        match key {
            winit::event::VirtualKeyCode::W => self.amount_forward = amount as i32 as f32,
            winit::event::VirtualKeyCode::S => self.amount_backward = amount as i32 as f32,
            winit::event::VirtualKeyCode::A => self.amount_left = amount as i32 as f32,
            winit::event::VirtualKeyCode::D => self.amount_right = amount as i32 as f32,
            _ => return false,
        }
        return true;
    }

    pub fn process_mouse(&mut self, delta_x: f32, delta_y: f32) -> bool {
        self.rotate_horizontal = delta_x;
        self.rotate_vertical = delta_y;
        return true;
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: std::time::Duration) {
        let dt = dt.as_secs_f32();

        let forward_factor = (self.amount_forward - self.amount_backward) * self.speed * dt;
        let right_factor = (self.amount_right - self.amount_left) * self.speed * dt;

        //we go in the direction the camera is pointing
        let direction = camera.direction();
        let right = camera.right();

        camera.position += direction * forward_factor;
        camera.position += right * right_factor;

        //rotate
        camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        //clamp pitch
        camera.pitch = Rad(camera.pitch.0.max(-SAFE_FRAC_PI_2).min(SAFE_FRAC_PI_2));
    }
}
