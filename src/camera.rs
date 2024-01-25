use cgmath::*;

#[derive(Copy, Clone, Debug)]
pub struct Camera {
    pub horizontal_angle: f32,
    pub vertical_angle: f32,
    pub position: Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
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

    pub fn update(&mut self, camera: &Camera) {
        let uniform = camera.view_proj_matrix();
        self.model = uniform.model;
        self.proj = uniform.proj;
        self.view = uniform.view;
    }
}

impl Camera {
    pub fn right(&self) -> Vector3<f32> {
        Vector3 {
            x: (self.horizontal_angle - 3.14f32 / 2.0f32).sin(),
            y: 0.0,
            z: (self.horizontal_angle - 3.14f32 / 2.0f32).cos(),
        }
        .normalize()
    }

    pub fn direction(&self) -> Vector3<f32> {
        Vector3 {
            x: self.vertical_angle.cos() * self.horizontal_angle.sin(),
            y: self.vertical_angle.sin(),
            z: self.vertical_angle.cos() * self.horizontal_angle.cos(),
        }
        .normalize()
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        let right = self.right();
        let direction = self.direction();
        let up = right.cross(direction);
        Matrix4::look_at_rh(
            Point3::from_vec(self.position),
            Point3::from_vec(self.position + direction),
            up,
        )
    }

    pub fn view_proj_matrix(&self) -> CameraUniform {
        // let elapsed = std::time::Instant::now().duration_since(start_time);
        let model = Matrix4::from_value(1 as f32);

        let mut proj = cgmath::perspective(Deg(self.fovy), self.aspect, self.znear, self.zfar);
        proj.y.y *= -1.0;

        let view = self.view_matrix();
        return CameraUniform {
            model: model.into(),
            view: (view).into(),
            proj: proj.into(),
        };
    }

    pub fn go_forward(&mut self, factor: f32) {
        self.position = self.position + (self.direction() * factor);
    }

    pub fn go_backward(&mut self, factor: f32) {
        self.position = self.position - (self.direction() * factor);
    }

    pub fn go_right(&mut self, factor: f32) {
        self.position = self.position + (self.right() * factor);
    }

    pub fn go_left(&mut self, factor: f32) {
        self.position = self.position - (self.right() * factor);
    }
}
