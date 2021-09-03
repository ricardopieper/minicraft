use cgmath::*;

#[derive(Copy, Clone, Debug)]
pub struct Camera {
    pub horizontal_angle: f32,
    pub vertical_angle: f32,
    pub position: Vector3<f32>
}

impl Camera {
    pub fn right(&self) -> Vector3<f32> {
        Vector3 {
            x: (self.horizontal_angle - 3.14f32 / 2.0f32).sin(),
            y: 0.0,
            z: (self.horizontal_angle - 3.14f32 / 2.0f32).cos()
        }.normalize()
    }

    pub fn direction(&self) -> Vector3<f32> {
        Vector3 {
            x: self.vertical_angle.cos() * self.horizontal_angle.sin(),
            y: self.vertical_angle.sin(),
            z: self.vertical_angle.cos() * self.horizontal_angle.cos()
        }.normalize()
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        let right = self.right();
        let direction = self.direction();
        let up = right.cross(direction);
        Matrix4::look_at_rh(
            Point3::from_vec(self.position),
            Point3::from_vec(self.position + direction),
            up
        )
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