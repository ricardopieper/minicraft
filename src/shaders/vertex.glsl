#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location=0) in vec3 position;
layout(location=1) in vec3 color; 
layout(location=2) in vec2 tex_coords; 
layout(location=3) in float light_intensity; 

layout(location = 0) out vec3 vertexColor;
layout(location = 1) out vec2 fragTexCoords;
layout(location = 2) out float fragLightIntensity;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
    vertexColor = color;
    fragTexCoords = tex_coords;
    fragLightIntensity = light_intensity;
}