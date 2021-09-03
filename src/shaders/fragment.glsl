#version 450
       
layout(location=0) out vec4 outColor;
layout(location=0) in vec3 vertexColor;
layout(location=1) in vec2 fragTexCoords;
layout(location=2) in float fragLightIntensity;

layout(set=0, binding=1) uniform sampler2D tex;

void main() {
    //outColor = vec4(vertexColor, 1); 
    outColor = vec4(texture(tex, fragTexCoords).xyz * fragLightIntensity, 1.0);
}