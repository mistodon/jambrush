#version 450
#extension GL_ARB_separate_shader_objects : enable

vec4 POSITIONS[6] = vec4[](
    vec4(0.0, 0.0, 0.0, 1.0),
    vec4(0.0, 1.0, 0.0, 1.0),
    vec4(1.0, 0.0, 0.0, 1.0),
    vec4(0.0, 1.0, 0.0, 1.0),
    vec4(1.0, 1.0, 0.0, 1.0),
    vec4(1.0, 0.0, 0.0, 1.0)
);

layout(location = 0) out vec4 vcolor;
layout(location = 1) out vec2 vuv;

// layout(binding = 0) uniform Uniforms {
// } uniforms;

layout(push_constant) uniform Pushies {
    mat4 transform;
    vec4 tint;
    vec2 uv_origin;
    vec2 uv_scale;
} pushies;

void main() {
    vec4 pos = POSITIONS[gl_VertexIndex];
    vcolor = pushies.tint;
    vuv = pushies.uv_origin + pushies.uv_scale * pos.xy;
    gl_Position = pushies.transform * pos;
}
