#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 vcolor;
layout(location = 1) in vec2 vuv;
layout(location = 2) in vec4 vdepth_uv_scale_add;

layout(location = 0) out vec4 target;

layout(set = 0, binding = 0) uniform texture2D color_map;
layout(set = 0, binding = 1) uniform sampler color_sampler;
layout(set = 1, binding = 0) uniform texture2D depth_map;
layout(set = 1, binding = 1) uniform sampler depth_sampler;

void main() {
    target = vcolor * texture(sampler2D(color_map, color_sampler), vuv);
    vec2 duv = vdepth_uv_scale_add.xy;
    float dscale = vdepth_uv_scale_add.z;
    float dadd = vdepth_uv_scale_add.w;
    gl_FragDepth = texture(sampler2D(depth_map, depth_sampler), duv).r * dscale + dadd;
}
