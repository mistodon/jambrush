#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 vcolor;
layout(location = 1) in vec2 vuv;
layout(location = 2) in vec2 vdepth_scale_add;

layout(location = 0) out vec4 target;

layout(set = 0, binding = 0) uniform texture2D color_map;
layout(set = 0, binding = 1) uniform sampler color_sampler;
layout(set = 1, binding = 0) uniform utexture2D depth_map;
layout(set = 1, binding = 1) uniform sampler depth_sampler;

void main() {
    target = vcolor * texture(sampler2D(color_map, color_sampler), vuv);
    float dscale = vdepth_scale_add.x;
    float dadd = vdepth_scale_add.y;
    uint depth = texture(usampler2D(depth_map, depth_sampler), vuv).r;
    gl_FragDepth = 1.0 - (float(depth) * (0.5 / 65535.0) * dscale + dadd);
}
