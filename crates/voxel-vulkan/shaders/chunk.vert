#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in uint in_material;
layout(location = 4) in uint in_ao;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    vec4 chunk_origin;
} pc;

layout(location = 0) out flat uint out_material;
layout(location = 1) out flat uint out_ao;
layout(location = 2) out vec3 out_normal;
layout(location = 3) out vec2 out_uv;

void main() {
    vec3 world_position = in_position + pc.chunk_origin.xyz;
    gl_Position = pc.view_proj * vec4(world_position, 1.0);
    out_material = in_material;
    out_ao = in_ao;
    out_normal = in_normal;
    out_uv = in_uv;
}
