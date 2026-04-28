#version 450

layout(location = 0) in flat uint in_material;
layout(location = 1) in flat uint in_ao;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

vec3 material_color(uint material) {
    if (material == 1) return vec3(0.42, 0.43, 0.45);
    if (material == 2) return vec3(0.43, 0.28, 0.16);
    if (material == 3) return vec3(0.22, 0.58, 0.20);
    if (material == 4) return vec3(0.10, 0.34, 0.74);
    if (material == 5) return vec3(0.78, 0.67, 0.42);
    if (material == 6) return vec3(0.36, 0.20, 0.10);
    if (material == 7) return vec3(0.13, 0.42, 0.16);
    if (material == 8) return vec3(0.88, 0.94, 0.96);
    return vec3(0.90, 0.15, 0.75);
}

float material_alpha(uint material) {
    if (material == 4) return 0.58;
    if (material == 7) return 0.68;
    return 1.0;
}

void main() {
    vec3 light_dir = normalize(vec3(0.35, 0.80, 0.45));
    float ndotl = max(dot(normalize(in_normal), light_dir), 0.20);
    float ao = 1.0 - clamp(float(in_ao) / 4.0, 0.0, 0.75);
    vec2 tile = abs(fract(in_uv) - vec2(0.5));
    float edge = 1.0 - 0.08 * step(0.47, max(tile.x, tile.y));
    out_color = vec4(material_color(in_material) * ndotl * ao * edge, material_alpha(in_material));
}
