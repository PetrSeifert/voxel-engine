use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let shader_dir = PathBuf::from("shaders");
    let shaders = [
        ("chunk.vert", shaderc::ShaderKind::Vertex),
        ("chunk.frag", shaderc::ShaderKind::Fragment),
        ("overlay.vert", shaderc::ShaderKind::Vertex),
        ("overlay.frag", shaderc::ShaderKind::Fragment),
    ];

    let compiler = shaderc::Compiler::new().expect("create shaderc compiler");
    let mut options = shaderc::CompileOptions::new().expect("create shaderc options");
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    for (name, kind) in shaders {
        let path = shader_dir.join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        let source = fs::read_to_string(&path).expect("read shader source");
        let artifact = compiler
            .compile_into_spirv(
                &source,
                kind,
                path.to_str().unwrap(),
                "main",
                Some(&options),
            )
            .expect("compile shader");
        fs::write(out_dir.join(format!("{name}.spv")), artifact.as_binary_u8())
            .expect("write shader spv");
    }
}
