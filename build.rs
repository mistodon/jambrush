extern crate glsl_to_spirv;

use std::error::Error;

fn main() {
    compile_shaders().unwrap();
}

fn compile_shaders() -> Result<(), Box<Error>> {
    use glsl_to_spirv::ShaderType;

    // Create destination path if necessary
    std::fs::create_dir_all("assets/compiled")?;

    for entry in std::fs::read_dir("assets")? {
        let entry = entry?;

        if entry.file_type()?.is_file() {
            let in_path = entry.path();

            // Support only vertex and fragment shaders currently
            let shader_type =
                in_path
                    .extension()
                    .and_then(|ext| match ext.to_string_lossy().as_ref() {
                        "vert" => Some(ShaderType::Vertex),
                        "frag" => Some(ShaderType::Fragment),
                        _ => None,
                    });

            if let Some(shader_type) = shader_type {
                use std::io::Read;

                println!(
                    "cargo:rerun-if-changed=assets/{}",
                    in_path.file_name().unwrap().to_string_lossy()
                );

                let source = std::fs::read_to_string(&in_path)?;
                let mut compiled_file = match glsl_to_spirv::compile(&source, shader_type) {
                    Err(s) => {
                        eprintln!("\nFAILED to compile: {}\n", in_path.display());
                        eprintln!("{}", s);
                        panic!("Shader compilation failed.");
                    }
                    Ok(result) => result,
                };

                let mut compiled_bytes = Vec::new();
                compiled_file.read_to_end(&mut compiled_bytes)?;

                let out_path = format!(
                    "assets/compiled/{}.spv",
                    in_path.file_name().unwrap().to_string_lossy()
                );

                std::fs::write(&out_path, &compiled_bytes)?;
            }
        }
    }

    Ok(())
}
