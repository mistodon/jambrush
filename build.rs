use std::error::Error;

fn main() {
    compile_shaders().unwrap();
}

fn compile_shaders() -> Result<(), Box<dyn Error>> {
    let mut compiler = shaderc::Compiler::new().unwrap();
    let options = shaderc::CompileOptions::new().unwrap();

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
                        "vert" => Some(shaderc::ShaderKind::Vertex),
                        "frag" => Some(shaderc::ShaderKind::Fragment),
                        _ => None,
                    });

            if let Some(shader_type) = shader_type {
                let file_name = in_path.file_name().unwrap().to_string_lossy();

                println!("cargo:rerun-if-changed=assets/{}", file_name);

                let source = std::fs::read_to_string(&in_path)?;
                let binary_result = compiler.compile_into_spirv(
                    &source,
                    shader_type,
                    &file_name,
                    "main",
                    Some(&options),
                );

                let compiled_bytes = match binary_result {
                    Err(s) => {
                        eprintln!("\nFAILED to compile: {}\n", in_path.display());
                        eprintln!("{}", s);
                        panic!("Shader compilation failed.");
                    }
                    Ok(result) => result,
                };

                let out_path = format!(
                    "assets/compiled/{}.spv",
                    in_path.file_name().unwrap().to_string_lossy()
                );

                std::fs::write(&out_path, compiled_bytes.as_binary_u8())?;
            }
        }
    }

    Ok(())
}
