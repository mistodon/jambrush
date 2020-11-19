use std::path::PathBuf;

use image::{DynamicImage, RgbaImage};

use structopt::StructOpt;

type DepthImage = image::ImageBuffer<image::Luma<u16>, Vec<u16>>;

#[derive(Debug, StructOpt)]
enum Command {
    Shaders,
    ToDepth {
        in_file: PathBuf,

        #[structopt(short)]
        out_file: Option<PathBuf>,
    },
    FromDepth {
        in_file: PathBuf,

        #[structopt(short)]
        out_file: Option<PathBuf>,
    },
}

fn main() {
    let opt = Command::from_args();
    match opt {
        Command::ToDepth { in_file, out_file } => {
            let img_bytes = std::fs::read(in_file).unwrap();
            let img = image::load_from_memory(&img_bytes).unwrap().to_rgba8();
            let (w, h) = img.dimensions();
            let mut depth = DepthImage::new(w, h);
            for (x, y, px) in img.enumerate_pixels() {
                let value = px.0[0] as u16 * 256;
                depth.put_pixel(x, y, image::Luma::<u16>([value]));
            }
            if let Some(out_file) = out_file {
                depth.save(&out_file).unwrap();
            } else {
                use std::io::Write;

                let depth = DynamicImage::ImageLuma16(depth);

                let mut stdout = std::io::stdout();
                stdout.write_all(&depth.to_bytes()).unwrap();
            }
        }
        Command::FromDepth { in_file, out_file } => {
            let img_bytes = std::fs::read(in_file).unwrap();
            let img = image::load_from_memory(&img_bytes).unwrap();
            let img = img.as_luma16().unwrap();
            let (w, h) = img.dimensions();
            let mut color = RgbaImage::new(w, h);
            for (x, y, px) in img.enumerate_pixels() {
                let value = (px.0[0] / 256) as u8;
                color.put_pixel(x, y, image::Rgba([value, value, value, 255]));
            }
            if let Some(out_file) = out_file {
                color.save(&out_file).unwrap();
            } else {
                use std::io::Write;

                let mut stdout = std::io::stdout();
                stdout.write_all(&color).unwrap();
            }
        }
        Command::Shaders => {
            compile_shaders().unwrap();
        }
    }
}

fn compile_shaders() -> Result<(), Box<dyn std::error::Error>> {
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
