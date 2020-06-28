use std::path::PathBuf;

use image::{DynamicImage, RgbaImage};
use structopt::StructOpt;

type DepthImage = image::ImageBuffer<image::Luma<u16>, Vec<u16>>;

#[derive(Debug, StructOpt)]
enum Command {
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
            let img = image::load_from_memory(&img_bytes).unwrap().to_rgba();
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
    }
}
