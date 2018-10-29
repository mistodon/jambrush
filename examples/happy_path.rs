extern crate image;
extern crate jambrush;
extern crate winit;

use winit::WindowEvent;

fn main() {
    use winit::{Event, EventsLoop, WindowBuilder};

    let mut events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
        .with_title("JamBrush - Happy path")
        .with_dimensions((1280, 720).into())
        .build(&events_loop)
        .unwrap();

    let mut jambrush = jambrush::JamBrushSystem::new(&window, (256, 144));

    let ship_sprite = {
        let image_bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/examples/wee_ship.png"
        )).unwrap();

        let image = image::load_from_memory(&image_bytes).unwrap().to_rgba();
        let (w, h) = image.dimensions();

        jambrush.load_sprite([w, h], &image)
    };

    let star_sprite = {
        let image_bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/examples/star.png"
        )).unwrap();

        let image = image::load_from_memory(&image_bytes).unwrap().to_rgba();
        let (w, h) = image.dimensions();

        jambrush.load_sprite([w, h], &image)
    };

    loop {
        let mut quitting = false;

        events_loop.poll_events(|event| {
            if let Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::CloseRequested => quitting = true,
                    WindowEvent::HiDpiFactorChanged(dpi) => {
                        jambrush.dpi_factor_changed(dpi);
                    }
                    WindowEvent::Resized(res) => {
                        jambrush.window_resized(res.into());
                    }
                    _ => {}
                }
            }
        });

        if quitting {
            break;
        }

        {
            let mut renderer =
                jambrush.start_rendering([0.0, 0.0, 0.0, 1.0], Some([0.1, 0.1, 0.1, 1.0]));
            renderer.sprite(&star_sprite, [0.0, 0.0], 0.0);
            renderer.sprite(&ship_sprite, [64.0, 16.0], 0.0);
            renderer.finish();
        }
    }

    jambrush.destroy();
}
