extern crate image;
extern crate jambrush;
extern crate winit;

use winit::WindowEvent;

fn main() {
    use winit::{Event, EventsLoop, WindowBuilder};

    let mut events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
        .with_title("JamBrush - Happy path")
        .with_dimensions((800, 450).into())
        .build(&events_loop)
        .unwrap();

    let mut jambrush = jambrush::JamBrushSystem::new(
        &window,
        &jambrush::JamBrushConfig {
            canvas_resolution: Some([256, 144]), // TODO: canvas_size ?
            max_texture_atlas_size: Some(1024),
            logging: true,
            debugging: true,
        },
    );

    let ship_sprite = {
        let image_bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/examples/wee_ship.png"
        ))
        .unwrap();

        jambrush.load_sprite(&image_bytes)
    };

    let star_sprite = {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/examples/star.png");
        jambrush.load_sprite_file(path)
    };

    let inconsolata = {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/examples/inconsolata_bold.ttf"
        );
        jambrush.load_font_file(path)
    };

    let spicyrice = {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/examples/spicyrice.ttf");
        jambrush.load_font_file(path)
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
            renderer.sprite(&star_sprite, ([0.0, 0.0], 0.0));
            renderer.sprite(&ship_sprite, ([64.0, 16.0], 0.0));

            renderer.text(
                &inconsolata,
                "Hello\nlittle\nspaceship\ngame",
                ([-1.0, 0.0], 14.0, 0.0, [0.0, 0.0, 0.5, 1.0]),
            );
            renderer.text(
                &inconsolata,
                "Hello\nlittle\nspaceship\ngame",
                ([1.0, 0.0], 14.0, 0.0, [0.0, 0.0, 0.5, 1.0]),
            );
            renderer.text(
                &inconsolata,
                "Hello\nlittle\nspaceship\ngame",
                ([0.0, -1.0], 14.0, 0.0, [0.0, 0.0, 0.5, 1.0]),
            );
            renderer.text(
                &inconsolata,
                "Hello\nlittle\nspaceship\ngame",
                ([0.0, 1.0], 14.0, 0.0, [0.0, 0.0, 0.5, 1.0]),
            );
            renderer.text(
                &inconsolata,
                "Hello\nlittle\nspaceship\ngame",
                ([0.0, 0.0], 14.0, 0.0, [1.0, 1.0, 1.0, 1.0]),
            );

            renderer.text(
                &spicyrice,
                "Some bigger text",
                ([40.0, 20.0], 43.0, 0.0, [1.0, 1.0, 1.0, 1.0]),
            );

            renderer.finish();
        }
    }

    jambrush.destroy();
}
