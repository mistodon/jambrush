extern crate image;
extern crate jambrush;
extern crate winit;

fn main() {
    use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

    let mut events_loop = EventsLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Gamma test")
        .with_dimensions((256, 144).into());

    let mut jambrush = jambrush::JamBrushSystem::new(
        window_builder,
        &events_loop,
        &jambrush::JamBrushConfig {
            canvas_size: Some([256, 144]),
            max_texture_atlas_size: Some(1024),
            logging: true,
            debugging: true,
            debug_texture_atlas: true,
        },
    );

    let white_sprite = jambrush.load_sprite_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/examples/white.png"
    ));
    let chart_sprite = jambrush.load_sprite_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/examples/gamma_test.png"
    ));

    let inconsolata = {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/examples/inconsolata_bold.ttf"
        );
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

        // Render
        {
            let mut renderer = jambrush.start_rendering([0.0, 0.0, 0.0, 1.0], Some([0.1, 0.1, 0.1, 1.0]));

            renderer.text(
                &inconsolata,
                14.0,
                "Colors in texture:",
                ([2.0, 2.0], 0.0, [1.0, 1.0, 1.0, 1.0]),
            );

            renderer.sprite(&chart_sprite, [0.0, 32.0]);

            renderer.text(
                &inconsolata,
                14.0,
                "Colors via tint:",
                ([2.0, 72.0], 0.0, [1.0, 1.0, 1.0, 1.0]),
            );

            for i in 0..8 {
                let green = (i + 1) as f32 / 8.0;
                let x = i as f32 * 32.0;

                renderer.sprite(&white_sprite, ([x, 104.0], [0.0, green, 0.0, 1.0]));
            }

            renderer.finish();
        }
    }

    jambrush.destroy();
}
