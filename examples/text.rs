extern crate image;
extern crate jambrush;
extern crate winit;

fn main() {
    use std::time::Instant;
    use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

    let mut events_loop = EventsLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Text")
        .with_dimensions((512, 512).into());

    let mut jambrush = jambrush::JamBrushSystem::new(
        window_builder,
        &events_loop,
        &jambrush::JamBrushConfig {
            canvas_size: Some([512, 512]),
            max_texture_atlas_size: Some(1024),
            logging: true,
            debugging: true,
            debug_texture_atlas: true,
        },
    );

    let inconsolata = {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/examples/inconsolata_bold.ttf"
        );
        jambrush.load_font_file(path)
    };

    let start_time = Instant::now();

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
            let dy = 38.0;

            let mut renderer =
                jambrush.start_rendering([0.0, 0.0, 0.0, 1.0], Some([0.1, 0.1, 0.1, 1.0]));
            renderer.text(
                &inconsolata,
                28.0,
                "Single line",
                ([0.0, dy * 0.0], [1.0, 1.0, 1.0, 1.0]),
            );
            renderer.text(
                &inconsolata,
                28.0,
                "Multi-\nline\nstring",
                ([0.0, dy * 1.0], [0.8, 0.8, 1.0, 1.0]),
            );

            {
                let mut cursor: jambrush::Cursor = [0.0, dy * 4.0].into();
                cursor = renderer.text(
                    &inconsolata,
                    28.0,
                    "Chaining ",
                    (cursor, [1.0, 0.5, 0.5, 1.0]),
                );
                cursor = renderer.text(&inconsolata, 28.0, "text ", (cursor, [1.0, 1.0, 0.5, 1.0]));
                renderer.text(
                    &inconsolata,
                    28.0,
                    "together!",
                    (cursor, [1.0, 0.5, 0.5, 1.0]),
                );
            }

            {
                let mut cursor = [0.0, dy * 5.0].into();
                cursor = renderer.text(
                    &inconsolata,
                    28.0,
                    "Chaining ",
                    (cursor, [1.0, 1.0, 1.0, 1.0]),
                );
                cursor = renderer.text(
                    &inconsolata,
                    28.0,
                    "multi-\nline text ",
                    (cursor, [0.5, 1.0, 0.5, 1.0]),
                );
                renderer.text(
                    &inconsolata,
                    28.0,
                    "together!",
                    (cursor, [1.0, 1.0, 1.0, 1.0]),
                );
            }

            {
                let mut cursor = [320.0, dy * 5.0].into();
                cursor = renderer.text(&inconsolata, 28.0, "Mid", (cursor, [1.0, 0.5, 0.5, 1.0]));
                cursor = renderer.text(&inconsolata, 28.0, "Word", (cursor, [1.0, 1.0, 0.5, 1.0]));
                renderer.text(&inconsolata, 28.0, "Chain", (cursor, [1.0, 0.5, 0.5, 1.0]));
            }

            {
                let elapsed = start_time.elapsed();
                let t = elapsed.as_secs() as f32 + elapsed.subsec_nanos() as f32 / 1_000_000_000.0;
                let x = 256.0 - (t * 0.5).cos() * 256.0;

                let start = [x, 256.0];
                let (end, text) = renderer.reflow_text(
                    &inconsolata,
                    28.0,
                    start,
                    512.0,
                    "Automatically re-flowed text",
                );
                renderer.text(&inconsolata, 28.0, &text, (start, [1.0, 1.0, 1.0, 1.0]));

                renderer.text(&inconsolata, 28.0, "!", (end, [1.0, 0.0, 0.0, 1.0]));
            }

            {
                let start = [0.0, 256.0 + dy * 3.0];
                let (end, text) = renderer.reflow_text(&inconsolata, 28.0, start, 512.0, "This is a block of automatically re-flowed text.\n\nIt contains explicit newlines as well");
                renderer.text(&inconsolata, 28.0, &text, (start, [0.8, 0.8, 1.0, 1.0]));

                renderer.text(&inconsolata, 28.0, "!", (end, [1.0, 0.0, 0.0, 1.0]));
            }

            renderer.finish();
        }
    }

    jambrush.destroy();
}
