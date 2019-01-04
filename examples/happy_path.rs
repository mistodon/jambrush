extern crate image;
extern crate jambrush;
extern crate winit;

fn main() {
    use std::time::Instant;
    use winit::{Event, EventsLoop, VirtualKeyCode, WindowBuilder, WindowEvent};

    let mut events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
        .with_title("JamBrush - Happy path")
        .with_dimensions((800, 450).into())
        .build(&events_loop)
        .unwrap();

    let mut jambrush = jambrush::JamBrushSystem::new(
        &window,
        &jambrush::JamBrushConfig {
            canvas_size: Some([256, 144]),
            max_texture_atlas_size: Some(1024),
            logging: true,
            debugging: true,
        },
    );

    let ship_sprite = jambrush.load_sprite_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/examples/wee_ship.png"
    ));
    let beastie_sprite = jambrush.load_sprite_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/examples/beastie.png"
    ));
    let bullet_sprite = jambrush.load_sprite_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/examples/bullet.png"
    ));

    let inconsolata = {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/examples/inconsolata_bold.ttf"
        );
        jambrush.load_font_file(path)
    };

    let mut last_frame_time = Instant::now();
    let mut ship_arc_t = 0.0;
    let mut bullets = vec![];

    loop {
        let mut quitting = false;
        let mut shooting = false;
        let now = Instant::now();
        let dt = now.duration_since(last_frame_time).subsec_nanos() as f32 / 1_000_000_000.0;
        last_frame_time = now;

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
                    WindowEvent::KeyboardInput { input, .. } => {
                        if let Some(VirtualKeyCode::Space) = input.virtual_keycode {
                            shooting = true;
                        }
                    }
                    _ => {}
                }
            }
        });

        if quitting {
            break;
        }

        // Update
        let ship_pos: [f32; 2];

        {
            ship_arc_t += dt;
            let cos_x = (ship_arc_t * 1.5).cos();
            let sin_y = (ship_arc_t * 6.0).sin();

            let [x0, x1] = [0.0, 224.0];
            let [y0, y1] = [112.0, 80.0];
            let w = (x1 - x0) / 2.0;
            let h = (y1 - y0) / 2.0;

            ship_pos = [x0 + w + w * cos_x, y0 + h + h * sin_y];
        }

        if shooting {
            bullets.push(ship_pos);
        }

        // Render
        {
            let mut renderer =
                jambrush.start_rendering([0.0, 0.0, 0.0, 1.0], Some([0.1, 0.1, 0.1, 1.0]));
            renderer.sprite(&ship_sprite, (ship_pos, 0.0));

            for &bullet in &bullets {
                renderer.sprite(&bullet_sprite, (bullet, 1.0));
            }

            renderer.text(
                &inconsolata,
                14.0,
                &format!("Score: {}", 0),
                ([0.0, 0.0], 0.0, [1.0, 1.0, 1.0, 1.0]),
            );

            renderer.finish();
        }
    }

    jambrush.destroy();
}
