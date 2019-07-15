extern crate image;
extern crate jambrush;
extern crate winit;

fn main() {
    use winit::{ElementState, Event, EventsLoop, VirtualKeyCode, WindowBuilder, WindowEvent};

    let mut events_loop = EventsLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Rebuild window")
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

    let inconsolata = {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/examples/inconsolata_bold.ttf"
        );
        jambrush.load_font_file(path)
    };

    loop {
        let mut quitting = false;
        let mut target_scale = None;

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
                    WindowEvent::KeyboardInput { input, .. }
                        if input.state == ElementState::Pressed =>
                    {
                        match input.virtual_keycode {
                            Some(VirtualKeyCode::Key1) => target_scale = Some(1),
                            Some(VirtualKeyCode::Key2) => target_scale = Some(2),
                            Some(VirtualKeyCode::Key3) => target_scale = Some(3),
                            _ => (),
                        }
                    }
                    _ => {}
                }
            }
        });

        if let Some(scale) = target_scale {
            let window_builder = WindowBuilder::new()
                .with_title("JamBrush - Rebuild window")
                .with_dimensions((256 * scale, 144 * scale).into());

            jambrush.rebuild_window(window_builder, &events_loop);
        }

        if quitting {
            break;
        }

        // Render
        {
            let mut renderer = jambrush.start_rendering([0.0, 0.0, 0.0, 1.0], None);

            renderer.text(
                &inconsolata,
                14.0,
                "Rebuild window test:",
                ([2.0, 72.0], 0.0, [1.0, 1.0, 1.0, 1.0]),
            );

            renderer.text(
                &inconsolata,
                14.0,
                "Press 1, 2, or 3 to resize window.",
                ([2.0, 90.0], 0.0, [1.0, 1.0, 1.0, 1.0]),
            );

            renderer.sprite(&white_sprite, ([64.0, 64.0], [0.5, 0.5, 0.5, 1.0]));

            renderer.finish();
        }
    }

    jambrush.destroy();
}
