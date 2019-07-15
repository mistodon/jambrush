extern crate image;
extern crate jambrush;
extern crate winit;

use winit::{Event, EventsLoop, VirtualKeyCode, WindowBuilder, WindowEvent};

struct Context {
    events_loop: EventsLoop,
    jambrush: jambrush::JamBrushSystem,
    white_sprite: jambrush::Sprite,
    inconsolata: jambrush::Font,
}

fn main() {
    let mut context: Option<Context> = None;
    let mut scale = 1;
    let mut target_scale = 1;

    loop {
        if (scale != target_scale) || context.is_none() {
            scale = target_scale;

            if let Some(Context { jambrush, .. }) = context.take() {
                jambrush.destroy();
            }

            let window_builder = WindowBuilder::new()
                .with_title("JamBrush - Full Recreate")
                .with_dimensions((256 * scale, 144 * scale).into());

            let events_loop = EventsLoop::new();

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

            context = Some(Context {
                events_loop,
                jambrush,
                white_sprite,
                inconsolata,
            });
        }

        let mut quitting = false;

        let Context {
            events_loop,
            jambrush,
            white_sprite,
            inconsolata,
        } = context.as_mut().unwrap();

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
                    WindowEvent::KeyboardInput { input, .. } => match input.virtual_keycode {
                        Some(VirtualKeyCode::Key1) => target_scale = 1,
                        Some(VirtualKeyCode::Key2) => target_scale = 2,
                        Some(VirtualKeyCode::Key3) => target_scale = 3,
                        _ => (),
                    },
                    _ => {}
                }
            }
        });

        if quitting {
            break;
        }

        // Render
        {
            let mut renderer = jambrush.start_rendering([0.0, 0.0, 0.0, 1.0], None);

            renderer.text(
                &inconsolata,
                14.0,
                "Full window recreation test:",
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

    let jambrush = context.unwrap().jambrush;
    jambrush.destroy();
}
