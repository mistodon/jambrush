use winit::{
    dpi::LogicalSize, event::VirtualKeyCode, event_loop::EventLoop, window::WindowBuilder,
};

struct Context {
    jambrush: jambrush::JamBrushSystem,
    white_sprite: jambrush::Sprite,
    inconsolata: jambrush::Font,
}

fn main() {
    let mut context: Option<Context> = None;
    let mut scale = 2;
    let mut target_scale = 2;

    let event_loop = EventLoop::new();

    event_loop.run(move |event, target, control_flow| {
        use winit::event::{Event, WindowEvent};
        use winit::event_loop::ControlFlow;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(dims) => {
                    if let Some(Context { jambrush, .. }) = &mut context {
                        jambrush.window_resized(dims.into());
                    }
                }
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    if let Some(Context { jambrush, .. }) = &mut context {
                        jambrush.dpi_factor_changed(scale_factor);
                    }
                }
                WindowEvent::KeyboardInput { input, .. } => match input.virtual_keycode {
                    Some(VirtualKeyCode::Key1) => target_scale = 1,
                    Some(VirtualKeyCode::Key2) => target_scale = 2,
                    Some(VirtualKeyCode::Key3) => target_scale = 3,
                    _ => (),
                },
                _ => (),
            },
            Event::MainEventsCleared => {
                if (scale != target_scale) || context.is_none() {
                    scale = target_scale;

                    context.take();

                    let window_builder = WindowBuilder::new()
                        .with_title("JamBrush - Full Recreate")
                        .with_inner_size(LogicalSize::<u32>::from((256 * scale, 144 * scale)));

                    let mut jambrush = jambrush::JamBrushSystem::new(
                        window_builder,
                        &target,
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
                        jambrush,
                        white_sprite,
                        inconsolata,
                    });
                }

                let Context { jambrush, .. } = context.as_ref().unwrap();
                jambrush.window().request_redraw();
            }
            Event::RedrawRequested(_) => {
                let Context {
                    jambrush,
                    white_sprite,
                    inconsolata,
                } = context.as_mut().unwrap();

                let mut renderer =
                    jambrush.start_rendering([0., 0., 0., 1.], Some([0.5, 0.5, 0.5, 1.]));

                renderer.text(
                    &inconsolata,
                    14.,
                    "Full window recreation test:",
                    ([2., 72.], 0., [1., 1., 1., 1.]),
                );

                renderer.text(
                    &inconsolata,
                    14.,
                    "Press 1, 2, or 3 to resize window.",
                    ([2., 90.], 0., [1., 1., 1., 1.]),
                );

                renderer.sprite(&white_sprite, ([64., 64.], [0.5, 0.5, 0.5, 1.]));

                renderer.finish();
            }
            _ => {}
        }
    });
}
