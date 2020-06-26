fn main() {
    use winit::{
        dpi::LogicalSize,
        event::{ElementState, VirtualKeyCode},
        event_loop::EventLoop,
        window::WindowBuilder,
    };

    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Depth culling")
        .with_inner_size(LogicalSize::<u32>::from((256, 144)));

    let mut jambrush = jambrush::JamBrushSystem::new(
        window_builder,
        &event_loop,
        &jambrush::JamBrushConfig {
            canvas_size: Some([256, 144]),
            max_texture_atlas_size: Some(1024),
            logging: true,
            debugging: true,
            debug_texture_atlas: true,
        },
    );

    let sprite = jambrush.load_sprite_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/examples/transparent_square.png"
    ));

    let mut depth_toggled = false;

    event_loop.run(move |event, _, control_flow| {
        use winit::event::{Event, WindowEvent};
        use winit::event_loop::ControlFlow;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(dims) => {
                    jambrush.window_resized(dims.into());
                }
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    jambrush.dpi_factor_changed(scale_factor);
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(VirtualKeyCode::Space) = input.virtual_keycode {
                        if input.state == ElementState::Pressed {
                            depth_toggled = !depth_toggled;
                        }
                    }
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                jambrush.window().request_redraw();
            }
            Event::RedrawRequested(_) => {
                let mut renderer =
                    jambrush.start_rendering([0., 0., 0., 1.], Some([0.1, 0.1, 0.1, 1.]));
                let [d0, d1, d2] = match depth_toggled {
                    true => [1., 2., 3.],
                    false => [3., 2., 1.],
                };

                renderer.sprite(&sprite, ([16., 16.], d0, [1., 0., 0., 1.]));
                renderer.sprite(&sprite, ([48., 48.], d1, [0., 1., 0., 1.]));
                renderer.sprite(&sprite, ([72., 72.], d2, [0., 0., 1., 1.]));

                renderer.finish();
            }
            _ => {}
        }
    });
}
