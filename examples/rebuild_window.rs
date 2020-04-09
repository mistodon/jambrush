fn main() {
    use winit::{
        dpi::LogicalSize,
        event::{ElementState, VirtualKeyCode},
        event_loop::EventLoop,
        window::WindowBuilder,
    };

    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Rebuild window")
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

    let mut target_scale = None;

    event_loop.run(move |event, target, control_flow| {
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
                _ => (),
            },
            Event::MainEventsCleared => {
                if let Some(scale) = target_scale.take() {
                    let window_builder = WindowBuilder::new()
                        .with_title("JamBrush - Rebuild window")
                        .with_inner_size(LogicalSize::<u32>::from((256 * scale, 144 * scale)));

                    jambrush.rebuild_window(window_builder, &target);
                }
                jambrush.window().request_redraw();
            }
            Event::RedrawRequested(_) => {
                let mut renderer = jambrush.start_rendering([0., 0., 0., 1.], None);

                renderer.text(
                    &inconsolata,
                    14.,
                    "Rebuild window test:",
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
