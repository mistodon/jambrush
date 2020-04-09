fn main() {
    use winit::{dpi::LogicalSize, event_loop::EventLoop, window::WindowBuilder};

    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Gamma test")
        .with_inner_size(LogicalSize::<u32>::from((256, 144_u32)));

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
                _ => (),
            },
            Event::MainEventsCleared => {
                jambrush.window().request_redraw();
            }
            Event::RedrawRequested(_) => {
                let mut renderer =
                    jambrush.start_rendering([0., 0., 0., 1.], Some([0.1, 0.1, 0.1, 1.]));

                renderer.text(
                    &inconsolata,
                    14.,
                    "Colors in texture:",
                    ([2., 2.], 0., [1., 1., 1., 1.]),
                );

                renderer.sprite(&chart_sprite, [0., 32.]);

                renderer.text(
                    &inconsolata,
                    14.,
                    "Colors via tint:",
                    ([2., 72.], 0., [1., 1., 1., 1.]),
                );

                for i in 0..8 {
                    let green = (i + 1) as f32 / 8.;
                    let x = i as f32 * 32.;

                    renderer.sprite(&white_sprite, ([x, 104.], [0., green, 0., 1.]));
                }

                renderer.finish();
            }
            _ => {}
        }
    });
}
