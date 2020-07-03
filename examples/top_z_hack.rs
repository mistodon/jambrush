fn main() {
    use winit::{
        dpi::LogicalSize,
        event_loop::EventLoop,
        window::WindowBuilder,
    };

    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Top Z hack")
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

    let opaque_sprite = jambrush.load_sprite_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/examples/opaque_square.png"
    ));

    let start_time = std::time::Instant::now();

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

                let t = start_time.elapsed().as_secs_f32();
                let d = t.sin() * 30000. + 30000.;

                renderer.sprite(&opaque_sprite, ([32., 24.], d, [1., 0., 0., 1.]));

                renderer.sprite(&opaque_sprite, jambrush::SpriteArgs {
                    pos: [32., 24.],
                    depth: 60000.,
                    tint: [0.5, 0., 0., 1.],
                    top_z_hack: Some(0.),
                    .. Default::default()
                });

                renderer.finish();
            }
            _ => {}
        }
    });
}
