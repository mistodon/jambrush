fn main() {
    use winit::{
        dpi::LogicalSize,
        event_loop::EventLoop,
        window::WindowBuilder,
    };

    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Depth mapping")
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

    let inconsolata = {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/examples/inconsolata_bold.ttf"
        );
        jambrush.load_font_file(path)
    };

    let white_sprite = jambrush.load_sprite_file(
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/examples/white.png"),
    );
    let cube_sprite = jambrush.load_sprite_file_with_depth(
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/examples/cube.png"),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/examples/cube_depth.png"
        ),
    );
    let ball_sprite = jambrush.load_sprite_file_with_depth(
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/examples/ball.png"),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/examples/ball_depth.png"
        ),
    );

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

                let elapsed = start_time.elapsed().as_secs_f32();
                let bg_depth = (elapsed.sin() / 2. + 0.5) * 60000.;

                renderer.text(
                    &inconsolata,
                    16.,
                    &format!("BG depth: {}", bg_depth),
                    ([1., 1.], 65000., [1., 0., 0., 1.]),
                );
                renderer.sprite(
                    &cube_sprite,
                    jambrush::SpriteArgs {
                        pos: [32., 48.],
                        depth_map: Some(jambrush::DepthMapArgs {
                            depth_offset: 16000.,
                            depth_scale: Some(32000.),
                        }),
                        ..Default::default()
                    },
                );
                renderer.sprite(
                    &ball_sprite,
                    jambrush::SpriteArgs {
                        pos: [128., 48.],
                        depth_map: Some(Default::default()),
                        ..Default::default()
                    },
                );

                renderer.sprite(&white_sprite, ([0., 0.], bg_depth, [256., 144.]));

                renderer.finish();
            }
            _ => {}
        }
    });
}
