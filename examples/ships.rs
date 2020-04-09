fn main() {
    use std::time::Instant;
    use winit::{
        dpi::LogicalSize, event::VirtualKeyCode, event_loop::EventLoop, window::WindowBuilder,
    };

    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Ships")
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

    let ship_sprite = jambrush.load_sprite_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/examples/wee_ship.png"
    ));
    let _beastie_sprite = jambrush.load_sprite_file(concat!(
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
    let mut ship_arc_t = 0.;
    let mut bullets = vec![];
    let mut shooting = false;
    let mut ship_pos = [0., 0.];

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
                        shooting = true;
                    }
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                let now = Instant::now();
                let dt = now.duration_since(last_frame_time).subsec_nanos() as f32 / 1_000_000_000.;
                last_frame_time = now;

                // Update
                {
                    ship_arc_t += dt;
                    let cos_x = (ship_arc_t * 1.5).cos();
                    let sin_y = (ship_arc_t * 6.).sin();

                    let [x0, x1] = [0., 224.];
                    let [y0, y1] = [112., 80.];
                    let w = (x1 - x0) / 2.;
                    let h = (y1 - y0) / 2.;

                    ship_pos = [x0 + w + w * cos_x, y0 + h + h * sin_y];
                }

                if shooting {
                    bullets.push(ship_pos);
                }

                jambrush.window().request_redraw();
                shooting = false;
            }
            Event::RedrawRequested(_) => {
                // Render
                let stats = jambrush.render_stats();
                let stats = format!("{:#?}", stats);
                let stats = stats.lines();
                {
                    let mut renderer =
                        jambrush.start_rendering([0., 0., 0., 1.], Some([0.1, 0.1, 0.1, 1.]));
                    renderer.sprite(&ship_sprite, (ship_pos, 0.));

                    for &bullet in &bullets {
                        renderer.sprite(&bullet_sprite, (bullet, 1.));
                    }

                    renderer.text(
                        &inconsolata,
                        14.,
                        &format!("Score: {}", 0),
                        ([0., 0.], 0., [1., 1., 1., 1.]),
                    );

                    for (i, line) in stats.enumerate() {
                        renderer.text(
                            &inconsolata,
                            14.,
                            line,
                            ([0., 15. + 15. * i as f32], 0., [1., 1., 0., 1.]),
                        );
                    }

                    renderer.finish();
                }
            }
            _ => {}
        }
    });
}
