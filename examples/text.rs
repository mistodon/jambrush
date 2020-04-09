fn main() {
    use std::time::Instant;
    use winit::{dpi::LogicalSize, event_loop::EventLoop, window::WindowBuilder};

    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Text")
        .with_inner_size(LogicalSize::<u32>::from((512, 512)));

    let mut jambrush = jambrush::JamBrushSystem::new(
        window_builder,
        &event_loop,
        &jambrush::JamBrushConfig {
            canvas_size: Some([512, 512]),
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

    let start_time = Instant::now();

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
                let dy = 38.;

                let mut renderer =
                    jambrush.start_rendering([0., 0., 0., 1.], Some([0.1, 0.1, 0.1, 1.]));
                renderer.text(
                    &inconsolata,
                    28.,
                    "Single line",
                    ([0., dy * 0.], [1., 1., 1., 1.]),
                );
                renderer.text(
                    &inconsolata,
                    28.,
                    "Multi-\nline\nstring",
                    ([0., dy * 1.], [0.8, 0.8, 1., 1.]),
                );

                {
                    let mut cursor: jambrush::Cursor = [0., dy * 4.].into();
                    cursor = renderer.text(
                        &inconsolata,
                        28.,
                        "Chaining ",
                        (cursor, [1., 0.5, 0.5, 1.]),
                    );
                    cursor = renderer.text(&inconsolata, 28., "text ", (cursor, [1., 1., 0.5, 1.]));
                    renderer.text(
                        &inconsolata,
                        28.,
                        "together!",
                        (cursor, [1., 0.5, 0.5, 1.]),
                    );
                }

                {
                    let mut cursor = [0., dy * 5.].into();
                    cursor = renderer.text(
                        &inconsolata,
                        28.,
                        "Chaining ",
                        (cursor, [1., 1., 1., 1.]),
                    );
                    cursor = renderer.text(
                        &inconsolata,
                        28.,
                        "multi-\nline text ",
                        (cursor, [0.5, 1., 0.5, 1.]),
                    );
                    renderer.text(
                        &inconsolata,
                        28.,
                        "together!",
                        (cursor, [1., 1., 1., 1.]),
                    );
                }

                {
                    let mut cursor = [320., dy * 5.].into();
                    cursor = renderer.text(&inconsolata, 28., "Mid", (cursor, [1., 0.5, 0.5, 1.]));
                    cursor = renderer.text(&inconsolata, 28., "Word", (cursor, [1., 1., 0.5, 1.]));
                    renderer.text(&inconsolata, 28., "Chain", (cursor, [1., 0.5, 0.5, 1.]));
                }

                {
                    let elapsed = start_time.elapsed();
                    let t = elapsed.as_secs() as f32 + elapsed.subsec_nanos() as f32 / 1_000_000_000.;
                    let x = 256. - (t * 0.5).cos() * 256.;

                    let start = [x, 256.];
                    let (end, text) = renderer.reflow_text(
                        &inconsolata,
                        28.,
                        start,
                        512.,
                        "Automatically re-flowed text",
                    );
                    renderer.text(&inconsolata, 28., &text, (start, [1., 1., 1., 1.]));

                    renderer.text(&inconsolata, 28., "!", (end, [1., 0., 0., 1.]));
                }

                {
                    let start = [0., 256. + dy * 3.];
                    let (end, text) = renderer.reflow_text(&inconsolata, 28., start, 512., "This is a block of automatically re-flowed text.\n\nIt contains explicit newlines as well");
                    renderer.text(&inconsolata, 28., &text, (start, [0.8, 0.8, 1., 1.]));

                    renderer.text(&inconsolata, 28., "!", (end, [1., 0., 0., 1.]));
                }

                renderer.finish();
            }
            _ => {}
        }
    });
}
