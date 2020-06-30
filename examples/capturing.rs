fn main() {
    use winit::{
        dpi::LogicalSize,
        event::{ElementState, VirtualKeyCode},
        event_loop::EventLoop,
        window::WindowBuilder,
    };

    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Capturing")
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
                    if input.state == ElementState::Pressed {
                        if let Some(VirtualKeyCode::Key1) = input.virtual_keycode {
                            jambrush.capture_to_file(jambrush::Capture::Canvas, "capturing_canvas.png");
                        } else if let Some(VirtualKeyCode::Key2) = input.virtual_keycode {
                            jambrush.capture_to_file(jambrush::Capture::TextureAtlas, "capturing_texture_atlas.png");
                        } else if let Some(VirtualKeyCode::Key3) = input.virtual_keycode {
                            jambrush.capture_to_file(jambrush::Capture::DepthTextureAtlas, "capturing_depth_texture_atlas.png");
                        } else if let Some(VirtualKeyCode::Key4) = input.virtual_keycode {
                            jambrush.capture_to_file(jambrush::Capture::DepthBuffer, "capturing_depth_buffer.png");
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

                renderer.text(
                    &inconsolata,
                    16.,
                    "Press:\n[1] Screenshot\n[2] Texture atlas\n[3] Depth texture atlas\n[4] Depth buffer",
                    [0., 0.],
                );
                renderer.sprite(&cube_sprite, [0., 60.]);
                renderer.sprite(&ball_sprite, jambrush::SpriteArgs {
                    pos: [96., 60.],
                    depth_mapped: true,
                    .. Default::default()
                });
                renderer.finish();
            }
            _ => {}
        }
    });
}
