fn main() {
    use winit::{
        dpi::LogicalSize,
        event::{ElementState, VirtualKeyCode},
        event_loop::EventLoop,
        window::WindowBuilder,
    };

    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("JamBrush - Pillars")
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

    let player_sprite = jambrush.load_sprite_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/examples/blue_friend.png"
    ));

    let height_map_bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/examples/pillar_heights.png"
    ));

    let _height_map_sprite = jambrush.load_sprite(height_map_bytes);

    let extruded_sprite = {
        type DepthImage = image::ImageBuffer<image::Luma<u16>, Vec<u16>>;

        let height_map = image::load_from_memory(height_map_bytes).unwrap().to_rgba();
        let (w, h) = height_map.dimensions();

        let mut color_map = image::RgbaImage::new(w, h);
        let mut depth_map = DepthImage::new(w, h);

        for (x, y, px) in height_map.enumerate_pixels() {
            color_map.put_pixel(x, y, [0, 64, 0, 255].into());
            let depth = y as u16;
            depth_map.put_pixel(x, y, [depth].into());
            let height = px.0[0] as u32;
            for i in 0..height {
                let tint = if i == (height - 1) {
                    [0, 255, 0, 255]
                } else {
                    [0, 196, 0, 255]
                };
                if i <= y {
                    color_map.put_pixel(x, y - i, tint.into());
                    depth_map.put_pixel(x, y - i, [depth + 0 as u16].into());
                }
            }
        }

        jambrush.load_sprite_rgba_with_depth([w, h], &color_map, false, &depth_map)
    };

    let mut player = [0., 0.];
    let mut inputs = vec![];

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
                    if let Some(key) = input.virtual_keycode {
                        let down = input.state == ElementState::Pressed;
                        if down {
                            inputs.push(key);
                        } else {
                            inputs.retain(|&k| k != key);
                        }
                    }
                    inputs.sort();
                    inputs.dedup();
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                let speed = 2.;
                for &key in &inputs {
                    match key {
                        VirtualKeyCode::A => player[0] -= speed,
                        VirtualKeyCode::D => player[0] += speed,
                        VirtualKeyCode::W => player[1] -= speed,
                        VirtualKeyCode::S => player[1] += speed,
                        _ => (),
                    }
                }
                jambrush.window().request_redraw();
            }
            Event::RedrawRequested(_) => {
                let mut renderer =
                    jambrush.start_rendering([0., 0., 0., 1.], Some([0.1, 0.1, 0.1, 1.]));

                renderer.sprite(
                    &extruded_sprite,
                    jambrush::SpriteArgs {
                        depth_mapped: true,
                        ..Default::default()
                    },
                );
                renderer.sprite(&player_sprite, (player, player[1] + 16.));

                renderer.finish();
            }
            _ => {}
        }
    });
}
