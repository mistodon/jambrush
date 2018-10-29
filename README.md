jambrush
===

A Rust drawing library that gets out of your way when you need it to.

Goals
---

1.  No non-rust runtime dependencies
2.  Very quick to get up and running
3.  Simple rendering of 2D sprites, shapes, and text
4.  Performant enough
5.  Immediate-mode
6.  Multi-layered so it's easy to incrementally replace with more sophisticated code

Non-goals
---

1.  A game engine
2.  3D rendering (at least not currently)
3.  Very high performance

Multiple levels of usage
---

1.  Manage the entire rendering context
2.  Manage just the swapchain, pipeline state, and rendering
3.  Just the swapchain, or just the pipeline state, or just the rendering

Should be able to gracefully phase out this library moving down the list. You can prototype using it, and when you need to improve the rendering code, incrementally remove it.

Features
---

1.  Rendering:
    - Sprites
    - 2D shapes
    - Text
2.  Resource management:
    - Load sprites at runtime (into an atlas)
        - Either from file
        - Or from data
    - Build-time function to pre-generate atlas
        - Again, multi-layered. Can be opinionated _or_ flexible.
3.  Rendering utilities:
    - Camera offset
4.  Screen effects:
    - Allow rendering to texture
    - Allow post-processing that image
    - Allow for pixel-perfect scaling
5.  Cross-platform:
    - Obviously works with multiple backends

Notes
---

If I'm going to make this, I need a good testing environment for all backends.
