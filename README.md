jambrush
===

A Rust drawing library that gets out of your way when you need it to.

Multiple levels of usage
---

1.  Manage the entire rendering context
2.  Manage enough render state to exist alongside other things

Should be able to gracefully phase out this library moving down the list.

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
